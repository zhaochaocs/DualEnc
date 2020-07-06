import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda

import re

import onmt
import onmt.io
from onmt.Utils import aeq


class CopyGenerator(nn.Module):
    """Generator module that additionally considers copying
    words directly from the source.

    The main idea is that we have an extended "dynamic dictionary".
    It contains `|tgt_dict|` words plus an arbitrary number of
    additional words introduced by the source sentence.
    For each source sentence we have a `src_map` that maps
    each source word to an index in `tgt_dict` if it known, or
    else to an extra word.

    The copy generator is an extended version of the standard
    generator that computse three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of instead copying a
      word from the source, computed using a bernoulli
    * :math:`p_{copy}` the probility of copying a word instead.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       tgt_dict (Vocab): output target dictionary

    """
    def __init__(self, input_size, tgt_dict):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, len(tgt_dict))
        self.linear_copy = nn.Linear(input_size, 1)
        self.linear_copy2 = nn.Linear(input_size, 3)
        self.tgt_dict = tgt_dict

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by compying
        source words.

        Args:
           hidden (`FloatTensor`): hidden outputs `[batch*tlen, input_size]`
           attn (`FloatTensor`): attn for each `[batch*tlen, input_size]`
           src_map (`FloatTensor`):
             A sparse indicator matrix mapping each source word to
             its index in the "extended" vocab containing.
             `[src_len, batch, extra_words]`
        """
        # CHECKS
        batch_by_tlen, _ = hidden.size()        #T*B, E
        if isinstance(attn, tuple):
            attn, attn2 = attn
            src_map, src_map2 = src_map
            batch_by_tlen_, slen = attn.size()  # T*B, T_src0
            batch_by_tlen_2, slen2 = attn2.size()  # T*B, T_src1
            slen_, batch, cvocab = src_map.size()
            slen_2, batch2, cvocab2 = src_map2.size()
            aeq(batch_by_tlen, batch_by_tlen_, batch_by_tlen_2)
            aeq(slen, slen_)
            aeq(slen2, slen_2)
        else:
            batch_by_tlen_, slen = attn.size()  #T*B, T_src
            slen_, batch, cvocab = src_map.size()
            aeq(batch_by_tlen, batch_by_tlen_)
            aeq(slen, slen_)
            attn2 = None

        # Original probabilities.
        logits = self.linear(hidden)    #T*B, V_tgt_size
        logits[:, self.tgt_dict.stoi[onmt.io.PAD_WORD]] = -float('inf')
        if all([re.match(r"(AGENT|PATIENT|BRIDGE)-\d", token, flags=0) is not None for token in self.tgt_dict.itos[4:25]]):
            logits[:, 4:25] = -float('inf')
        prob = F.softmax(logits, dim=1)     #T*B, V_tgt_size

        if attn2 is None:
            # Probability of copying p(z=1) batch.
            p_copy = F.sigmoid(self.linear_copy(hidden))#T*B, 1
            # Probibility of not copying: p_{word}(w) * (1 - p(z))
            out_prob = torch.mul(prob,  1 - p_copy.expand_as(prob))#T*B, V_tgt_size

            mul_attn = torch.mul(attn, p_copy.expand_as(attn))#T*B, T_src
            copy_prob = torch.bmm(mul_attn.view(-1, batch, slen)
                                  .transpose(0, 1),
                                  src_map.transpose(0, 1)).transpose(0, 1)
            copy_prob = copy_prob.contiguous().view(-1, cvocab)#T*B, V_src_size(extend)

            return torch.cat([out_prob, copy_prob], 1), 0
        else:
            # Probability of copying p(z=1) batch.
            p_copy = F.softmax(self.linear_copy2(hidden), dim=1)  # T*B, 3
            # Probibility of not copying: p_{word}(w) * (1 - p(z))
            out_prob = torch.mul(prob, p_copy[:,0].unsqueeze(1).expand_as(prob))  # T*B, V_tgt_size
            # Probability of copying from src1
            mul_attn = torch.mul(attn, p_copy[:,1].unsqueeze(1).expand_as(attn))  # T*B, T_src
            copy_prob = torch.bmm(mul_attn.view(-1, batch, slen)
                                  .transpose(0, 1),
                                  src_map.transpose(0, 1)).transpose(0, 1)
            copy_prob = copy_prob.contiguous().view(-1, cvocab)  # T*B, V_src_size(extend)
            # Probability of copying from src2
            mul_attn2 = torch.mul(attn2, p_copy[:,2].unsqueeze(1).expand_as(attn2))  # T*B, T_src2
            copy_prob2 = torch.bmm(mul_attn2.view(-1, batch, slen2)
                                  .transpose(0, 1),
                                  src_map2.transpose(0, 1)).transpose(0, 1)
            copy_prob2 = copy_prob2.contiguous().view(-1, cvocab2)  # T*B, V_src_size(extend)
            return torch.cat([out_prob, copy_prob, copy_prob2], 1), cvocab  # return the vocab_size of the first src


class CopyGeneratorCriterion(object):
    def __init__(self, vocab_size, force_copy, pad, eps=1e-20):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = vocab_size
        self.offset2 = 0
        self.pad = pad

    def __call__(self, scores, align, target):
        if isinstance(align, tuple):
            assert self.offset2 > 0
            align, align2 = align
        else:
            align2 = None

        # Compute unks in align and target for readability              scores: B*T, |V|
        align_unk = align.eq(0).float()
        align_not_unk = align.ne(0).float()
        target_unk = target.eq(0).float()
        target_not_unk = target.ne(0).float()

        # Copy probability of tokens in source
        out = scores.gather(1, align.view(-1, 1) + self.offset).view(-1)
        # Set scores for unk to 0 and add eps
        out = out.mul(align_not_unk) + self.eps
        # Get scores for tokens in target
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)

        if align2 is not None:
            align_unk2 = align2.eq(0).float()
            align_not_unk2 = align2.ne(0).float()
            # Copy probability of tokens in source
            out2 = scores.gather(1, align2.view(-1, 1) + self.offset + self.offset2).view(-1)
            # Set scores for unk to 0 and add eps
            out2 = out2.mul(align_not_unk2) + self.eps


        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            if align2 is None:
                # Add score for non-unks in target
                out = out + tmp.mul(target_not_unk)
                # Add score for when word is unk in both align and tgt
                out = out + tmp.mul(align_unk).mul(target_unk)
            else:
                # Add score for non-unks in target
                out = out + out2 + tmp.mul(target_not_unk)
                # Add score for when word is unk in both align and tgt
                out = out + tmp.mul(align_unk).mul(target_unk).mul(align_unk2)
        else:
            if align2 is None:
                # Forced copy. Add only probability for not-copied tokens
                out = out + tmp.mul(align_unk)
            else:
                out = out + out2 + tmp.mul(align_unk).mul(align_unk2)

        # Drop padding.
        loss = -out.log().mul(target.ne(self.pad).float())
        return loss


class CopyGeneratorLossCompute(onmt.Loss.LossComputeBase):
    """
    Copy Generator Loss Computation.
    """
    def __init__(self, generator, tgt_vocab,
                 force_copy, normalize_by_length,
                 eps=1e-20):
        super(CopyGeneratorLossCompute, self).__init__(
            generator, tgt_vocab)

        # We lazily load datasets when there are more than one, so postpone
        # the setting of cur_dataset.
        self.cur_dataset = None
        self.force_copy = force_copy
        self.normalize_by_length = normalize_by_length
        self.criterion = CopyGeneratorCriterion(len(tgt_vocab), force_copy,
                                                self.padding_idx)

    def _make_shard_state(self, batch, output, range_, attns):
        """ See base class for args description. """
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        if not len(attns.get("copy2")):
            return {
                "output": output,
                "target": batch.tgt[range_[0] + 1: range_[1]],
                "copy_attn": attns.get("copy"),
                "align": batch.alignment[range_[0] + 1: range_[1]]
            }
        else:
            return {
                "output": output,
                "target": batch.tgt[range_[0] + 1: range_[1]],
                "copy_attn": attns.get("copy"),
                "copy_attn2": attns.get("copy2"),
                "align": batch.alignment[range_[0] + 1: range_[1]],
                "align2": batch.alignment2[range_[0] + 1: range_[1]],
            }



    def _compute_loss(self, batch, output, target, copy_attn, align, copy_attn2=None, align2=None):
        """
        Compute the loss. The args must match self._make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = target.view(-1)  # Tt * B -> -1
        if copy_attn2 is not None:
            align = (align.view(-1), align2.view(-1))
            scores, offset2 = self.generator(self._bottle(output),  # Tt * B * E
                                    (self._bottle(copy_attn), self._bottle(copy_attn2)),  # Tt * B * Ts
                                    (batch.src_map, batch.src_map2))  # Ts * B * V_src_size
            self.criterion.offset2 = offset2
        else:
            align = align.view(-1)        # Tt * B -> -1
            scores, _ = self.generator(self._bottle(output),  # Tt * B * E
                                    self._bottle(copy_attn),  # Tt * B * Ts
                                    batch.src_map)  # Ts * B * V_src_size

        loss = self.criterion(scores, align, target)
        scores_data = scores.data.clone()
        if copy_attn2 is None:
            scores_data = onmt.io.TextDataset.collapse_copy_scores2(
                        self._unbottle(scores_data, batch.batch_size),
                        batch, self.tgt_vocab, self.cur_dataset.src_vocabs)
        else:
            scores_data = onmt.io.TextDataset.collapse_copy_scores2(
                self._unbottle(scores_data, batch.batch_size),
                batch, self.tgt_vocab, (self.cur_dataset.src_vocabs, self.cur_dataset.src_vocabs2), offset2)

        scores_data = self._bottle(scores_data)

        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.data.clone()


        if copy_attn2 is not None:
            align, align2 = align
            correct_mask = target_data.eq(0) * align.data.ne(0)
            correct_copy = (align.data + len(self.tgt_vocab)) * correct_mask.long()
            target_data = target_data + correct_copy

            correct_mask2 = target_data.eq(0) * align.data.eq(0) * align2.data.ne(0)
            correct_copy2 = (align2.data + len(self.tgt_vocab) + offset2) * correct_mask2.long()
            target_data = target_data + correct_copy2

        else:
            correct_mask = target_data.eq(0) * align.data.ne(0)
            correct_copy = (align.data + len(self.tgt_vocab)) * correct_mask.long()
            target_data = target_data + correct_copy


        # Compute sum of perplexities for stats
        loss_data = loss.sum().data.clone()
        stats = self._stats(loss_data, scores_data, target_data)

        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            # Compute Sequence Lengths
            pad_ix = batch.dataset.fields['tgt'].vocab.stoi[onmt.io.PAD_WORD]
            tgt_lens = batch.tgt.ne(pad_ix).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss, stats
