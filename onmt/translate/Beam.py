from __future__ import division

import collections
import itertools
import re

from nltk.corpus import stopwords
import numpy as np
import torch

from onmt.translate import Penalties


stops = set(stopwords.words('english'))

UNSEEN_CATEGORIES = ['Athlete', 'Artist', 'MeanOfTransportation', 'CelestialBody', 'Politician']
SEEN_CATEGORIES = ['Astronaut', 'Building', 'Monument', 'University', 'SportsTeam',
                   'WrittenWork', 'Food', 'ComicsCharacter', 'Airport', 'City']


class Beam(object):
    """
    Class for managing the internals of the beam search process.

    Takes care of beams, back pointers, and scores.

    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """
    def __init__(self, size, pad, bos, eos,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0,
                 stepwise_penalty=False):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                        .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

        # Apply Penalty at every step
        self.stepwise_penalty = stepwise_penalty

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_probs, attn_out):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words), K is the beam size
        * `attn_out`- attention at the last step

        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1)
        if self.stepwise_penalty:
            self.global_scorer.update_score(self, attn_out)
        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20
        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + \
                self.scores.unsqueeze(1).expand_as(word_probs)
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20
        else:
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                            True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))
        self.attn.append(attn_out.index_select(0, prev_k))
        self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """
    def __init__(self, alpha, beta, cov_penalty, length_penalty):
        self.alpha = alpha
        self.beta = beta
        penalty_builder = Penalties.PenaltyBuilder(cov_penalty,
                                                   length_penalty)
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty()
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        """
        Rescores a prediction based on penalty functions
        """
        normalized_probs = self.length_penalty(beam,
                                               logprobs,
                                               self.alpha)
        if not beam.stepwise_penalty:
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"],
                                       self.beta)
            normalized_probs -= penalty

        return normalized_probs

    def update_score(self, beam, attn):
        """
        Function to update scores of a Beam that is not finished
        """
        if "prev_penalty" in beam.global_state.keys():
            beam.scores.add_(beam.global_state["prev_penalty"])
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"] + attn,
                                       self.beta)
            beam.scores.sub_(penalty)

    def update_global_state(self, beam):
        "Keeps the coverage vector as sum of attentions"
        if len(beam.prev_ks) == 1:
            beam.global_state["prev_penalty"] = beam.scores.clone().fill_(0.0)
            beam.global_state["coverage"] = beam.attn[-1]
            self.cov_total = beam.attn[-1].sum(1)
        else:
            self.cov_total += torch.min(beam.attn[-1],
                                        beam.global_state['coverage']).sum(1)
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])

            prev_penalty = self.cov_penalty(beam,
                                            beam.global_state["coverage"],
                                            self.beta)
            beam.global_state["prev_penalty"] = prev_penalty


def coverage_metric(candidates, plan, n_best):
    # how many words in vocab_set is covered by a candidate?
    entity_set = set(["".join(entity) for entity in re.findall(r"(agent|patient|bridge)(-\d)", plan, flags=0)])
    vocab_counter = collections.Counter()
    for cand in candidates:
        cand_list = [v for v in cand.split() if len(v) > 1 and
                     v not in stops]
        cand_entity_list = [v for v in cand_list if re.match(r"(agent|patient|bridge)(-\d)", v, flags=0)]
        cand_vocab_list = [v for v in cand_list if not re.match(r"(agent|patient|bridge)(-\d)", v, flags=0)]
        vocab_counter.update(cand_vocab_list)
    vocab_set = set([v for v, c in vocab_counter.items() if c >= (n_best/2)])
    cand_coverage = [0.0 for _ in range(len(candidates))]
    for i, cand in enumerate(candidates):
        cand_list = [v for v in cand.split() if len(v) > 1 and
                     v not in stops]
        cand_entity_list = [v for v in cand_list if re.match(r"(agent|patient|bridge)(-\d)", v, flags=0)]
        cand_vocab_list = [v for v in cand_list if not re.match(r"(agent|patient|bridge)(-\d)", v, flags=0)]
        entity_coverage = len(entity_set & set(cand_entity_list)) / len(entity_set) if len(entity_set) else 0
        vocab_coverage = len(vocab_set & set(cand_vocab_list)) / len(vocab_set) if len(vocab_set) else 0
        if len(entity_set):
            cand_coverage[i] = (entity_coverage * 2 + vocab_coverage) / 3
        else:
            cand_coverage[i] = vocab_coverage
    return np.array(cand_coverage)


def repetition_metric(candidates, plan):
    # the freq of a word appeared in a candidate should be less than the freq of this word in the plan
    subj = [p.split('|')[0].strip() for p in plan.split('< tsp >')]
    pred = [p.split('|')[1].strip() for p in plan.split('< tsp >')]
    obj = [p.split('|')[2].strip() for p in plan.split('< tsp >')]
    plan = [entity.split() for entity in subj+obj+pred]
    plan = [w for p in plan for w in p if len(w) > 1 and w not in stops]

    plan_counter = collections.Counter(plan)
    repetition = [0 for _ in range(len(candidates))]
    for i, cand in enumerate(candidates):
        cand_list = [v for v in cand.split() if len(v) > 1 and v not in stops]
        cand_counter = collections.Counter(cand_list)
        for v, c in cand_counter.items():
            if len(v) > 1 and plan_counter.get(v) is not None:
                if c > plan_counter.get(v):
                    repetition[i] += c - plan_counter.get(v)
    repetition = 1 - np.array(repetition) / len(plan_counter)
    return np.array(repetition)


def grouped(iterable, n):
    return itertools.zip_longest(*[iter(iterable)]*n)


def select_beam(preds_out, triple_file, n_best):

    triples = []
    with open(triple_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if not len(line):
                continue
            triples.append(line)

    realization = []

    for i, candidates in enumerate(grouped(preds_out, n_best)):
        triple = triples[i]
        candidates = [cand.strip() for cand in candidates]
        coverage, repetition = coverage_metric(candidates, triple, n_best), repetition_metric(candidates, triple)
        score = coverage + repetition

        max_idx = np.argmax(score)
        realization.append(candidates[max_idx])

    return realization
