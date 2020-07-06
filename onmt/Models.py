from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import onmt
from onmt.Utils import aeq
from onmt.my_modules.GCN import GCNLayer, RGCNLayer


def rnn_factory(rnn_type, **kwargs):
    # Use pytorch version when available.
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        # SRU doesn't support PackedSequence.
        no_pack_padded_seq = True
        rnn = onmt.modules.SRU(**kwargs)
    else:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn, no_pack_padded_seq


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`onmt.Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Memory_Bank]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """
    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, lengths=None, encoder_state=None):
        """
        Args:
            src (:obj:`LongTensor`):
               padded sequences of sparse indices `[src_len x batch x nfeat]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`
            encoder_state (rnn-class specific):
               initial encoder_state state.

        Returns:
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                * memory bank for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError


class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

    def forward(self, src, lengths=None, encoder_state=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, lengths, encoder_state)

        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()
        mean = emb.mean(0).expand(self.num_layers, batch, emb_dim)
        memory_bank = emb
        encoder_final = (mean, mean)
        return encoder_final, memory_bank


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    def forward(self, src, lengths=None, encoder_state=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, lengths, encoder_state)

        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)
        # encoder_final: (h_n, c_n) ([layer_num=2, 64, 256],[layer_num, batch_size, h_dim])
        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]    # seq_len * batch_size * h_dim or h_dim = num_directions * h_dim

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for i in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs


class GCNEncoder_DGL(EncoderBase):

    def __init__(self, embeddings,
                 num_inputs, num_units,
                 num_labels,
                 num_layers=1,
                 in_arcs=True,
                 out_arcs=True,
                 batch_first=False,
                 residual='',
                 use_gates=True,
                 use_glus=False,
                 morph_embeddings=None,
                 use_birnn=False, rnn_layer=1):
        """

        :param embeddings:
        :param num_inputs:
        :param num_units:
        :param num_labels:
        :param num_layers:
        :param in_arcs: whether to include the in_arcs
        :param out_arcs:
        :param batch_first:
        :param residual:
        :param use_gates:
        :param use_glus:
        :param morph_embeddings:
        """
        super(GCNEncoder_DGL, self).__init__()
        if isinstance(embeddings, tuple):
            self.embeddings, self.cpt_embeddings = embeddings
        else:
            self.embeddings = embeddings
            self.cpt_embeddings = None
        self.num_layers = num_layers
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.residual = (residual == '' or residual == 'residual')
        self.use_gates = use_gates
        self.use_glus = use_glus
        # self.feature_type = feature_type
        self.use_birnn = use_birnn
        self.rnn_layer = rnn_layer

        self.edge_to_id = {e: i for i, e in enumerate(['sp', 'po', 'op', 'ps', 'll', 'ne'])}

        self.embedding_size = self.embeddings.embedding_size


        if morph_embeddings is not None:
            self.morph_embeddings = morph_embeddings
            self.emb_morph_emb = nn.Linear(num_inputs+morph_embeddings.embedding_size, num_inputs)

        self.emb_transform = nn.Linear(self.num_inputs+3, self.num_units)

        self.H_1 = torch.nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        nn.init.xavier_normal_(self.H_1)
        self.H_2 = torch.nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        nn.init.xavier_normal_(self.H_2)
        self.H_3 = torch.nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        nn.init.xavier_normal_(self.H_3)
        self.H_4 = torch.nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        nn.init.xavier_normal_(self.H_4)

        self.layers = nn.ModuleList()
        # self.layers.append(RGCNLayer(in_feats, hidden_size, activation=F.relu))
        for _ in range(num_layers):
            self.layers.append(RGCNLayer(num_units, num_units, edge_to_id=self.edge_to_id, activation=F.relu))


    def init_forward(self, G):
        def message_func(edges):
            msg = edges.src['emb'] * edges.data['norm']
            return {"msg": msg}

        def reduce_func(nodes):
            # The argument is a batch of nodes.
            # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
            return {'emb': torch.sum(nodes.mailbox['msg'], dim=1)}

        ne_edges = (G.edata['type'] == self.edge_to_id['ne']).squeeze().nonzero().squeeze()

        # ne_nodes = (G.ndata['spo'][:, 1] == 0).squeeze().nonzero().squeeze()
        ne_nodes = (G.ndata['spo'][:, 0] + G.ndata['spo'][:, 2] > 0).squeeze().nonzero().squeeze()

        G.send(ne_edges, message_func)
        G.recv(ne_nodes, reduce_func)
        emb = G.ndata.pop('emb')
        return emb


    def forward(self, G):

        batch_size, G_sizes = G.batch_size, G.batch_num_nodes
        G_sizes_tensor = torch.tensor(G_sizes)

        if torch.cuda.is_available():
            G.ndata["global_id"] = G.ndata.pop("global_id").cuda()
            G.ndata['spo'] = G.ndata.pop('spo').cuda()

            G.edata['type'] = G.edata.pop('type').cuda()
            G.edata['norm'] = G.edata.pop('norm').cuda()

            G_sizes_tensor = G_sizes_tensor.cuda()

        G.ndata['emb'] = self.embeddings(G.ndata['global_id'].unsqueeze(1).unsqueeze(2)).squeeze(1)  # n, d
        G.ndata['emb'] = self.init_forward(G)
        G.ndata['h'] = torch.cat((G.ndata['emb'], G.ndata['spo'].float()), dim=1)  # n, d(+4)
        # G.ndata['h'] = torch.cat((G.ndata['spo'].float(), G.ndata['access'].float()), dim=1)  # n, d(+4)
        G.ndata['h'] = self.emb_transform(G.ndata['h'])
        prev_h = G.ndata['h']

        for i, layer in enumerate(self.layers):
            layer(G)
            if self.residual and not i == self.num_layers - 1:
                G.ndata['h'] += prev_h
                prev_h = G.ndata['h']
        h = G.ndata.pop('h')  # n, d

        # h_p = h[G.ndata['spo'][:, 1].byte()]  # p_num, p_feature
        #
        # h_p = self.predict(h_p, G_sizes)
        #
        h_p = torch.split(h, G_sizes, dim=0)
        encoder_final = torch.cat([torch.sum(hpi, dim=0, keepdim=True) for hpi in h_p], dim=0) / G_sizes_tensor.float().unsqueeze(1)    # batch_size * h
        memory_bank = []
        node_size = max(G_sizes)
        for g_size, hpi in zip(G_sizes, h_p):
            m = nn.ConstantPad2d((0, 0, 0, node_size-g_size), 0)
            memory_bank.append(m(hpi).unsqueeze(0))

        memory_bank = torch.cat(memory_bank, dim=0).transpose(1, 0)

        h_1 = torch.mm(encoder_final, self.H_1).view((1, batch_size, self.num_units))  # [1, b, h]
        h_2 = torch.mm(encoder_final, self.H_2).view((1, batch_size, self.num_units))
        h_3 = torch.mm(encoder_final, self.H_3).view((1, batch_size, self.num_units))
        h_4 = torch.mm(encoder_final, self.H_4).view((1, batch_size, self.num_units))
        h__1 = torch.cat([h_1, h_2], dim=0)  # [2, b, h]
        h__2 = torch.cat([h_3, h_4], dim=0)  # [2, b, h]

        if self.use_birnn:
            h_1 = h_1.split(self.num_units // 2, 2)
            h_1 = torch.cat(h_1, dim=0)
            h_2 = h_2.split(self.num_units // 2, 2)
            h_2 = torch.cat(h_2, dim=0)
            if self.rnn_layer == 1:
                return (h_1, h_2), memory_bank
            elif self.rnn_layer == 2:
                h_3 = h_3.split(self.num_units // 2, 2)
                h_3 = torch.cat(h_3, dim=0)
                h_4 = h_4.split(self.num_units // 2, 2)
                h_4 = torch.cat(h_4, dim=0)
                h_1 = torch.cat([h_1, h_2], dim=0)  # [2, b, h]
                h_2 = torch.cat([h_3, h_4], dim=0)  # [2, b, h]
                return (h_1, h_2), memory_bank
            else:
                raise ValueError()
        else:
            return (h__1, h__2), memory_bank

class GCNEncoder(EncoderBase):

    def __init__(self, embeddings,
                 num_inputs, num_units,
                 num_labels,
                 num_layers=1,
                 in_arcs=True,
                 out_arcs=True,
                 batch_first=False,
                 residual='',
                 use_gates=True,
                 use_glus=False,
                 morph_embeddings=None,
                 use_birnn=False, rnn_layer=1):
        """

        :param embeddings:
        :param num_inputs:
        :param num_units:
        :param num_labels:
        :param num_layers:
        :param in_arcs: whether to include the in_arcs
        :param out_arcs:
        :param batch_first:
        :param residual:
        :param use_gates:
        :param use_glus:
        :param morph_embeddings:
        """
        super(GCNEncoder, self).__init__()
        self.embeddings = embeddings
        self.num_layers = num_layers
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.residual = residual
        self.use_gates = use_gates
        self.use_glus = use_glus
        self.use_birnn = use_birnn
        self.rnn_layer = rnn_layer

        self.embedding_size = self.embeddings.embedding_size


        if morph_embeddings is not None:
            self.morph_embeddings = morph_embeddings
            self.emb_morph_emb = nn.Linear(num_inputs+morph_embeddings.embedding_size, num_inputs)

        self.H_1 = torch.nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        nn.init.xavier_normal_(self.H_1)
        self.H_2 = torch.nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        nn.init.xavier_normal_(self.H_2)
        self.H_3 = torch.nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        nn.init.xavier_normal_(self.H_3)
        self.H_4 = torch.nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        nn.init.xavier_normal_(self.H_4)

        self.gcn_layers = []
        if residual == '' or residual == 'residual':

            for i in range(self.num_layers):
                if i == 0:
                    gcn = GCNLayer(self.embedding_size, num_units, num_labels,
                                   in_arcs=in_arcs, out_arcs=out_arcs, use_gates=self.use_gates,
                                   use_glus=self.use_glus)
                else:
                    gcn = GCNLayer(num_inputs, num_units, num_labels,
                               in_arcs=in_arcs, out_arcs=out_arcs, use_gates=self.use_gates,
                               use_glus=self.use_glus)
                self.gcn_layers.append(gcn)

            self.gcn_seq = nn.Sequential(*self.gcn_layers)

        elif residual == 'dense':
            for i in range(self.num_layers):
                input_size = num_inputs + (i * num_units)
                gcn = GCNLayer(input_size, num_units, num_labels,
                               in_arcs=in_arcs, out_arcs=out_arcs, use_gates=self.use_gates,
                               use_glus=self.use_glus)
                self.gcn_layers.append(gcn)

            self.gcn_seq = nn.Sequential(*self.gcn_layers)

    def forward(self, src, lengths=None, arc_tensor_in=None, arc_tensor_out=None,
                label_tensor_in=None, label_tensor_out=None,
                mask_in=None, mask_out=None,  # batch* t, degree
                mask_loop=None, sent_mask=None, morph=None, morph_mask=None):
        if morph is None:

            embeddings = self.embeddings(src)
        else:
            embeddings = self.embeddings(src)  # [t,b,e]
            morph_size = morph.data.size()  # [B,t,max_m]
            embeddings_m = self.morph_embeddings(morph.view(morph_size[0] * morph_size[1],
                                                            morph_size[2], 1))  # [B*t,max_m, m_e]
            embeddings_m = embeddings_m.view((morph_size[0], morph_size[1], morph_size[2],
                                              embeddings_m.data.size()[2]))  # [B,t,max_m, m_e]
            embeddings_m = embeddings_m.permute(3, 0, 1, 2).contiguous()  # [m_e ,B , max_m, t]
            masked_morph = embeddings_m * morph_mask  # [m_e ,B , max_m, t]*[B,t,max_m] = [m_e , B, t, max_m]

            morph_sum = masked_morph.sum(3).permute(2, 1, 0).contiguous()  # [t,B,m_e]

            embeddings = torch.cat([embeddings, morph_sum], dim=2)

            embeddings = torch.nn.functional.relu(self.emb_morph_emb(embeddings))



        if self.residual == '':

            for g, gcn in enumerate(self.gcn_layers):
                if g == 0:
                    memory_bank = gcn(embeddings, lengths, arc_tensor_in, arc_tensor_out,
                                                 label_tensor_in, label_tensor_out,
                                                 mask_in, mask_out,
                                                 mask_loop, sent_mask)  # [t, b, h]

                else:
                    memory_bank = gcn(memory_bank, lengths, arc_tensor_in, arc_tensor_out,
                                                 label_tensor_in, label_tensor_out,
                                                 mask_in, mask_out,
                                                 mask_loop, sent_mask)  # [t, b, h]
        elif self.residual == 'residual':

            for g, gcn in enumerate(self.gcn_layers):
                if g == 0:
                    memory_bank = gcn(embeddings, lengths, arc_tensor_in, arc_tensor_out,
                                                 label_tensor_in, label_tensor_out,
                                                 mask_in, mask_out,
                                                 mask_loop, sent_mask)  # [t, b, h]

                elif g == 1:
                    prev_memory_bank = embeddings+memory_bank
                    memory_bank = gcn(prev_memory_bank, lengths, arc_tensor_in, arc_tensor_out,
                                      label_tensor_in, label_tensor_out,
                                      mask_in, mask_out,
                                      mask_loop, sent_mask)  # [t, b, h]

                else:
                    prev_memory_bank = prev_memory_bank + memory_bank
                    memory_bank = gcn(prev_memory_bank, lengths, arc_tensor_in, arc_tensor_out,
                                                 label_tensor_in, label_tensor_out,
                                                 mask_in, mask_out,
                                                 mask_loop, sent_mask)  # [t, b, h]

        elif self.residual == 'dense':
            for g, gcn in enumerate(self.gcn_layers):
                if g == 0:
                    memory_bank = gcn(embeddings, lengths, arc_tensor_in, arc_tensor_out,
                                                 label_tensor_in, label_tensor_out,
                                                 mask_in, mask_out,
                                                 mask_loop, sent_mask)  # [t, b, h]

                elif g == 1:
                    prev_memory_bank = torch.cat([embeddings, memory_bank], dim=2)
                    memory_bank = gcn(prev_memory_bank, lengths, arc_tensor_in, arc_tensor_out,
                                      label_tensor_in, label_tensor_out,
                                      mask_in, mask_out,
                                      mask_loop, sent_mask)  # [t, b, h]

                else:
                    prev_memory_bank = torch.cat([prev_memory_bank, memory_bank], dim=2)
                    memory_bank = gcn(prev_memory_bank, lengths, arc_tensor_in, arc_tensor_out,
                                                 label_tensor_in, label_tensor_out,
                                                 mask_in, mask_out,
                                                 mask_loop, sent_mask)  # [t, b, h]



        batch_size = memory_bank.size()[1]
        result_ = memory_bank.permute(2, 1, 0)  # [h,b,t]
        res_sum = result_.sum(2)  # [h,b]
        sent_mask = sent_mask.permute(1, 0).contiguous()  # [b,t]
        mask_sum = sent_mask.sum(1)  # [b]
        encoder_final = res_sum / mask_sum  # [h, b]
        encoder_final = encoder_final.permute(1, 0)  # [b, h]

        h_1 = torch.mm(encoder_final, self.H_1).view((1, batch_size, self.num_units))  # [1, b, h]
        h_2 = torch.mm(encoder_final, self.H_2).view((1, batch_size, self.num_units))
        h_3 = torch.mm(encoder_final, self.H_3).view((1, batch_size, self.num_units))
        h_4 = torch.mm(encoder_final, self.H_4).view((1, batch_size, self.num_units))
        h__1 = torch.cat([h_1, h_2], dim=0)  # [2, b, h]
        h__2 = torch.cat([h_3, h_4], dim=0)  # [2, b, h]
        if self.use_birnn:
            h_1 = h_1.split(self.num_units//2, 2)
            h_1 = torch.cat(h_1, dim=0)
            h_2 = h_2.split(self.num_units // 2, 2)
            h_2 = torch.cat(h_2, dim=0)
            if self.rnn_layer == 1:
                return (h_1, h_2), memory_bank
            elif self.rnn_layer == 2:
                h_3 = h_3.split(self.num_units // 2, 2)
                h_3 = torch.cat(h_3, dim=0)
                h_4 = h_4.split(self.num_units // 2, 2)
                h_4 = torch.cat(h_4, dim=0)
                h_1 = torch.cat([h_1, h_2], dim=0)  # [2, b, h]
                h_2 = torch.cat([h_3, h_4], dim=0)  # [2, b, h]
                return (h_1, h_2), memory_bank
            else:
                raise ValueError()
        else:
            return (h__1, h__2), memory_bank


class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Memory_Bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.Models.DecoderState`):
                 decoder state object to initialize the decoder
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Check
        if not isinstance(state, tuple):
            assert isinstance(state, RNNDecoderState)
        else:
            assert isinstance(state[0], RNNDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        if not isinstance(memory_bank, tuple):
            _, memory_batch, _ = memory_bank.size()
        else:
            _, memory_batch, _ = memory_bank[0].size()
        aeq(tgt_batch, memory_batch)
        # END

        # Run the forward pass of the RNN.
        decoder_final, decoder_outputs, attns = self._run_forward_pass(
            tgt, memory_bank, state, memory_lengths=memory_lengths)     # final: (h_T, c_T)    outputs: list of (B*E)

        # Update the state with the result.
        final_output = decoder_outputs[-1]
        coverage = None
        if "coverage" in attns:
            coverage = attns["coverage"][-1].unsqueeze(0)
        state.update_state(decoder_final, final_output.unsqueeze(0), coverage)

        # Concatenates sequence of tensors along a new dimension.
        decoder_outputs = torch.stack(decoder_outputs)
        for k in attns:
            if len(attns[k]):
                attns[k] = torch.stack(attns[k])

        return decoder_outputs, state, attns

    def init_decoder_state(self, src, memory_bank, encoder_final):
        def _fix_enc_hidden(h):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h    # 2, 64, 256

        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size,
                                   tuple([_fix_enc_hidden(enc_hid)
                                         for enc_hid in encoder_final]))
        else:  # GRU
            return RNNDecoderState(self.hidden_size,
                                   _fix_enc_hidden(encoder_final))


class StdRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """
    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                 [len x batch x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            decoder_final (Variable): final hidden state from the decoder.
            decoder_outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        attns = {}
        emb = self.embeddings(tgt)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, decoder_final = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, decoder_final = self.rnn(emb, state.hidden)

        # Check
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        # END

        # Calculate the attention.
        decoder_outputs, p_attn = self.attn(
            rnn_output.transpose(0, 1).contiguous(),
            memory_bank.transpose(0, 1),
            memory_lengths=memory_lengths
        )
        attns["std"] = p_attn

        # Calculate the context gate.
        if self.context_gate is not None:
            decoder_outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                decoder_outputs.view(-1, decoder_outputs.size(2))
            )
            decoder_outputs = \
                decoder_outputs.view(tgt_len, tgt_batch, self.hidden_size)

        decoder_outputs = self.dropout(decoder_outputs)
        return decoder_final, decoder_outputs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Memory_Bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = state.input_feed.squeeze(0)
        input_feed_batch, _ = input_feed.size()
        tgt_len, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": [], "std2": []}
        if self._copy:
            attns["copy"] = []
            attns["copy2"] = []
        if self._coverage:
            attns["coverage"] = []
            attns["coverage2"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(emb.split(1)):        # T*B*E
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)   # B, E -> B, 2E

            rnn_output, hidden = self.rnn(decoder_input, hidden)    # rnn_output: 64*256    hidden: (1*64*256, 1*64*256)
            decoder_output, p_attn = self.attn(
                rnn_output,
                (memory_bank[0].transpose(0,1), memory_bank[1].transpose(0,1)) if isinstance(memory_bank, tuple) \
                    else memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths)
            if isinstance(p_attn, tuple):
                p_attn2, p_attn = p_attn            # set the 2nd attention as the main attention
            else:
                p_attn2 = None
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            decoder_outputs += [decoder_output]
            attns["std"] += [p_attn]
            if p_attn2 is not None:
                attns['std2'] += [p_attn2]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output,
                        (memory_bank[0].transpose(0, 1), memory_bank[1].transpose(0, 1)) \
                        if isinstance(memory_bank, tuple) else memory_bank.transpose(0, 1))

                if isinstance(copy_attn, tuple):
                    copy_attn2, copy_attn = copy_attn           # set the 2nd attention as the main attention
                    attns["copy"] += [copy_attn]
                    attns["copy2"] += [copy_attn2]
                else:
                    copy_attn2 = None
                    attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
                if attns["std2"] is not None:
                    attns["copy2"] = attns["std2"]
        # Return result.
        return hidden, decoder_outputs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, multigpu=False, encoder2=None):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder2 = encoder2

    def forward(self, src, tgt, lengths, dec_state=None, src2=None, lengths2=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_final, memory_bank = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(src, memory_bank, enc_final)
        if src2 is not None and self.encoder2 is not None:
            enc_final2, memory_bank2 = self.encoder2(src2, lengths2)
            enc_state2 = self.decoder.init_decoder_state(src2, memory_bank2, enc_final2)

            enc_final = (enc_final, enc_final2)
            memory_bank = (memory_bank, memory_bank2)
            enc_state = (enc_state, enc_state2)
            lengths = (lengths, lengths2)


        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state


class NMTModelGCN(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, multigpu=False, encoder2=None):
        self.multigpu = multigpu
        super(NMTModelGCN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder2 = encoder2

    def forward(self, src, tgt, lengths, adj_arc_in, adj_arc_out, adj_lab_in,
                adj_lab_out, mask_in, mask_out, mask_loop, mask_sent, morph=None,
                mask_morph=None, dec_state=None, src2=None, lengths2=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        if isinstance(self.encoder, RNNEncoder):
            src, lengths = src2, lengths2
            enc_final, memory_bank = self.encoder(src2, lengths2)  # (h_n, c_n): [1,64,256], (T, B, E)
        else:
            enc_final, memory_bank = self.encoder(src, lengths, adj_arc_in, adj_arc_out,
                                                adj_lab_in, adj_lab_out,
                                                mask_in, mask_out, mask_loop, mask_sent, morph, mask_morph)   #enc_final: (2*b*e, 2*b*e)  memory_bank: (T*b*e)
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)

        if src2 is not None and self.encoder2 is not None:
            if isinstance(self.encoder2, RNNEncoder):
                enc_final2, memory_bank2 = self.encoder2(src2, lengths2)    # (h_n, c_n): [1,64,256], (T, B, E)
            else:
                src2 = src
                lengths2 = lengths
                enc_final2, memory_bank2 = self.encoder2(src, lengths, adj_arc_in, adj_arc_out,
                                                      adj_lab_in, adj_lab_out,
                                                      mask_in, mask_out, mask_loop, mask_sent, morph,
                                                      mask_morph)  # enc_final: (2*b*e, 2*b*e)  memory_bank: (T*b*e)
            enc_state2 = self.decoder.init_decoder_state(src2, memory_bank2, enc_final2)

            enc_final = (enc_final, enc_final2)
            memory_bank = (memory_bank, memory_bank2)
            # enc_state = (enc_state, enc_state2)
            enc_state = self.state_fusion(enc_state, enc_state2)
            lengths = (lengths, lengths2)


        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         state=enc_state if dec_state is None else dec_state,
                         memory_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state

    def state_fusion(self, state1, state2):
        hidden_fused = []
        for i, (h1, h2) in enumerate(zip(state1.hidden, state2.hidden)):
            hidden_fused.append(h1 + h2)
        state1.hidden = tuple(hidden_fused)
        return state1


class NMTModelGCN_DGL(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, multigpu=False, encoder2=None):
        self.multigpu = multigpu
        super(NMTModelGCN_DGL, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder2 = encoder2

    def forward(self, src, tgt, batched_dgl_graph, lengths, morph=None,
                mask_morph=None, dec_state=None, src2=None, lengths2=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        if isinstance(self.encoder, RNNEncoder):
            src, lengths = src2, lengths2
            enc_final, memory_bank = self.encoder(src2, lengths2)  # (h_n, c_n): [1,64,256], (T, B, E)
        else:
            enc_final, memory_bank = self.encoder(batched_dgl_graph)   #enc_final: (2*b*e, 2*b*e)  memory_bank: (T*b*e)
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)

        if src2 is not None and self.encoder2 is not None:
            if isinstance(self.encoder2, RNNEncoder):
                enc_final2, memory_bank2 = self.encoder2(src2, lengths2)    # (h_n, c_n): [1,64,256], (T, B, E)
            else:
                src2 = src
                lengths2 = lengths

                enc_final2, memory_bank2 = self.encoder2(batched_dgl_graph)    # (h_n, c_n): [1,64,256], (T, B, E)
            enc_state2 = self.decoder.init_decoder_state(src2, memory_bank2, enc_final2)

            enc_final = (enc_final, enc_final2)
            memory_bank = (memory_bank, memory_bank2)
            # enc_state = (enc_state, enc_state2)
            enc_state = self.state_fusion(enc_state, enc_state2)
            lengths = (lengths, lengths2)


        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         state=enc_state if dec_state is None else dec_state,
                         memory_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state

    def state_fusion(self, state1, state2):
        hidden_fused = []
        for i, (h1, h2) in enumerate(zip(state1.hidden, state2.hidden)):
            hidden_fused.append(h1 + h2)
        state1.hidden = tuple(hidden_fused)
        return state1


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach()

    def beam_update(self, idx, positions, beam_size):
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, hidden_size, rnnstate):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = self.hidden[0].size(1)         # 2, batch_size, hidden_size
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(self.hidden[0].data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]
