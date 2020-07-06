import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import torch.nn.functional as F

try:
    import dgl.function as fn
except:
    pass

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, edge_to_id, bias=True,
                 activation=None):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.edge_to_id = edge_to_id
        self.num_rels = len(edge_to_id) - 1
        self.bias = bias
        self.activation = activation


        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat, self.out_feat))

        # add bias
        if not self.bias is None:
            self.bias = nn.Parameter(torch.Tensor(self.num_rels, out_feat))


        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if not self.bias is None:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        def message_func(edges):
            # edges.data['norm'] is the in_dgr-1 of the target node
            msg = torch.matmul(edges.src['h'], self.weight)   # [e,h] * [r,h,h] = [r,e,h]
            rel_idx = edges.data['type']    # e,1
            rel_idx_weight = rel_idx.unsqueeze(0).expand(-1,-1, self.out_feat)
            msg_transform = msg.gather(dim=0, index = rel_idx_weight)  #1,e,h
            msg = msg_transform.squeeze(0)
            if not self.bias is None:
                rel_idx_bias = rel_idx.expand(-1, self.out_feat)
                msg_bias = self.bias.gather(dim=0, index=rel_idx_bias)  #e,h
                msg += msg_bias

            msg = msg * edges.data['norm']
            return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        other_edges = (1 - (g.edata['type'] == self.edge_to_id['ne']).int()).squeeze().nonzero().squeeze()

        g.send(other_edges, message_func)
        g.recv(reduce_func=fn.sum(msg='msg', out='h'), apply_node_func=apply_func)


# Define the GCNLayer module
class GCNLayer_dgl(nn.Module):
    def __init__(self, in_feats, out_feats, activation=F.relu):
        super(GCNLayer_dgl, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def apply_func(self, node):
        h = self.linear(node.data['h'])
        if not self.activation is None:
            h = self.activation(h)
        return {'h' : h}

    def forward(self, g):
        # g: graph
        # inputs: node_num * emb_size
        # g is the graph and the inputs is the input node features
        # first set the node features
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'), self.apply_func)


class GCNLayer(nn.Module):
    """ Graph convolutional neural network encoder.

    """
    def __init__(self,
                 num_inputs, num_units,
                 num_labels,
                 in_arcs=True,
                 out_arcs=True,
                 batch_first=False,
                 use_gates=True,
                 use_glus=False):
        super(GCNLayer, self).__init__()

        self.in_arcs = in_arcs
        self.out_arcs = out_arcs

        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_labels = num_labels
        self.batch_first = batch_first

        self.glu = nn.GLU(3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_gates = use_gates
        self.use_glus = use_glus
        #https://www.cs.toronto.edu/~yujiali/files/talks/iclr16_ggnn_talk.pdf
        #https://arxiv.org/pdf/1612.08083.pdf

        if in_arcs:
            self.V_in = Parameter(torch.Tensor(num_labels, self.num_inputs, self.num_units))
            nn.init.xavier_normal(self.V_in)

            self.b_in = Parameter(torch.Tensor(num_labels, self.num_units))
            nn.init.constant(self.b_in, 0)

            if self.use_gates:
                self.V_in_gate = Parameter(torch.Tensor(self.num_inputs, 1))
                nn.init.xavier_normal(self.V_in_gate)
                self.b_in_gate = Parameter(torch.Tensor(num_labels, 1))
                nn.init.constant(self.b_in_gate, 1)

        if out_arcs:
            self.V_out = Parameter(torch.Tensor(num_labels, self.num_inputs, self.num_units))
            nn.init.xavier_normal(self.V_out)

            self.b_out = Parameter(torch.Tensor(num_labels, self.num_units))
            nn.init.constant(self.b_out, 0)

            if self.use_gates:
                self.V_out_gate = Parameter(torch.Tensor(self.num_inputs, 1))
                nn.init.xavier_normal(self.V_out_gate)
                self.b_out_gate = Parameter(torch.Tensor(num_labels, 1))
                nn.init.constant(self.b_out_gate, 1)

        self.W_self_loop = Parameter(torch.Tensor(self.num_inputs, self.num_units))

        nn.init.xavier_normal(self.W_self_loop)

        if self.use_gates:
            self.W_self_loop_gate = Parameter(torch.Tensor(self.num_inputs, 1))
            nn.init.xavier_normal(self.W_self_loop_gate)

    def forward(self, src, lengths=None, arc_tensor_in=None, arc_tensor_out=None,
                label_tensor_in=None, label_tensor_out=None,
                mask_in=None, mask_out=None,  # batch* t, degree
                mask_loop=None, sent_mask=None):

        if not self.batch_first:
            encoder_outputs = src.permute(1, 0, 2).contiguous()
        else:
            encoder_outputs = src.contiguous()
        batch_size = encoder_outputs.size()[0]
        seq_len = encoder_outputs.size()[1]
        max_degree = 1
        input_ = encoder_outputs.view((batch_size * seq_len, self.num_inputs))  # [b* t, h]

        if self.in_arcs:
            # import pdb
            # pdb.set_trace()
            input_in = torch.matmul(input_, self.V_in)  # [b* t, h] * [n,h,h] = [n,b*t, h]
            # first_in: get the index of node_2, and select its embedding from input_in
            # arc_tensor_in[0] * seq_len + arc_tensor_in[1]: b* t* degr
            first_in = input_in.index_select(1, arc_tensor_in[0] * seq_len + arc_tensor_in[1])  # [b* t* degr, h]
            label_tensor_in_idx = label_tensor_in.unsqueeze(-1).expand(-1,-1,self.num_units)
            first_in = first_in.gather(dim=0, index=label_tensor_in_idx).squeeze(0)
            # second_in: get the index of the labels and select the bias parameter from b_in
            second_in = self.b_in.index_select(0, label_tensor_in[0])  # [b* t* degr, h]
            in_ = first_in + second_in
            degr = int(first_in.size()[0] / batch_size / seq_len)   # max_degree: set as 10

            in_ = in_.view((batch_size, seq_len, degr, self.num_units)) # b, t, degr, h

            if self.use_glus:
                # gate the information of each neighbour, self nodes are in here too.
                in_ = torch.cat((in_, in_), 3)
                in_ = self.glu(in_)

            if self.use_gates:
                # compute gate weightsll
                input_in_gate = torch.mm(input_, self.V_in_gate)  # [b* t, h] * [h,h] = [b*t, h]
                first_in_gate = input_in_gate.index_select(0, arc_tensor_in[0] * seq_len + arc_tensor_in[1])  # [b* t* mxdeg, h]
                second_in_gate = self.b_in_gate.index_select(0, label_tensor_in[0])
                in_gate = (first_in_gate + second_in_gate).view((batch_size, seq_len, degr))

            max_degree += degr

        if self.out_arcs:
            input_out = torch.matmul(input_, self.V_out)  # [b* t, h] * [h,h] = [b* t, h]
            first_out = input_out.index_select(1, arc_tensor_out[0] * seq_len + arc_tensor_out[1])  # [b* t* mxdeg, h]
            label_tensor_out_idx = label_tensor_out.unsqueeze(-1).expand(-1,-1,self.num_units)
            first_out = first_out.gather(dim=0, index=label_tensor_out_idx).squeeze(0)
            second_out = self.b_out.index_select(0, label_tensor_out[0])

            degr = int(first_out.size()[0] / batch_size / seq_len)
            max_degree += degr

            out_ = (first_out + second_out).view((batch_size, seq_len, degr, self.num_units))


            if self.use_glus:
                # gate the information of each neighbour, self nodes are in here too.
                out_ = torch.cat((out_, out_), 3)
                out_ = self.glu(out_)

            if self.use_gates:
                # compute gate weights
                input_out_gate = torch.mm(input_, self.V_out_gate)  # [b* t, h] * [h,h] = [b* t, h]
                first_out_gate = input_out_gate.index_select(0, arc_tensor_out[0] * seq_len + arc_tensor_out[1])  # [b* t* mxdeg, h]
                second_out_gate = self.b_out_gate.index_select(0, label_tensor_out[0])
                out_gate = (first_out_gate + second_out_gate).view((batch_size, seq_len, degr))


        same_input = torch.mm(encoder_outputs.view(-1, encoder_outputs.size(2)), self.W_self_loop). \
            view(encoder_outputs.size(0), encoder_outputs.size(1), -1)  # b, t, h
        same_input = same_input.view(encoder_outputs.size(0), encoder_outputs.size(1), 1, self.W_self_loop.size(1)) # b, t, 1, h
        if self.use_gates:
            same_input_gate = torch.mm(encoder_outputs.view(-1, encoder_outputs.size(2)), self.W_self_loop_gate) \
                .view(encoder_outputs.size(0), encoder_outputs.size(1), -1)


        # # in_arcs:
        # potentials_in = in_.view((batch_size * seq_len,
        #                                -1, self.num_units))  # [b*t,  deg, h]
        # if self.use_gates:
        #     potentials_gate_in = in_gate.view((batch_size * seq_len,
        #                                      -1)) # [b* t,  deg]
        # mask_soft_in = mask_in  # [b* t, deg]
        # mask_len_in = mask_in.sum(dim=1) + 1e-5    # b*t
        # if self.use_gates:
        #     probs_det_ = (self.sigmoid(potentials_gate_in) * mask_soft_in).unsqueeze(2)  # [b * t, deg, 1]
        #     potentials_masked_in = potentials_in * probs_det_  # [b * t, deg,h]
        # else:
        #     potentials_masked_in = potentials_in * mask_soft_in.unsqueeze(2)# [b * t, deg, h]
        # potentials_masked_in_ = potentials_masked_in.sum(dim=1) / mask_len_in.unsqueeze(1)# [b * t, h]

        # # out arcs
        # potentials_out = out_.view((batch_size * seq_len,
        #                           -1, self.num_units))  # [b*t,  deg, h]
        # if self.use_gates:
        #     potentials_gate_out = out_gate.view((batch_size * seq_len,
        #                                        -1))  # [b* t,  deg]
        # mask_soft_out = mask_out  # [b* t, deg]
        # mask_len_out = mask_out.sum(dim=1)+ 1e-5  # b*t
        # if self.use_gates:
        #     probs_det_ = (self.sigmoid(potentials_gate_out) * mask_soft_out).unsqueeze(2)  # [b * t, deg, 1]
        #     potentials_masked_out = potentials_out * probs_det_  # [b * t, deg,h]
        # else:
        #     potentials_masked_out = potentials_out * mask_soft_out.unsqueeze(2)  # [b * t, deg, h]
        # potentials_masked_out_ = potentials_masked_out.sum(dim=1) / mask_len_out.unsqueeze(1)

        # # loop arcs
        # potentials_loop = same_input.view((batch_size * seq_len,
        #                           -1, self.num_units))  # [b, t,  1, h]
        # if self.use_gates:
        #     potentials_gate_loop = same_input_gate # [b, t,  mxdeg, h]
        # mask_soft_loop = mask_loop  # [b* t, 1]
        # if self.use_gates:
        #     probs_det_ = (self.sigmoid(potentials_gate_loop) * mask_soft_loop).unsqueeze(2)  # [b * t, deg, 1]
        #     potentials_masked_loop = potentials_loop * probs_det_  # [b * t, 1,h]
        # else:
        #     potentials_masked_loop = potentials_loop * mask_soft_loop.unsqueeze(2)  # [b * t, 1, h]
        # potentials_masked_loop_ = potentials_masked_loop.sum(dim=1)  # [b * t, h]

        # # potentials_resh = potentials.view((batch_size * seq_len,
        # #                                    max_degree, self.num_units,))  # [h, b * t, mxdeg]  [b * t, mxdeg, h]?
        # #
        # #
        # # if self.use_gates:
        # #     potentials_r = potentials_gate.view((batch_size * seq_len,
        # #                                          max_degree))  # [b * t, mxdeg]
        # #
        # #     probs_det_ = (self.sigmoid(potentials_r) * mask_soft).unsqueeze(2)  # [b * t, mxdeg]
        # #     potentials_masked = potentials_resh * probs_det_  # [b * t, mxdeg,h]
        # # else:
        # #     # NO Gates
        # #     potentials_masked = potentials_resh * mask_soft.unsqueeze(2)    # [b * t, mxdeg, h]



        # potentials_masked_ = torch.cat((potentials_masked_in_.unsqueeze(1), \
        #                                 potentials_masked_out_.unsqueeze(1), \
        #                                 potentials_masked_loop_.unsqueeze(1)), dim=1).sum(dim=1)  # [b * t, h]
        # potentials_masked_ = self.relu(potentials_masked_)  # [b * t, h]

        # result_ = potentials_masked_.view((batch_size, seq_len, self.num_units))  # [ b, t, h]

        # result_ = result_ * sent_mask.permute(1, 0).contiguous().unsqueeze(2)  # [b, t, h]

        # memory_bank = result_.permute(1, 0, 2).contiguous()  # [t, b, h]

        # return memory_bank

        if self.in_arcs and self.out_arcs:
            potentials = torch.cat((in_, out_, same_input), dim=2)  # [b, t,  mxdeg, h]
            if self.use_gates:
                potentials_gate = torch.cat((in_gate, out_gate, same_input_gate), dim=2)  # [b, t,  mxdeg, h]
            mask_soft = torch.cat((mask_in, mask_out, mask_loop), dim=1)  # [b* t, mxdeg]
        elif self.out_arcs:
            potentials = torch.cat((out_, same_input), dim=2)  # [b, t,  2*mxdeg+1, h]
            if self.use_gates:
                potentials_gate = torch.cat((out_gate, same_input_gate), dim=2)  # [b, t,  mxdeg, h]
            mask_soft = torch.cat((mask_out, mask_loop), dim=1)  # [b* t, mxdeg]
        elif self.in_arcs:
            potentials = torch.cat((in_, same_input), dim=2)  # [b, t,  2*mxdeg+1, h]
            if self.use_gates:
                potentials_gate = torch.cat((in_gate, same_input_gate), dim=2)  # [b, t,  mxdeg, h]
            mask_soft = torch.cat((mask_in, mask_loop), dim=1)  # [b* t, mxdeg]
        else:
            potentials = same_input  # [b, t,  2*mxdeg+1, h]
            if self.use_gates:
                potentials_gate = same_input_gate # [b, t,  mxdeg, h]
            mask_soft = mask_loop  # [b* t, mxdeg]

        potentials_resh = potentials.view((batch_size * seq_len,
                                           max_degree, self.num_units,))  # [h, b * t, mxdeg]  [b * t, mxdeg, h]?


        if self.use_gates:
            potentials_r = potentials_gate.view((batch_size * seq_len,
                                                 max_degree))  # [b * t, mxdeg]

            probs_det_ = (self.sigmoid(potentials_r) * mask_soft).unsqueeze(2)  # [b * t, mxdeg]
            potentials_masked = potentials_resh * probs_det_  # [b * t, mxdeg,h]
        else:
            # NO Gates
            potentials_masked = potentials_resh * mask_soft.unsqueeze(2)



        potentials_masked_ = potentials_masked.sum(dim=1)  # [b * t, h]
        potentials_masked_ = self.relu(potentials_masked_)  # [b * t, h]

        result_ = potentials_masked_.view((batch_size, seq_len, self.num_units))  # [ b, t, h]

        result_ = result_ * sent_mask.permute(1, 0).contiguous().unsqueeze(2)  # [b, t, h]

        memory_bank = result_.permute(1, 0, 2).contiguous()  # [t, b, h]

        return memory_bank
