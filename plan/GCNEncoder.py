#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   File Name：     GCNEncoder
   Description :
   Author :       zhaochaocs
   date：          10/24/19
"""

import numpy as np

from gensim.models import KeyedVectors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from onmt.my_modules.GCN import RGCNLayer

def load_bin_vec(fname, vocab, emb_size=200):
    """
    Loads word vecs from bin file
    """
    word_vecs = torch.zeros((len(vocab), emb_size))
    xavier_uniform_(word_vecs)

    wv_from_bin = KeyedVectors.load_word2vec_format(fname, binary=False)
    print('load word2vec done...')
    vocab_hit_count, vocab_whole = 0, 0
    for v, i in vocab.items():
        if "_" not in v:
            vocab_whole += 1
        if v in wv_from_bin.wv.vocab.keys():
            vocab_hit_count += 1
            word_vecs[i] = torch.tensor(wv_from_bin[v])
    print("Hit {0:.2f} ({1:d} / {2:d}) of vocabs from the pretrained emb ...".format(vocab_hit_count / vocab_whole,
                                                                               vocab_hit_count, vocab_whole))
    return word_vecs

class AveEmbEncoder(nn.Module):

    def __init__(self, vocab_size, emb_size, emb_weight=None):
        super(AveEmbEncoder, self).__init__()
        if emb_weight is None:
            self.lookup = nn.Embedding(vocab_size, emb_size)
            xavier_uniform_(self.lookup.weight.data)
        else:
            emb_weight = torch.FloatTensor(emb_weight)
            assert emb_weight.size()[0] == vocab_size
            assert emb_weight.size()[1] == emb_size
            self.lookup = nn.Embedding(vocab_size, emb_size, padding_idx=0).from_pretrained(emb_weight, freeze=False)

        self.emb_size = emb_size

    def forward(self, input_x):
        inputs_len = (input_x != 0).sum(dim=1, keepdim=True).float()  # batch_size
        x_wrd = self.lookup(input_x)  # batch_size * sent_len * emb_size
        return x_wrd.sum(1) / inputs_len


class RNN_Encoder(nn.Module):

    def __init__(self, emb_size, lstm_hidden_dim=150, lstm_num_layers=1):
        super(RNN_Encoder, self).__init__()

        self.hidden_dim = lstm_hidden_dim
        self.num_layers = lstm_num_layers

        # gru
        self.bigru = nn.GRU(emb_size, self.hidden_dim, batch_first=True, \
                            dropout=0.5, num_layers=lstm_num_layers, bidirectional=True)

        self.drop_layer = nn.Dropout(p=0.5)

    def forward(self, input_x, input_mask):

        # input: batch * sent_len
        batch_size, sent_len = input_x.size()
        input_len = (input_x != 0).sum(1).float()   # batch
        input_len_sorted, perm_idx = input_len.sort(0, descending=True)
        _, unperm_idx = perm_idx.sort(0)
        input_sorted = input_x[perm_idx]        # batch * sent_len
        input_x_emb = self.lookup(input_sorted)       # batch * sent_len * emb_size

        x_packed = pack_padded_sequence(input=input_x_emb, lengths=input_len_sorted.masked_fill(input_len_sorted == 0, 1),
                                        batch_first=True)
        gru_out, hn = self.bigru(x_packed)
        hn = hn.permute(1, 0, 2).reshape(sent_len, -1)  #seq_len, 2 * hidden_size WHY PERMUTE???
        gru_out, output_len = pad_packed_sequence(gru_out, batch_first=True)  # batch, sent_len, 2 * hidden_size
        hn = hn[unperm_idx]
        gru_out = gru_out[unperm_idx]  # batch, sent_len, 2h

        gru_out = self.drop_layer(gru_out)  # batch, doc_len, 2h

        return gru_out, hn


class GCNConfig:
    def __init__(self, vocab_size, emb_size, in_feats, hidden_size, num_classes, edge_to_id,
                 num_hidden_layers=0, emb_weight=None, freeze=False, residual=True):
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.emb_weight = emb_weight
        assert emb_weight in ['pre_train', None]

        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.edge_to_id = edge_to_id
        self.num_hidden_layers = num_hidden_layers

        self.freeze = freeze
        self.residual = residual



class GCN(nn.Module):
    def __init__(self, config, **kwargs):
        super(GCN, self).__init__()

        self.edge_to_id = config.edge_to_id
        self.residual = config.residual

        self.mention_encoder = AveEmbEncoder(config.vocab_size, config.emb_size, config.emb_weight)

        if config.emb_weight is None:
            self.lookup = nn.Embedding(config.vocab_size, config.emb_size, padding_idx=0)
            xavier_uniform_(self.lookup.weight.data)
            self.lookup.weight.data[0] = torch.zeros(config.emb_size)
            self.lookup.weight.data[1] = torch.zeros(config.emb_size)
            if config.freeze:
                print("Emb is randomly initialized. It would be better to make it trainable!")
        else:
            assert 'emb_weight' in kwargs
            emb_weight = torch.FloatTensor(kwargs['emb_weight'])
            assert emb_weight.size()[0] == config.vocab_size
            assert emb_weight.size()[1] == config.emb_size
            emb_weight[0] = torch.zeros(config.emb_size)
            emb_weight[1] = torch.zeros(config.emb_size)
            self.lookup = nn.Embedding(config.vocab_size, config.emb_size, padding_idx=0).from_pretrained(emb_weight, freeze=config.freeze)

        self.emb_size = config.emb_size

        self.emb_transform = nn.Linear(config.in_feats, config.hidden_size)

        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList()
        # self.layers.append(RGCNLayer(in_feats, hidden_size, activation=F.relu))
        for _ in range(config.num_hidden_layers):
            self.layers.append(RGCNLayer(config.hidden_size, config.hidden_size, config.edge_to_id, activation=F.relu))
        # self.layers.append(RGCNLayer(hidden_size, num_classes, activation=None))

        # self.out_act = nn.Sigmoid()
        self.output_layer = nn.Linear(config.hidden_size, config.num_classes)

        self.output_layer2 = nn.Bilinear(config.hidden_size, config.hidden_size, 1)

        self.output_layer_split = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()


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

    def ave_emb_encoder(self, inputs):
        inputs_len = (inputs != 0).sum(1, keepdim=True).float()   # batch_size
        x_wrd = self.lookup(inputs)  # batch_size * sent_len * emb_size
        return x_wrd.sum(1)/inputs_len

    def forward(self, G, G_sizes):
        # init the features
        if torch.cuda.is_available():
            G.ndata["global_id"] = G.ndata.pop("global_id").cuda()
            # G.ndata["global_hypernym_id"] = G.ndata.pop("global_hypernym_id").cuda()    #n, max_level
            # G.ndata["global_hypernym_mask"] = G.ndata.pop("global_hypernym_mask").cuda()       #n, max_level
            G.ndata['spo'] = G.ndata.pop('spo').cuda()
            G.ndata['access'] = G.ndata.pop('access').cuda()
            G.ndata['pre_access'] = G.ndata.pop('pre_access').cuda()
            G.ndata['node_mention'] = G.ndata.pop('node_mention').cuda()

            G.edata['type'] = G.edata.pop('type').cuda()
            G.edata['norm'] = G.edata.pop('norm').cuda()

        G.ndata['emb'] = self.lookup(G.ndata['global_id'])  # n, d
        G.ndata['emb'] = self.init_forward(G)

        # emb2 = self.ave_emb_encoder(G.ndata["node_mention"])
        # G.ndata['emb'] = self.mention_encoder(G.ndata["node_mention"])      # batch * len -> batch * len * emb_size

        # emb2 *= G.ndata['spo'].sum(dim=1, keepdim=True).gt(0).float()
        # if self.eval():
        #     print((G.ndata['emb'] - emb2).sum())
        # G.ndata['emb'] = emb2


        G.ndata['h'] = torch.cat((G.ndata['emb'], G.ndata['spo'].float(), G.ndata['access'].float(),
                                  G.ndata['pre_access'].float()), dim=1)  # n, d(+4)
        # G.ndata['h'] = torch.cat((G.ndata['spo'].float(), G.ndata['access'].float()), dim=1)  # n, d(+4)
        G.ndata['h'] = self.emb_transform(G.ndata['h'])
        prev_h = G.ndata['h']

        for i, layer in enumerate(self.layers):
            layer(G)
            if self.residual and not i == self.num_hidden_layers - 1:
                G.ndata['h'] += prev_h
                prev_h = G.ndata['h']
        h = G.ndata.pop('h')  # n, d

        h_p = h[G.ndata['spo'][:, 1].bool()]  # p_num, p_feature

        h_p = self.predict(h_p, G_sizes)

        h_p = torch.split(h_p, G_sizes, dim=0)
        h_p_out = torch.cat([F.log_softmax(hpi, dim=0) for hpi in h_p], dim=0)

        return h_p_out.reshape(-1)

    def predict(self, h_p, G_sizes):  # h_p: n*d representation of predicates
        # h_p = self.output_layer(h_p)

        proj_mat = self.get_proj_mat(G_sizes)  # n, n
        if torch.cuda.is_available():
            proj_mat = proj_mat.cuda()
        h_pool = torch.matmul(h_p.transpose(1, 0), proj_mat).transpose(0, 1)  # n, d
        # output of predicate emission
        output = self.output_layer2(h_p, h_pool)
        return output.squeeze(1)

    def get_proj_mat(self, G_sizes):
        node_size = sum(G_sizes)
        mat = np.zeros((node_size, node_size))  # n * n
        start_ptr, end_ptr = 0, 0
        for s in G_sizes:
            end_ptr = start_ptr + s
            mat[start_ptr:end_ptr, start_ptr:end_ptr] = np.ones((s, s)) * 1.0 / s
            start_ptr = end_ptr
        return torch.from_numpy(mat).float()