#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 @author: Chao
 @contact: zhaochaocs@gmail.com
 @time: 5/9/2019 11:00 AM
"""
import argparse
import opts
import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import dgl

import torch
import torch.nn as nn

from plan.GCNEncoder import GCN, load_bin_vec, GCNConfig
from plan.dataset import Dataset
from plan.plan_eval import eval_acc_strict_loop, eval_strict, eval_strict2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.manual_seed(1)  # cpu
torch.cuda.manual_seed(1)  # gpu
np.random.seed(1)  # numpy
torch.backends.cudnn.deterministic = True  # cudnn


# import traceback
# import warnings
# import sys
#
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#
#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))
#
# warnings.showwarning = warn_with_traceback


# with open('delex_dict.json') as data_file:
#     delex_dict = json.load(data_file)

def parse_args():
    parser = argparse.ArgumentParser(
        description='plan.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.plan_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    return opt


def batch_train(net, batched_data):
    batched_graphs, batched_sizes, batched_labels, batched_split = zip(*batched_data)
    G = dgl.batch(batched_graphs)
    L = torch.cat([torch.tensor(label) for label in batched_labels])

    # for G, G_sizes, L in train_data.graph_batch_generator(triple_batch_size, True):
    if torch.cuda.is_available():
        L = L.cuda()
    logits = net(G, batched_sizes)

    # we only compute loss for labeled nodes
    loss = bce_loss(logits, L) / L.sum()
    return loss


def batch_test(net, batched_data, pred_path_dict):
    batched_graphs, batched_sizes, batched_ids, batched_local_predicates, batched_node_feature_ids = zip(*batched_data)
    G = dgl.batch(batched_graphs)
    cur_batch_size, cur_triple_size = len(batched_sizes), batched_sizes[0]
    local_pred = torch.zeros(cur_batch_size, cur_triple_size).long()

    for step in range(cur_triple_size):
        with torch.no_grad():
            logits = net(G, batched_sizes)  # n
            logits = logits.cpu()
        logits = logits.reshape(-1, cur_triple_size)

        max_value, max_idx = torch.max(logits, 1)
        local_pred[:, step] = max_idx
        accessed = torch.zeros(cur_batch_size, cur_triple_size).long()
        accessed[range(cur_batch_size), max_idx] = 1
        accessed = accessed.reshape(-1, 1)
        if torch.cuda.is_available():
            accessed = accessed.cuda()
        G.ndata["access"][G.ndata['spo'][:, 1].bool()] += accessed

        G.ndata["pre_access"] *= 0
        G.ndata["pre_access"][G.ndata['spo'][:, 1].bool()] += accessed

    local_pred = local_pred.tolist()
    # global_pred = batched_node_feature_ids.gather(index=local_pred, dim=1).tolist()

    # max_idx = max_idx.tolist()
    for i, id in enumerate(batched_ids):
        tripleset_id = id.strip()
        pred_path_dict[tripleset_id] = local_pred[i]


def eval_plan(src_file, pred_path_dict):
    lex_id_file = src_file[:-13] + 'translate.lexid.txt'
    golden_file = src_file[:-13] + 'src-rel-order.txt'
    pred_file = src_file[:-13] + 'src-rel-order-pred.txt'
    triple_file = src_file[:-14] + '.triple'
    with open(lex_id_file, 'r', encoding='utf-8') as fr_id, \
            open(pred_file, 'w', encoding='utf-8') as fw_tp:
        for id in fr_id:
            id = id.strip()
            if int(id.split("_")[1]) < triple_len_filter:
                fw_tp.write("0\n")
            else:
                plan = list(map(str, pred_path_dict[id]))
                fw_tp.write(" ".join(plan) + '\n')

    for mode in ['seen', 'unseen', 'whole']:
        acc, bleu = eval_strict2(pred_file, golden_file, lex_id_file, triple_file,
                                 baseline_path_dict=None, mode=mode)
        return acc, bleu


if __name__ == '__main__':

    opt = parse_args()

    edge_to_id = {e: i for i, e in enumerate(['sp', 'po', 'op', 'ps', 'll', 'ne'])}
    triple_len_filter = 2

    node_to_id, id_to_node = Dataset.build_node_mapping([opt.train_src, opt.test_src])

    # process data (delexicalization and idlization)
    train_data = Dataset()
    train_data.read_examples(type="train", opt=opt, node_to_id=node_to_id, triple_len_filter=triple_len_filter)
    print('Load the training data done...')
    dev_data = Dataset()
    dev_data.read_examples(type="valid", opt=opt, node_to_id=node_to_id, triple_len_filter=triple_len_filter)
    print('Load the dev data done...')
    test_data = Dataset()
    test_data.read_examples(type="test", opt=opt, node_to_id=node_to_id, triple_len_filter=triple_len_filter)
    print('Load the test data done...')


    # emb_weight = load_bin_vec(fname="/home/chao/tools/glove/glove.6B.100d.w2v.txt", vocab=node_to_id, emb_size=100)
    emb_weight = None

    # init model and data
    model_config = GCNConfig(vocab_size=len(node_to_id), emb_size=100, emb_weight=emb_weight, freeze=True,
              in_feats=105, hidden_size=100, num_classes=1, edge_to_id=edge_to_id, num_hidden_layers=2)
    net = GCN(model_config)
    if torch.cuda.is_available():
        net = net.cuda()

    bce_loss = nn.KLDivLoss(reduction="sum")
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    triple_batch_size = 100
    graph_batch_size = 100

    train_graphs, train_triple_sizes, train_labels, train_split = train_data.load_dataset()
    dev_graphs, dev_triple_sizes, dev_ids, dev_local_predicates, dev_node_feature_ids = dev_data.load_test_dataset()
    test_graphs, test_triple_sizes, test_ids, test_local_predicates, test_node_feature_ids = test_data.load_test_dataset()

    random_baselines_dev = dev_data.generate_random_plan()
    random_baselines_test = test_data.generate_random_plan()

    prev_acc = 0

    # train the model
    for epoch in range(3):
        net.train()
        loss_value = 0

        with tqdm(total=train_data.graph_size / triple_batch_size) as pbar:
            for batched_data in Dataset._batch_generator(list(zip(train_graphs, train_triple_sizes, train_labels, train_split)),
                                                         batch_size=triple_batch_size, shuffle=True):
                pbar.update(1)
                loss = batch_train(net, batched_data)
                loss_value += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print('Epoch %d | Loss : %.4f' % (epoch, loss_value))

        # eval on valid set
        pred_path_dict = dict()
        for batched_data in Dataset.data_batch_generator(
                data=list(zip(dev_graphs, dev_triple_sizes, dev_ids, dev_local_predicates, dev_node_feature_ids)),
                batch_size=100, shuffle=False, group=True):
            batch_test(net, batched_data, pred_path_dict)
        acc, bleu = eval_plan(opt.valid_src, pred_path_dict)

        if acc > prev_acc:
            prev_acc = acc
            if os.path.exists(opt.model_output):
                os.remove(opt.model_output)
            torch.save(net.state_dict(), opt.model_output)

    with open('data/plan_word_mapping.pkl', 'wb') as fwb:
        pickle.dump([node_to_id, id_to_node], fwb)
    with open('data/plan_config.pkl', 'wb') as fwb:
        pickle.dump(model_config, fwb)