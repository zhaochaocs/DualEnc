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

from plan.GCNEncoder import GCN, load_bin_vec
from plan.dataset import Dataset
from plan.plan_eval import eval_acc_strict_loop, eval_strict, eval_strict2

from plan_train import batch_test

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

# Define a 2-layer GCN model



if __name__ == '__main__':

    opt = parse_args()

    edge_to_id = {e: i for i, e in enumerate(['sp', 'po', 'op', 'ps', 'll', 'ne'])}
    triple_len_filter = 2

    with open('data/plan_config.pkl', 'rb') as frb:
        plan_config = pickle.load(frb)
    with open('data/plan_word_mapping.pkl', 'rb') as frb:
        node_to_id, id_to_node = pickle.load(frb)

    # process data (delexicalization and idlization)
    test_data = Dataset()
    test_data.read_examples(type="test", opt=opt, node_to_id=node_to_id, triple_len_filter=triple_len_filter)
    print('Load the test data done...')

    net = GCN(plan_config)
    net.load_state_dict(torch.load(opt.model))
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()

    test_graphs, test_triple_sizes, test_ids, test_local_predicates, test_node_feature_ids = test_data.load_test_dataset()

    random_baselines = test_data.generate_random_plan()

    # test the model
    golden_path_dict = defaultdict(list)
    pred_path_dict = dict()
    pred_path_local_dict = defaultdict(list)

    for batched_data in Dataset.data_batch_generator(
            data=list(zip(test_graphs, test_triple_sizes, test_ids, test_local_predicates, test_node_feature_ids)),
            batch_size=100, shuffle=False, group=True):
        batch_test(net, batched_data, pred_path_dict)

    lex_id_file = opt.test_src[:-13] + 'translate.lexid.txt'
    golden_file = opt.test_src[:-13] + 'src-rel-order.txt'
    pred_file = opt.test_src[:-13] + 'src-rel-order-pred.txt'
    triple_file = opt.test_src[:-14] + '.triple'
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
                                 baseline_path_dict=random_baselines, mode=mode)


