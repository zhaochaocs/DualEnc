#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   File Name：     dataset
   Description :
   Author :       zhaochaocs
   date：          10/24/19
"""

import copy
from collections import Counter
from operator import itemgetter

import numpy as np
from tqdm import tqdm

from plan.example import Example


class Dataset:
    def __init__(self):
        self.data = []

    @property
    def example_size(self):
        return len(self.data)
    
    @property
    def graph_size(self):
        return sum([len(example.predicates) for example in self.data])

    @classmethod
    def build_node_mapping(cls, train_srcs):
        node_count = Counter()
        node_to_id = {"PAD": 0, 'UNK': 1}

        for train_src in train_srcs:
            with open(train_src, 'r', encoding='utf-8') as fr_nodes:
                for nodes in fr_nodes:
                    nodes = nodes.strip().split()
                    node_count.update(nodes)

        node_to_id.update({node: i + 2 for i, (node, _) in enumerate(node_count.most_common())})
        id_to_node = {i: n for n, i in node_to_id.items()}
        return node_to_id, id_to_node

    def read_examples(self, opt, type, node_to_id, triple_len_filter):
        assert type in ['train', 'valid', 'test']
        if type == "train":
            lex_id_file = opt.train_src[:-13] + 'translate.lexid.txt'
            src_file, node1_file, node2_file, label_file, order_file = \
                opt.train_src, opt.train_node1, opt.train_node2, opt.train_label, opt.train_plan
            split_file = opt.train_split
        elif type == "valid":
            lex_id_file = opt.valid_src[:-13] + 'translate.lexid.txt'
            src_file, node1_file, node2_file, label_file = \
                opt.valid_src, opt.valid_node1, opt.valid_node2, opt.valid_label
        else:
            lex_id_file = opt.test_src[:-13] + 'translate.lexid.txt'
            src_file, node1_file, node2_file, label_file = \
                opt.test_src, opt.test_node1, opt.test_node2, opt.test_label

        with open(lex_id_file, 'r', encoding='utf-8') as fr_lexid, \
                open(src_file, 'r', encoding='utf-8') as fr_nodes, \
                open(node1_file, 'r', encoding='utf-8') as fr_nodes1, \
                open(node2_file, 'r', encoding='utf-8') as fr_nodes2, \
                open(label_file, 'r', encoding='utf-8') as fr_labels:
            for lex_id, nodes, nodes1, nodes2, labels in \
                    zip(fr_lexid, fr_nodes, fr_nodes1, fr_nodes2, fr_labels):
                lex_id = lex_id.strip()
                if not int(lex_id.split('_')[1]) >= triple_len_filter:
                    continue
                nodes = nodes.strip().split()
                nodes1 = list(map(int, nodes1.strip().split()))
                nodes2 = list(map(int, nodes2.strip().split()))
                labels = labels.strip().split()

                predicates = []
                for node2, label in zip(nodes2, labels):
                    if label == 'A0':
                        predicates.append(node2)

                example = Example(lex_id=lex_id,
                                  predicates=predicates,
                                  nodes=nodes,
                                  node_feature_ids=[node_to_id.get(token, 1) for token in nodes],       # UNK: 1
                                  nodes1=nodes1,
                                  nodes2=nodes2,
                                  labels=labels)

                self.data.append(example)

        # add golden plan to training data
        if type == 'train':
            cur_example_id = -1
            with open(lex_id_file, 'r', encoding='utf-8') as fr_lexid, \
                    open(order_file, 'r', encoding='utf-8') as fr_plan, \
                    open(split_file, 'r', encoding='utf-8') as fr_split:
                for lex_id, orders, split_ in zip(fr_lexid, fr_plan, fr_split):
                    lex_id = lex_id.strip()
                    plan = orders.strip()
                    split_ = split_.strip()
                    plan = list(map(int, plan.split()))
                    split_ = list(map(int, split_.split()))
                    assert len(plan) == len(split_)
                    if not int(lex_id.split('_')[1]) >= triple_len_filter:
                        continue
                    cur_example_id += 1
                    assert self.data[cur_example_id].entry_id == lex_id
                    self.data[cur_example_id].reset_predicates(plan)
                    self.data[cur_example_id].set_split(split_)


    def load_dataset(self):
        train_graphs, train_triple_sizes, train_labels, train_splits = [], [], [], []
        for i, example in enumerate(self.data):
            if i % 2000 == 0:
                print("Reading {} / {} files done ... ".format(i, len(self.data)))
            graphs, triple_sizes = example.build_rdf_graph()
            train_graphs += graphs
            train_triple_sizes += triple_sizes
            train_labels += example.get_label(encoding='one-hot')
            train_splits += [0] + example.split_[:-1]
        return train_graphs, train_triple_sizes, train_labels, train_splits

    def load_test_dataset(self):
        test_graphs, test_triple_sizes = [], []
        test_ids, test_local_predicates, test_node_feature_ids = [], [], []

        for i, example in enumerate(self.data):
            if i % 2000 == 0:
                print("Reading {} / {} files done ... ".format(i, len(self.data)))
            graph, triple_size = example.build_rdf_graph(accessed_predicates=[])
            test_graphs += graph
            test_triple_sizes += triple_size
            # test_labels.append(example.get_label(encoding='global-id'))
            test_ids.append(example.get_id(lex=True))
            test_local_predicates.append(sorted(example.get_predicates()))
            test_node_feature_ids.append(example.node_feature_ids)
        return test_graphs, test_triple_sizes, test_ids, test_local_predicates, test_node_feature_ids

    def generate_random_plan(self):
        random_baselines = {}
        for i, example in enumerate(self.data):
            entry_id = example.entry_id
            random_baselines[entry_id] = {}
            for sys in ["None", "random_walk", "dfs", "bfs"]:
                random_nodes = example.get_random_plan(sys)
                # random_nodes = [example.node_feature_ids.index(n) for n in random_nodes]
                plan = np.argsort(random_nodes)
                random_baselines[entry_id][sys] = plan

        return random_baselines

    @classmethod
    def _batch_generator(cls, data, batch_size, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """

        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

        if shuffle and data_size > 1:
            perm = np.random.permutation(data_size)
            data = itemgetter(*perm)(data)

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            batched_data = copy.deepcopy(list(data[start_index:end_index]))
            yield batched_data

    @classmethod
    def data_batch_generator(cls, data, batch_size, shuffle=True, group=True):
        if group:
            data_group = {i: [] for i in range(1, 8)}
            for d in data:
                data_group[d[1]].append(d)
        else:
            data_group = {0: data}
        for _, data in data_group.items():
            if len(data):
                yield from cls._batch_generator(data, batch_size, shuffle)
