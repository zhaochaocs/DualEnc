#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   File Name：     example
   Description :
   Author :       zhaochaocs
   date：          10/24/19
"""
import collections
import copy
import random

import dgl
import networkx as nx
import numpy as np
import torch

edge_to_id = {e: i for i, e in enumerate(['sp', 'po', 'op', 'ps', 'll', 'ne'])}


class Example:
    def __init__(self, lex_id, predicates, nodes, nodes1, nodes2, labels, node_feature_ids, lex=None):
        self.entry_id = lex_id  # entryID_entrySize_entryCategory_lexID
        self.predicates = predicates    # local idlized list of predicates (golden order)
        self.nodes = nodes  # list of nodes
        self.node_feature_ids = node_feature_ids       # global nodes id
        self.nodes1 = nodes1
        self.nodes2 = nodes2
        self.labels = labels

        self.node_size = len(nodes)
        self.triple_size = labels.count('A1')

        self.predicate_size = len(predicates)
            # len(set(self.get_predicates()))  # might smaller than real triple size due to the incomplete plan

        assert self.predicate_size == self.triple_size

        if self.predicate_size == 0:
            print(lex)

        self.nx_graph = self.create_nx_graph().to_undirected()

    def get_id(self, lex=True):
        if lex:
            return self.entry_id
        else:
            return self.entry_id.rsplit("_", 1)[0]

    def reset_predicates(self, plan):
        self.predicates = [self.predicates[p] for p in plan]

    def set_split(self, split_):
        self.split_ = split_

    def get_predicates(self, mode='local', idlize=True):
        if mode == 'global':
            if not idlize:
                return self.get_predicates(idlize=False)
            else:
                return [self.node_feature_ids[local_idx]
                        for local_idx in self.get_predicates()]
        elif mode == 'local':
            if idlize:
                return self.predicates
            else:
                return [self.nodes[p] for p in self.predicates]
        else:
            raise TypeError('Invalid mode!')

    def get_label(self, encoding='one-hot'):
        local_predicates = self.get_predicates(mode='local', idlize=True)
        # assert local_predicates == sorted(local_predicates)

        if encoding == 'p-id':
            return local_predicates
        elif encoding == 'global-id':
            return self.get_predicates(mode='global', idlize=True)
        elif encoding == 'one-hot':
            label = torch.zeros(self.node_size, self.predicate_size)
            for t in range(0, self.predicate_size):
                label[local_predicates[t], t] = 1
            label = torch.index_select(label, dim=0, index=torch.tensor(sorted(local_predicates)).long())
            return label.permute(1, 0).tolist()
        else:
            raise TypeError('Invalid encoding mode!')

    def build_rdf_graph(self, accessed_predicates=None):
        """
        build the graph sequences according to the tripleset
        At each step, we set a special token to mark the nodes that have already been accessed
        :param tripleset:
        :param accessed_predicates: the predicates that has already been visited, using local node index.
        Notice that None and [] are different
        :return:
        """
        # store the batch info
        g_list = []
        triple_size_list = []
        # match between global and local nodes

        # globalID_localID_dict, local_triple_list = localize(tripleset)
        local_predicate_list = self.get_predicates(mode="local", idlize=True)

        self.dgl_graph = self.create_graph()

        if accessed_predicates is None:
            for t in range(0, self.predicate_size):
                # construct the graph
                g = copy.deepcopy(self.dgl_graph)
                # assign the access info
                for i in range(t):  # during the t-th step, we have accessed the 0 to t-1 predicates
                    local_predicate = local_predicate_list[i]
                    # g.nodes[local_predicate].data['access'] = torch.tensor([[1]])
                    g.nodes[local_predicate].data['access'] += 1
                    if i == t-1:
                        g.nodes[local_predicate].data['pre_access'] += 1

                # batch graph
                g_list.append(g)
                triple_size_list.append(self.predicate_size)

            return g_list, triple_size_list
        else:
            g = copy.deepcopy(self.dgl_graph)
            assert g.ndata['spo'][:,1].sum().item() == self.predicate_size
            for p in accessed_predicates:
                g.nodes[p].data['access'] = torch.tensor([[1]])
            if len(accessed_predicates):
                g.nodes[accessed_predicates[-1]].data['pre_access'] = torch.tensor([[1]])
            g_list.append(g)
            triple_size_list.append(self.predicate_size)
            return g_list, triple_size_list


    def create_nx_graph(self):
        edge_list = []
        g = nx.DiGraph()
        for node_1, node_2, edge in zip(self.nodes1, self.nodes2, self.labels):

            if edge == "A0":
                edge_list.append((node_1, node_2))
            elif edge == "A1":
                edge_list.append((node_2, node_1))

        g.add_edges_from(edge_list)
        return g

    def create_graph(self):
        g = dgl.DGLGraph()
        node_size = self.node_size
        g.add_nodes(node_size)
        g.ndata['global_id'] = torch.tensor(self.node_feature_ids)
        g.ndata['spo'] = torch.zeros(self.node_size, 3).float()
        edge_list = []
        edge_type_list = []
        node_mention = [[self.node_feature_ids[i]] for i in range(node_size)]

        for node_1, node_2, edge in zip(self.nodes1, self.nodes2, self.labels):

            if edge == "A0":
                g.nodes[node_1].data['spo'] += torch.tensor([1.0, 0, 0])
                g.nodes[node_2].data['spo'] += torch.tensor([0, 0.5, 0])

                edge_list.append((node_1, node_2))
                edge_list.append((node_2, node_1))
                edge_list.append((node_1, node_1))
                edge_list.append((node_2, node_2))
                # edge_list.append((node_1, node_1))
                edge_type_list += [edge_to_id[e] for e in ['sp', 'ps', 'll', 'll']]
            elif edge == "A1":
                g.nodes[node_1].data['spo'] += torch.tensor([0, 0, 1.0])
                g.nodes[node_2].data['spo'] += torch.tensor([0, 0.5, 0])

                edge_list.append((node_1, node_2))
                edge_list.append((node_2, node_1))
                edge_list.append((node_1, node_1))
                edge_list.append((node_2, node_2))
                # edge_list.append((node_1, node_1))
                edge_type_list += [edge_to_id[e] for e in ['op', 'po', 'll', 'll']]
            elif edge == 'NE':
                edge_list.append((node_1, node_2))
                edge_type_list += [edge_to_id["ne"]]
                node_mention[node_2].append(self.node_feature_ids[node_1])
            else:
                raise ValueError("Do not support the edge type {}".format(edge))

        new_edge_list, new_edge_type_list = [], []
        added_edge = set()
        for edge, edge_type in zip(edge_list, edge_type_list):
            # if edge_type == edge_to_id["ll"]:
            #     continue
            edge_id = "{}-{}-{}".format(edge[0], edge[1], edge_type)
            if edge_id not in added_edge:
                added_edge.add(edge_id)
                new_edge_list.append(edge)
                new_edge_type_list.append(edge_type)

        src, dst = tuple(zip(*new_edge_list))
        g.add_edges(src, dst)

        g.ndata['global_id'] = torch.tensor(self.node_feature_ids)
        g.ndata['access'] = torch.zeros(self.node_size, 1).long()
        g.ndata['pre_access'] = torch.zeros(self.node_size, 1).long()
        g.ndata['spo'] = torch.gt(g.ndata['spo'], 0).float()
        g.edata['type'] = torch.tensor(new_edge_type_list).reshape(-1, 1)

        # node_mention = [torch.tensor(m) for m in node_mention]
        # We set the max mention len as 15
        max_mention_len = 15
        node_mention_tensor = torch.zeros((self.node_size, max_mention_len), dtype=torch.long)
        # g.ndata['node_mention'] = torch.nn.utils.rnn.pad_sequence(node_mention, batch_first=True, padding_value=0)
        for idx, mention in enumerate(node_mention):
            if len(mention) > max_mention_len:
                mention = mention[:max_mention_len]
            node_mention_tensor[idx, :len(mention)] = torch.tensor(mention, dtype=torch.long)
        g.ndata['node_mention'] = node_mention_tensor

        dst_in_deg = {}
        for dst_node in set(dst):
            if dst_node not in dst_in_deg:
                dst_in_deg[dst_node] = {}
            for edge_type in range(len(edge_to_id)):
                if edge_type not in dst_in_deg[dst_node]:
                    dst_in_deg[dst_node][edge_type] = 0
        for dst_node, edge_type in zip(dst, new_edge_type_list):
            dst_in_deg[dst_node][edge_type] += 1

        e_norm = [1.0 / dst_in_deg[dst_node][e_type]
                  for dst_node, e_type in zip(dst, new_edge_type_list)]
        g.edata['norm'] = torch.tensor(e_norm).reshape(-1, 1)

        return g

    def get_random_plan(self, walk_func=None):
        if walk_func is None or walk_func == "None":
            plan = copy.deepcopy(self.get_predicates(mode="local", idlize=True))
            np.random.shuffle(plan)
            return plan
        else:
            assert walk_func in ['random_walk', "dfs", "bfs"]
            if not nx.is_connected(self.nx_graph):
                graphs = [self.nx_graph.subgraph(c) for c in nx.connected_components(self.nx_graph)]
            else:
                graphs = [self.nx_graph]

            if walk_func == "random_walk":
                random_list = flat_list([random_walk(g) for g in graphs])
            elif walk_func == "dfs":
                random_list = flat_list([dfs(g) for g in graphs])
            else:
                random_list = flat_list([bfs(g) for g in graphs])
            local_random_plan = [n for n in random_list if n in self.predicates]
            # return [self.node_feature_ids[p] for p in local_random_plan]
            return local_random_plan


def random_walk(G):
    graph_size = len(list(G.nodes))
    assert graph_size
    visited = collections.OrderedDict()
    cur_node = random.choice(list(G.nodes))
    visited[cur_node] = 1
    while not len(visited) == graph_size:
        next_node = random.choice(list(G.neighbors(cur_node)))
        if next_node not in visited:
            visited[next_node] = 1
        cur_node = next_node
    return list(visited.keys())

def dfs(G):
    """
    dfs for an undirected graph
    :param G:
    :return:
    """
    def dfs_util(cur_node, visited):
        neighbors = list(G.neighbors(cur_node))
        random.shuffle(neighbors)
        for n in neighbors:
            if n not in visited:
                visited[n] = 1
                dfs_util(n, visited)

    graph_size = len(list(G.nodes))
    assert graph_size
    visited = collections.OrderedDict()
    cur_node = random.choice(list(G.nodes))
    visited[cur_node] = 1
    dfs_util(cur_node, visited)
    return list(visited.keys())


def bfs(G):
    """
    bfs for an undirected graph
    :param G:
    :return:
    """

    graph_size = len(list(G.nodes))
    assert graph_size
    visited = collections.OrderedDict()
    cur_node = random.choice(list(G.nodes))
    visited[cur_node] = 1
    queue = []
    queue.append(cur_node)

    while queue:
        cur_node = queue.pop(0)
        neighbors = list(G.neighbors(cur_node))
        random.shuffle(neighbors)
        for n in neighbors:
            if n not in visited:
                queue.append(n)
                visited[n] = 1
    return list(visited.keys())


def flat_list(lists):
    flatted_list = []
    for l in lists:
        if not isinstance(l, list):
            flatted_list.append(l)
        else:
            flatted_list += flat_list(l)
    return flatted_list


if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (1, 4), (4, 5)])
    G = G.to_undirected()
    for _ in range(20):
        print(dfs(G))









