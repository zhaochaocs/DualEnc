import nltk
import numpy as np
import copy
import random
import pickle
import collections
import pandas as pd


pd.set_option('display.expand_frame_repr', False)
pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', -1)

UNSEEN_CATEGORIES = ['Athlete', 'Artist', 'MeanOfTransportation', 'CelestialBody', 'Politician']
SEEN_CATEGORIES = ['Astronaut', 'Building', 'Monument', 'University', 'SportsTeam',
                   'WrittenWork', 'Food', 'ComicsCharacter', 'Airport', 'City']


def step(a):
    return (np.heaviside(a, 0.5) - 0.5)*2

def _n_gram_extract(plan, n):
    n_gram_list = list(zip(*[plan[i:] for i in range(n)]))
    n_gram_list = set(n_gram_list)
    return list(map(tuple, n_gram_list))

def remove_duplicate(pred, ref):
    ref_count = collections.Counter(ref)
    pred_count = dict()
    new_pred = []
    for p in pred:
        pred_count[p] = pred_count.get(p, 0) + 1
        if pred_count.get(p) <= ref_count[p] or p not in ref_count:
            new_pred.append(p)
    return new_pred

def BLEU_2(hypothesis, references, ref_len):

    ref_idx, _ = max(enumerate(references), key=(lambda x: len(x[1])))
    hypothesis = remove_duplicate(hypothesis, references[ref_idx])

    p = nltk.translate.bleu_score.modified_precision(references, hypothesis, 2)
    hyper_len = len(hypothesis)
    # ref_len = nltk.translate.bleu_score.closest_ref_length(references, hyper_len)
    bp = nltk.translate.bleu_score.brevity_penalty(ref_len, hyper_len)
    bleu = bp * p
    return bleu

def n_gram_prec(pred, refs, n=2, thres=1, begin=True, end=True):
    assert isinstance(pred, list)
    assert isinstance(refs, list)
    assert all(isinstance(ref, list) for ref in refs)

    if begin:
        pred = ['<BEGIN>'] + pred
        refs = [['<BEGIN>'] + ref for ref in refs]
    if end:
        pred = pred + ['<END>']
        refs = [ref + ['<END>'] for ref in refs]

    n_gram_in_prec = _n_gram_extract(pred, n)
    n_gram_in_refs = list()
    for ref in refs:
        n_gram_in_refs += _n_gram_extract(ref, n)

    if not len(n_gram_in_prec) or not len(n_gram_in_refs):
        return 0

    n_gram_in_refs = collections.Counter(n_gram_in_refs)

    hit_count = 0
    for n_gram in n_gram_in_prec:
        if n_gram_in_refs[n_gram] >= thres:
            hit_count += 1
    return hit_count / len(n_gram_in_prec)


def eval_acc_strict_loop(pred_path_dict, golden_path_dict, loop=10):
    stat_acc, stat_bleu = np.zeros((5, 8)), np.zeros((5, 8))
    for epoch in range(loop):
        cur_stat_acc, cur_stat_bleu = eval_acc_strict(pred_path_dict, golden_path_dict)
        stat_acc += cur_stat_acc
        stat_bleu += cur_stat_bleu

    np.set_printoptions(formatter={'float_kind': lambda x: "%.2f" % x})
    metric_name = ['acc', 'bleu']
    for i, stat in enumerate([stat_acc, stat_bleu]):
        print('normalized {} after {} loops :: '.format(metric_name[i], loop))

        stat = stat / loop

        print(stat)


def eval_strict2(pred_file, golden_file, lex_id_file, triple_file, baseline_path_dict=None, mode='seen'):
    pred_path_dict, golden_path_dict = {}, collections.defaultdict(list)
    new_baseline_path_dict = None if not baseline_path_dict else {}

    with open(lex_id_file, 'r', encoding='utf-8') as fr_id, \
         open(golden_file, 'r', encoding='utf-8') as fr_g, \
         open(pred_file, 'r', encoding='utf-8') as fr_p, \
         open(triple_file, 'r', encoding='utf-8') as fr_t:
        for id, golden, pred, triples in zip(fr_id, fr_g, fr_p, fr_t):
            id, golden, pred, triples = id.strip(), list(map(int, golden.strip().split())), \
                                        list(map(int, pred.strip().split())), triples.strip()
            triple_size = int(id.split('_')[1])
            id_short = id.rsplit('_', 1)[0]
            if triple_size < 2:
                golden_path_dict[id_short] = [[0], [0], [0]]
                pred_path_dict[id_short] = [0]
                if baseline_path_dict and id_short not in new_baseline_path_dict:
                    new_baseline_path_dict[id_short] = {}
                    for baseline in list(list(baseline_path_dict.values())[0].keys()):
                        new_baseline_path_dict[id_short][baseline] = [0]
                continue
            assert len(golden) == len(set(golden)) and len(golden) <= max(golden) + 1
            if not (len(pred) == len(set(pred)) and len(pred) == max(pred) + 1):
                print('warning::{}'.format(id))
            triples = [triple.strip().replace(' ', '_') for triple in triples.split(" < tsp > ")]
            # pred_path_dict[id_short].append(pred)
            golden_path_dict[id_short].append([triples[idx] for idx in golden])
            pred = [triples[idx] for idx in pred]
            if id_short not in pred_path_dict:
                pred_path_dict[id_short] = pred
            else:
                pass
                # assert pred_path_dict[id_short] == pred

            if baseline_path_dict and id_short not in new_baseline_path_dict:
                new_baseline_path_dict[id_short] = {}
                baselines = baseline_path_dict[id]
                for baseline, order in baselines.items():
                    new_baseline_path_dict[id_short][baseline] = [triples[idx] for idx in order]

    return eval_strict(pred_path_dict, golden_path_dict, baseline_path_dict=new_baseline_path_dict, mode=mode, thres=2)


def eval_strict(pred_path_dict, golden_path_dict, baseline_path_dict=None, mode='seen', thres=2):
    assert mode in ['seen', 'unseen', 'whole']
    if baseline_path_dict is None:
        baseline_path_dict = {}
        baselines = []
    else:
        baselines = list(list(baseline_path_dict.values())[0].keys())
        
    acc_results = pd.DataFrame(0.0, index= baselines + ["GCN", "total"],
                           columns=["{}-triple".format(i) for i in range(thres, 8)])
    bleu_results = pd.DataFrame(0.0, index=baselines + ["GCN", "total"],
                               columns=["{}-triple".format(i) for i in range(thres, 8)])

    # for significance test
    acc_instance_results = pd.DataFrame(0.0, index=list(golden_path_dict.keys()),
                               columns=["GCN"]+baselines)
    bleu_instance_results = pd.DataFrame(0.0, index=list(golden_path_dict.keys()),
                               columns=["GCN"]+baselines)


    for item, golden_path in golden_path_dict.items():
        # print("Golden::{}".format('\n'.join([print_path(p) for p in golden_path])))
        triple_size = int(item.split('_')[1])
        item_category = item.split('_')[2]
        assert item_category in SEEN_CATEGORIES or item_category in UNSEEN_CATEGORIES

        if mode == 'seen' and item_category in UNSEEN_CATEGORIES:
            continue
        if mode == "unseen" and item_category in SEEN_CATEGORIES:
            continue
        if triple_size < thres:
            continue

        acc_results.at["total", "{}-triple".format(triple_size)] += 1.0
        bleu_results.at["total", "{}-triple".format(triple_size)] += 1.0

        pred_path = pred_path_dict[item]
        if not all([set(ref).issubset(set(pred_path)) for ref in golden_path]):
            print('warning::{}'.format(item))
        baseline_paths = [baseline_path_dict[item][baseline] for baseline in baselines]
        
        for system, pred_out in zip(["GCN"]+baselines, [pred_path] + baseline_paths):
            acc = sum([1 if pred_out == ref else 0 for ref in golden_path]) > 0 and len(pred_out) > 1
            bleu = BLEU_2(copy.deepcopy(pred_out), copy.deepcopy(golden_path), triple_size)
            # bleu = n_gram_prec(copy.deepcopy(pred_out), copy.deepcopy(golden_path), thres=1, begin=False, end=False)
            
            acc_results.at[system, "{}-triple".format(triple_size)] += float(acc)
            bleu_results.at[system, "{}-triple".format(triple_size)] += float(bleu)

            acc_instance_results.at[item, system] = float(acc)
            bleu_instance_results.at[item, system] = float(bleu)
            
          

    acc_results['total'] = acc_results.sum(axis=1)
    bleu_results['total'] = bleu_results.sum(axis=1)
    
    for system in ["GCN"]+baselines:
        acc_results.loc[system] = acc_results.loc[system] / (acc_results.loc['total'] + 1e-8)
        bleu_results.loc[system] = bleu_results.loc[system] / (bleu_results.loc['total'] + 1e-8)
    
    return acc_results.at['GCN', 'total'], bleu_results.at['GCN', 'total']


def eval_acc_strict(pred_path_dict, golden_path_dict):
    stat_acc, stat_bleu = np.zeros((5, 7)), np.zeros((5, 7))
    for item, golden_path in golden_path_dict.items():
        # print("Golden::{}".format('\n'.join([print_path(p) for p in golden_path])))
        triple_size = max([len(path) - 1 for path in golden_path])
        if not all([len(path) == triple_size+1 for path in golden_path]) or triple_size < 1 or not len(golden_path) == 3:  # the human refs are not of the same len
            continue
        # if min([len(path) - 1 for path in golden_path]) < 1:
        #     continue
        max_ind = np.argmax(np.array([len(path) - 1 for path in golden_path]))
        # total += 1
        stat_acc[2][triple_size] += 1
        stat_acc[4][triple_size] += 1
        pred_path = pred_path_dict[item]

        rand_path = copy.deepcopy(golden_path[max_ind])
        np.random.shuffle(rand_path)

        if len(golden_path) > 1:
            j = random.sample(range(len(golden_path)), 1)[0]
            ref_path = [path for i, path in enumerate(golden_path) if not i == j]
            human_path = golden_path[j]
            # if not all([len(path) == len(human_path)for path in golden_path]):  # the human refs are not of the same len
            #     stat_acc[5][triple_size] += 1
        else:
            ref_path = golden_path
            human_path = golden_path[0]

        for i, path in zip([0, 1, 3], [rand_path, pred_path, human_path]):
            if i == 3:  # human agreement
                acc = sum([1 if human_path == ref else 0 for ref in ref_path]) > 0
                bleu = n_gram_prec(human_path, ref_path, thres=1)
            else:
                acc = sum([1 if path==ref else 0 for ref in golden_path]) > 1
                bleu = n_gram_prec(path, golden_path, thres=2)

            stat_acc[i][triple_size] += acc
            stat_bleu[i][triple_size] += bleu

    stat_bleu[2] = stat_acc[2]
    stat_bleu[4] = stat_acc[4]
    # stat_bleu[5] = stat_acc[5]


    stat_sum = stat_acc.sum(axis=1, keepdims=True)
    stat_acc = np.concatenate((stat_acc, stat_sum), axis=1)
    stat_sum = stat_bleu.sum(axis=1, keepdims=True)
    stat_bleu = np.concatenate((stat_bleu, stat_sum), axis=1)

    for i, stat in enumerate([stat_acc, stat_bleu]):

        stat[0] = stat[0] / (stat[2] + 0.0000001)
        stat[1] = stat[1] / (stat[2] + 0.0000001)
        stat[3] = stat[3] / (stat[4] + 0.0000001)
    return stat_acc, stat_bleu

    np.set_printoptions(formatter={'float_kind': lambda x: "%.2f" % x})
    print(stat_acc)
    print('\n')
    print(stat_bleu)


