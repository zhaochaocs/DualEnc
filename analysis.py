#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from os import system
import glob
import sys
import re
import itertools
import operator

import torch
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

import onmt.io
import opts

class Instance():
    def __init__(self, _id, refs, pred):
        self.id = _id
        self.refs = refs
        self.pred = pred
        self.bleu, self.meteor = None, None

    def set_bleu(self, score):
        self.bleu = score

    def set_meteor(self, score):
        self.meteor = score

def get_results_mean_std(log_file='test.txt'):
    epoch_re = r"begin to train ... Loop ::  (\d+)\n"
    bleu_re = r"BLEU = (\d+\.\d+),.*\n"
    meteor_re = r"Final score:\s+(0\.\d+)\n"
    ter_re = r"Total TER: (\d+\.\d+) \(.*\)\n"

    with open(log_file, 'r') as fr:
        content = fr.read()
        epochs = list(map(int, re.findall(epoch_re, content)))
        bleus = list(map(float, re.findall(bleu_re, content)))
        meteors = list(map(float, re.findall(meteor_re, content)))
        ters = list(map(float, re.findall(ter_re, content)))
        if len(bleus) == 2 * len(ters):
            bleus = np.array([b for i, b in enumerate(bleus) if i % 2 == 1])
        else:
            bleus = np.array(bleus)
        assert len(epochs) == len(bleus) == len(meteors) == len(ters)

        idx = bleus.argsort()[-10:][::-1]
        bleus = np.array(bleus)[idx]
        meteors = np.array(meteors)[idx]
        ters = np.array(ters)[idx]

        print("BLEU: mean {0:.4f} std {1:.4f}".format(np.mean(bleus), np.std(bleus)))
        print("METEOR: mean {0:.4f} std {1:.4f}".format(np.mean(meteors), np.std(meteors)))
        print("TER: mean {0:.4f} std {1:.4f}".format(np.mean(ters), np.std(ters)))

def get_results(results_dir, ref_num=3):
    """
    get blue results for single instance and dump the instance into json dict
    :param results_dir:
    :param ref_num:
    :return:
    """
    instance_dict = {}

    seg_ids = []
    preds = []
    refs = []

    for file in os.listdir(results_dir):
        if file.endswith('ter.txt'):
            ter_file = os.path.join(results_dir, file)
        elif file.endswith("relex.txt"):
            pred_file = os.path.join(results_dir, file)

    with open(ter_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if not len(line):
                continue
            seg_id = line.rsplit('(', 1)[1][:-1]
            seg_ids.append(seg_id)

    with open('data/golden/test-all-notdelex-reference0.lex', 'r', encoding='utf-8') as fr0, \
            open('data/golden/test-all-notdelex-reference1.lex', 'r', encoding='utf-8') as fr1, \
            open('data/golden/test-all-notdelex-reference2.lex', 'r', encoding='utf-8') as fr2:
        for ref0, ref1, ref2 in zip(fr0, fr1, fr2):
            refs_instance = []
            for ref in [ref0, ref1, ref2]:
                ref = ref.strip()
                if len(ref):
                    refs_instance.append(ref)
            assert len(refs_instance)
            refs.append(refs_instance)

    with open(pred_file, 'r', encoding='utf-8') as fr:
        for pred in fr:
            pred = pred.strip()
            if len(pred):
                preds.append(pred)

    try:
        assert len(refs) == len(preds) == len(seg_ids) == 971
    except:
        raise ValueError("Ref file and pred file have different length!")

    with open('bleu.res', 'w', encoding='utf-8') as fw:
        for seg_id, pred, ref3 in zip(seg_ids, preds, refs):
            score = sentence_bleu(ref3, pred)
            fw.write("Segment {} score:\t{}\n".format(seg_id, score))
            instance = Instance(_id=seg_id, refs=ref3, pred=pred)
            instance.set_bleu(score)
            instance_dict[seg_id] = instance
    return instance_dict


def case_analysis(results_dir1, results_dir2, res1_name, res2_name):
    """
    output the top 10 better instances according to the bleu
    :param results_dir1:
    :param results_dir2:
    :return:
    """
    instance_dict1 = get_results(results_dir1)
    instance_dict2 = get_results(results_dir2)
    _case_analysis(instance_dict1, instance_dict2, './output', res1_name, res2_name)


def get_bleu(instance_dict):
    hyps = []
    refs = []
    for instance in instance_dict.values():
        hyps.append(instance.pred)
        refs.append(instance.refs)
    bleu = corpus_bleu(refs, hyps)
    return bleu


def get_bleu_single(hyp, refs):
    reference = 'Two person be in a small race car drive by a green hill .'
    output = 'Two person in race uniform in a street car .'

    with open('output', 'w', encoding='utf-8') as output_file:
        output_file.write(hyp)

    with open('reference', 'w', encoding='utf-8') as reference_file:
        reference_file.write('\n'.join(refs))


    system('./multi-bleu.perl reference < output')


def _case_analysis(instance_dict1, instance_dict2, out_dir, res1_name, res2_name):
    bleu1 = get_bleu(instance_dict1)
    print("BLEU of {} is {}".format(res1_name, bleu1))
    bleu2 = get_bleu(instance_dict2)
    print("BLEU of {} is {}".format(res2_name, bleu2))
    instance_bleu_diff = {seg_id: instance_dict1[seg_id].bleu - instance_dict2[seg_id].bleu
                          for seg_id in instance_dict1.keys()}
    top_diffs = sorted(instance_bleu_diff.items(), key=operator.itemgetter(1), reverse=True)
    top_seg_ids = [diff[0] for diff in top_diffs[:10]]

    with open(os.path.join(out_dir, "{}_beat_{}".format(res1_name, res2_name)), 'w', encoding='utf-8') as fw:
        for seg_id in top_seg_ids:
            pred1 = instance_dict1[seg_id].pred
            pred2 = instance_dict2[seg_id].pred
            refs = "\n\t*\t".join(instance_dict1[seg_id].refs)
            bleu1, bleu2 = instance_dict1[seg_id].bleu, instance_dict2[seg_id].bleu
            output = "{}\nPRED1: {}\t{}\nPRED2: {}\t{}\nREFS: {}\n\n".format(seg_id, bleu1, pred1, bleu2, pred2, refs)
            fw.write(output)

def case_analysis2(id, results1, results2, ref1, ref2, ref3):
    refs = []
    with open(ref1, 'r', encoding='utf-8') as fr1, open(ref2, 'r', encoding='utf-8') as fr2, open(ref3, 'r', encoding='utf-8') as fr3:
        for line1, line2, line3 in zip(fr1, fr2, fr3):
            refs_instances = []
            for ref in [line1, line2, line3]:
                ref = ref.strip()
                if len(ref):
                    refs_instances.append(ref)
            refs.append(refs_instances)

    instance_dict1, instance_dict2 = {}, {}
    with open(id, 'r', encoding='utf-8') as fr_id, open(results1, 'r', encoding='utf-8') as fr1, open(results2, 'r', encoding='utf-8') as fr2:
        for i, (seg_id, pred1, pred2) in enumerate(zip(fr_id, fr1, fr2)):
            instance1 = Instance(_id=seg_id, refs=refs[i], pred=pred1)
            instance1.set_bleu(sentence_bleu(refs[i], pred1))
            instance2 = Instance(_id=seg_id, refs=refs[i], pred=pred2)
            instance2.set_bleu(sentence_bleu(refs[i], pred2))

            instance_dict1[seg_id] = instance1
            instance_dict2[seg_id] = instance2







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Results analysis')
    parser.add_argument("-c", "--compare", help="whether to compare. If true, two dirs should be provided",
                        action="store_true")
    parser.add_argument('--results_dir1', help='results_dir1')
    parser.add_argument('--results_dir2', help='results_dir2')
    parser.add_argument('--results_name1', default=None, help='results_name1')
    parser.add_argument('--results_name2', default=None, help='results_name2')
    args = parser.parse_args()

    if not args.compare:
        print("Eval results for {} ::".format(args.results_dir1))
        if Path(args.results_dir1).is_file():
            get_results_mean_std(args.results_dir1)
        else:
            get_results_mean_std(os.path.join(args.results_dir1, 'myout'))
    else:
        if args.results_name1 is None:
            args.results_name1 = os.path.basename(os.path.normpath(args.results_dir1))
            args.results_name2 = os.path.basename(os.path.normpath(args.results_dir2))

        if args.compare:
            case_analysis(args.results_dir1, args.results_dir2, args.results_name1, args.results_name2)
        else:
            print("Eval results for {} ::".format(args.results_dir1))
            if Path(args.results_dir1).is_file():
                get_results_mean_std(args.results_dir1)
            else:
                get_results_mean_std(os.path.join(args.results_dir1, 'myout'))




