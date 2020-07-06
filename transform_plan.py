#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from itertools import zip_longest
import pickle
import os
import glob
import sys
import re
import itertools
import operator
from random import shuffle

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):

        self.vocab = vocab
        with open(vocab, 'r', encoding='utf-8') as fr:
            self.vocab = [line.rstrip('\n') for line in fr]
            self.vocab = set(self.vocab)
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text, return_orig=False):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        output_token_orig = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(token)
                output_token_orig.append(token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            sub_token_orig = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True

                    break
                sub_tokens.append(cur_substr)
                sub_token_orig.append(cur_substr[2:] if start > 0 else cur_substr)
                start = end

            if is_bad:
                output_tokens.append(token)
                output_token_orig.append(token)
            else:
                output_tokens.extend(sub_tokens)
                output_token_orig.extend(sub_token_orig)
        if return_orig:
            return output_tokens, output_token_orig
        else:
            return output_tokens


def parsing_plan(plan, plan_order):
    """
    parse the plan from string to list of triples, where each triple is a list of [s,p,o]
    :param plan: string, the same as those in .triple file
    :param plan_order: list, plan order
    :return:
    """
    tripleset = plan.split(' < tsp > ')
    tripleset = [[t.strip() for t in triple.split(' | ')] for triple in tripleset]
    assert len(tripleset) >= len(plan_order)
    assert all([len(triple) == 3 for triple in tripleset])
    tripleset = [tripleset[i] for i in plan_order]
    return tripleset


def transform_plan(plan, plan_order, entity, relation, separate_rel, separate_ent, cut, finish, shuffle_plan):
    if shuffle_plan:
        plan = random_plan(plan)

    if not entity and cut:
        print("no entity included! set cut as false")
        cut = False

    if cut:
        return transform_plan2(plan, plan_order, entity, relation, separate_rel, separate_ent, cut, finish)

    tripleset = parsing_plan(plan, plan_order)
    if not separate_rel:
        for triple in tripleset:
            triple[1] = "_".join(triple[1].split())
    if not separate_ent:
        for triple in tripleset:
            triple[0] = "_".join(triple[0].split())
            triple[2] = "_".join(triple[2].split())

    if not entity:
        tripleset = [[triple[1]] for triple in tripleset]

    new_tripleset = []
    for i, triple in enumerate(tripleset):
        if len(triple) == 3:
            triple = "S| {} P| {} O| {}".format(triple[0], triple[1], triple[2])
            if finish:
                triple += " F|" if i < len(tripleset)-1 else " FF|"
        elif len(triple) == 1:
            triple = "P| {}".format(triple[0])
        new_tripleset.append(triple)
    return " ".join(new_tripleset)


def transform_plan2(plan, plan_order, entity, relation, separate_rel, separate_ent, cut, finish):
    if not entity and cut:
        print("no entity included! set cut as false")
        cut = False

    tripleset = parsing_plan(plan, plan_order)
    if not separate_rel:
        for triple in tripleset:
            triple[1] = "_".join(triple[1].split())
    if not separate_ent:
        for triple in tripleset:
            triple[0] = "_".join(triple[0].split())
            triple[2] = "_".join(triple[2].split())
    if not entity:
        tripleset = [[triple[1]] for triple in tripleset]

    new_tripleset = []

    prev_triple, next_triple = None, None
    for i, triple in enumerate(tripleset):
        subj_label, pred_label, obj_label = "S|", "P|", "O|"

        if i > 0:
            prev_triple = tripleset[i-1]
        if i < len(tripleset)-1:
            next_triple = tripleset[i+1]

        if next_triple is not None and triple[2] == next_triple[0]:
            obj_label = "B|"
        if prev_triple is not None and (triple[0] == prev_triple[0] or triple[0] == prev_triple[2]):
            new_tripleset.append("{} {} {} {}".format(pred_label, triple[1], obj_label, triple[2]))
        else:
            new_tripleset.append("{} {} {} {} {} {}".format(subj_label, triple[0], pred_label, triple[1], obj_label, triple[2]))

        if finish:
            new_tripleset[-1] += " F|" if i < len(tripleset)-1 else " FF|"

    return " ".join(new_tripleset)


def random_plan(plan):
    tmp = [triple.strip() for triple in plan.split("S|") if len(triple.strip())]
    shuffle(tmp)
    randomed_plan = " ".join(["S| " + triple for triple in tmp])
    return randomed_plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform plan')
    parser.add_argument("--plan_pkl_file", default=None, help="plan pkl file")
    parser.add_argument("--id_file", help="id file")
    parser.add_argument("--plan_txt_file", default=None, help="original plan file, nondelex.triple")
    parser.add_argument("--plan_order_file", default=None, help="plan order file, rel-order.txt")
    parser.add_argument("--new_plan_txt_file", help="new plan file")
    parser.add_argument("--target_file", default=None, help="new plan file")
    parser.add_argument("-e", "--entity", action="store_true", help="keep entity")
    parser.add_argument("-r", "--relation", action="store_true", help="keep relation")
    parser.add_argument("-sr", "--separate_rel", action="store_true", help="separate entity tokens")
    parser.add_argument("-se", "--separate_ent", action="store_true", help="separate relation tokens")
    parser.add_argument("-c", "--cut", action="store_true", help="cut bridge entities")
    parser.add_argument("-f", "--finish", action="store_true", help="add a special finish token after each triple")
    parser.add_argument("-d", "--random", action="store_true", help="create random (shuffled) plans")
    parser.add_argument("--bpe", action="store_true", help="bpe tokenization")
    parser.add_argument("--base", action="store_true", help="no s p o role")

    args = parser.parse_args()

    if args.bpe:
        tokenizer = WordpieceTokenizer(vocab='data/vocab.txt')

    if args.plan_txt_file is not None:

        with open(args.plan_txt_file, 'r', encoding='utf-8') as fr, \
             open(args.plan_order_file, 'r', encoding='utf-8') as fr2, \
                open(args.new_plan_txt_file, 'w', encoding='utf-8') as fw:
            for line, line2 in zip(fr, fr2):
                line = line.strip()
                line2 = line2.strip()
                if not len(line):
                    continue
                assert len(line2) > 0
                plan_order = list(map(int, line2.split()))
                new_plan = transform_plan(line, plan_order, args.entity, args.relation, args.separate_rel, args.separate_ent,
                                          args.cut, args.finish, args.random)
                if args.bpe:
                    assert args.target_file is not None
                    new_plan = " ".join(tokenizer.tokenize(new_plan))
                if args.base:
                    for role in ["S|", "P|", "O|", "B|", "F|", "FF|"]:
                        new_plan = new_plan.replace(role, "")
                    new_plan = new_plan.strip()
                fw.write(new_plan + '\n')

    else:
        assert args.plan_pkl_file is not None

        with open(args.plan_pkl_file, 'rb') as frb_tp:
            triple_reordered_dict = pickle.load(frb_tp)

        with open(args.id_file, 'r', encoding='utf-8') as fr_id, \
                open(args.new_plan_txt_file, 'w', encoding='utf-8') as fw_tp:
            for id in fr_id:
                id = id.strip()
                if not len(id):
                    continue
                tripleset_id = id.rsplit("_", 1)[0]
                plan = triple_reordered_dict[tripleset_id].strip()

                new_plan = transform_plan(plan, args.entity, args.relation, args.separate_rel, args.separate_ent,
                                          args.cut, args.finish, args.random)
                if args.bpe:
                    assert args.target_file is not None
                    new_plan = " ".join(tokenizer.tokenize(new_plan))
                if args.base:
                    for role in ["S|", "P|", "O|", "B|", "F|", "FF|"]:
                        new_plan = new_plan.replace(role, "")
                    new_plan = new_plan.strip()
                fw_tp.write(new_plan + '\n')


    if args.bpe:
        target_list = []
        with open(args.target_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                if not len(line):
                    continue
                new_line = " ".join(tokenizer.tokenize(line))
                target_list.append(new_line)
        with open(args.target_file, 'w', encoding='utf-8') as fw:
            for target in target_list:
                fw.write(target + '\n')
