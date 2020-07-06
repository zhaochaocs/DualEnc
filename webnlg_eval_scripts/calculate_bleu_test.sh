#!/bin/bash

# compute BLEU
encoder=$1
# if delex: delex; notdelex
if_delex=$2
# eval data: dev; test
eval_data=$3
# part: seen; unseen; all
part=$4

export TEST_TARGETS_REF0=data/golden/${eval_data}-${part}-notdelex-reference0.lex
export TEST_TARGETS_REF1=data/golden/${eval_data}-${part}-notdelex-reference1.lex
export TEST_TARGETS_REF2=data/golden/${eval_data}-${part}-notdelex-reference2.lex
# export TEST_TARGETS_REF3=../data/webnlg/all-notdelex-reference3.lex
# export TEST_TARGETS_REF4=../data/webnlg/all-notdelex-reference4.lex
# export TEST_TARGETS_REF5=../data/webnlg/all-notdelex-reference5.lex
# export TEST_TARGETS_REF6=../data/webnlg/all-notdelex-reference6.lex
# export TEST_TARGETS_REF7=../data/webnlg/all-notdelex-reference7.lex

export TEST_PRED=data/webnlg/pred-${eval_data}-webnlg-${part}-${encoder}-${if_delex}-relex.txt

./webnlg_eval_scripts/multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} < ${TEST_PRED}

