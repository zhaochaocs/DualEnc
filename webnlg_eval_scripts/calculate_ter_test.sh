#!/bin/bash

# compute BLEU
encoder=$1
# if delex: delex; notdelex
if_delex=$2
# eval data: dev; test
eval_data=$3

export TEST_TARGETS_REF=data/golden/${eval_data}-all-notdelex-reference-ter.lex

export TEST_PRED=data/webnlg/pred-${eval_data}-webnlg-all-${encoder}-${if_delex}-relex-ter.txt

java -jar eval-tools/tercom-0.7.25/tercom.7.25.jar -r ${TEST_TARGETS_REF} -h ${TEST_PRED} > ter.res
