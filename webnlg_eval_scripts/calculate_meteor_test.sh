#!/bin/bash

# compute BLEU
encoder=$1
# if delex: delex; notdelex
if_delex=$2
# eval data: dev; test
eval_data=$3

export TEST_TARGETS_REF=data/golden/${eval_data}-all-notdelex-reference-meteor.lex

export TEST_PRED=data/webnlg/pred-${eval_data}-webnlg-all-${encoder}-${if_delex}-relex.txt

java -Xmx2G -jar eval-tools/meteor-1.5/meteor-1.5.jar ${TEST_PRED} ${TEST_TARGETS_REF} -l en -norm -r 3 > meteor.res
