#!/usr/bin/env bash

set -e

# Please download data from https://github.com/ThiagoCF05/webnlg/tree/master/data/v1.5/en and then
# put the three folders under data/webnlg

# Please select if delex: delex; notdelex
if_delex="notdelex"

cd data/webnlg/
python3 ../../webnlg_eval_scripts/webnlg_gcnonmt_input.py -i ./ -c all -l -e
python3 ../../webnlg_eval_scripts/webnlg_gcnonmt_input.py -i ./ -p test -c all -l -e
cd ../..

python3 plan_train.py -model_output data/planner.pt \
        -train_src data/webnlg/train-webnlg-all-${if_delex}-src-nodes.txt \
        -train_src2 data/webnlg/train-webnlg-all-${if_delex}-src.txt \
        -train_label data/webnlg/train-webnlg-all-${if_delex}-src-labels.txt \
        -train_node1 data/webnlg/train-webnlg-all-${if_delex}-src-node1.txt \
        -train_node2 data/webnlg/train-webnlg-all-${if_delex}-src-node2.txt \
        -train_plan data/webnlg/train-webnlg-all-${if_delex}-src-rel-order.txt \
        -train_split data/webnlg/train-webnlg-all-${if_delex}-src-split.txt \
        -valid_src data/webnlg/dev-webnlg-all-${if_delex}-src-nodes.txt \
        -valid_src2 data/webnlg/dev-webnlg-all-${if_delex}-src.txt \
        -valid_label data/webnlg/dev-webnlg-all-${if_delex}-src-labels.txt \
        -valid_node1 data/webnlg/dev-webnlg-all-${if_delex}-src-node1.txt \
        -valid_node2 data/webnlg/dev-webnlg-all-${if_delex}-src-node2.txt \
        -test_src data/webnlg/test-webnlg-all-${if_delex}-src-nodes.txt \
        -test_src2 data/webnlg/test-webnlg-all-${if_delex}-src.txt \
        -test_label data/webnlg/test-webnlg-all-${if_delex}-src-labels.txt \
        -test_node1 data/webnlg/test-webnlg-all-${if_delex}-src-node1.txt \
        -test_node2 data/webnlg/test-webnlg-all-${if_delex}-src-node2.txt

python3 plan_predict.py -model data/planner.pt \
        -test_src data/webnlg/test-webnlg-all-${if_delex}-src-nodes.txt \
        -test_src2 data/webnlg/test-webnlg-all-${if_delex}-src.txt \
        -test_label data/webnlg/test-webnlg-all-${if_delex}-src-labels.txt \
        -test_node1 data/webnlg/test-webnlg-all-${if_delex}-src-node1.txt \
        -test_node2 data/webnlg/test-webnlg-all-${if_delex}-src-node2.txt
