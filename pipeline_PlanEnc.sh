#!/usr/bin/env bash

set -e

# Please select encoder: rnn; gcn
encoder="rnn"
encoder2="rnn"
data_type="text"

# Please select if delex: delex; notdelex
if_delex="notdelex"

# use_glove: yes; no; if no, remove pretrain args!!!
use_glove="no"
emb_size=256
hidden_size=256

# eval data: dev; test
eval_data="test"

model_id="${encoder}_${if_delex}"
save_model_name="tmp_${model_id}_"

model_output_dir="output/${encoder}-${encoder2}_${if_delex}"

# preprocessing, converting the webnlg data into inputs of the OpenNMT
python3 transform_plan.py --id_file data/webnlg/train-webnlg-all-${if_delex}-translate.lexid.txt \
               --plan_txt_file data/webnlg/train-webnlg-all-${if_delex}.triple \
               --plan_order_file data/webnlg/train-webnlg-all-${if_delex}-src-rel-order.txt \
               --new_plan_txt_file data/webnlg/train-webnlg-all-${if_delex}-plan.txt \
               -e -r -se -sr
python3 transform_plan.py --id_file data/webnlg/dev-webnlg-all-${if_delex}-translate.lexid.txt \
               --plan_txt_file data/webnlg/dev-webnlg-all-${if_delex}.triple \
               --plan_order_file data/webnlg/dev-webnlg-all-${if_delex}-src-rel-order.txt \
               --new_plan_txt_file data/webnlg/dev-webnlg-all-${if_delex}-plan.txt \
               -e -r -se -sr

# preprocessing, convert the sequence data into torchtext
python3 preprocess.py \
            -train_src data/webnlg/train-webnlg-all-${if_delex}-plan.txt \
            -train_src2 data/webnlg/train-webnlg-all-${if_delex}-plan.txt \
            -train_tgt data/webnlg/train-webnlg-all-${if_delex}-tgt.txt \
            -valid_src data/webnlg/dev-webnlg-all-${if_delex}-plan.txt \
            -valid_src2 data/webnlg/dev-webnlg-all-${if_delex}-plan.txt \
            -valid_tgt data/webnlg/dev-webnlg-all-${if_delex}-tgt.txt \
            -src_seq_length 500 -tgt_seq_length 500 \
            -save_data data/${model_id}_exp -data_type ${data_type} -dynamic_dict
echo "Finish preprocessing... Output files are in data/${model_id}_exp"

## train the model

for (( seed=1; seed<=10; seed++ ))
do
    echo "begin to train ... Loop :: " $seed

    python3 train.py -data data/${model_id}_exp -save_model data/${save_model_name} \
                -encoder_type rnn -rnn_type LSTM  -layers 2 \
                -word_vec_size ${emb_size} -rnn_size ${hidden_size} \
                -epochs 20 -optim adam -learning_rate 0.001 -learning_rate_decay 0.7 -seed $seed\
                -gpuid 0 -start_checkpoint_at 15 \
                -copy_attn -brnn

    echo "Begin to translate..."

    model_path=$(head -1 data/${save_model_name}.txt)

    python3 transform_plan.py --id_file data/webnlg/test-webnlg-all-${if_delex}-translate.lexid.txt \
           --plan_txt_file data/webnlg/test-webnlg-all-${if_delex}.triple \
           --plan_order_file data/webnlg/test-webnlg-all-${if_delex}-src-rel-order-pred.txt \
           --new_plan_txt_file data/webnlg/test-webnlg-all-${if_delex}-plan.txt \
           -e -r -se -sr

    # translating
    python3 translate.py -model ${model_path} -data_type ${data_type} \
                -src data/webnlg/${eval_data}-webnlg-all-${if_delex}-plan.txt \
                -src2 data/webnlg/${eval_data}-webnlg-all-${if_delex}-plan.txt \
                -src_triple data/webnlg/${eval_data}-webnlg-all-${if_delex}.triple \
                -output data/webnlg/pred-${eval_data}-webnlg-all-${encoder}-${if_delex}.txt \
                -gpu 0 -batch_size 256 \
                -replace_unk -report_bleu


    cd data/webnlg/
    python3 ../../webnlg_eval_scripts/webnlg_gcnonmt_relexicalise.py -i ./ -p ${eval_data} -c seen \
        -f pred-${eval_data}-webnlg-all-${encoder}-${if_delex}.txt -l
    cd ../..

done
