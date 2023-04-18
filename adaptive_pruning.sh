#! /bin/bash

set -ex

task=$1
finetune_dir="finetune_"$task"_bert"
ee_dir="early_exit_"$task"_bert"
token_cont_dir="token_contribution_"$task"_bert"
log_dir="logs_"$task

python train_adaptive_pruning/finetune_task.py \
       --task=$task \
       --model_type='bert-base-uncased' \
       --output_dir=$finetune_dir \
       --per_device_train_batch_size=16 \
       --per_device_eval_batch_size=64 \
       --warmup_steps=500 \
       --weight_decay=0.01 \
       --num_train_epochs=3 > $log_dir"/finetune_"$task"_bert.txt"

python train_adaptive_pruning/compute_head_importance.py \
       --task=$task \
       --model_type=$finetune_dir \
       --data_type=val \
       --batch_size=8 

python train_adaptive_pruning/train_early_exit.py \
       --task=$task\
       --model_type=$finetune_dir\
       --output_dir=$ee_dir\
       --per_device_eval_batch_size=16\
       --per_device_eval_batch_size=64\
       --warmup_steps=500\
       --weight_decay=0.01\
       --num_train_epochs=3\
       --early_pooler_hidden_size=32 > $log_dir"/early_exit_"$task"_bert.txt"       
       
python train_adaptive_pruning/compute_token_contribution.py \
       --task=$task \
       --model_type=$finetune_dir \
       --batch_size=8 
    
python train_adaptive_pruning/train_token_contribution.py \
       --task=$task \
       --model_type=$finetune_dir \
       --output_dir=$token_cont_dir \
       --per_device_train_batch_size=8 \
       --per_device_eval_batch_size=32 \
       --learning_rate=5e-2 \
       --num_train_epochs=5 \
       --early_pooler_hidden_size=32 > $log_dir"/predict_token_contribution_"$task"_bert.txt"              
