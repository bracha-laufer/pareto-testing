#!/bin/bash

set -ex

task=$1
res_folder=$task"_pruning_results" 
ee_dir="early_exit_"$task"_bert"
token_cont_dir="token_contribution_"$task"_bert"

mkdir -p - $res_folder

python evaluate_pruning.py\
       --task=$task\
       --data_type='test'\
       --model_type1=$ee_dir\
       --model_type2=$token_cont_dir\
       --res_folder=$res_folder\
       --n_ee=18\
       --min_th_ee=0.0\
       --max_th_ee=0.8\
       --n_token=20\
       --n_head=18\
       --min_th_head=0.0\
       --max_th_head=0.166\
       --per_device_eval_batch_size=425  #64 

python calibrate_control_acc_reduce_cost.py\
       --task=$task\
       --data_type='test'\
       --res_folder=$res_folder\
       --n_trials=100\
       --delta=0.1\
       --alphas=0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2   