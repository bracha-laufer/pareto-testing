#! /bin/bash

set -ex

task=$1

log_dir="logs_"$task
mkdir -p - $log_dir

python data_utils/process_data.py \
       --task $task \
       --model_type 'bert-base-uncased' > $log_dir"/process_data.txt"

python data_utils/split_data.py \
       --task $task \
       --n_all 25000 \
       --n_val 5000 \
       --seed 0