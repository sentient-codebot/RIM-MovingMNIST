#!/bin/bash
echo Running on $HOSTNAME
source ~/.bashrc
conda activate pytorch

experiment_name="MMNIST_SLOTPRED"
cfg_json="configs/rim/rim_slot.json"
core="RIM"
should_resume="true"
save_freq=25
num_hidden=6
num_slots=6
k=6
dataset_dir='data'
batch_size=16
lr=0.0005

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq \
    --k $k --num_hidden $num_hidden --num_slots $num_slots \
    --dataset_dir $dataset_dir \
    --batch_size $batch_size --lr $lr
