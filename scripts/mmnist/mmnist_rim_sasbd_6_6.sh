#!/bin/bash
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/.bashrc
conda activate pytorch

experiment_name="MMNIST_SASBD_6_6"
cfg_json="configs/rim/rim_slot.json"
dataset_dir='data'
core="RIM"
should_resume="false"
save_freq=25
num_hidden=6
num_slots=6
k=6
spotlight="false"
decode_hidden="false"

DISABLE_ARTIFACT=1 python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq --k $k --num_hidden $num_hidden --num_slots $num_slots --spotlight_bias $spotlight --dataset_dir $dataset_dir \
    --decode_hidden $decode_hidden \

