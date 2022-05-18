#!/bin/bash
echo Running on $HOSTNAME
source ~/.bashrc
conda activate ~/.conda/envs/pytorch


experiment_name="SPRITES_SASBD"
cfg_json="configs/rim/rim_slot.json"
core="RIM"
should_resume="false"
save_freq=10
num_hidden=5
num_slots=5
k=10
task="spritesmot"
batch_size=16
epochs=400

dataset_dir="/tudelft.net/staff-umbrella/nanthesis/data"


python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq \
    --batch_size $batch_size --epochs $epochs\
    --k $k --num_hidden $num_hidden --num_slots $num_slots \
    --task $task \
    --dataset_dir $dataset_dir
