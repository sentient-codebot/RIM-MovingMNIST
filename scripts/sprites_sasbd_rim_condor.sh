#!/bin/bash
echo Running on $HOSTNAME
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

experiment_name="SPRITES_SASBD"
cfg_json="configs/rim/rim_slot.json"
core="RIM"
should_resume="false"
save_freq=10
num_hidden=4
num_slots=4
k=10
task="spritesmot"
batch_size=16
epochs=300

dataset_dir="/home/nnan/sprites/train"


python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq \
    --batch_size $batch_size --epochs $epochs\
    --k $k --num_hidden $num_hidden --num_slots $num_slots \
    --task $task \
    --dataset_dir $dataset_dir
