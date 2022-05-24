#!/bin/bash
echo Running on $HOSTNAME
source ~/.bashrc
conda activate pytorch

proj_dir="/tudelft.net/staff-umbrella/nanthesis/data"
scratch_dir='/scratch/cristianmeo/data'
if [ -d $proj_dir ] 
then 
    dataset_dir=$proj_dir
else
    dataset_dir=$scratch_dir
fi

experiment_name="SPRITES_SISASBD"
cfg_json="configs/rim/rim_slot.json"
use_past_slots='true'
core="RIM"
should_resume="false"
save_freq=25
k=6

DISABLE_ARTIFACT=1 python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq \
    --k $k \
    --dataset_dir $dataset_dir \
    --use_past_slots $use_past_slots \
    --task "spritesmot"
