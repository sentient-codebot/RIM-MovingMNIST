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

experiment_name="MMNIST_SEPBASIC"
cfg_json="configs/rim/rim_basic.json"
core="RIM"
should_resume="false"
save_freq=25
num_hidden=6
num_slots=3
k=6
decoder_type="SEP_BASIC"

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq \
    --k $k --num_hidden $num_hidden --num_slots $num_slots \
    --decoder_type $decoder_type \
    --dataset_dir $dataset_dir
