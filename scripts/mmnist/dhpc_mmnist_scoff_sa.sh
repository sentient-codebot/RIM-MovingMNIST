#!/bin/bash
echo Running on $HOSTNAME
source ~/.bashrc
conda activate pytorch

dataset_dir='/scratch/cristianmeo/data'

experiment_name="MMNIST_SACATBASIC"
cfg_json="configs/scoff/scoff_slot.json"
should_resume="false"
save_freq=50

decoder_type="CAT_BASIC"

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --should_resume $should_resume --save_frequency $save_freq \
    --dataset_dir $dataset_dir \
    --decoder_type $decoder_type

