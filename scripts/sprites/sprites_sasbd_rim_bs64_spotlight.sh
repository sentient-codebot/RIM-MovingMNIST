#!/bin/bash
echo Running on $HOSTNAME
source ~/.bashrc
conda activate ~/anaconda3


experiment_name="SPRITES_SEP_BASIC_BS64"
cfg_json="configs/rim/rim_slot.json"
core="RIM"
should_resume="false"
save_freq=10
num_hidden=5
num_slots=5
k=10
decoder_type="SEP_BASIC"
task="spritesmot"
batch_size=64
epochs=400
decode_hidden="false"
dataset_dir="/home/cristianmeo/Datasets"
spotlight_bias="False"


python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq \
    --batch_size $batch_size --epochs $epochs\
    --k $k --num_hidden $num_hidden --num_slots $num_slots \
    --task $task \
    --dataset_dir $dataset_dir \
    --decode_hidden $decode_hidden \
    --spotlight_bias $spotlight_bias \
    --decoder_type $decoder_type
