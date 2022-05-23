#!/bin/zsh
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/.bashrc
conda activate ~/anaconda3

experiment_name="MMNIST_SASBD_SPOTLIGHT"
cfg_json="configs/rim/rim_slot.json"
dataset_dir='/home/cristianmeo/movingmnist'
core="RIM"
should_resume="false"
save_freq=25
num_hidden=3
num_slots=3
k=3
spotlight="true"
decode_hidden="false"
decoder_type="SEP_BASIC"

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq --k $k --num_hidden $num_hidden --num_slots $num_slots --spotlight_bias $spotlight --dataset_dir $dataset_dir --decode_hidden $decode_hidden --decoder_type $decoder_type
