#!/bin/zsh
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

experiment_name="MMNIST_SASBD_SPOTLIGHT"
cfg_json="configs/rim/rim_slot.json"
dataset_dir='/home/cristianmeo/movingmnist'
core="RIM"
should_resume="false"
save_freq=25
num_hidden=6
num_slots=3
k=6
spotlight="true"
decode_hidden="false"

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq --k $k --num_hidden $num_hidden --num_slots $num_slots --spotlight_bias $spotlight --dataset_dir $dataset_dir
