#!/bin/zsh
echo Running on $HOSTNAME
#source ../../cas_env/bin/activate

experiment_name="MMNIST_GRU"
cfg_json="configs/rim_complete/rim_complete.json"
core="GRU"
should_resume="false"
save_freq=25

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --model_persist_frequency $save_freq
