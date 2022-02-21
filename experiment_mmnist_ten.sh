#!/bin/zsh
echo Running on $HOSTNAME
#source ../../cas_env/bin/activate
#10 RIMs, maybe will specialize over 10 digits

experiment_name="MMNIST_complete"
cfg_json="configs/rim_complete/rim_complete.json"
core="RIM"
should_resume="false"
save_freq=25
num_units=10
k=3

nohup python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --model_persist_frequency $save_freq -num_units $num_units -k $k
