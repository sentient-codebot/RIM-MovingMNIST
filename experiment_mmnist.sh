#!/bin/zsh
echo Running on $HOSTNAME
#source ../../cas_env/bin/activate

experiment_name="MMNIST_complete"
cfg_json="configs/rim_complete/rim_complete.json"
core="RIM"
should_resume="false"
save_freq=25
loss_fn='MSE' # MMNIST, MSE by default

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --model_persist_frequency $save_freq --loss_fn $loss_fn
