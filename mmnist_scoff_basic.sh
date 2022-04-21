#!/bin/zsh
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

experiment_name="MMNIST_basic"
cfg_json="configs/scoff/scoff_basic.json"
should_resume="true"
save_freq=25

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --should_resume $should_resume --save_frequency $save_freq 
