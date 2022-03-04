#!/bin/zsh
echo Running on $HOSTNAME
#source ../../cas_env/bin/activate
conda activate pytorch

experiment_name="MMNIST_dropout"
cfg_json="configs/rim_complete/rim_complete_dropout.json"
core="RIM"
should_resume="false"
save_freq=25
version=0
num_units=10
k=3
rim_dropout=0.3

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --model_persist_frequency $save_freq --version $version --num_units $num_units --k $k --rim_dropout $rim_dropout
