#!/bin/zsh
echo Running on $HOSTNAME
#source ../../cas_env/bin/activate

experiment_name="MMNIST_dropout"
cfg_json="configs/rim_complete/rim_complete_dropout.json"
core="RIM"
should_resume="false"
save_freq=25
version=3
num_units=1000
k=10
hidden_size=20

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --model_persist_frequency $save_freq --version $version --k $k --num_units $num_units --hidden_size $hidden_size
