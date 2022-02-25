#!/bin/zsh
echo Running on $HOSTNAME
source ../../cas_env/bin/activate

experiment_name="MMNIST_dropout"
cfg_json="configs/rim_complete/rim_complete_dropout.json"
version=3 # default:0. ver3: drop at 0.3 f1 0.59
core="RIM"
should_resume="true"

python3 saliency_map.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --version $version
