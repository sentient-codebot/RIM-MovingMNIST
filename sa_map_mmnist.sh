#!/bin/zsh
echo Running on $HOSTNAME
source ../../cas_env/bin/activate

experiment_name="MMNIST_complete"
cfg_json="configs/rim_complete/rim_complete.json"
core="RIM"
should_resume="true"

python3 saliency_map.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume
