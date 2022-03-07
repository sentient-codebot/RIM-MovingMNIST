#!/bin/zsh
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

experiment_name="MMNIST_sparse"
cfg_json="configs/rim_complete/rim_complete.json"
core="RIM"
should_resume="false"
save_freq=25
sparse="true"
loss_fn="BCE"

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --model_persist_frequency $save_freq --sparse $sparse --loss_fn $loss_fn
