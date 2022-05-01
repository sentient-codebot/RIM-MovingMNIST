#!/bin/zsh
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

experiment_name="MMNIST_SASBD_OF_competition"
cfg_json="configs/scoff/scoff_slot.json"
should_resume="false"
save_freq=50
k=6
sbd_mem_efficient="true"

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --should_resume $should_resume --save_frequency $save_freq \
    --k $k \
    --sbd_mem_efficient $sbd_mem_efficient
