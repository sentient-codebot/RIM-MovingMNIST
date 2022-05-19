#!/bin/zsh
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

experiment_name="MMNIST_SEPBASIC"
cfg_json="configs/rim/rim_basic.json"
core="RIM"
should_resume="false"
save_freq=25
num_hidden=6
num_slots=3
k=6
decoder_type="SEP_BASIC"

DEBUG=1 python3 test_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq \
    --k $k --num_hidden $num_hidden --num_slots $num_slots \
    --decoder_type $decoder_type
