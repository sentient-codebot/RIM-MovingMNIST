#!/bin/zsh
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

experiment_name="MMNIST_basic_nonflatten_sep_sbd"
cfg_json="configs/rim/rim_basic.json"
core="RIM"
should_resume="false"
save_freq=25
encoder="NONFLATTEN"
decoder="SEP_SBD"

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq --encoder_type $encoder --decoder_type $decoder
