#!/bin/zsh
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

experiment_name="MMNIST_ind_dec_pos_emb_norm_attn_prob"
cfg_json="configs/rim_complete/rim_complete.json"
core="RIM"
should_resume="false"
save_freq=25
k=3
bs=32

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --model_persist_frequency $save_freq --k $k --batch_size $bs
