#!/bin/zsh
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

experiment_name="PRETRAIN_MMNIST_SLOT"
cfg_json="configs/rim/rim_slot.json"
core="RIM"
should_resume="false"
save_freq=25
k=3
bs=32
dataset_dir="./data"
sbd_mem_efficient="true"

python3 pretrain_autoencoder.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq --k $k --batch_size $bs \
    --dataset_dir $dataset_dir \
    --sbd_mem_efficient $sbd_mem_efficient
