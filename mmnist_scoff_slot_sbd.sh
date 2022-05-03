#!/bin/zsh
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

experiment_name="MMNIST_SASBD_PREENC_SA"
cfg_json="configs/scoff/scoff_slot.json"
should_resume="false"
save_freq=50
k=6
sbd_mem_efficient="true"
bs=32
load_sa="true"
dataset_dir="./data"

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --should_resume $should_resume --save_frequency $save_freq \
    --k $k \
    --sbd_mem_efficient $sbd_mem_efficient \
    --batch_size $bs \
    --load_trained_slot_attention $load_sa \
    --dataset_dir $dataset_dir
