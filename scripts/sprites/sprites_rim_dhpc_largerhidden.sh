#!/bin/bash
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/.bashrc
conda activate /tudelft.net/staff-umbrella/nanthesis/.conda/envs/pytorch

# wandb login cc879c952bfc023d10e378c7a850ba349227cd1c

experiment_name="SPRITES_LARGERHIDDEN"
cfg_json="configs/rim/rim_basic.json"
use_slot_attention="false"
encoder_type="NONFLATTEN"
decoder_type="SEP_BASIC"
core="RIM"
should_resume="false"
save_freq=10
num_hidden=5
hidden_size=200
num_slots=5
k=5
task="SPRITESMOT"
dataset_dir="/tudelft.net/staff-umbrella/nanthesis/data"
batch_size=64
epochs=400


DISABLE_ARTIFACT=1 python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq \
    --batch_size $batch_size --epochs $epochs \
    --use_slot_attention $use_slot_attention --encoder_type $encoder_type --decoder_type $decoder_type \
    --k $k --num_hidden $num_hidden --num_slots $num_slots \
    --hidden_size $hidden_size \
    --dataset_dir $dataset_dir --task $task
