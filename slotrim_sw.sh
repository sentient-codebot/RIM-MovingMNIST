#!/bin/zsh
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

experiment_name="MMNIST_3SLOT_6RIM_SW"
cfg_json="configs/rim/rim_slot.json"
core="RIM"
should_resume="false"
save_freq=25
num_slots=3
num_hidden=6
k=6
use_sw="true"
memory_size=100
num_memory_slots=3
batch_size=32

python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq \
    --batch_size $batch_size \
    --num_slots $num_slots \
    --k $k --num_hidden $num_hidden \
    --use_sw $use_sw --memory_size $memory_size --num_memory_slots $num_memory_slots
