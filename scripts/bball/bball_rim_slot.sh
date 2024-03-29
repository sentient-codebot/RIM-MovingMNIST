#!/bin/zsh
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

experiment_name="BBALL_slotrim_3_6_678ball"
cfg_json="configs/rim/rim_slot.json"
core="RIM"
should_resume="false"
save_freq=10
num_hidden=9
num_slots=9
k=6
task="bball"
ball_options='transfer'
ball_trainset="4ball"
ball_testset="678ball"
batch_size=64
epochs=200


python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq \
    --batch_size $batch_size --epochs $epochs\
    --k $k --num_hidden $num_hidden --num_slots $num_slots \
    --task $task --ball_options $ball_options --ball_trainset $ball_trainset --ball_testset $ball_testset
