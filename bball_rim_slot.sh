#!/bin/zsh
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

experiment_name="BBALL_slotrim_3_6_678ball"
cfg_json="configs/rim/rim_slot.json"
core="RIM"
should_resume="false"
save_freq=25
num_hidden=6
num_slots=9
k=6
task="bball"
ball_options='transfer'
ball_trainset="678ball"
ball_testset="4ball"
batch_size=32


python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq \
    --batch_size $batch_size \
    --k $k --num_hidden $num_hidden --num_slots $num_slots \
    --task $task --ball_options $ball_options --ball_trainset $ball_trainset --ball_testset $ball_testset
