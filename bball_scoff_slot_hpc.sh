#!/bin/bash
echo Running on $HOSTNAME
# source ../../cas_env/bin/activate
source ~/.bashrc
conda activate /scratch/cristianmeo/.conda/envs/pytorch

wandb login cc879c952bfc023d10e378c7a850ba349227cd1c

experiment_name="BBALL_4_678_SASBD"
cfg_json="configs/scoff/scoff_slot.json"
core="SCOFF"
should_resume="false"
save_freq=10
num_hidden=9
num_slots=9
k=9
task="bball"
dataset_dir="/scratch/cristianmeo/bouncingball"
ball_options='transfer'
ball_trainset="4ball"
ball_testset="678ball"
batch_size=64
epochs=300


DISABLE_ARTIFACT=1 python3 train_mmnist.py --experiment_name $experiment_name --cfg_json $cfg_json --core $core --should_resume $should_resume --save_frequency $save_freq \
    --batch_size $batch_size --epochs $epochs\
    --k $k --num_hidden $num_hidden --num_slots $num_slots \
    --task $task --ball_options $ball_options --ball_trainset $ball_trainset --ball_testset $ball_testset \
    --dataset_dir $dataset_dir
