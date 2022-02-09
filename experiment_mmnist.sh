#!/bin/zsh
echo Running on $HOSTNAME
#source ../../cas_env/bin/activate

train_dataset="balls3curtain64.h5"
hidden_size=100
should_save_csv="False"
lr=0.0007
num_units=6
k=4
num_input_heads=1
version=0
rnn_cell="GRU"
input_key_size=64
input_value_size=2040
input_query_size=64
input_dropout=0.1
comm_key_size=32
comm_value_size=32
comm_query_size=32
num_comm_heads=4
comm_dropout=0.1
experiment_name="MMNIST_complete"
batch_size=64
epochs=200
should_resume="False"
batch_frequency_to_log_heatmaps=10
model_persist_frequency=10

nohup python3 train_mmnist.py --train_dataset $train_dataset --hidden_size $hidden_size --should_save_csv $should_save_csv --lr $lr --num_units $num_units --k $k --num_input_heads $num_input_heads --version $version --rnn_cell $rnn_cell --input_key_size $input_key_size --input_value_size $input_value_size --input_query_size $input_query_size --input_dropout $input_dropout --comm_key_size $comm_key_size --comm_value_size $comm_value_size --comm_query_size $comm_query_size --num_comm_heads $num_comm_heads --comm_dropout $comm_dropout --experiment_name $experiment_name --batch_size $batch_size --epochs $epochs --should_resume $should_resume --batch_frequency_to_log_heatmaps $batch_frequency_to_log_heatmaps --model_persist_frequency $model_persist_frequency
