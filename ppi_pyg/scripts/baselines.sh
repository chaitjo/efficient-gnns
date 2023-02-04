
#!/bin/bash

tmux new -s baselines -d
tmux send-keys "conda activate ogb" C-m

runs=5

############################

gnn=student
num_layers=5
hidden_channels=68

training=supervised
expt_name=supervised
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
wait" C-m

training=kd
expt_name=kd
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
wait" C-m

############################

gnn=gcn
num_layers=2
hidden_channels=256

training=supervised
expt_name=supervised
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
wait" C-m

training=kd
expt_name=kd
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
wait" C-m

############################

gnn=sage
num_layers=2
hidden_channels=256

training=supervised
expt_name=supervised
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
wait" C-m

training=kd
expt_name=kd
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
wait" C-m

############################

gnn=gat
num_layers=2
hidden_channels=64  # 4 heads

training=supervised
expt_name=supervised
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
wait" C-m

training=kd
expt_name=kd
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels &
wait" C-m


tmux send-keys "tmux kill-session -t baselines" C-m
