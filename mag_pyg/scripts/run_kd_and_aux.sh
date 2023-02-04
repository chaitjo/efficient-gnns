
#!/bin/bash

# Experimental setup:
# - We tune the loss balancing weights α, β in for all techniques on the validation set when using both logit-based and representation distillation together. 
# - For KD, we tune α ∈ {0.8, 0.9}, τ1 ∈ {4, 5}. 
# - For FitNet, AT, LSP, and GSP, we tune β ∈ {100, 1000, 10000} and the kernel for LSP, GSP. 
# - For G-CRD, we tune β ∈ {0.01, 0.05}, τ2 ∈ {0.05, 0.075, 0.1} and the projection head. 
# When comparing representation distillation methods, we set α to 0 in order to ablate performance and reduce β by one order of magnitude.

tmux new -s mag -d
tmux send-keys "conda activate ogb" C-m

runs=5

num_layers=2
hidden_channels=32

############################

training=fitnet
beta=1

tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta &
wait" C-m

############################

training=at
beta=100

tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta &
wait" C-m

# ############################

expt_name=graph_saint/lpw-rbf
training=lpw
kernel=rbf
beta=1

tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta --kernel $kernel --expt_name $expt_name &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta --kernel $kernel --expt_name $expt_name &
wait" C-m

############################

expt_name=graph_saint/lpw-cosine
training=lpw
kernel=cosine
beta=1

tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta --kernel $kernel --expt_name $expt_name &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta --kernel $kernel --expt_name $expt_name &
wait" C-m

############################

expt_name=graph_saint/lpw-poly
training=lpw
kernel=poly
beta=1

tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta --kernel $kernel --expt_name $expt_name &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta --kernel $kernel --expt_name $expt_name &
wait" C-m

############################

expt_name=graph_saint/gpw-cosine
training=gpw
kernel=cosine
beta=1
max_samples=24576

tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta --kernel $kernel --expt_name $expt_name --max_samples $max_samples --expt_name $expt_name &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta --kernel $kernel --expt_name $expt_name --max_samples $max_samples --expt_name $expt_name &
wait" C-m

############################

expt_name=graph_saint/gpw-poly
training=gpw
kernel=poly
beta=1
max_samples=24576

tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta --kernel $kernel --expt_name $expt_name --max_samples $max_samples --expt_name $expt_name &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta --kernel $kernel --expt_name $expt_name --max_samples $max_samples --expt_name $expt_name &
wait" C-m

############################

expt_name=graph_saint/nce
training=nce
beta=0.1
nce_T=0.075
max_samples=24576

tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta --nce_T $nce_T --max_samples $max_samples --expt_name $expt_name &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --num_layers $num_layers --hidden_channels $hidden_channels --training $training --beta $beta --nce_T $nce_T --max_samples $max_samples --expt_name $expt_name &
wait" C-m

###########################

tmux send-keys "tmux kill-session -t mag" C-m
