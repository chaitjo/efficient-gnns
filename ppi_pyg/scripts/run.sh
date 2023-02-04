
#!/bin/bash

# Experimental setup:
# - We tune the loss balancing weights α, β in for all techniques on the validation set when using both logit-based and representation distillation together. 
# - For KD, we tune α ∈ {0.8, 0.9}, τ1 ∈ {4, 5}. 
# - For FitNet, AT, LSP, and GSP, we tune β ∈ {100, 1000, 10000} and the kernel for LSP, GSP. 
# - For G-CRD, we tune β ∈ {0.01, 0.05}, τ2 ∈ {0.05, 0.075, 0.1} and the projection head. 
# When comparing representation distillation methods, we set α to 0 in order to ablate performance and reduce β by one order of magnitude.

tmux new -s all -d
tmux send-keys "conda activate ogb" C-m

runs=5

gnn=student
num_layers=5
hidden_channels=68

############################

training=fitnet
expt_name=fitnet
beta=1000

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
wait" C-m

############################

training=at
expt_name=at
beta=1000

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
wait" C-m

############################

training=lpw
expt_name=lpw-rbf
beta=100
kernel=rbf

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=lpw
expt_name=lpw-poly
beta=100
kernel=poly

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=lpw
expt_name=lpw-cosine
beta=100
kernel=cosine

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=lpw
expt_name=lpw-l2
beta=100
kernel=l2

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=nce
expt_name=nce
beta=0.1
nce_T=0.075
max_samples=16384
proj_dim=256

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --nce_T $nce_T --max_samples $max_samples --proj_dim $proj_dim &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --nce_T $nce_T --max_samples $max_samples --proj_dim $proj_dim &
wait" C-m














gnn=gcn
num_layers=2
hidden_channels=256

############################

training=fitnet
expt_name=fitnet
beta=1000

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
wait" C-m

############################

training=at
expt_name=at
beta=1000

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
wait" C-m

############################

training=lpw
expt_name=lpw-rbf
beta=100
kernel=rbf

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=lpw
expt_name=lpw-poly
beta=100
kernel=poly

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=lpw
expt_name=lpw-cosine
beta=100
kernel=cosine

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=lpw
expt_name=lpw-l2
beta=100
kernel=l2

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=nce
expt_name=nce
beta=0.1
nce_T=0.075
max_samples=16384
proj_dim=256

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --nce_T $nce_T --max_samples $max_samples --proj_dim $proj_dim &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --nce_T $nce_T --max_samples $max_samples --proj_dim $proj_dim &
wait" C-m








gnn=sage
num_layers=2
hidden_channels=256

############################

training=fitnet
expt_name=fitnet
beta=1000

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
wait" C-m

############################

training=at
expt_name=at
beta=1000

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
wait" C-m

############################

training=lpw
expt_name=lpw-rbf
beta=100
kernel=rbf

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=lpw
expt_name=lpw-poly
beta=100
kernel=poly

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=lpw
expt_name=lpw-cosine
beta=100
kernel=cosine

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=lpw
expt_name=lpw-l2
beta=100
kernel=l2

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=nce
expt_name=nce
beta=0.1
nce_T=0.075
max_samples=16384
proj_dim=256

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --nce_T $nce_T --max_samples $max_samples --proj_dim $proj_dim &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --nce_T $nce_T --max_samples $max_samples --proj_dim $proj_dim &
wait" C-m








gnn=gat
num_layers=2
hidden_channels=64

############################

training=fitnet
expt_name=fitnet
beta=1000

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
wait" C-m

############################

training=at
expt_name=at
beta=1000

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta &
wait" C-m

############################

training=lpw
expt_name=lpw-rbf
beta=100
kernel=rbf

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=lpw
expt_name=lpw-poly
beta=100
kernel=poly

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=lpw
expt_name=lpw-cosine
beta=100
kernel=cosine

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=lpw
expt_name=lpw-l2
beta=100
kernel=l2

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --kernel $kernel &
wait" C-m

############################

training=nce
expt_name=nce
beta=0.1
nce_T=0.075
max_samples=16384
proj_dim=256

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --nce_T $nce_T --max_samples $max_samples --proj_dim $proj_dim &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --gnn $gnn --num_layers $num_layers --hidden_channels $hidden_channels --beta $beta --nce_T $nce_T --max_samples $max_samples --proj_dim $proj_dim &
wait" C-m





tmux send-keys "tmux kill-session -t all" C-m
