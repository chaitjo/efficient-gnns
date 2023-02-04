
#!/bin/bash

# Experimental setup:
# - We tune the loss balancing weights α, β in for all techniques on the validation set when using both logit-based and representation distillation together. 
# - For KD, we tune α ∈ {0.8, 0.9}, τ1 ∈ {4, 5}. 
# - For FitNet, AT, LSP, and GSP, we tune β ∈ {100, 1000, 10000} and the kernel for LSP, GSP. 
# - For G-CRD, we tune β ∈ {0.01, 0.05}, τ2 ∈ {0.05, 0.075, 0.1} and the projection head. 
# When comparing representation distillation methods, we set α to 0 in order to ablate performance and reduce β by one order of magnitude.

tmux new -s expt -d
tmux send-keys "conda activate ogb" C-m

gnn=sage

runs=5

num_layers=2

############################

training=supervised
expt_name=sage/supervised
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn &
wait" C-m

training=kd
expt_name=sage/kd
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn &
wait" C-m

training=fitnet
expt_name=sage/fitnet
beta=1000
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta &
wait" C-m

training=at
expt_name=sage/at
beta=100000
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta &
wait" C-m

training=gpw
expt_name=sage/gpw-rbf
beta=100000
max_samples=2048
proj_dim=128
kernel=rbf
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
wait" C-m

training=gpw
expt_name=sage/gpw-l2
beta=1
max_samples=2048
proj_dim=128
kernel=l2
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
wait" C-m

training=gpw
expt_name=sage/gpw-poly
beta=100
max_samples=4096
proj_dim=128
kernel=poly
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
wait" C-m

training=gpw
expt_name=sage/gpw-cosine
beta=100
max_samples=4096
proj_dim=128
kernel=cosine
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
wait" C-m

training=lpw
expt_name=sage/lpw-rbf
beta=100
max_samples=2048
proj_dim=128
kernel=rbf
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
wait" C-m

training=lpw
expt_name=sage/lpw-l2
beta=100
max_samples=2048
proj_dim=128
kernel=l2
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
wait" C-m

training=lpw
expt_name=sage/lpw-poly
beta=100
max_samples=4096
proj_dim=128
kernel=poly
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
wait" C-m

training=lpw
expt_name=sage/lpw-cosine
beta=100
max_samples=4096
proj_dim=128
kernel=cosine
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
wait" C-m

training=nce
expt_name=sage/nce
beta=0.1
nce_T=0.075
max_samples=16384
proj_dim=256
tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --nce_T $nce_T &
python gnn.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --nce_T $nce_T &
wait" C-m

############################



tmux send-keys "tmux kill-session -t expt" C-m
