
#!/bin/bash

tmux new -s expt -d
tmux send-keys "conda activate ogb" C-m

runs=5

num_layers=2



gnn=sage


############################

training=fitnet
expt_name=sage-rerun/fitnet
beta=100
tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta &
wait" C-m

training=at
expt_name=sage-rerun/at
beta=10000
tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta &
wait" C-m

training=gpw
expt_name=sage-rerun/gpw-cosine
beta=10
max_samples=4096
proj_dim=128
kernel=cosine
tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
wait" C-m

training=lpw
expt_name=sage-rerun/lpw-cosine
beta=100
max_samples=4096
proj_dim=128
kernel=cosine
tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
wait" C-m

training=nce
expt_name=sage-rerun/nce
beta=0.01
nce_T=0.05
max_samples=16384
proj_dim=256
tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --nce_T $nce_T &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --nce_T $nce_T &
wait" C-m

############################











gnn=gcn

############################

training=fitnet
expt_name=gcn-rerun/fitnet
beta=100
tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta &
wait" C-m

training=at
expt_name=gcn-rerun/at
beta=10000
tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta &
wait" C-m

training=gpw
expt_name=gcn-rerun/gpw-cosine
beta=10
max_samples=4096
proj_dim=128
kernel=cosine
tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
wait" C-m

training=lpw
expt_name=gcn-rerun/lpw-cosine
beta=100
max_samples=4096
proj_dim=128
kernel=cosine
tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --kernel $kernel &
wait" C-m

training=nce
expt_name=gcn-rerun/nce
beta=0.01
nce_T=0.05
max_samples=16384
proj_dim=256
tmux send-keys "
python gnn_kd_and_aux.py --device 0 --seed 0 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --nce_T $nce_T &
python gnn_kd_and_aux.py --device 1 --seed 5 --runs $runs --training $training --expt_name $expt_name --num_layers $num_layers --gnn $gnn --beta $beta --max_samples $max_samples --proj_dim $proj_dim --nce_T $nce_T &
wait" C-m

############################





tmux send-keys "tmux kill-session -t expt" C-m
