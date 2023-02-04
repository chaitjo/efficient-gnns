
#!/bin/bash

tmux new -s expt -d
tmux send-keys "conda activate ogb" C-m

runs=5


############################

tmux send-keys "
python train_teacher.py --device 0 --seed 0 --runs $runs &
python train_teacher.py --device 1 --seed 5 --runs $runs &
wait" C-m

tmux send-keys "tmux kill-session -t expt" C-m
