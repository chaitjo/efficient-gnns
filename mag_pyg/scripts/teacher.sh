
#!/bin/bash

tmux new -s expt -d
tmux send-keys "conda activate ogb" C-m

runs=5


############################

tmux send-keys "
python gnn.py --device 0 --seed 0 --runs $runs --num_layers 3 --hidden_channels 512 &
python gnn.py --device 1 --seed 5 --runs $runs --num_layers 3 --hidden_channels 512 &
wait" C-m

tmux send-keys "tmux kill-session -t expt" C-m
