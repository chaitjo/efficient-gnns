#!/bin/bash

tmux new -s expt -d
tmux send-keys "conda activate ogb" C-m

tmux send-keys "
python3 gat.py --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --save-pred --gpu 0 --seed 0 --n-runs 5 --expt-name gat-3L250x3h &
python3 gat.py --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --save-pred --gpu 1 --seed 5 --n-runs 5 --expt-name gat-3L250x3h &
wait" C-m

tmux send-keys "
python3 gat.py --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --save-pred --gpu 0 --seed 0 --n-runs 5 --n-heads 4 --expt-name gat-3L250x4h &
python3 gat.py --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --save-pred --gpu 1 --seed 5 --n-runs 5 --n-heads 4 --expt-name gat-3L250x4h &
wait" C-m

tmux send-keys "tmux kill-session -t expt" C-m
