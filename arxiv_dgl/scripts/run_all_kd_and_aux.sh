#!/bin/bash

# Experimental setup:
# - We tune the loss balancing weights α, β in for all techniques on the validation set when using both logit-based and representation distillation together. 
# - For KD, we tune α ∈ {0.8, 0.9}, τ1 ∈ {4, 5}. 
# - For FitNet, AT, LSP, and GSP, we tune β ∈ {100, 1000, 10000} and the kernel for LSP, GSP. 
# - For G-CRD, we tune β ∈ {0.01, 0.05}, τ2 ∈ {0.05, 0.075, 0.1} and the projection head. 
# When comparing representation distillation methods, we set α to 0 in order to ablate performance and reduce β by one order of magnitude.

tmux new -s expt -d
tmux send-keys "conda activate ogb" C-m







training=kd

tmux send-keys "
python3 sign.py --eval-ev 10 --R 5 --input-d 0.1 --num-h 512 --dr 0.5 --lr 0.001 --eval-b 100000 --num-runs 5 --seed 0 --gpu 0 --training $training --expt_name $training &
python3 sign.py --eval-ev 10 --R 5 --input-d 0.1 --num-h 512 --dr 0.5 --lr 0.001 --eval-b 100000 --num-runs 5 --seed 5 --gpu 1 --training $training --expt_name $training &
wait" C-m






training=fitnet
beta=1000

tmux send-keys "
python3 sign.py --eval-ev 10 --R 5 --input-d 0.1 --num-h 512 --dr 0.5 --lr 0.001 --eval-b 100000 --num-runs 5 --seed 0 --gpu 0 --training $training --expt_name $training --beta $beta &
python3 sign.py --eval-ev 10 --R 5 --input-d 0.1 --num-h 512 --dr 0.5 --lr 0.001 --eval-b 100000 --num-runs 5 --seed 5 --gpu 1 --training $training --expt_name $training --beta $beta &
wait" C-m



training=at
beta=100

tmux send-keys "
python3 sign.py --eval-ev 10 --R 5 --input-d 0.1 --num-h 512 --dr 0.5 --lr 0.001 --eval-b 100000 --num-runs 5 --seed 0 --gpu 0 --training $training --expt_name $training --beta $beta &
python3 sign.py --eval-ev 10 --R 5 --input-d 0.1 --num-h 512 --dr 0.5 --lr 0.001 --eval-b 100000 --num-runs 5 --seed 5 --gpu 1 --training $training --expt_name $training --beta $beta &
wait" C-m




training=gpw
beta=1000000
max_samples=2048
proj_dim=128

tmux send-keys "
python3 sign.py --eval-ev 10 --R 5 --input-d 0.1 --num-h 512 --dr 0.5 --lr 0.001 --eval-b 100000 --num-runs 5 --seed 0 --gpu 0 --training $training --expt_name $training --beta $beta --max_samples $max_samples --proj_dim $proj_dim &
python3 sign.py --eval-ev 10 --R 5 --input-d 0.1 --num-h 512 --dr 0.5 --lr 0.001 --eval-b 100000 --num-runs 5 --seed 5 --gpu 1 --training $training --expt_name $training --beta $beta --max_samples $max_samples --proj_dim $proj_dim &
wait" C-m



training=nce
beta=0.1
nce_T=0.075
max_samples=16384
proj_dim=256
expt_name=nce

tmux send-keys "
python3 sign.py --eval-ev 10 --R 5 --input-d 0.1 --num-h 512 --dr 0.5 --lr 0.001 --eval-b 100000 --num-runs 5 --seed 0 --gpu 0 --training $training --expt_name $expt_name --beta $beta --max_samples $max_samples --proj_dim $proj_dim --nce_T $nce_T &
python3 sign.py --eval-ev 10 --R 5 --input-d 0.1 --num-h 512 --dr 0.5 --lr 0.001 --eval-b 100000 --num-runs 5 --seed 5 --gpu 1 --training $training --expt_name $expt_name --beta $beta --max_samples $max_samples --proj_dim $proj_dim --nce_T $nce_T &
wait" C-m



tmux send-keys "tmux kill-session -t expt" C-m
