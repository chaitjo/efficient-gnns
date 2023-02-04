# Knowledge Distillation for GNNs (ARXIV with DGL)

**Dataset**: ARXIV

**Library**: DGL

This repository contains code to benchmark knowledge distillation for GNNs on the ARXIV dataset, developed in the DGL framework.
The main purpose of the DGL codebase is to:
- Train teacher models (which are 3 layer GATs) which were giving OOM errors when trained via PyG. Code adapted from: https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv
- Train graph-agnostic SIGN models, for which the best open-source implementation was in DGL: https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/sign

Note that other student GNNs like GCN/GraphSage are trained on ARXIV via the PyG implementation in `arxiv_pyg`.

## Directory Structure

```
.
├── dataset                     # automatically created by OGB data downloaders
├── checkpoints                 # saves GAT teacher checkpoints
├── features                    # saves GAT teacher final hidden features
├── logits                      # saves GAT teacher predicted logits
├── output                      # saves GAT teacher predicted outputs
├── logs                        # logging directory for SIGN models
|
├── scripts                     # scripts to conduct full experiments and reproduce results
│   ├── gat-teachers.sh         # trains GAT teachers for 10 seeds
│   ├── run_all_kd_and_aux.sh   # script to benchmark KD techniques
│   └── run_all.sh              # script to benchmark KD techniques
|
├── README.md
|
├── criterion.py                # KD loss functions
├── gat.py                      # trains GAT teacher and dumps checkpoints, features, logits, outputs, etc.
├── load_checkpoint.py          # loads and evaluates a GAT checkpoint on the test set
├── models.py                   # model definitions for GAT and GCN
├── sign.py                     # trains SIGN models with/without KD
├── submit.py                   # load and evaluate GAT model predictions on the test set
├── submit_sign.py              # load and evaluate SIGN models on the test set
├── test_timing_gat.py          # this script was used to measure inference time
└── test_timing_sign.py         # this script was used to measure inference time
```

## Example Usage

For full usage, each file has accompanying flags and documentation.
Also see the `scripts` folder for reproducing results.
