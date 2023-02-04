#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import os
import random
import time
import glob

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from models import GAT

epsilon = 1 - math.log(2)

device = None

dataset = "ogbn-arxiv"
n_node_feats, n_classes = 0, 0


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def load_data(dataset):
    global n_node_feats, n_classes

    data = DglNodePropPredDataset(name=dataset)
    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph):
    global n_node_feats

    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph


def gen_model(args):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats

    model = GAT(
        n_node_feats_,
        n_classes,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        attn_drop=args.attn_drop,
        edge_drop=args.edge_drop,
        use_attn_dst=not args.no_attn_dst,
        use_symmetric_norm=args.use_norm,
    )

    return model


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def add_labels(feat, labels, idx):
    onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


@torch.no_grad()
def evaluate(args, model, graph, labels, train_idx, val_idx, test_idx, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if args.use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([val_idx, test_idx])
        for _ in range(args.n_label_iters):
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, feat)

    train_loss = custom_loss_function(pred[train_idx], labels[train_idx])
    val_loss = custom_loss_function(pred[val_idx], labels[val_idx])
    test_loss = custom_loss_function(pred[test_idx], labels[test_idx])

    model_feat = model.feat

    return (
        evaluator(pred[train_idx], labels[train_idx]),
        evaluator(pred[val_idx], labels[val_idx]),
        evaluator(pred[test_idx], labels[test_idx]),
        train_loss,
        val_loss,
        test_loss,
        pred,
        model_feat
    )


def count_parameters(args):
    model = gen_model(args)
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def main():
    global device, n_node_feats, n_classes, epsilon

    argparser = argparse.ArgumentParser(
        "GAT checkpoint on ogbn-arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument(
        "--use-labels", action="store_true", help="Use labels in the training set as input features."
    )
    argparser.add_argument("--n-label-iters", type=int, default=1, help="number of label iterations")
    argparser.add_argument("--mask-rate", type=float, default=0.5, help="mask rate")
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--use-norm", action="store_true", help="Use symmetrically normalized adjacency matrix.")
    argparser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=3, help="number of layers")
    argparser.add_argument("--n-heads", type=int, default=3, help="number of heads")
    argparser.add_argument("--n-hidden", type=int, default=250, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.75, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.25, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention drop rate")
    argparser.add_argument("--edge-drop", type=float, default=0.3, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--checkpoint-files", type=str, default="./checkpoints/*.pt", help="address of checkpoint files")
    args = argparser.parse_args()

    if not args.use_labels and args.n_label_iters > 0:
        raise ValueError("'--use-labels' must be enabled when n_label_iters > 0")

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset)
    graph = preprocess(graph)

    graph, labels, train_idx, val_idx, test_idx = map(
        lambda x: x.to(device), (graph, labels, train_idx, val_idx, test_idx)
    )

    evaluator_wrapper = lambda pred, labels: evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]

    # run
    val_accs, test_accs = [], []

    for file in glob.iglob(args.checkpoint_files):
        print("load:", file)
        checkpoint = torch.load(file)

        # define model and optimizer
        model = gen_model(args).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred, model_feat = evaluate(
            args, model, graph, labels, train_idx, val_idx, test_idx, evaluator_wrapper
        )
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        n_running = file.split('/')[-1][:-3]
        os.makedirs(f"./logits", exist_ok=True)
        torch.save(pred, f"./logits/{n_running}.pt")
        os.makedirs(f"./features", exist_ok=True)
        torch.save(model_feat, f"./features/{n_running}.pt")

    print(args)
    print()
    print(f"Run {len(val_accs)} times")
    print(f"Average val accuracy: {np.mean(val_accs)*100:.2f} ± {np.std(val_accs)*100:.2f}")
    print(f"Average test accuracy: {np.mean(test_accs)*100:.2f} ± {np.std(test_accs)*100:.2f}")
    print(f"Number of params: {count_parameters(args)}")


if __name__ == "__main__":
    main()
