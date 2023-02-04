import argparse
import glob

import numpy as np
import torch
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

device = None

dataset = "ogbn-arxiv"
n_node_feats, n_classes = 0, 0


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


def evaluate(labels, pred, train_idx, val_idx, test_idx, evaluator):
    return (
        evaluator(pred[train_idx], labels[train_idx]),
        evaluator(pred[val_idx], labels[val_idx]),
        evaluator(pred[test_idx], labels[test_idx]),
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--pred-files", type=str, default="./output/*.pt", help="address of prediction files")
    args = argparser.parse_args()

    # load data
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset)
    evaluator_wrapper = lambda pred, labels: evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]

    val_accs, test_accs = [], []

    for pred_file in glob.iglob(args.pred_files):
        print("load:", pred_file)
        pred = torch.load(pred_file)
        train_acc, val_acc, test_acc = evaluate(labels, pred, train_idx, val_idx, test_idx, evaluator_wrapper)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    print(args)
    print()
    print(f"Run {len(val_accs)} times")
    print(f"Average val accuracy: {np.mean(val_accs)*100:.2f} ± {np.std(val_accs)*100:.2f}")
    print(f"Average test accuracy: {np.mean(test_accs)*100:.2f} ± {np.std(test_accs)*100:.2f}")
    