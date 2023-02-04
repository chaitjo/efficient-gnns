import argparse
import random
import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import dgl
import dgl.function as fn

from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

import nvidia_smi
nvidia_smi.nvmlInit()



def get_ogb_evaluator(dataset):
    """
    Get evaluator from Open Graph Benchmark based on dataset
    """
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
        "y_true": labels.view(-1, 1),
        "y_pred": preds.view(-1, 1),
    })["acc"]


def convert_mag_to_homograph(g, device):
    """
    Featurize node types that don't have input features (i.e. author,
    institution, field_of_study) by averaging their neighbor features.
    Then convert the graph to a undirected homogeneous graph.
    """
    src_writes, dst_writes = g.all_edges(etype="writes")
    src_topic, dst_topic = g.all_edges(etype="has_topic")
    src_aff, dst_aff = g.all_edges(etype="affiliated_with")
    new_g = dgl.heterograph({
        ("paper", "written", "author"): (dst_writes, src_writes),
        ("paper", "has_topic", "field"): (src_topic, dst_topic),
        ("author", "aff", "inst"): (src_aff, dst_aff)
    })
    new_g = new_g.to(device)
    new_g.nodes["paper"].data["feat"] = g.nodes["paper"].data["feat"]
    new_g["written"].update_all(fn.copy_u("feat", "m"), fn.mean("m", "feat"))
    new_g["has_topic"].update_all(fn.copy_u("feat", "m"), fn.mean("m", "feat"))
    new_g["aff"].update_all(fn.copy_u("feat", "m"), fn.mean("m", "feat"))
    g.nodes["author"].data["feat"] = new_g.nodes["author"].data["feat"]
    g.nodes["institution"].data["feat"] = new_g.nodes["inst"].data["feat"]
    g.nodes["field_of_study"].data["feat"] = new_g.nodes["field"].data["feat"]

    # Convert to homogeneous graph
    # Get DGL type id for paper type
    target_type_id = g.get_ntype_id("paper")
    g = dgl.to_homogeneous(g, ndata=["feat"])
    g = dgl.add_reverse_edges(g, copy_ndata=True)
    # Mask for paper nodes
    g.ndata["target_mask"] = g.ndata[dgl.NTYPE] == target_type_id
    return g


def load_dataset(name, device):
    """
    Load dataset and move graph and features to device
    """
    if name not in ["ogbn-products", "ogbn-arxiv", "ogbn-mag"]:
        raise RuntimeError("Dataset {} is not supported".format(name))
    dataset = DglNodePropPredDataset(name=name)
    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx["train"]
    val_nid = splitted_idx["valid"]
    test_nid = splitted_idx["test"]
    g, labels = dataset[0]
    g = g.to(device)
    if name == "ogbn-arxiv":
        g = dgl.add_reverse_edges(g, copy_ndata=True)
        g = dgl.add_self_loop(g)
        g.ndata['feat'] = g.ndata['feat'].float()
    elif name == "ogbn-mag":
        # MAG is a heterogeneous graph. The task is to make prediction for
        # paper nodes
        labels = labels["paper"]
        train_nid = train_nid["paper"]
        val_nid = val_nid["paper"]
        test_nid = test_nid["paper"]
        g = convert_mag_to_homograph(g, device)
    else:
        g.ndata['feat'] = g.ndata['feat'].float()
    n_classes = dataset.num_classes
    labels = labels.squeeze()
    evaluator = get_ogb_evaluator(name)

    print(f"# Nodes: {g.number_of_nodes()}\n"
          f"# Edges: {g.number_of_edges()}\n"
          f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n"
          f"# Classes: {n_classes}")

    return g, labels, n_classes, train_nid, val_nid, test_nid, evaluator


class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x


class SIGN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, num_hops, n_layers,
                 dropout, input_drop):
        super(SIGN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = nn.Dropout(input_drop)
        for hop in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, n_layers, dropout))
        self.project = FeedForwardNet(num_hops * hidden, hidden, out_feats,
                                      n_layers, dropout)

    def forward(self, feats):
        feats = [self.input_drop(feat) for feat in feats]
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            hidden.append(ff(feat))
        self.out_feat = self.dropout(self.prelu(torch.cat(hidden, dim=-1)))
        out = self.project(self.out_feat)
        return out

    def reset_parameters(self):
        for ff in self.inception_ffs:
            ff.reset_parameters()
        self.project.reset_parameters()


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def neighbor_average_features(g, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    g.ndata["feat_0"] = g.ndata["feat"]
    for hop in range(1, args.R + 1):
        g.update_all(fn.copy_u(f"feat_{hop-1}", "msg"),
                     fn.mean("msg", f"feat_{hop}"))
    res = []
    for hop in range(args.R + 1):
        res.append(g.ndata.pop(f"feat_{hop}"))

    if args.dataset == "ogbn-mag":
        # For MAG dataset, only return features for target node types (i.e.
        # paper nodes)
        target_mask = g.ndata["target_mask"]
        target_ids = g.ndata[dgl.NID][target_mask]
        num_target = target_mask.sum().item()
        new_res = []
        for x in res:
            feat = torch.zeros((num_target,) + x.shape[1:],
                               dtype=x.dtype, device=x.device)
            feat[target_ids] = x[target_mask]
            new_res.append(feat)
        res = new_res
    return res


def prepare_data(device, args):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """
    data = load_dataset(args.dataset, device)
    g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
    in_feats = g.ndata['feat'].shape[1]
    feats = neighbor_average_features(g, args)
    labels = labels.to(device)
    # move to device
    train_nid = train_nid.to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)
    return feats, labels, in_feats, n_classes, \
        train_nid, val_nid, test_nid, evaluator


def test(model, feats, labels, test_loader, evaluator, train_nid, val_nid, test_nid, device_id):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
    model.eval()
    device = labels.device
    preds = []
    logits = []
    iter_time = 0
    max_gpu_used = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
    for batch in test_loader:
        iter_start_time = time.time()
        batch_feats = [feat[batch].to(device) for feat in feats]
        batch_logits = model(batch_feats)
        iter_time += (time.time() - iter_start_time)
        gpu_used = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
        if gpu_used >= max_gpu_used:
            max_gpu_used = gpu_used

        preds.append(torch.argmax(batch_logits, dim=-1))
        logits.append(batch_logits)
    
    # Concat mini-batch prediction results along node dimension
    preds = torch.cat(preds, dim=0)
    logits = torch.cat(logits, dim=0)
    train_res = evaluator(preds[train_nid], labels[train_nid])
    val_res = evaluator(preds[val_nid], labels[val_nid])
    test_res = evaluator(preds[test_nid], labels[test_nid])
    
    return train_res, val_res, test_res, iter_time, max_gpu_used


def run(run_idx, args, data, device):
    feats, labels, in_size, num_classes, \
        train_nid, val_nid, test_nid, evaluator = data
    test_loader = torch.utils.data.DataLoader(
        torch.arange(labels.shape[0]), batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False)

    # Initialize model and optimizer for each run
    num_hops = args.R + 1
    model = SIGN(in_size, args.num_hidden, num_classes, num_hops,
                 args.ff_layer, args.dropout, args.input_dropout)
    model = model.to(device)
    total_param = get_n_params(model)
    print("# Params:", total_param)

    with torch.no_grad():
        train_res, val_res, test_res, iter_time, max_gpu_used = test(
            model, feats, labels, test_loader, evaluator, train_nid, val_nid, test_nid, args.gpu)
    
    return iter_time, max_gpu_used


def main(args):
    seed(42)

    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)

    with torch.no_grad():
        data = prepare_data(device, args)
    
    t_iter_time = []
    t_gpu_used = []
    for i in range(args.num_runs):
        iter_time, gpu_used = run(i, args, data, device)
        t_iter_time.append(iter_time)
        t_gpu_used.append(gpu_used)

    print(f"Avg. Iteration Time: {np.mean(t_iter_time[1:]):.3f}s ± {np.std(t_iter_time[1:]):.3f}")
    print(f"Avg. GPU Used: {np.mean(t_gpu_used) * 1e-9:.1f}GB ± {np.std(t_gpu_used) * 1e-9:.1f}")


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIGN")
    
    # Experimental settings
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-runs", type=int, default=10,
                        help="number of times to repeat the experiment")

    # SIGN settings
    parser.add_argument("--num-hidden", type=int, default=512)
    parser.add_argument("--ff-layer", type=int, default=2,
                        help="number of feed-forward layers")
    parser.add_argument("--R", type=int, default=5,
                        help="number of hops")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--input-dropout", type=float, default=0.1,
                        help="dropout on input features")
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--eval-batch-size", type=int, default=100000,
                        help="evaluation batch size")

    args = parser.parse_args()

    print(args)
    main(args)