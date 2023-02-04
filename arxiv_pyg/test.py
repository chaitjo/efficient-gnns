import argparse
import numpy as np
import time
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import subgraph

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger

import nvidia_smi
nvidia_smi.nvmlInit()


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            self.out_feat = x
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            self.out_feat = x
        x = self.convs[-1](x, adj_t)
        return x


@torch.no_grad()
def test(model, data, split_idx, evaluator, device):
    model.eval()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device)

    iter_start_time = time.time()
    out = model(data.x, data.adj_t)
    iter_time = time.time() - iter_start_time
    gpu_used = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc, iter_time, gpu_used


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # dgl.random.seed(seed)


def main():
    seed(42)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()

    if args.gnn == 'sage':
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    elif args.gnn == 'gcn':
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)
    else:
        raise ValueError('Invalid GNN type')

    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')

    if args.checkpoint != "":
        checkpoint = torch.load(f"{args.checkpoint}/results.pt", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    t_iter_time = []
    t_gpu_used = []
    for run in range(args.runs):
        torch.cuda.empty_cache()
        model.reset_parameters()
        
        result = test(model, data, split_idx, evaluator, args.device)
        
        logger.add_result(run, result[:3])

        train_acc, valid_acc, test_acc, iter_time, gpu_used = result
        t_iter_time.append(iter_time)
        t_gpu_used.append(gpu_used)
        print(f'Run: {run + 1:02d}, '
                f'Train: {100 * train_acc:.2f}%, '
                f'Valid: {100 * valid_acc:.2f}% '
                f'Test: {100 * test_acc:.2f}% '
                f'Time: {iter_time:.3f}s, '
              f'GPU: {gpu_used * 1e-9:.1f}GB' )
    
    logger.print_statistics()
    print(f"Avg. Iteration Time: {np.mean(t_iter_time[1:]):.3f}s ± {np.std(t_iter_time[1:]):.3f}")
    print(f"Avg. GPU Used: {np.mean(t_gpu_used) * 1e-9:.1f}GB ± {np.std(t_gpu_used) * 1e-9:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    
    # Experiment settings
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default="")
    
    # GNN settings
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--runs', type=int, default=10) 
    
    args = parser.parse_args()
    
    print(args)
    main()