import argparse
import numpy as np
import time
import os
import random
from copy import copy
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter

from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, subgraph
from torch_geometric.data import Data, GraphSAINTRandomWalkSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger

import nvidia_smi
nvidia_smi.nvmlInit()


class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types):
        super(RGCNConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.rel_lins = ModuleList([
            Linear(in_channels, out_channels, bias=False)
            for _ in range(num_edge_types)
        ])

        self.root_lins = ModuleList([
            Linear(in_channels, out_channels, bias=True)
            for _ in range(num_node_types)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        for lin in self.root_lins:
            lin.reset_parameters()

    def forward(self, x, edge_index, edge_type, node_type):
        out = x.new_zeros(x.size(0), self.out_channels)

        for i in range(self.num_edge_types):
            mask = edge_type == i
            out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))

        for i in range(self.num_node_types):
            mask = node_type == i
            out[mask] += self.root_lins[i](x[mask])

        return out

    def message(self, x_j, edge_type: int):
        return self.rel_lins[edge_type](x_j)


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types):
        super(RGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(num_nodes_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Create embeddings for all node types that do not come with features.
        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        I, H, O = in_channels, hidden_channels, out_channels  # noqa

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(RGCNConv(I, H, num_node_types, num_edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(H, H, num_node_types, num_edge_types))
        self.convs.append(RGCNConv(H, O, self.num_node_types, num_edge_types))

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx):
        # Create global node feature matrix.
        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=node_type.device)

        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]]

        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]]

        return h

    def forward(self, x_dict, edge_index, edge_type, node_type,
                local_node_idx):

        x = self.group_input(x_dict, node_type, local_node_idx)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type, node_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                self.out_feat = x

        return x

    def inference(self, x_dict, edge_index_dict, key2int):
        # We can perform full-batch inference on GPU.

        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)
        for key, emb in self.emb_dict.items():
            x_dict[int(key)] = emb

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)

        iter_start_time = time.time()
        for i, conv in enumerate(self.convs):
            out_dict = {}

            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)

            for keys, adj_t in adj_t_dict.items():
                src_key, target_key = keys[0], keys[-1]
                out = out_dict[key2int[target_key]]
                tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                out.add_(conv.rel_lins[key2int[keys]](tmp))

            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    F.relu_(out_dict[j])

            x_dict = out_dict
        
        iter_time = time.time() - iter_start_time

        return x_dict, iter_time


@torch.no_grad()
def test(model, data, x_dict, edge_index_dict, key2int, split_idx, evaluator, device):
    model.eval()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device)

    # iter_start_time = time.time()
    out, iter_time = model.inference(x_dict, edge_index_dict, key2int)
    # iter_time = time.time() - iter_start_time
    gpu_used = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used

    out = out[key2int['paper']]

    y_pred = out.argmax(dim=-1, keepdim=True).cpu()
    y_true = data.y_dict['paper']

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
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


def main():
    seed(42)

    dataset = PygNodePropPredDataset(name='ogbn-mag')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-mag')
    logger = Logger(args.runs, args)

    # We do not consider those attributes for now.
    data.node_year_dict = None
    data.edge_reltype_dict = None

    print(data)

    edge_index_dict = data.edge_index_dict

    # We need to add reverse edges to the heterogeneous graph.
    r, c = edge_index_dict[('author', 'affiliated_with', 'institution')]
    edge_index_dict[('institution', 'to', 'author')] = torch.stack([c, r])

    r, c = edge_index_dict[('author', 'writes', 'paper')]
    edge_index_dict[('paper', 'to', 'author')] = torch.stack([c, r])

    r, c = edge_index_dict[('paper', 'has_topic', 'field_of_study')]
    edge_index_dict[('field_of_study', 'to', 'paper')] = torch.stack([c, r])

    # Convert to undirected paper <-> paper relation.
    edge_index = to_undirected(edge_index_dict[('paper', 'cites', 'paper')])
    edge_index_dict[('paper', 'cites', 'paper')] = edge_index

    # We convert the individual graphs into a single big one, so that sampling
    # neighbors does not need to care about different edge types.
    # This will return the following:
    # * `edge_index`: The new global edge connectivity.
    # * `edge_type`: The edge type for each edge.
    # * `node_type`: The node type for each node.
    # * `local_node_idx`: The original index for each node.
    # * `local2global`: A dictionary mapping original (local) node indices of
    #    type `key` to global ones.
    # `key2int`: A dictionary that maps original keys to their new canonical type.
    out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
    edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out

    homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
                    node_type=node_type, local_node_idx=local_node_idx,
                    num_nodes=node_type.size(0))

    homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
    homo_data.y[local2global['paper']] = data.y_dict['paper']

    homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
    homo_data.train_mask[local2global['paper'][split_idx['train']['paper']]] = True

    print(homo_data)

    # Map informations to their canonical type.
    x_dict = {}
    for key, x in data.x_dict.items():
        x_dict[key2int[key]] = x

    num_nodes_dict = {}
    for key, N in data.num_nodes_dict.items():
        num_nodes_dict[key2int[key]] = N

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    model = RGCN(128, args.hidden_channels, dataset.num_classes, args.num_layers,
                args.dropout, num_nodes_dict, list(x_dict.keys()),
                len(edge_index_dict.keys())).to(device)
    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')
    total_param = total_param - 1134649*128 - 59965*128 - 8740*128
    print(f'GNN parameters: {total_param}')
    print(f'Embedding parameters: {1134649*128 - 59965*128 - 8740*128}')

    if args.checkpoint != "":
        checkpoint = torch.load(f"{args.checkpoint}/results.pt", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    x_dict = {k: v.to(device) for k, v in x_dict.items()}

    t_iter_time = []
    t_gpu_used = []
    for run in range(args.runs):
        torch.cuda.empty_cache()
        model.reset_parameters()
        result = test(model, data, x_dict, edge_index_dict, key2int, split_idx, evaluator, args.device)
        
        logger.add_result(run, result[:3])
        train_acc, valid_acc, test_acc, iter_time, gpu_used = result
        t_iter_time.append(iter_time)
        t_gpu_used.append(gpu_used)
        print(f'Run: {run + 1:02d}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}%, '
              f'Test: {100 * test_acc:.2f}%, '
              f'Time: {iter_time:.3f}s, '
              f'GPU: {gpu_used * 1e-9:.1f}GB' )
    
    logger.print_statistics()
    print(f"Avg. Iteration Time: {np.mean(t_iter_time[1:]):.3f}s ± {np.std(t_iter_time[1:]):.3f}")
    print(f"Avg. GPU Used: {np.mean(t_gpu_used) * 1e-9:.1f}GB ± {np.std(t_gpu_used)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBN-MAG (GraphSAINT)')

    # Experiment settings
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default="")

    # GNN settings
    parser.add_argument('--gnn', type=str, default='rgcn')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--walk_length', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=30)
    
    args = parser.parse_args()
    
    print(args)
    main()
