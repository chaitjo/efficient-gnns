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

from criterion import *


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

        return x_dict


def train(model, train_loader, x_dict, optimizer, args, device, teacher_model, student_proj=None, teacher_proj=None):
    model.train()
    if student_proj:
        student_proj.train()
    if teacher_proj:
        teacher_proj.train()
    if teacher_model:
        teacher_model.eval()

    pbar = tqdm(total=args.num_steps * args.batch_size)
    total_loss = total_examples = 0
    total_loss_cls = total_loss_aux = 0
    
    for data in train_loader:
        data = data.to(device)
        out = model(x_dict, data.edge_index, data.edge_attr, 
                    data.node_type, data.local_node_idx)
        out = out[data.train_mask]
        labels = data.y[data.train_mask].squeeze()
        
        if args.training == 'supervised':
            loss = F.cross_entropy(out, labels)
            loss_cls = loss
            loss_aux = loss*0
        elif args.training == 'kd':
            with torch.no_grad():
                teacher_out = teacher_model(x_dict, data.edge_index, data.edge_attr, data.node_type, data.local_node_idx)[data.train_mask]
            loss, loss_cls, loss_aux = kd_criterion(
                out, labels, teacher_out, args.alpha, args.kd_T
            )
        elif args.training == 'fitnet':
            with torch.no_grad():
                teacher_out = teacher_model(x_dict, data.edge_index, data.edge_attr, data.node_type, data.local_node_idx)[data.train_mask]
                teacher_out_feat = teacher_model.out_feat[data.train_mask]
            out_feat = student_proj(model.out_feat[data.train_mask])
            teacher_out_feat = teacher_proj(teacher_out_feat)
            _, _, loss_aux = fitnet_criterion(
                out, labels, out_feat, teacher_out_feat, args.beta
            )
            loss, loss_cls, _ = kd_criterion(
                out, labels, teacher_out, args.alpha, args.kd_T
            )
            loss = loss + args.beta* loss_aux
        elif args.training == 'at':
            with torch.no_grad():
                teacher_out = teacher_model(x_dict, data.edge_index, data.edge_attr, data.node_type, data.local_node_idx)[data.train_mask]
                teacher_out_feat = teacher_model.out_feat[data.train_mask]
            out_feat = model.out_feat[data.train_mask]
            _, _, loss_aux = at_criterion(
                out, labels, out_feat, teacher_out_feat, args.beta
            )
            loss, loss_cls, _ = kd_criterion(
                out, labels, teacher_out, args.alpha, args.kd_T
            )
            loss = loss + args.beta* loss_aux
        elif args.training == 'gpw':
            with torch.no_grad():
                teacher_out = teacher_model(x_dict, data.edge_index, data.edge_attr, data.node_type, data.local_node_idx)[data.train_mask]
                teacher_out_feat = teacher_model.out_feat[data.train_mask]
            out_feat = model.out_feat[data.train_mask]
            # teacher_out_feat = teacher_proj(teacher_out_feat)
            _, _, loss_aux = gpw_criterion(
                out, labels, out_feat, teacher_out_feat, 
                args.kernel, args.beta, args.max_samples
            )
            loss, loss_cls, _ = kd_criterion(
                out, labels, teacher_out, args.alpha, args.kd_T
            )
            loss = loss + args.beta* loss_aux
        elif args.training == 'lpw':
            with torch.no_grad():
                teacher_out = teacher_model(x_dict, data.edge_index, data.edge_attr, data.node_type, data.local_node_idx)[data.train_mask]
                teacher_out_feat = teacher_model.out_feat[data.train_mask]
            out_feat = model.out_feat[data.train_mask]
            # teacher_out_feat = teacher_out_feat 
            edge_index = subgraph(data.train_mask.nonzero().squeeze(1), data.edge_index, relabel_nodes=True)[0]
            _, _, loss_aux = lpw_criterion(
                out, labels, out_feat, teacher_out_feat, 
                edge_index, args.kernel, args.beta
            )
            loss, loss_cls, _ = kd_criterion(
                out, labels, teacher_out, args.alpha, args.kd_T
            )
            loss = loss + args.beta* loss_aux
        elif args.training == 'nce':
            with torch.no_grad():
                teacher_out = teacher_model(x_dict, data.edge_index, data.edge_attr, data.node_type, data.local_node_idx)[data.train_mask]
                teacher_out_feat = teacher_model.out_feat[data.train_mask]
            out_feat = student_proj(model.out_feat[data.train_mask])
            teacher_out_feat = teacher_proj(teacher_out_feat)
            _, _, loss_aux = nce_criterion(
                out, labels, out_feat, teacher_out_feat, 
                args.beta, args.nce_T, args.max_samples
            )
            loss, loss_cls, _ = kd_criterion(
                out, labels, teacher_out, args.alpha, args.kd_T
            )
            loss = loss + args.beta* loss_aux
        else:
            raise NotImplementedError
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_examples = data.train_mask.sum().item()
        total_loss += loss.item() * num_examples
        total_loss_cls += loss_cls.item() * num_examples
        total_loss_aux += loss_aux.item() * num_examples
        total_examples += num_examples
        pbar.update(args.batch_size)

    pbar.close()

    return total_loss / total_examples, total_loss_cls / total_examples, total_loss_aux / total_examples


@torch.no_grad()
def test(model, data, x_dict, edge_index_dict, key2int, split_idx, evaluator):
    model.eval()

    out = model.inference(x_dict, edge_index_dict, key2int)
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

    return train_acc, valid_acc, test_acc


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
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

    train_loader = GraphSAINTRandomWalkSampler(homo_data,
                                            batch_size=args.batch_size,
                                            walk_length=args.num_layers,
                                            num_steps=args.num_steps,
                                            sample_coverage=0,
                                            save_dir=dataset.processed_dir)

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

    x_dict = {k: v.to(device) for k, v in x_dict.items()}

    for run in range(args.runs):
        seed(args.seed + run)

        model.reset_parameters()

        teacher_model = None
        if args.training != 'supervised':
            teacher_model = RGCN(128, 512, dataset.num_classes, 3, 0.5, num_nodes_dict, list(x_dict.keys()), len(edge_index_dict.keys())).to(device)
            checkpoint = torch.load(f"{args.teacher_path}/seed{args.seed + run}/results.pt", map_location=torch.device('cpu'))
            teacher_model.load_state_dict(checkpoint['model_state_dict'])
            teacher_model.eval()

        if args.training in ["nce", "fitnet"]:
            student_proj = torch.nn.Sequential(
                torch.nn.Linear(args.hidden_channels, args.proj_dim), 
                torch.nn.BatchNorm1d(args.proj_dim), 
                torch.nn.ReLU()
            ).to(device)

            teacher_proj = torch.nn.Sequential(
                torch.nn.Linear(512, args.proj_dim), 
                torch.nn.BatchNorm1d(args.proj_dim), 
                torch.nn.ReLU()
            ).to(device)

            optimizer = torch.optim.Adam([
                {'params': model.parameters(), 'lr': args.lr},
                {'params': student_proj.parameters(), 'lr': args.lr},
                {'params': teacher_proj.parameters(), 'lr': args.lr}
            ])
        else:
            student_proj, teacher_proj = None, None
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Create Tensorboard logger
        start_time_str = time.strftime("%Y%m%dT%H%M%S")
        log_dir = os.path.join(
            "/data/user/mag_pyg_logs/logs_kd_and_aux",
            args.expt_name,
            f"{args.gnn}-L{args.num_layers}h{args.hidden_channels}-Student-{args.training}",
            f"{start_time_str}-GPU{args.device}-seed{args.seed+run}"
        )
        os.makedirs(log_dir, exist_ok=True)
        
        # Start training
        best_epoch = 0
        best_train = 0
        best_val = 0
        best_test = 0
        # best_model_sd = None
        # best_optimizer_sd = None
        for epoch in range(1, 1 + args.epochs):
            train_loss, train_loss_cls, train_loss_aux = train(
                model, train_loader, x_dict, optimizer, args, device, 
                teacher_model, student_proj, teacher_proj
            )
            torch.cuda.empty_cache()
            
            result = test(model, data, x_dict, edge_index_dict, key2int, split_idx, evaluator)
            
            logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss_total: {train_loss:.4f}, '
                  f'Loss_cls: {train_loss_cls:.4f}, '
                  f'Loss_aux: {train_loss_aux:.8f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}%, '
                  f'Test: {100 * test_acc:.2f}%')

            if valid_acc > best_val:
                best_epoch = epoch
                best_train = train_acc
                best_val = valid_acc
                best_test = test_acc
                # best_model_sd = model.state_dict()
                # best_optimizer_sd = optimizer.state_dict()

        logger.print_statistics(run)
        torch.save({
            'args': args,
            'total_param': total_param,
            'BestEpoch': best_epoch,
            'Train': best_train,
            'Validation': best_val, 
            'Test': best_test,
            # 'model_state_dict': best_model_sd,
            # 'optimizer_state_dict': best_optimizer_sd,
        }, os.path.join(log_dir, "results.pt"))
    
    logger.print_statistics()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBN-MAG (GraphSAINT)')

    # Experiment settings
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--training', type=str, default="supervised")
    parser.add_argument('--expt_name', type=str, default="graph_saint")
    parser.add_argument('--teacher_path', type=str, default="/data/user/mag_pyg_logs/logs/graph_saint/rgcn-L3h512-Teacher")

    # GNN settings
    parser.add_argument('--gnn', type=str, default='rgcn')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--walk_length', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=30)
    
    # KD settings
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='alpha parameter for KD (default: 0.5)')
    parser.add_argument('--kd_T', type=float, default=4.0,
                        help='temperature parameter for KD (default: 1.0)')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='beta parameter for auxiliary distillation losses (default: 0.5)')
    
    # NCE/auxiliary distillation settings
    parser.add_argument('--nce_T', type=float, default=0.075,
                        help='temperature parameter for NCE (default: 0.075)')
    parser.add_argument('--max_samples', type=int, default=16384,
                        help='maximum samples for NCE/GPW (default: 8192)')
    parser.add_argument('--proj_dim', type=int, default=128,
                        help='common projection dimensionality for NCE/FitNet (default: 128)')
    parser.add_argument('--kernel', type=str, default='poly',
                        help='kernel for LPW: cosine, polynomial, l2, rbf (default: rbf)')

    args = parser.parse_args()
    
    print(args)
    main()
