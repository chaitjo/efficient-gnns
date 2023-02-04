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

from criterion import *


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


class ProjectionGCD(torch.nn.Module):
    def __init__(self, hidden_channels, proj_dim):
        super(ProjectionGCD, self).__init__()
        self.conv = GCNConv(hidden_channels, proj_dim)
        self.bn = torch.nn.BatchNorm1d(proj_dim)
    
    def forward(self, x, adj_t):
        x = self.conv(x, adj_t)
        x = self.bn(x)
        x = F.relu(x)
        return x


def train(model, data, train_idx, optimizer, args, teacher_out_feat, teacher_logits, student_proj=None, teacher_proj=None, edge_index=None):
    model.train()
    if student_proj:
        student_proj.train()
    if teacher_proj:
        teacher_proj.train()
    
    out = model(data.x, data.adj_t)[train_idx]
    labels = data.y.squeeze(1)[train_idx]
    
    if args.training == 'supervised':
        loss = F.cross_entropy(out, labels)
        loss_cls = loss
        loss_aux = loss*0
    elif args.training == 'kd':
        loss, loss_cls, loss_aux = kd_criterion(
            out, labels, teacher_logits[train_idx], args.alpha, args.kd_T
        )
    elif args.training == 'fitnet':
        out_feat = student_proj(model.out_feat[train_idx])
        teacher_out_feat = teacher_proj(teacher_out_feat[train_idx])
        _, _, loss_aux = fitnet_criterion(
            out, labels, out_feat, teacher_out_feat, args.beta
        )
        loss, loss_cls, _ = kd_criterion(
            out, labels, teacher_logits[train_idx], args.alpha, args.kd_T
        )
        loss = loss + args.beta* loss_aux
    elif args.training == 'at':
        out_feat = model.out_feat[train_idx]
        teacher_out_feat = teacher_out_feat[train_idx]
        _, _, loss_aux = at_criterion(
            out, labels, out_feat, teacher_out_feat, args.beta
        )
        loss, loss_cls, _ = kd_criterion(
            out, labels, teacher_logits[train_idx], args.alpha, args.kd_T
        )
        loss = loss + args.beta* loss_aux
    elif args.training == 'gpw':
        out_feat = student_proj(model.out_feat[train_idx])
        teacher_out_feat = teacher_proj(teacher_out_feat[train_idx])
        _, _, loss_aux = gpw_criterion(
            out, labels, out_feat, teacher_out_feat, 
            args.kernel, args.beta, args.max_samples
        )
        loss, loss_cls, _ = kd_criterion(
            out, labels, teacher_logits[train_idx], args.alpha, args.kd_T
        )
        loss = loss + args.beta* loss_aux
    elif args.training == 'lpw':
        out_feat = model.out_feat[train_idx]
        teacher_out_feat = teacher_out_feat[train_idx] 
        _, _, loss_aux = lpw_criterion(
            out, labels, out_feat, teacher_out_feat, 
            edge_index, args.kernel, args.beta
        )
        loss, loss_cls, _ = kd_criterion(
            out, labels, teacher_logits[train_idx], args.alpha, args.kd_T
        )
        loss = loss + args.beta* loss_aux
    elif args.training == 'nce':
        out_feat = student_proj(model.out_feat[train_idx])
        teacher_out_feat = teacher_proj(teacher_out_feat[train_idx])
        _, _, loss_aux = nce_criterion(
            out, labels, out_feat, teacher_out_feat, 
            args.beta, args.nce_T, args.max_samples
        )
        loss, loss_cls, _ = kd_criterion(
            out, labels, teacher_logits[train_idx], args.alpha, args.kd_T
        )
        loss = loss + args.beta* loss_aux
    elif args.training == 'gcd':
        out_feat = student_proj(model.out_feat, data.adj_t)[train_idx]
        teacher_out_feat = teacher_proj(teacher_out_feat, data.adj_t)[train_idx]
        _, _, loss_aux = nce_criterion(
            out, labels, out_feat, teacher_out_feat, 
            args.beta, args.nce_T, args.max_samples
        )
        loss, loss_cls, _ = kd_criterion(
            out, labels, teacher_logits[train_idx], args.alpha, args.kd_T
        )
        loss = loss + args.beta* loss_aux
    else:
        raise NotImplementedError
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), loss_cls.item(), loss_aux.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
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

    return out, (train_acc, valid_acc, test_acc)


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
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    edge_index = None
    if args.training in ['lpw', 'nce-edges', 'nce-labels-edges']:
        edge_index = torch.stack(data.adj_t.coo()[:2])
        edge_index = subgraph(train_idx, edge_index, relabel_nodes=True)[0]

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

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        seed(args.seed + run)

        model.reset_parameters()
        
        teacher_out_feat, teacher_logits = None, None
        if args.training != 'supervised':
            teacher_out_feat = torch.load(f"../arxiv_dgl/features/gat-3L250x3h/{args.seed + run}.pt").to(device)
            teacher_logits = torch.load(f"../arxiv_dgl/logits/gat-3L250x3h/{args.seed + run}.pt").to(device)

        if args.training in ["nce", "fitnet", "gpw", "gcd"]:
            if args.training == 'gcd':
                student_proj = ProjectionGCD(args.hidden_channels, args.proj_dim).to(device)
                teacher_proj = ProjectionGCD(750, args.proj_dim).to(device)

            else:
                student_proj = torch.nn.Sequential(
                    torch.nn.Linear(args.hidden_channels, args.proj_dim), 
                    torch.nn.BatchNorm1d(args.proj_dim), 
                    torch.nn.ReLU()
                ).to(device)

                teacher_proj = torch.nn.Sequential(
                    torch.nn.Linear(750, args.proj_dim), 
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
            "logs/kd_and_aux",
            args.expt_name,
            f"{args.gnn}-L{args.num_layers}h{args.hidden_channels}-Student-{args.training}",
            f"{start_time_str}-GPU{args.device}-seed{args.seed+run}"
        )
        tb_logger = SummaryWriter(log_dir)
        
        # Start training
        best_epoch = 0
        best_train = 0
        best_val = 0
        best_test = 0
        best_logits = None
        for epoch in range(1, 1 + args.epochs):
            train_loss, train_loss_cls, train_loss_aux = train(
                model, data, train_idx, optimizer, 
                args, teacher_out_feat, teacher_logits,
                student_proj, teacher_proj, edge_index
            )
            
            logits, result = test(model, data, split_idx, evaluator)
            
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss_total: {train_loss:.4f}, '
                      f'Loss_cls: {train_loss_cls:.4f}, '
                      f'Loss_aux: {train_loss_aux:.8f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
                
                # Log statistics to Tensorboard, etc.
                tb_logger.add_scalar('loss/train', train_loss, epoch)
                tb_logger.add_scalar('loss/cls', train_loss_cls, epoch)
                tb_logger.add_scalar('loss/aux', train_loss_aux, epoch)
                tb_logger.add_scalar('acc/train', train_acc, epoch)
                tb_logger.add_scalar('acc/valid', valid_acc, epoch)
                tb_logger.add_scalar('acc/test', test_acc, epoch)

                if valid_acc > best_val:
                    best_epoch = epoch
                    best_train = train_acc
                    best_val = valid_acc
                    best_test = test_acc
                    best_logits = logits

        logger.print_statistics(run)
        torch.save({
            'args': args,
            'total_param': total_param,
            'BestEpoch': best_epoch,
            'Train': best_train,
            'Validation': best_val, 
            'Test': best_test, 
            'logits': best_logits,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(log_dir, "results.pt"))
    
    logger.print_statistics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    
    # Experiment settings
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--training', type=str, default="supervised")
    parser.add_argument('--expt_name', type=str, default="debug")

    # GNN settings
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    
    # KD settings
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='alpha parameter for KD (default: 0.5)')
    parser.add_argument('--kd_T', type=float, default=4.0,
                        help='temperature parameter for KD (default: 1.0)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='beta parameter for auxiliary distillation losses (default: 0.5)')
    
    # NCE/auxiliary distillation settings
    parser.add_argument('--nce_T', type=float, default=0.075,
                        help='temperature parameter for NCE (default: 0.075)')
    parser.add_argument('--max_samples', type=int, default=8192,
                        help='maximum samples for NCE/GPW (default: 8192)')
    parser.add_argument('--proj_dim', type=int, default=256,
                        help='common projection dimensionality for NCE/FitNet (default: 150)')
    parser.add_argument('--kernel', type=str, default='rbf',
                        help='kernel for LPW: cosine, polynomial, l2, rbf (default: rbf)')

    args = parser.parse_args()
    
    print(args)
    main()