import argparse
import numpy as np
import time
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import subgraph
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader

from sklearn.metrics import f1_score

from logger import Logger

from criterion import *


class TeacherNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TeacherNet, self).__init__()
        self.conv1 = GATConv(in_channels, 256, heads=4)
        self.lin1 = torch.nn.Linear(in_channels, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(4 * 256, out_channels, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        self.out_feat = x
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x


class StudentNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StudentNet, self).__init__()
        self.conv1 = GATConv(in_channels, 68, heads=2)
        self.lin1 = torch.nn.Linear(in_channels, 2 * 68)
        self.conv2 = GATConv(2 * 68, 68, heads=2)
        self.lin2 = torch.nn.Linear(2 * 68, 2 * 68)
        self.conv3 = GATConv(2 * 68, 68, heads=2)
        self.lin3 = torch.nn.Linear(2 * 68, 2 * 68)
        self.conv4 = GATConv(2 * 68, 68, heads=2)
        self.lin4 = torch.nn.Linear(2 * 68, 2 * 68)
        self.conv5 = GATConv(2 * 68, out_channels, heads=2, concat=False)
        self.lin5 = torch.nn.Linear(2 * 68, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()
        self.lin5.reset_parameters()

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = F.elu(self.conv3(x, edge_index) + self.lin3(x))
        x = F.elu(self.conv4(x, edge_index) + self.lin4(x))
        self.out_feat = x
        x = self.conv5(x, edge_index) + self.lin5(x)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, heads=4):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels* heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads* hidden_channels, hidden_channels, heads=heads))
            self.lins.append(
                torch.nn.Linear(hidden_channels* heads, hidden_channels* heads))
        self.convs.append(GATConv(heads* hidden_channels, out_channels, heads=heads, concat=False))
        self.lins.append(torch.nn.Linear(hidden_channels* heads, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        for i, (conv, lin) in enumerate(zip(self.convs[:-1], self.lins[:-1])):
            x = conv(x, adj_t) + lin(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            self.out_feat = x
        x = self.convs[-1](x, adj_t) + self.lins[-1](x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False))

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


def train(model, teacher_model, train_dataset, train_loader, optimizer, args, device, student_proj=None, teacher_proj=None):
    model.train()
    if student_proj:
        student_proj.train()
    if teacher_proj:
        teacher_proj.train()
    if teacher_model:
        teacher_model.eval()

    avg_loss = 0
    avg_loss_cls = 0
    avg_loss_aux = 0
    
    for step, batch in enumerate(train_loader):
        batch = batch.to(device)
        labels = batch.y
        out = model(batch.x, batch.edge_index)

        if args.training == 'supervised':
            loss = F.binary_cross_entropy_with_logits(out, labels)
            loss_cls = loss
            loss_aux = loss*0
        elif args.training == 'kd':
            with torch.no_grad():
                teacher_out = teacher_model(batch.x, batch.edge_index)
            loss, loss_cls, loss_aux = kd_criterion(
                out, labels, teacher_out, args.alpha, args.kd_T
            )
        elif args.training == 'fitnet':
            with torch.no_grad():
                teacher_out = teacher_model(batch.x, batch.edge_index)
                teacher_out_feat = teacher_model.out_feat
            out_feat = student_proj(model.out_feat)
            teacher_out_feat = teacher_proj(teacher_out_feat)
            loss, loss_cls, loss_aux = fitnet_criterion(
                out, labels, out_feat, teacher_out_feat, args.beta
            )
        elif args.training == 'at':
            with torch.no_grad():
                teacher_out = teacher_model(batch.x, batch.edge_index)
                teacher_out_feat = teacher_model.out_feat
            out_feat = model.out_feat
            loss, loss_cls, loss_aux = at_criterion(
                out, labels, out_feat, teacher_out_feat, args.beta
            )
        elif args.training == 'gpw':
            with torch.no_grad():
                teacher_out = teacher_model(batch.x, batch.edge_index)
                teacher_out_feat = teacher_model.out_feat
            out_feat = model.out_feat
            teacher_out_feat = teacher_out_feat
            loss, loss_cls, loss_aux = gpw_criterion(
                out, labels, out_feat, teacher_out_feat, 
                args.kernel, args.beta, args.max_samples
            )
        elif args.training == 'lpw':
            with torch.no_grad():
                teacher_out = teacher_model(batch.x, batch.edge_index)
                teacher_out_feat = teacher_model.out_feat
            out_feat = model.out_feat
            teacher_out_feat = teacher_out_feat 
            loss, loss_cls, loss_aux = lpw_criterion(
                out, labels, out_feat, teacher_out_feat, 
                batch.edge_index, args.kernel, args.beta
            )
        elif args.training == 'nce':
            with torch.no_grad():
                teacher_out = teacher_model(batch.x, batch.edge_index)
                teacher_out_feat = teacher_model.out_feat
            out_feat = student_proj(model.out_feat)
            teacher_out_feat = teacher_proj(teacher_out_feat)
            loss, loss_cls, loss_aux = nce_criterion(
                out, labels, out_feat, teacher_out_feat, 
                args.beta, args.nce_T, args.max_samples
            )
        else:
            raise NotImplementedError
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.detach().item()
        avg_loss_cls += loss_cls.detach().item()
        avg_loss_aux += loss_aux.detach().item()

    avg_loss /= (step + 1)
    avg_loss_cls /= (step + 1)
    avg_loss_aux /= (step + 1) 
    return avg_loss, avg_loss_cls, avg_loss_aux


@torch.no_grad()
def test(model, dataset, loader, device):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    train_dataset = PPI('data/PPI/', split='train')
    val_dataset = PPI('data/PPI/', split='val')
    test_dataset = PPI('data/PPI/', split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    if args.gnn == 'teacher':
        model = TeacherNet(train_dataset.num_features, train_dataset.num_classes).to(device)
        args.heads = 4  # hardcode heads for student_proj
    elif args.gnn == 'student':
        model = StudentNet(train_dataset.num_features, train_dataset.num_classes).to(device)
        args.heads = 2  # hardcode heads for student_proj
    elif args.gnn == 'gat':
        model = GAT(train_dataset.num_features, args.hidden_channels,
                    train_dataset.num_classes, args.num_layers,
                    args.dropout, args.heads).to(device)
    elif args.gnn == 'sage':
        model = SAGE(train_dataset.num_features, args.hidden_channels,
                     train_dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
        args.heads = 1  # hardcode heads for student_proj
    elif args.gnn == 'gcn':
        model = GCN(train_dataset.num_features, args.hidden_channels,
                    train_dataset.num_classes, args.num_layers,
                    args.dropout).to(device)
        args.heads = 1  # hardcode heads for student_proj
    else:
        raise ValueError('Invalid GNN type')

    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        seed(args.seed + run)

        model.reset_parameters()
        
        teacher_model = None
        if args.training != 'supervised':
            teacher_model = TeacherNet(train_dataset.num_features, train_dataset.num_classes).to(device)
            checkpoint = torch.load(f"{args.teacher_path}/seed{args.seed + run}/checkpoint.pt")
            teacher_model.load_state_dict(checkpoint['model_state_dict'])
            teacher_model.eval()

        if args.training in ["nce", "fitnet"]:
            student_proj = torch.nn.Sequential(
                torch.nn.Linear(args.hidden_channels* args.heads, args.proj_dim), 
                torch.nn.BatchNorm1d(args.proj_dim), 
                torch.nn.ReLU()
            ).to(device)

            teacher_proj = torch.nn.Sequential(
                torch.nn.Linear(1024, args.proj_dim), 
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
            "logs",
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
        for epoch in range(1, 1 + args.epochs):
            train_loss, train_loss_cls, train_loss_aux = train(
                model, teacher_model, train_dataset, train_loader, 
                optimizer, args, device, student_proj, teacher_proj
            )
            
            train_acc = test(model, train_dataset, train_loader, device)
            valid_acc = test(model, val_dataset, val_loader, device)
            test_acc = test(model, test_dataset, test_loader, device)
            logger.add_result(run, (train_acc, valid_acc, test_acc))

            if epoch % args.log_steps == 0:
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

        logger.print_statistics(run)
        torch.save({
            'args': args,
            'total_param': total_param,
            'BestEpoch': best_epoch,
            'Train': best_train,
            'Validation': best_val, 
            'Test': best_test, 
        }, os.path.join(log_dir, "results.pt"))
    
    logger.print_statistics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPI')
    
    # Experiment settings
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--training', type=str, default="supervised")
    parser.add_argument('--expt_name', type=str, default="debug")
    parser.add_argument('--teacher_path', type=str, default="checkpoints")

    # GNN settings
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    
    # KD settings
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha parameter for KD (default: 0.5)')
    parser.add_argument('--kd_T', type=float, default=1.0,
                        help='temperature parameter for KD (default: 1.0)')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='beta parameter for auxiliary distillation losses (default: 0.0)')
    
    # NCE/auxiliary distillation settings
    parser.add_argument('--nce_T', type=float, default=0.075,
                        help='temperature parameter for NCE (default: 0.075)')
    parser.add_argument('--max_samples', type=int, default=8192,
                        help='maximum samples for NCE/GPW (default: 8192)')
    parser.add_argument('--proj_dim', type=int, default=256,
                        help='common projection dimensionality for NCE/FitNet (default: 256)')
    parser.add_argument('--kernel', type=str, default='rbf',
                        help='kernel for LPW: cosine, polynomial, l2, rbf (default: rbf)')

    args = parser.parse_args()
    
    print(args)
    main()