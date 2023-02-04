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


def train(model, train_dataset, train_loader, optimizer, args, device):
    model.train()
    
    avg_loss = 0
    
    for step, batch in enumerate(train_loader):
        batch = batch.to(device)
        labels = batch.y
        out = model(batch.x, batch.edge_index)

        loss = F.binary_cross_entropy_with_logits(out, labels)
        loss_cls = loss
        loss_aux = loss*0
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.detach().item()
        
    avg_loss /= (step + 1)
    return avg_loss


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

    model = TeacherNet(train_dataset.num_features, train_dataset.num_classes).to(device)
    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        seed(args.seed + run)

        model.reset_parameters()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Create Tensorboard logger
        log_dir = os.path.join(
            "checkpoints",
            f"seed{args.seed+run}"
        )
        tb_logger = SummaryWriter(log_dir)
        
        # Start training
        best_epoch = 0
        best_train = 0
        best_val = 0
        best_test = 0
        for epoch in range(1, 1 + args.epochs):
            train_loss = train(
                model, train_dataset, train_loader, 
                optimizer, args, device
            )
            
            train_acc = test(model, train_dataset, train_loader, device)
            valid_acc = test(model, val_dataset, val_loader, device)
            test_acc = test(model, test_dataset, test_loader, device)
            logger.add_result(run, (train_acc, valid_acc, test_acc))

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {train_loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
                
                # Log statistics to Tensorboard, etc.
                tb_logger.add_scalar('loss/train', train_loss, epoch)
                tb_logger.add_scalar('acc/train', train_acc, epoch)
                tb_logger.add_scalar('acc/valid', valid_acc, epoch)
                tb_logger.add_scalar('acc/test', test_acc, epoch)

                if valid_acc > best_val:
                    best_epoch = epoch
                    best_train = train_acc
                    best_val = valid_acc
                    best_test = test_acc

                    torch.save({
                        'args': args,
                        'total_param': total_param,
                        'BestEpoch': best_epoch,
                        'Train': best_train,
                        'Validation': best_val, 
                        'Test': best_test,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(log_dir, "checkpoint.pt"))

        logger.print_statistics(run)
        
    logger.print_statistics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPI')
    
    # Experiment settings
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument('--log_steps', type=int, default=1)
    
    # GNN settings
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    
    args = parser.parse_args()
    
    print(args)
    main()