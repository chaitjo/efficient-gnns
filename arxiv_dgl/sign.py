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

from criterion import *


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


def train(model, feats, labels, optimizer, train_loader, args, 
          teacher_out_feat, teacher_logits, student_proj=None, teacher_proj=None):
    model.train()
    if student_proj:
        student_proj.train()
    if teacher_proj:
        teacher_proj.train()
    
    device = labels.device
    
    avg_loss = 0; loss = None
    avg_loss_cls = 0; loss_cls = None
    avg_loss_aux = 0; loss_aux = None

    for step, batch in enumerate(train_loader):
        batch_feats = [x[batch].to(device) for x in feats]
        
        if args.training == 'supervised':
            loss = F.cross_entropy(model(batch_feats), labels[batch])
        elif args.training == 'kd':
            batch_logits = model(batch_feats)
            teacher_batch_logits = teacher_logits[batch].to(device)
            loss, loss_cls, loss_aux = kd_criterion(
                batch_logits, labels[batch], teacher_batch_logits, args.alpha, args.kd_T
            )
        elif args.training == 'fitnet':
            batch_logits = model(batch_feats)
            batch_out_feat = student_proj(model.out_feat)
            teacher_batch_out_feat = teacher_proj(teacher_out_feat[batch].to(device))
            loss, loss_cls, loss_aux = fitnet_criterion(
                batch_logits, labels[batch], batch_out_feat, teacher_batch_out_feat, args.beta
            )
        elif args.training == 'at':
            batch_logits = model(batch_feats)
            batch_out_feat = model.out_feat
            teacher_batch_out_feat = teacher_out_feat[batch].to(device)
            loss, loss_cls, loss_aux = at_criterion(
                batch_logits, labels[batch], batch_out_feat, teacher_batch_out_feat, args.beta
            )
        elif args.training == 'gpw':
            batch_logits = model(batch_feats)
            batch_out_feat = student_proj(model.out_feat)
            teacher_batch_out_feat = teacher_proj(teacher_out_feat[batch].to(device))
            loss, loss_cls, loss_aux = gpw_criterion(
                batch_logits, labels[batch], batch_out_feat, teacher_batch_out_feat, 
                args.kernel, args.beta, args.max_samples
            )
        elif args.training == 'nce':
            batch_logits = model(batch_feats)
            batch_out_feat = student_proj(model.out_feat)
            teacher_batch_out_feat = teacher_proj(teacher_out_feat[batch].to(device))
            loss, loss_cls, loss_aux = nce_criterion(
                batch_logits, labels[batch], batch_out_feat, teacher_batch_out_feat, 
                args.beta, args.nce_T, args.max_samples
            )
        else:
            raise NotImplementedError
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.detach().item()
        avg_loss_cls += loss_cls.detach().item() if loss_cls else 0
        avg_loss_aux += loss_aux.detach().item() if loss_aux else 0

    avg_loss /= (step + 1)
    avg_loss_cls /= (step + 1)
    avg_loss_aux /= (step + 1) 
    return avg_loss, avg_loss_cls, avg_loss_aux


def train_kd_and_aux(model, feats, labels, optimizer, train_loader, args, 
          teacher_out_feat, teacher_logits, student_proj=None, teacher_proj=None):
    model.train()
    if student_proj:
        student_proj.train()
    if teacher_proj:
        teacher_proj.train()
    
    device = labels.device
    
    avg_loss = 0; loss = None
    avg_loss_cls = 0; loss_cls = None
    avg_loss_aux = 0; loss_aux = None

    for step, batch in enumerate(train_loader):
        batch_feats = [x[batch].to(device) for x in feats]
        
        if args.training == 'supervised':
            loss = F.cross_entropy(model(batch_feats), labels[batch])
        elif args.training == 'kd':
            batch_logits = model(batch_feats)
            teacher_batch_logits = teacher_logits[batch].to(device)
            loss, loss_cls, loss_aux = kd_criterion(
                batch_logits, labels[batch], teacher_batch_logits, args.alpha, args.kd_T
            )
        elif args.training == 'fitnet':
            batch_logits = model(batch_feats)
            batch_out_feat = student_proj(model.out_feat)
            teacher_batch_out_feat = teacher_proj(teacher_out_feat[batch].to(device))
            _, _, loss_aux = fitnet_criterion(
                batch_logits, labels[batch], batch_out_feat, teacher_batch_out_feat, args.beta
            )
            teacher_batch_logits = teacher_logits[batch].to(device)
            loss, loss_cls, _ = kd_criterion(
                batch_logits, labels[batch], teacher_batch_logits, args.alpha, args.kd_T
            )
            loss = loss + args.beta* loss_aux
        elif args.training == 'at':
            batch_logits = model(batch_feats)
            batch_out_feat = model.out_feat
            teacher_batch_out_feat = teacher_out_feat[batch].to(device)
            _, _, loss_aux = at_criterion(
                batch_logits, labels[batch], batch_out_feat, teacher_batch_out_feat, args.beta
            )
            teacher_batch_logits = teacher_logits[batch].to(device)
            loss, loss_cls, _ = kd_criterion(
                batch_logits, labels[batch], teacher_batch_logits, args.alpha, args.kd_T
            )
            loss = loss + args.beta* loss_aux
        elif args.training == 'gpw':
            batch_logits = model(batch_feats)
            batch_out_feat = student_proj(model.out_feat)
            teacher_batch_out_feat = teacher_proj(teacher_out_feat[batch].to(device))
            _, _, loss_aux = gpw_criterion(
                batch_logits, labels[batch], batch_out_feat, teacher_batch_out_feat, 
                args.kernel, args.beta, args.max_samples
            )
            teacher_batch_logits = teacher_logits[batch].to(device)
            loss, loss_cls, _ = kd_criterion(
                batch_logits, labels[batch], teacher_batch_logits, args.alpha, args.kd_T
            )
            loss = loss + args.beta* loss_aux
        elif args.training == 'nce':
            batch_logits = model(batch_feats)
            batch_out_feat = student_proj(model.out_feat)
            teacher_batch_out_feat = teacher_proj(teacher_out_feat[batch].to(device))
            _, _, loss_aux = nce_criterion(
                batch_logits, labels[batch], batch_out_feat, teacher_batch_out_feat, 
                args.beta, args.nce_T, args.max_samples
            )
            teacher_batch_logits = teacher_logits[batch].to(device)
            loss, loss_cls, _ = kd_criterion(
                batch_logits, labels[batch], teacher_batch_logits, args.alpha, args.kd_T
            )
            loss = loss + args.beta* loss_aux
        else:
            raise NotImplementedError
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.detach().item()
        avg_loss_cls += loss_cls.detach().item() if loss_cls else 0
        avg_loss_aux += loss_aux.detach().item() if loss_aux else 0

    avg_loss /= (step + 1)
    avg_loss_cls /= (step + 1)
    avg_loss_aux /= (step + 1) 
    return avg_loss, avg_loss_cls, avg_loss_aux


def test(model, feats, labels, test_loader, evaluator, train_nid, val_nid, test_nid):
    model.eval()
    device = labels.device
    preds = []
    logits = []
    for batch in test_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        batch_logits = model(batch_feats)
        preds.append(torch.argmax(batch_logits, dim=-1))
        logits.append(batch_logits)
    # Concat mini-batch prediction results along node dimension
    preds = torch.cat(preds, dim=0)
    logits = torch.cat(logits, dim=0)
    train_res = evaluator(preds[train_nid], labels[train_nid])
    val_res = evaluator(preds[val_nid], labels[val_nid])
    test_res = evaluator(preds[test_nid], labels[test_nid])
    return logits, (train_res, val_res, test_res)


def run(run_idx, args, data, device, teacher_out_feat=None, teacher_logits=None):
    feats, labels, in_size, num_classes, \
        train_nid, val_nid, test_nid, evaluator = data
    train_loader = torch.utils.data.DataLoader(
        train_nid, batch_size=args.batch_size, shuffle=True, drop_last=False)
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

    if args.training in ["nce", "fitnet", "gpw"]:
        student_proj = torch.nn.Sequential(
            torch.nn.Linear(args.num_hidden * num_hops, args.proj_dim), 
            torch.nn.BatchNorm1d(args.proj_dim), 
            torch.nn.ReLU()
        ).to(device)

        teacher_proj = torch.nn.Sequential(
            torch.nn.Linear(750, args.proj_dim), 
            torch.nn.BatchNorm1d(args.proj_dim), 
            torch.nn.ReLU()
        ).to(device)

        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': student_proj.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': teacher_proj.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}
        ])
    else:
        student_proj, teacher_proj = None, None
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Create Tensorboard logger
    start_time_str = time.strftime("%Y%m%dT%H%M%S")
    log_dir = os.path.join(
        "logs/kd_and_aux",
        args.expt_name,
        f"{start_time_str}-GPU{args.gpu}-seed{args.seed+run_idx}"
    )
    tb_logger = SummaryWriter(log_dir)

    # Start training
    best_epoch = 0
    best_train = 0
    best_val = 0
    best_test = 0
    best_logits = None
    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
        # train_loss, train_loss_cls, train_loss_aux = train(
        #     model, feats, labels, optimizer, train_loader, 
        #     args, teacher_out_feat, teacher_logits,
        #     student_proj, teacher_proj
        # )
        train_loss, train_loss_cls, train_loss_aux = train_kd_and_aux(
            model, feats, labels, optimizer, train_loader, 
            args, teacher_out_feat, teacher_logits,
            student_proj, teacher_proj
        )
        
        if epoch % args.eval_every == 0:
            with torch.no_grad():
                logits, acc = test(model, feats, labels, test_loader, evaluator,
                                   train_nid, val_nid, test_nid)
            end = time.time()
            log = "Epoch {}, Time(s): {:.4f}, ".format(epoch, end - start)
            log += "Acc: Train {:.4f}, Val {:.4f}, Test {:.4f}, ".format(*acc)
            log += "Loss: Total {:.2f}, Cls {:.2f}, Aux {:.8f}".format(train_loss, train_loss_cls, train_loss_aux)
            print(log)

            # Log statistics to Tensorboard, etc.
            tb_logger.add_scalar('loss/train', train_loss, epoch)
            tb_logger.add_scalar('loss/cls', train_loss_cls, epoch)
            tb_logger.add_scalar('loss/aux', train_loss_aux, epoch)
            tb_logger.add_scalar(f'acc/train', acc[0], epoch)
            tb_logger.add_scalar(f'acc/valid', acc[1], epoch)
            tb_logger.add_scalar(f'acc/test', acc[2], epoch)

            if acc[1] > best_val:
                best_epoch = epoch
                best_train = acc[0]
                best_val = acc[1]
                best_test = acc[2]
                best_logits = logits

    print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
        best_epoch, best_val, best_test))

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
    
    return best_val, best_test


def main(args):
    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)

    with torch.no_grad():
        data = prepare_data(device, args)
    
    val_accs = []
    test_accs = []
    for i in range(args.num_runs):
        seed(args.seed + i)
        print(f"Run {i} start training")

        teacher_out_feat, teacher_logits = None, None
        if args.training != 'supervised':
            teacher_out_feat = torch.load(f"./features/gat-3L250x3h/{args.seed + i}.pt").to(device)
            teacher_logits = torch.load(f"./logits/gat-3L250x3h/{args.seed + i}.pt").to(device)

        best_val, best_test = run(i, args, data, device, teacher_out_feat, teacher_logits)
        val_accs.append(best_val)
        test_accs.append(best_test)

    print(f"Average val accuracy: {np.mean(val_accs):.4f}, "
          f"std: {np.std(val_accs):.4f}")
    print(f"Average test accuracy: {np.mean(test_accs):.4f}, "
          f"std: {np.std(test_accs):.4f}")


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
    parser.add_argument('--training', type=str, default="supervised",
                        help='training type: supervised, kd, nce, fitnet (default: supervised)')
    parser.add_argument('--expt_name', type=str, default="debug",
                        help='experiment name to output result')
    
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
    main(args)