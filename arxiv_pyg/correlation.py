import os
import numpy as np
import time
import random

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import subgraph

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances

from gnn import *


def pairwise_euclidean_dist(A, eps=1e-10):
    # Fast pairwise euclidean distance on GPU
    # TODO why is this not fast?
	sqrA = torch.sum(torch.pow(A, 2), 1, keepdim=True).expand(A.shape[0], A.shape[0])
	return torch.sqrt(
		sqrA - 2*torch.mm(A, A.t()) + sqrA.t() + eps
	)


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


def fast_linear_CKA(X, Y):
    L_X = centering(np.dot(X, X.T))
    L_Y = centering(np.dot(Y, Y.T))
    hsic = np.sum(L_X * L_Y)
    var1 = np.sqrt(np.sum(L_X * L_X))
    var2 = np.sqrt(np.sum(L_Y * L_Y))

    return hsic / (var1 * var2)


# Load ARXIV dataset

device = 'cuda:0'  # 'cpu'  # f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())

data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)

split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)
valid_idx = split_idx['valid'].to(device)
test_idx = split_idx['test'].to(device)

edge_index = torch.stack(data.adj_t.coo()[:2])
edge_index = subgraph(valid_idx, edge_index, relabel_nodes=True)[0]
src, dst = edge_index

evaluator = Evaluator(name='ogbn-arxiv')

###

class args:
    # Experiment settings
    device = -1
    seed = 0
    
    gnn = "sage"  #"sage"

    # metric = "cosine-local"
    
    num_layers = 2
    hidden_channels = 256
    dropout = 0.5

checkpoint_dirs = {
    # Example
    "GCD": "/home/user/molecules/arxiv_pyg/logs_gcd/kd_and_aux/gcd/sage-L2h256-Student-gcd",
    "NCE": "/home/user/molecules/arxiv_pyg/logs_kd_and_aux/sage-rerun/nce/sage-L2h256-Student-nce",
    "LPW": "/home/user/molecules/arxiv_pyg/logs_kd_and_aux/sage/lpw-cosine/sage-L2h256-Student-lpw",
    "GPW": "/home/user/molecules/arxiv_pyg/logs_kd_and_aux/sage-rerun/gpw-cosine/sage-L2h256-Student-gpw",
    "AT": "/home/user/molecules/arxiv_pyg/logs_kd_and_aux/sage-rerun/at/sage-L2h256-Student-at",
    "FitNet": "/home/user/molecules/arxiv_pyg/logs_kd_and_aux/sage/fitnet/sage-L2h256-Student-fitnet",
    "KD": "/home/user/molecules/arxiv_pyg/logs/sage/kd/sage-L2h256-Student-kd",
    "Sup.": "/home/user/molecules/arxiv_pyg/logs/sage/supervised/sage-L2h256-Student-supervised"
}
    
###

with torch.no_grad():

    for method, checkpoint_dir in zip(checkpoint_dirs.keys(), checkpoint_dirs.values()):
        print(checkpoint_dir)

        student_checkpoints = {}
        for root, dirs, files in os.walk(checkpoint_dir):
            if 'results.pt' in files:
                seed = int(root[-1])
                student_checkpoints[seed] = os.path.join(root, 'results.pt')

        corr_list = []
        corr_list_local = []
        cka_list = []
        for run in range(10):
            torch.cuda.empty_cache()

            # Teacher features and pairwise distances
            f_t = torch.load(f"../arxiv_dgl/features/gat-3L250x3h/{run}.pt", map_location=torch.device('cpu')).to(device)
            f_t = f_t[valid_idx]
            
            # if args.metric == "cosine":
            #     f_t = F.normalize(f_t, p=2, dim=-1)
            #     pw_t = 1 - torch.mm(f_t, f_t.transpose(0, 1)).cpu().numpy()
            #     np.fill_diagonal(pw_t, 0)
            #     pw_t = squareform(pw_t)
            
            # elif args.metric == "cosine-local":
            #     f_t = F.normalize(f_t, p=2, dim=-1)
            #     pw_t = 1 - F.cosine_similarity(f_t[src], f_t[dst]).cpu().numpy()

            # elif args.metric == 'cka':
            #     f_t = F.normalize(f_t, p=2, dim=-1).cpu().numpy()
            
            # else:
            #     # pw_t = pairwise_euclidean_dist(f_t)
            #     # pw_t = pairwise_distances(f_t.cpu().numpy(), metric='euclidean', n_jobs=32)
            #     pw_t = pdist(f_t.cpu().numpy(), 'euclidean')

            f_t = F.normalize(f_t, p=2, dim=-1)
            pw_t = 1 - torch.mm(f_t, f_t.transpose(0, 1)).cpu().numpy()
            np.fill_diagonal(pw_t, 0)
            pw_t = squareform(pw_t)
            pw_t_local = 1 - F.cosine_similarity(f_t[src], f_t[dst]).cpu().numpy()
            
            ###
            
            # Student features and pairwise distances
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

            checkpoint = torch.load(student_checkpoints[run], map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            model(data.x, data.adj_t)
            f_s = model.out_feat[valid_idx]
            
            f_s = F.normalize(f_s, p=2, dim=-1)
            pw_s = 1 - torch.mm(f_s, f_s.transpose(0, 1)).cpu().numpy()
            np.fill_diagonal(pw_s, 0)
            pw_s = squareform(pw_s)
            pw_s_local = 1 - F.cosine_similarity(f_s[src], f_s[dst]).cpu().numpy()
            
            # Compute correlation
            corr_list.append(pearsonr(pw_t, pw_s)[0])
            corr_list_local.append(pearsonr(pw_t_local, pw_s_local)[0])
            cka_list.append(fast_linear_CKA(f_s.cpu().numpy(), f_t.cpu().numpy()))
            
        # print(corr_list)
        print(f"{method}: {np.mean(corr_list):.4f} +- {np.std(corr_list):.4f}, {np.mean(corr_list_local):.4f} +- {np.std(corr_list_local):.4f}, {np.mean(cka_list):.4f} +- {np.std(cka_list):.4f}")
