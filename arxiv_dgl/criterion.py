import torch
import torch.nn.functional as F
import numpy as np

from torch_geometric.utils import softmax


def kd_criterion(logits, labels, teacher_logits, alpha=0.9, T=4):
    """Logit-based KD, [Hinton et al., 2015](https://arxiv.org/abs/1503.02531)
    """
    loss_cls = F.cross_entropy(logits, labels)

    loss_kd = F.kl_div(
        F.log_softmax(logits/ T, dim=1), 
        F.softmax(teacher_logits/ T, dim=1),
        log_target=False
    )

    loss = loss_kd* (alpha* T* T) + loss_cls* (1-alpha)
    
    return loss, loss_cls, loss_kd


def fitnet_criterion(logits, labels, feat, teacher_feat, beta=1000):
    """FitNet, [Romero et al., 2014](https://arxiv.org/abs/1412.6550)
    """
    loss_cls = F.cross_entropy(logits, labels)

    feat = F.normalize(feat, p=2, dim=-1)
    teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)

    loss_fitnet = F.mse_loss(feat, teacher_feat)

    loss = loss_cls + beta* loss_fitnet

    return loss, loss_cls, loss_fitnet


def at_criterion(logits, labels, feat, teacher_feat, beta=1000):
    """Attention Transfer, [Zagoruyko and Komodakis, 2016](https://arxiv.org/abs/1612.03928)
    """
    loss_cls = F.cross_entropy(logits, labels)

    feat = feat.pow(2).sum(-1)
    teacher_feat = teacher_feat.pow(2).sum(-1)

    loss_at = F.mse_loss(
        F.normalize(feat, p=2, dim=-1),
        F.normalize(teacher_feat, p=2, dim=-1)
    )

    loss = loss_cls + beta* loss_at

    return loss, loss_cls, loss_at


def gpw_criterion(logits, labels, feat, teacher_feat, kernel='cosine', beta=1, max_samples=8192):
    """Global Structure Preserving loss, [Joshi et al., TNNLS 2022](https://arxiv.org/abs/2111.04964)
    """
    loss_cls = F.cross_entropy(logits, labels)

    if max_samples < feat.shape[0]:
        sampled_inds = np.random.choice(feat.shape[0], max_samples, replace=False)
        feat = feat[sampled_inds]
        teacher_feat = teacher_feat[sampled_inds]
    
    pw_sim = None
    teacher_pw_sim = None
    if kernel == 'cosine':
        feat = F.normalize(feat, p=2, dim=-1)
        teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)
        pw_sim = torch.mm(feat, feat.transpose(0, 1)).flatten()
        teacher_pw_sim = torch.mm(teacher_feat, teacher_feat.transpose(0, 1)).flatten()
    elif kernel == 'poly':
        feat = F.normalize(feat, p=2, dim=-1)
        teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)
        pw_sim = torch.mm(feat, feat.transpose(0, 1)).flatten()**2
        teacher_pw_sim = torch.mm(teacher_feat, teacher_feat.transpose(0, 1)).flatten()**2
    elif kernel == 'l2':
        pw_sim = (feat.unsqueeze(0) - feat.unsqueeze(1)).norm(p=2, dim=-1).flatten()
        teacher_pw_sim = (teacher_feat.unsqueeze(0) - teacher_feat.unsqueeze(1)).norm(p=2, dim=-1).flatten()
    elif kernel == 'rbf':
        pw_sim = torch.exp(-0.5* ((feat.unsqueeze(0) - feat.unsqueeze(1))**2).sum(dim=-1).flatten())
        teacher_pw_sim = torch.exp(-0.5* ((teacher_feat.unsqueeze(0) - teacher_feat.unsqueeze(1))**2).sum(dim=-1).flatten())
    else:
        raise NotImplementedError

    loss_gpw = F.mse_loss(pw_sim, teacher_pw_sim)

    loss = loss_cls + beta* loss_gpw

    return loss, loss_cls, loss_gpw


def nce_criterion(logits, labels, feat, teacher_feat, beta=0.5, nce_T=0.075, max_samples=8192):
    """Graph Contrastive Representation Distillation, [Joshi et al., TNNLS 2022](https://arxiv.org/abs/2111.04964)
    """
    loss_cls = F.cross_entropy(logits, labels)

    if max_samples < feat.shape[0]:
        sampled_inds = np.random.choice(feat.shape[0], max_samples, replace=False)
        feat = feat[sampled_inds]
        teacher_feat = teacher_feat[sampled_inds]
    
    feat = F.normalize(feat, p=2, dim=-1)
    teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)

    nce_logits = torch.mm(feat, teacher_feat.transpose(0, 1))
    nce_labels = torch.arange(feat.shape[0]).to(feat.device)

    loss_nce = F.cross_entropy(nce_logits/ nce_T, nce_labels)
    
    loss = loss_cls + beta* loss_nce

    return loss, loss_cls, loss_nce
