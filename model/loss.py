import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target, weight=torch.tensor([0.1, 0.1, 0.8]))

def mse_loss(output, target):
    return F.mse_loss(output, target)