import torch.nn.functional as F
import torch


def nll_loss_weighted(output, target):
    return F.nll_loss(
        output,
        target,
        weight=torch.tensor([0.1, 0.1, 0.8]).to(output.device, dtype=torch.float32),
    )
