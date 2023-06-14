import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_scheduler(optimizer):
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)
    return scheduler