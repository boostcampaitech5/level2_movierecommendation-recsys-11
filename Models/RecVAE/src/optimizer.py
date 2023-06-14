import torch
from torch.optim import Adam


def get_optimizer(args, params, model):
    optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer