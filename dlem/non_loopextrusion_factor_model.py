import torch
import numpy as np
from dlem import util

class NonLoopExt(torch.nn.Module):
    def __init__(self, in_features, rank_k, patch_size):
        super(NonLoopExt, self).__init__()
        self.rank_k = rank_k
        self.in_features = in_features
        self.patch_size = patch_size
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_features, out_channels=4,
                            kernel_size=5, dilation=2, padding=4),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(4),
            torch.nn.MaxPool1d(kernel_size=10),
            torch.nn.Conv1d(in_channels=4, out_channels=rank_k,
                            kernel_size=5, dilation=2, padding=4),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(rank_k),
            torch.nn.MaxPool1d(kernel_size=10),
            torch.nn.Conv1d(in_channels=rank_k, out_channels=rank_k,
                            kernel_size=5, dilation=2, padding=4),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(rank_k),
            torch.nn.MaxPool1d(kernel_size=10),
            torch.nn.Conv1d(in_channels=rank_k, out_channels=rank_k,
                            kernel_size=5, dilation=2, padding=4),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(rank_k),
            torch.nn.MaxPool1d(kernel_size=10),
        )
        self.triu_i = np.concatenate([np.arange(patch_size-n) for n in range(1, patch_size)])
        self.triu_j = np.concatenate([np.arange(n, patch_size) for n in range(1, patch_size)])
    def forward(self, x):
        x = self.layers(x)
        b, c, l = x.shape
        x = x.reshape(b, c, l, 1) + x.reshape(b, c, 1, l)
        return x.sum(axis=1)[:, self.triu_i, self.triu_j]
