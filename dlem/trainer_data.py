"""Contains Pytorch Lightning trainer class for DLEM.
"""
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

class LitTrainerData(L.LightningDataModule):
    """Trainer class for DLEM.
    """
    def __init__(self, dataset_training, dataset_test, dataset_val, batch_size, overlap, offsets):
        super().__init__()
        self.dataset_training = dataset_training
        self.dataset_test = dataset_test
        self.dataset_val = dataset_val
        self.batch_size = batch_size
        self.overlap = overlap
        self.offsets = offsets

    def on_train_epoch_start(self, current_epoch):
        self.dataset_training.rearrange_patches(self.overlap,
                                        self.offsets[current_epoch % len(self.offsets)])

    def train_dataloader(self):
        return DataLoader(self.dataset_training, batch_size=self.batch_size, shuffle=True, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False)
