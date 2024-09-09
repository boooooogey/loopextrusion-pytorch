"""PyTorch dataset class for reading and processing data from a Cooler file."""
import torch
from torch.utils.data import Dataset
import numpy as np
from .util import (get_contact_map,
                   return_chrom_len_list,
                   return_chrom_list,
                   return_region_from_index)

class DLEMDataset(Dataset):
    """
    A dataset class for reading and processing data from a Cooler file.
    Args:
        cooler_file (str): The path to the Cooler file.
        resolution (int): The resolution of the data.
        window_size (int): The size of the sliding window.
        stride (int): The stride of the sliding window.
        do_adaptive_coarsegrain (bool, optional): Whether to perform adaptive coarse-graining. 
        Defaults to True.
    """
    def __init__(self,
                 cooler_file:str,
                 resolution:int,
                 window_size:int,
                 stride:int,
                 do_adaptive_coarsegrain=True):

        self.cooler_file = cooler_file
        self.resolution = resolution
        self.do_adaptive_coarsegrain = do_adaptive_coarsegrain
        self.window_size = window_size
        self.stride = stride
        self.chromosome_list = return_chrom_list(cooler_file)
        self.chromosome_len_list = return_chrom_len_list(cooler_file,
                                                         resolution,
                                                         window_size,
                                                         stride)

        self.start_indices = np.cumsum(self.chromosome_len_list)
        self.len = self.start_indices[-1]  # Total number of windows
        self.start_indices = np.insert(self.start_indices, 0, 0)[:-1]

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.len

    def __getitem__(self, idx):

        # Get the chromosome, start, end from the index
        chromosome, start, end = return_region_from_index(idx,
                                                          self.start_indices,
                                                          self.chromosome_list,
                                                          self.window_size,
                                                          self.stride)
        start = start * self.resolution
        end = end * self.resolution

        # Get the contact map and sum_zero for the specified region
        contact_map, sum_zero = get_contact_map(self.cooler_file,
                                                self.resolution,
                                                chromosome,
                                                start,
                                                end,
                                                self.do_adaptive_coarsegrain)

        # Convert the contact map and sum_zero to torch tensors
        contact_map = torch.from_numpy(contact_map)

        return contact_map, sum_zero, chromosome, start, end
