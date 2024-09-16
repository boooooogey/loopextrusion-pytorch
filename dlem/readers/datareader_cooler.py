"""PyTorch dataset class for reading and processing data from a Cooler file."""
from typing import List, Tuple, Union
from numpy.typing import ArrayLike
import torch
from torch.utils.data import Dataset
import numpy as np
from dlem.readers.util import (get_contact_map,
                               return_chrom_len_list,
                               return_chrom_list,
                               return_region_from_index,
                               return_chrom_size_list,
                               return_patch_bin_range,
                               return_chromosome_length)

def find_genomic_position(indx:int,
                          start_indices:ArrayLike,
                          chromosome_list:List[str],
                          window_size:int,
                          stride:int,
                          resolution:int,
                          cooler_file:str) -> Tuple[str, int, int]:
    """Find the genomic position corresponding to the given index.

    Args:
        indx (int): The index of the region.
        start_indices (ArrayLike): The start indices of each chromosome.
        chromosome_list (List[str]): The list of chromosome names.
        window_size (int): The size of the sliding window.
        stride (int): The stride of the sliding window.
        resolution (int): The resolution of the data.
        cooler_file (str): The path to the Cooler file.

    Returns:
        Tuple[str, int, int]: A tuple containing the chromosome name, start position, and end position.
    """
    chromosome, start, end = return_region_from_index(indx,
                                                      start_indices,
                                                      chromosome_list,
                                                      window_size,
                                                      stride)

    start = start * resolution
    end = min(end * resolution, return_chromosome_length(cooler_file,
                                                         resolution,
                                                         chromosome))
    return chromosome, start, end


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
                 chrom_subset:Union[List[str],None]=None,
                 do_adaptive_coarsegrain=True):

        self.cooler_file = cooler_file
        self.resolution = resolution
        self.do_adaptive_coarsegrain = do_adaptive_coarsegrain
        self.window_size = window_size
        self.stride = stride
        if chrom_subset is None:
            self.chromosome_list = return_chrom_list(cooler_file)
        else:
            self.chromosome_list = chrom_subset

        self.chromosome_len_list = return_chrom_len_list(self.cooler_file,
                                                         self.chromosome_list,
                                                         self.resolution,
                                                         self.window_size,
                                                         self.stride)

        self.start_indices = np.cumsum(self.chromosome_len_list)
        self.len = self.start_indices[-1]  # Total number of windows
        self.start_indices = np.insert(self.start_indices, 0, 0)[:-1]

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.len

    def __getitem__(self, indx):
        # Get the chromosome, start, end from the index
        chromosome, start, end = find_genomic_position(indx,
                                                       self.start_indices,
                                                       self.chromosome_list,
                                                       self.window_size,
                                                       self.stride,
                                                       self.resolution,
                                                       self.cooler_file)

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

    def return_chrom_size_list(self) -> List[Tuple[str, int]]:
        """
        Returns a list of tuples containing the chromosome names and their corresponding lengths.
        Returns:
            list: A list of tuples where each tuple contains the chromosome name and its length.
        """
        return return_chrom_size_list(self.cooler_file, self.resolution)

    def return_chrom_positions(self, indx:int) -> List[Tuple[str, int, int]]:
        """
        Returns the chromosome and the start and end positions of the region corresponding to the
        given index.
        Args:
            indx (int): The index of the region.
        Returns:
            list: A list of tuples where each tuple contains the chromosome name and the start and
            end positions of the region.
        """
        # Get the chromosome, start, end from the index
        chromosome, start, end = find_genomic_position(indx,
                                                       self.start_indices,
                                                       self.chromosome_list,
                                                       self.window_size,
                                                       self.stride,
                                                       self.resolution,
                                                       self.cooler_file)

        chr_arr, start_arr, end_arr = return_patch_bin_range(chromosome,
                                                             start, end, self.window_size)

        index_out = np.arange(indx * self.window_size, (indx + 1) * self.window_size)

        return chr_arr, start_arr, end_arr, index_out
 