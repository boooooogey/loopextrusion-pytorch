"""
This module contains utility functions for reading and processing contact maps from cooler files.
"""
from typing import List, Tuple
import cooler
import numpy as np
from cooltools.lib.numutils import adaptive_coarsegrain, interp_nan
from numpy.typing import ArrayLike
import numpy as np

def check_chromosome_length(c: cooler.Cooler, chromosome: str, start: int, end: int) -> None:
    """Check if the given chromosome and positions are valid.

    Args:
        c (cooler.Cooler): The cooler object.
        chromosome (str): The chromosome of interest.
        start (int): The start position of the region.
        end (int): The end position of the region.

    Raises:
        ValueError: If the given chromosome is invalid.
        ValueError: If the given start or end position is invalid.
    """
    # Check if the given chromosome is valid
    if chromosome not in c.chromnames:
        raise ValueError("Invalid chromosome")

    # Get the length of the given chromosome
    chromosome_length = c.chromsizes[chromosome]

    # Check if the given start and end positions are within the chromosome's length
    if start < 0 or start > chromosome_length:
        raise ValueError("Invalid start or end position")

def return_chromosome_length(cooler_file:str, resolution:int, chromosome: str) -> int:
    """Return the length of the given chromosome.

    Args:
        cooler_file (str): The path to the cooler file.
        resolution (int): The resolution of the contact map.
        chromosome (str): The chromosome of interest.

    Returns:
        int: The length of the chromosome.
    """
    c = cooler.Cooler(cooler_file + f"::resolutions/{resolution}")
    # Get the length of the given chromosome
    chromosome_length = c.chromsizes[chromosome]

    return chromosome_length

def get_output_length(input_length: int, kernel_size: int, stride: int) -> int:
    """Calculate the output length given an input length, kernel size, and stride.

    Args:
        input_length (int): The length of the input.
        kernel_size (int): The size of the kernel.
        stride (int): The stride value.

    Returns:
        int: The output length.
    """
    output_length = (input_length - kernel_size) // stride + 1
    return output_length

def get_input_indices(output_index: int, kernel_size: int, stride: int) -> Tuple[int, int]:
    """Get the start and end indices in the input file given the index in the output of convolution.

    Args:
        output_index (int): The index in the output of convolution.
        kernel_size (int): The size of the kernel.
        stride (int): The stride value.

    Returns:
        Tuple[int, int]: The start and end indices in the input file.
    """
    start_index = output_index * stride
    end_index = start_index + kernel_size
    return start_index, end_index

def return_chrom_len_list(cooler_file:str,
                          chrom_list:List[str],
                          resolution:int,
                          window_size:int,
                          stride:int) -> List[int]:
    """Return the list of number of windows for all the given chromosomes.

    Args:
        cooler_file (str): The path to the cooler file.
        chrom_list (List[str]): The list of chromosome names.
        resolution (int): The resolution of the contact map.
        window_size (int): The size of the window.
        stride (int): The stride value.

    Returns:
        List[int]: The list of window numbers for each chromosome.
    """
    c = cooler.Cooler(cooler_file + f"::resolutions/{resolution}")
    chrom_window_list = []
    for chrom in chrom_list:
        chrom_window_list.append(get_output_length(len(c.bins().fetch(chrom)), window_size, stride))
    return chrom_window_list

def return_chrom_size_list(cooler_file:str,
                           resolution:int) -> List[Tuple[str, int]]:
    """Return the list of sizes for all the chromosomes.

    Args:
        cooler_file (str): The path to the cooler file.
        resolution (int): The resolution of the contact map.
        window_size (int): The size of the window.
        stride (int): The stride value.

    Returns:
        list: The list of window numbers for each chromosome.
    """
    c = cooler.Cooler(cooler_file + f"::resolutions/{resolution}")
    chrom_size_dict = dict(c.chromsizes)
    return [(k, int(chrom_size_dict[k])) for k in chrom_size_dict]


def get_contact_map(
    cooler_file: str,
    resolution: int,
    chromosome: str,
    start: int,
    end: int,
    do_adaptive_coarsegrain: bool = True,
) -> ArrayLike:
    """Get the contact map for a specified region.

    Args:
        cooler_file (str): The path to the cooler file.
        resolution (int): The resolution of the contact map.
        chromosome (str): The chromosome of interest.
        start (int): The start position of the region.
        end (int): The end position of the region.
        do_adaptive_coarsegrain (bool, optional): Whether to apply adaptive coarse-graining to the
        contact map. Defaults to True.

    Returns:
        ArrayLike: The contact map as a numpy array.
    """

    # Load the cooler file
    c = cooler.Cooler(cooler_file + f"::resolutions/{resolution}")

    check_chromosome_length(c, chromosome, start, end)

    # Define the region of interest
    region = (chromosome, start, end)

    # Get the contact matrix for the specified region
    contact_map = c.matrix(balance=True).fetch(region)

    if do_adaptive_coarsegrain:
        # Apply adaptive coarse-graining to the contact map
        # Mute numpy warnings
        np.seterr(all='ignore')

        # Apply adaptive coarse-graining to the contact map
        contact_map = adaptive_coarsegrain(contact_map,
                                           c.matrix(balance=False).fetch(region),
                                           cutoff=3, max_levels=8)

        # Reset numpy error handling behavior
        np.seterr(all='warn')

    sum_zero = np.sum(np.nansum(contact_map, axis=0) == 0)/contact_map.shape[0]

    if np.isnan(contact_map).any():
        # Interpolate NaN values
        contact_map = interp_nan(contact_map)

    return contact_map, sum_zero

def return_chrom_list(cooler_file: str) -> list:
    """Return the list of chromosome names from a cooler file.

    Args:
        cooler_file (str): The path to the cooler file.

    Returns:
        list: The list of chromosome names.
    """
    # Load the cooler file
    res = cooler.fileops.list_coolers(cooler_file)[0]
    c = cooler.Cooler(cooler_file + f"::{res}")

    # Get the chromosome names
    chromosome_names = c.chromnames

    return chromosome_names

def get_total_bins(cooler_file: str, resolution: int) -> int:
    """Get the total number of bins for the given resolution from the cooler file.

    Args:
        cooler_file (str): The path to the cooler file.
        resolution (int): The resolution of the contact map.

    Returns:
        int: The total number of bins.
    """
    # Load the cooler file
    c = cooler.Cooler(cooler_file + f"::resolutions/{resolution}")

    # Get the total number of bins
    total_bins = c.info["nbins"]

    return total_bins

def find_chromosome(indx:int, start_indices:List[int]) -> int:
    """Find the chromosome index for the given bin index.

    Args:
        indx (int): The bin index.
        start_indices (List[int]): The list of start indices of each chromosome.

    Returns:
        int: The chromosome index.
    """
    return np.searchsorted(start_indices, indx, side='right') - 1

def return_region_from_index(index:int,
                             start_indices:List[int],
                             chrom_names:List[str],
                             window_size:int,
                             stride:int) -> Tuple[str, int, int]:
    """Return the region (chromosome, start, end) corresponding to the given index.

    Args:
        index (int): The index of the region.
        start_indices (List[int]): The list of start indices of each chromosome.
        chrom_names (List[str]): The list of chromosome names.
        window_size (int): The size of the window.
        stride (int): The stride value.

    Returns:
        Tuple[str, int, int]: The region (chromosome, start, end).
    """
    chrom_indx = find_chromosome(index, start_indices)
    chrom = chrom_names[chrom_indx]
    window_index = index - start_indices[chrom_indx]
    start, end = get_input_indices(window_index, window_size, stride)
    return chrom, start, end

def return_patch_bin_range(chromosome:str,
                           start:int,
                           end:int,
                           window_size:int) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Return the range of patch bins given the start and end positions and window size.

    Args:
        chromosome (str): The chromosome of interest.
        start (int): The start position of the region.
        end (int): The end position of the region.
        window_size (int): The size of the window.

    Returns:
        Tuple[ArrayLike, ArrayLike, ArrayLike]: The range of patch bins as a tuple of start and end arrays.
    """
    patch_range = np.arange(start, end+1, (end - start) / window_size, dtype=int)
    chromosome_range = np.array([chromosome] * (len(patch_range)-1))
    return (chromosome_range, patch_range[:-1], patch_range[1:]) # start, end
