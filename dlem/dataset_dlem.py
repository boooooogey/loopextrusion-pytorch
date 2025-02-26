from typing import Tuple, List
import os
import glob
import warnings
import numpy as np
import pandas as pd
import pyranges as pr
from torch.utils.data import Dataset
import cooler
from cooltools.lib.numutils import adaptive_coarsegrain, interp_nan
from numpy.typing import ArrayLike
import pyBigWig

def flip_diag_row(mat:ArrayLike) -> ArrayLike:
    """Swap row and diagonal elements of a given matrix.

    Args:
        mat (ArrayLike): given matrix.

    Returns:
        ArrayLike: row diagonal swapped matrix. 
    """
    n = mat.shape[0]
    ii = np.arange(n)
    iy = ii.reshape(1,-1) * np.ones(n).reshape(-1,1)
    ix = (ii[::-1].reshape(-1,1) - ii[::-1].reshape(1,-1)) % n
    return mat[ix.astype(int), iy.astype(int)]

def diagonal_normalize(mat:ArrayLike) -> ArrayLike:
    """Center each diagonal mean at 0.

    Args:
        mat (ArrayLike): input patch.

    Returns:
        ArrayLike: matrix with 0 centered diagonals.
    """
    mat = flip_diag_row(mat)
    centers = np.mean(mat, where=np.triu(np.ones_like(mat, dtype=bool)), axis=1)
    mat = flip_diag_row(mat - centers.reshape(-1, 1))
    return np.triu(mat) + np.triu(mat, 1).T

def return_diagonal_ordered_triu_indices(n:int) -> Tuple[ArrayLike, ArrayLike]:
    """Return the indices of the upper triangular part of the matrix in the order of diagonals.

    Args:
        n (int): size of the matrix. The matrix is assumed to be square.

    Returns:
        Tuple[ArrayLike, ArrayLike]: indices of the upper triangular part of the matrix in the order
        of diagonals.
    """
    first = np.concatenate([np.arange(n-i) for i in range(n)])
    second = np.concatenate([np.arange(i, n) for i in range(n)])
    return first, second

def read_contact_from_cooler(file:str,
                             res:int,
                             chrom:str,
                             start:int,
                             end:int) -> ArrayLike:
    """Read contact map from cooler files for the region provided.

    Args:
        file (str): cooler path.
        res (int): resolution of the data.
        chrom (str): chromosome for the region.
        start (int): start genomic locus.
        end (int): end genomic locus.

    Returns:
        ArrayLike: contact map for the patch.
    """
    region = (chrom, start, end)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clr =  cooler.Cooler(file + f'::resolutions/{res}')
        patch = adaptive_coarsegrain(clr.matrix(balance=True).fetch(region),
                                    clr.matrix(balance=False).fetch(region),
                                    cutoff=3, max_levels=8)
        patch = interp_nan(patch)
        patch = diagonal_normalize(np.log(patch))
    return patch.astype(np.float32)

def read_chromosome_lengths(file:str, res:int) -> dict:
    """Return chromosome lengths from the cooler file.

    Args:
        file (str): mcool file path.
        res (int): resolution of the contact map.

    Returns:
        dict: Dictionary of chromosome lengths.
    """
    clr = cooler.Cooler(file+f'::resolutions/{res}')
    return dict(clr.chromsizes)

def return_patch_regions(chrom_sizes:dict,
                         patch_size:int,
                         resolution:int,
                         overlap:int,
                         offset:int) -> pd.DataFrame:
    """Return the regions for given patch size, overlap and offset.

    Args:
        chrom_sizes (dict): Chromosome sizes.
        patch_size (int): The size of the patch.
        resolution (int): resolution of the contact map from mcool file.
        overlap (int): how much overlap between patches.
        offset (int): offset the patches. To generate different views of the data.

    Returns:
        pd.DataFrame: The regions for the patches as (chrom, start, end).
    """

    starts = []
    ends = []
    chroms = []
    overlap = overlap * resolution
    patch_size = patch_size * resolution
    offset = offset * resolution
    for chrom in chrom_sizes:
        binned_chrom_size  = chrom_sizes[chrom] // 2_000  * 2_000
        start = np.clip(np.array(range(offset,
                                       binned_chrom_size - patch_size,
                                       patch_size - overlap)), 0, binned_chrom_size)
        end = np.clip(np.array(range(offset + patch_size,
                                     binned_chrom_size,
                                     patch_size - overlap)), 0, binned_chrom_size)
        starts.append(start)
        ends.append(end)
        chroms += [chrom] * len(start)

    starts = np.concatenate(starts)
    ends = np.concatenate(ends)
    return pd.DataFrame({'chrom': chroms, 'start': starts, 'end': ends})

def convert_diagonals_to_mat(diagonals:ArrayLike, n:int) -> ArrayLike:
    """Converts diagonals in vector from to matrix form.

    Args:
        diagonals (ArrayLike): Vectorized diagonals. [d_1... d_n].
        n (int): size of the matrix.

    Returns:
        ArrayLike: Matrix form of the diagonals.
    """
    mat = np.zeros((n, n), dtype=np.float32)
    first, second = return_diagonal_ordered_triu_indices(n)
    mat[first, second] = diagonals
    mat[second, first] = diagonals
    return mat

def filter_regions(regions:pd.DataFrame, zero_regions:pd.DataFrame) -> pd.DataFrame:
    """Filter out the regions overlapping with the zero regions.

    Args:
        regions (pd.DataFrame): patch regions to be filtered.
        zero_regions (pd.DataFrame): Regions of zero coverage from the mcool file.

    Returns:
        pd.DataFrame: Filtered regions.
    """
    regions = regions.copy()
    regions.columns = ['Chromosome', 'Start', 'End']
    regions["ID"] = regions.index
    regions = pr.PyRanges(regions)

    zero_regions = zero_regions.copy()
    zero_regions.columns = ['Chromosome', 'Start', 'End']
    zero_regions = pr.PyRanges(zero_regions)

    overlap = regions.join(zero_regions)
    return regions[~regions.df["ID"].isin(overlap.df["ID"].unique())].df.drop(columns="ID")

def return_bigwig_region(file:str, chrom:str, start:int, end:int) -> ArrayLike:
    """Return the values from the bigwig file for the given region.

    Args:
        file (str): bigwig file path.
        chrom (str): chromosome.
        start (int): start genomic locus.
        end (int): end genomic locus.

    Returns:
        ArrayLike: epigenetic values.
    """
    with pyBigWig.open(file) as bw:
        val = np.array(bw.values(chrom, start, end), dtype=np.float32)
        val = (val - np.nanmean(val)) / np.nanstd(val)
        val[np.isnan(val)] = 0
    return val

def return_bigwig_regions(files:List[str], chrom:str, start:int, end:int) -> ArrayLike:
    """Return a matrix of epigenetic values for the given regions. Each row corresponds to a
    different file.

    Args:
        files (List[str]): List of bigwig files.
        chrom (str): chromosome.
        start (int): start genomic locus.
        end (int): end genomic locus.

    Returns:
        ArrayLike: Matrix of epigenetic values. 
    """
    return np.stack([return_bigwig_region(file, chrom, start, end) for file in files], axis=0)

class DlemData(Dataset):
    """loading patches from mcool file for given resolution"""
    def __init__(self,
                 path,
                 resolution,
                 patch_size,
                 chrom_selection=None,
                 chrom_filter=None,
                 subselection=None,
                 overlap=0,
                 offset=0):

        self.path = path

        self.cell_types = sorted(os.listdir(os.path.join(path, "cell_types")))
        if subselection is not None:
            self.cell_types = subselection
        self.avoid_regions_file = os.path.join(path, "avoid.tsv")

        self.resolution = resolution
        self.patch_size = patch_size
        self.start = 0
        self.stop = patch_size
        self.overlap = overlap
        self.offset = offset

        self.chromosome_lengths = read_chromosome_lengths(os.path.join(path,
                                                                       "cell_types",
                                                                       self.cell_types[0],
                                                                       "contactmaps.mcool"),
                                                                       resolution)

        if chrom_selection is not None:
            self.chromosome_lengths = {chrom: self.chromosome_lengths[chrom]
                                       for chrom in chrom_selection}
        if chrom_filter is not None:
            self.chromosome_lengths = {chrom: size
                                       for chrom, size in self.chromosome_lengths.items()
                                       if chrom not in chrom_filter}

        self.avoid_regions = pd.read_csv(self.avoid_regions_file, sep='\t')

        self.patches = return_patch_regions(self.chromosome_lengths,
                                            patch_size,
                                            resolution,
                                            overlap,
                                            offset)

        self.patches = filter_regions(self.patches, self.avoid_regions)
        self.diagonal_ind = return_diagonal_ordered_triu_indices(patch_size)

        self.bigwig_files = dict()

        for cell_type in self.cell_types:
            self.bigwig_files[cell_type] = sorted([file
                                        for ext in ['*.bigwig', '*.bigWig', '*.bw']
                                        for file in glob.glob(os.path.join(path,
                                                                           "cell_types",
                                                                           cell_type,
                                                                           "tracks",
                                                                           ext))])
        # Ensure all lists in the dictionary have the same length
        track_lengths = [len(files) for files in self.bigwig_files.values()]
        if not all(track_lengths[0] == np.array(track_lengths)):
            raise ValueError("All cell types must have the same number of tracks!")
        self.track_dim = track_lengths[0]

        self.seq_features_files = sorted([file
                                        for ext in ['*.bigwig', '*.bigWig', '*.bw']
                                        for file in glob.glob(os.path.join(path,
                                                                           'seq_features',
                                                                           ext))])

    def rearrange_patches(self, overlap:int, offset:int):
        """Given new overlap and offset, rearrange the patches.

        Args:
            overlap (int): overlap between patches.
            offset (int): offset between patches.
        """
        self.overlap = overlap
        self.offset = offset
        self.patches = return_patch_regions(self.chromosome_lengths,
                                            self.patch_size,
                                            self.resolution,
                                            overlap,
                                            offset)

        self.patches = filter_regions(self.patches, self.avoid_regions)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        region = tuple(self.patches.iloc[idx])
        seq_features = return_bigwig_regions(self.seq_features_files, *region)
        if len(self.cell_types) == 1:
            patch = read_contact_from_cooler(os.path.join(self.path,
                                                          "cell_types",
                                                          self.cell_types[0],
                                                          "contactmaps.mcool"),
                                            self.resolution,
                                            *region)
            tracks = return_bigwig_regions(self.bigwig_files[self.cell_types[0]], *region)
            return seq_features, patch[self.diagonal_ind], tracks
        else:
            patches = [read_contact_from_cooler(os.path.join(self.path,
                                                             "cell_types",
                                                             cell_type,
                                                             "contactmaps.mcool"),
                                               self.resolution,
                                               *region)[self.diagonal_ind]
                                               for cell_type in self.cell_types]
            tracks = [return_bigwig_regions(self.bigwig_files[cell_type], *region)
                      for cell_type in self.cell_types]
            return seq_features, patches, tracks, self.cell_types
