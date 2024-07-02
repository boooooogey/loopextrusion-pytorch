"""
Data loader implementation for DLEM data.
"""
import os
from typing import Tuple, List, Union
import pandas as pd
import numpy as np
import torch
from numpy.typing import ArrayLike
from pyfaidx import Fasta
from seqdataloader.batchproducers.coordbased.core import Coordinates
from .. import util
from jax.nn import one_hot
import jax.numpy as jnp

def return_coord_from_bed(file:str) -> List[Coordinates]:
    """Converts sampled region into list of Coordinates objects from seqdataloader module. 

    Args:
        chrom (str): name of the chromosome
        idx (int): start index of the sample
        sample_size (int): sample size
        bin_size (int): length of a bin in HiC contact matrix.

    Returns:
        List[Coordinates]: List of coordinates to be read from the genome.
    """
    bed = pd.read_csv(file, sep='\t', header=None)
    return [Coordinates(chr, start, stop) for i, (chr, start, stop, fold) in bed.iterrows()]

class DLEMDataset(torch.utils.data.Dataset):
    """DLEM dataset to interface data batching and reading.
    """
    def __init__(self, path:str, genome_path:str, sub_select:Union[ArrayLike, None]=None) -> None:
        """
        Args:
            path (str): Dataset folder path.
            genome_path (str): Path to genome fasta file.
        """
        self.json_path = path
        self.args = util.read_json(os.path.join(path, 'meta.json'))
        self.patches = np.memmap(os.path.join(path, 'contactmaps.dat'),
                                 dtype='float32',
                                 mode = 'r',
                                 shape=(self.args['SAMPLE_NUM'], self.args['PATCH_LEN']))

        self.tracks = np.memmap(os.path.join(path, 'features.dat'),
                                dtype='float32',
                                mode = 'r',
                                shape=(self.args['SAMPLE_NUM'],
                                       self.args['FEA_DIM'],
                                       self.args['PATCH_DIM'])
        )
        self.folds = pd.read_csv(os.path.join(path, "sequences.bed"),
                                 sep="\t",
                                 header=None).iloc[:,3].to_numpy()

        self.regions = return_coord_from_bed(os.path.join(path, "regions.bed"))
        self.fasta_path = genome_path
        self.sub_select = sub_select

    def __getitem__(self, index:int) -> Tuple[ArrayLike, ArrayLike]:
        lookup = {"A":0, "C":1, "G":2, "T":3, "N":4}
        co = self.regions[index]
        genome_object = Fasta(self.fasta_path,
                              read_ahead=co.end-co.start,
                              sequence_always_upper=True)
        tracks = np.array(self.tracks[index])
        if self.sub_select is not None:
            tracks = tracks[self.sub_select]
        tracks[np.isnan(tracks)] = 0
        seq = genome_object[co.chrom][co.start:co.end].seq
        nums = jnp.array([lookup[i] for i in seq])
        one_hot_emb = torch.tensor(np.array(one_hot(nums, 4)))
        return np.array(self.patches[index]), tracks, one_hot_emb.type(torch.float32).T

    def __len__(self)->int:
        return int(self.args["SAMPLE_NUM"])

    @property
    def stop_diag(self) -> int:
        """return stop diagonal"""
        return int(self.args["STOP_DIAG"])

    @property
    def start_diag(self) -> int:
        """return start diagonal"""
        return int(self.args["START_DIAG"])

    @property
    def data_folds(self) -> ArrayLike:
        """return folds"""
        return self.folds

    @property
    def patch_dim(self) -> ArrayLike:
        """return patch dimensions"""
        return int(self.args['PATCH_DIM'])

    @property
    def feature_dim(self) -> ArrayLike:
        """return feature dimensions"""
        if self.sub_select is None:
            return self.args['FEA_DIM']
        else:
            return len(self.sub_select)
