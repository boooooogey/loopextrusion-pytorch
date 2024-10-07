"""
Data loader implementation for DLEM data.
"""
import os
from typing import Tuple, List, Union
import pandas as pd
import numpy as np
import torch
from numpy.typing import ArrayLike
from .. import util

class DLEMDataset(torch.utils.data.Dataset):
    """DLEM dataset to interface data batching and reading.
    """
    def __init__(self, path:str, sub_select:Union[ArrayLike, None]=None) -> None:
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
        self.seqs = np.memmap(
            os.path.join(path, 'sequences.dat'),
            dtype='float32',
            mode = 'r',
            shape=(self.args['SAMPLE_NUM'],
                   4,
                   self.args['RES'] * self.args['PATCH_DIM'] + 2 * self.args["SEQUENCE_OFFSET"])
        )
        self.folds = pd.read_csv(os.path.join(path, "regions.bed"),
                                 sep="\t",
                                 header=None).iloc[:,3].to_numpy()

        self.sub_select = sub_select

    def __getitem__(self, index:int) -> Tuple[ArrayLike, ArrayLike]:
        tracks = np.array(self.tracks[index])
        if self.sub_select is not None:
            tracks = tracks[self.sub_select]
        tracks[np.isnan(tracks)] = 0
        patch = torch.from_numpy(np.array(self.patches[index]))
        tracks = torch.from_numpy(np.array(tracks))
        offset = self.args["SEQUENCE_OFFSET"]
        one_hot_emb = torch.from_numpy(np.array(self.seqs[index]))[:, offset:-offset]
        return patch, tracks, one_hot_emb

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
