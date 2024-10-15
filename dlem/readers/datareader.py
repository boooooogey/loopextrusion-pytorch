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
    def __init__(self, path:str, subselection:Union[List[int], None] = None) -> None:
        """
        Args:
            path (str): Dataset folder path.
        """
        self.json_path = path
        self.args = util.read_json(os.path.join(path, 'meta.json'))
        if subselection is not None:
            self.subselection = subselection
        else:
            self.subselection = np.arange(self.args['FEA_DIM'])
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

    def __getitem__(self, index:int) -> Tuple[ArrayLike, ArrayLike]:
        tracks = np.array(self.tracks[index])
        tracks[np.isnan(tracks)] = 0
        return np.array(self.patches[index]), tracks[self.subselection]

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
        return len(self.subselection)#int(self.args['FEA_DIM'])
