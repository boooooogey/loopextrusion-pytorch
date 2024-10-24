"""
The dataset classes used in training DLEM models.
"""
import os
from typing import Union
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch
from numpy.typing import ArrayLike
import pyBigWig
from dlem import util

class DLEMDataset(torch.utils.data.Dataset, ABC):
    """This class is the blueprint for reading contactmaps, one hot embedded sequences and tracks.
    """
    def __init__(self, path:str):
        """
        Args:
            path (str): path to the directory where the contactmaps, sequences and tracks are kept.
        """
        self.path = path
        self.args = util.read_json(os.path.join(path, 'meta.json'))
        self.region_bed = pd.read_csv(os.path.join(path, "regions.bed"),
                                 sep="\t",
                                 header=None)
        self.region_bed.columns = ["chr", "start", "end", "fold", "fold_index"]

        self.folds = self.region_bed.iloc[:,3].to_numpy()
        self.fold_labels = np.unique(self.region_bed.iloc[:,3].to_numpy()) 
        self.fold_nums = dict(self.region_bed.value_counts("fold"))

    @abstractmethod
    def __getitem__(self, index:int) -> ArrayLike:
        """Reading samples from  the dataset depending on the format"""

    def __len__(self)->int:
        return int(self.args["SAMPLE_NUM"])

    @property
    def data_folds(self) -> ArrayLike:
        """return folds"""
        return self.folds

    @property
    def genomic_region_length(self) -> ArrayLike:
        """return patch dimensions"""
        return int(self.args['PATCH_EDGE_SIZE'])

class SeqDataset(DLEMDataset):
    """Reads the squences from memmap. The sequences are stored as one-hot encoded.
    """
    def __init__(self, path:str, shift:int=0):
        """
        Args:
            path (str): path to the dataset where sequences are stored in memmap format. The dataset
            is expected to be separated into folds.
            shift (int): shift the sequence by a random number between 0 and shift.
        """
        super(SeqDataset, self).__init__(path)

        self.seqs = dict()
        for fold in self.fold_labels:
            self.seqs[fold] = np.memmap(
                os.path.join(path, f'sequences.{fold}.dat'),
                dtype='int32',
                mode = 'r',
                shape=(self.fold_nums[fold],
                    self.args['PATCH_EDGE_SIZE'] + 2 * self.args["SEQUENCE_OFFSET"])
            )

        if shift >= 0 and shift > self.args['SEQUENCE_OFFSET']:
            raise ValueError("Shift must be between 0 and sequence offset")
        self.shift = shift


    def __getitem__(self, index:int) -> ArrayLike:
        fold = self.region_bed.iloc[index]["fold"]
        fold_index = self.region_bed.iloc[index]["fold_index"]
        offset = self.args["SEQUENCE_OFFSET"]
        shift = np.random.choice([-1,1]) * int(np.random.rand() * self.shift)
        row_index = np.array(self.seqs[fold][fold_index])[(offset+shift):(-offset+shift)]
        column_index = np.arange(len(row_index))
        mask = row_index != 4
        one_hot_emb = np.zeros((4, len(row_index)), dtype=np.float32)
        one_hot_emb[row_index[mask], column_index[mask]] = 1
        one_hot_emb = torch.from_numpy(one_hot_emb)
        return one_hot_emb

class ContactmapDataset(DLEMDataset):
    """Reads the squences from memmap. The sequences are stored as one-hot encoded.
    """
    def __init__(self, path:str, resolution:int):
        """
        Args:
            path (str): path to the dataset where contactmaps are stored in memmap format.
            The dataset is expected to be separated into folds. The contact maps are stored in a
            vectorized form. The elements in the same diagonals are stored contiguously.
            resolution (int): resolution of the contact maps. Hopefully there are more than one
            option.
        """
        super(ContactmapDataset, self).__init__(path)
        self.resolution = resolution

        self.sample_size = dict(zip(self.args["RES"],
                                    self.args["DIAGONALIZED_SIZE"]))[self.resolution]
        self.start_diag = self.args["START_DIAG"] // self.resolution
        self.stop_diag = self.args["STOP_DIAG"] // self.resolution
        self.contact_map_edge_length = self.args["PATCH_EDGE_SIZE"] // self.resolution


        self.contactmaps = dict()
        for fold in self.fold_labels:
            self.contactmaps[fold] = np.memmap(
                os.path.join(path, f"res_{self.resolution}", f'contactmaps.{fold}.dat'),
                dtype='float32',
                mode = 'r',
                shape=(self.fold_nums[fold],
                       self.sample_size)
            )

    def __getitem__(self, index:int) -> ArrayLike:
        fold = self.region_bed.iloc[index]["fold"]
        fold_index = self.region_bed.iloc[index]["fold_index"]
        contactmap = np.array(self.contactmaps[fold][fold_index])
        return contactmap

    @property
    def size(self) -> int:
        """return patch length in vectorized form."""
        return int(self.sample_size)

    @property
    def start(self) -> int:
        """At what diagonal to start"""
        return self.start_diag

    @property
    def stop(self) -> int:
        """At what diagonal to stop"""
        return self.stop_diag

    @property
    def patch_edge(self) -> int:
        """Edge length of the contactmap at the given resolution."""
        return self.contact_map_edge_length

class TrackDataset(DLEMDataset):
    """Reads the squences from memmap. The sequences are stored as one-hot encoded.
    """
    def __init__(self, path:str, subselection:Union[ArrayLike, None]=None):
        """
        Args:
            path (str): path to the dataset where tracks are stored in memmap format.
            The dataset is expected to be separated into folds. The tracks are stored in a
            vectorized form.
            track_name (str): name of the track to be read.
        """
        super(TrackDataset, self).__init__(path)
        self.track_dir = os.path.join(path, "tracks")
        self.track_files = os.listdir(self.track_dir)
        if subselection is not None:
            self.track_files = self.track_files[subselection]

        self.tracks = []
        for track in self.track_files:
            self.tracks.append(pyBigWig.open(os.path.join(self.track_dir, track), 'r'))

    def __getitem__(self, index:int) -> ArrayLike:
        region = self.region_bed.iloc[index]
        start = region["start"]
        end = region["end"]
        tracks = np.empty((len(self.tracks), end - start), dtype=np.float32)
        for i, track in enumerate(self.tracks):
            tracks[i] = np.array(track.values(region["chr"], start, end))
        return tracks

    @property
    def track_name(self) -> str:
        """return track name"""
        return self.track_files

class CombinedDataset(torch.utils.data.Dataset):
    """Reads the squences from memmap. The sequences are stored as one-hot encoded.
    """
    def __init__(self, seq_dataset:torch.utils.data.Dataset,
                       contactmap_dataset:torch.utils.data.Dataset,
                       track_dataset:Union[torch.utils.data.Dataset, None]=None):
        """
        Args:
            path (str): path to the dataset where contactmaps are stored in memmap format.
            The dataset is expected to be separated into folds. The contact maps are stored in a
            vectorized form. The elements in the same diagonals are stored contiguously.
            resolution (int): resolution of the contact maps. Hopefully there are more than one
            option.
        """
        if track_dataset is not None:
            self.datasets = [seq_dataset, contactmap_dataset, track_dataset]
        else:
            self.datasets = [seq_dataset, contactmap_dataset]
        if not all([dataset.path == self.datasets[0].path for dataset in self.datasets[1:]]):
            raise ValueError("All datasets must have the same path")
        self.args = self.datasets[0].args

    def __getitem__(self, index:int) -> ArrayLike:
        region = self.datasets[0].region_bed.iloc[index]
        if not all(region.equals(dataset.region_bed.iloc[index]) for dataset in self.datasets[1:]):
            raise ValueError("All datasets must have the same regions!")
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self)->int:
        return len(self.datasets[0])

    @property
    def data_folds(self) -> ArrayLike:
        """return folds"""
        return self.datasets[0].data_folds

    @property
    def patch_dim(self) -> int:
        """Patch dimension"""
        return self.datasets[1].patch_edge

    @property
    def start(self) -> ArrayLike:
        """At what diagonal to start"""
        return self.datasets[1].start

    @property
    def stop(self) -> ArrayLike:
        """At what diagonal to stop"""
        return self.datasets[1].stop

    @property
    def track_dim(self) -> int:
        """Number of tracks"""
        if len(self.datasets) < 3:
            return 0
        return len(self.datasets[2].track_name)
