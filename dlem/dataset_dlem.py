"""
The dataset classes used in training DLEM models.
"""
import os
from typing import Union, List, Tuple
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch
from numpy.typing import ArrayLike
import pyBigWig
from dlem import util

def _standardize(x:ArrayLike, stats:Union[Tuple[float, float],None]=None) -> ArrayLike:
    if stats is None:
        return (x - x.mean()) / x.std()
    else:
        return (x - stats[0]) / stats[1]

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
        self.cell_lines = self.args["CELL_LINES"]

    @abstractmethod
    def __getitem__(self, index:int) -> ArrayLike:
        """Reading samples from  the dataset depending on the format"""

    def __len__(self)->int:
        return int(self.args["SAMPLE_NUM"])

    @property
    def cell_line_list(self) -> List[str]:
        """return the cell lines present in the dataset"""
        return self.cell_lines

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
    def __init__(self, path:str, resolution:int, select_cell_lines:Union[List[str],None]=None):
        """
        Args:
            path (str): path to the dataset where contactmaps are stored in memmap format.
            The dataset is expected to be separated into folds. The contact maps are stored in a
            vectorized form. The elements in the same diagonals are stored contiguously.
            resolution (int): resolution of the contact maps. Hopefully there are more than one
            option.
            select_cell_lines (Union[List[str],None]): select the cell lines to be used in the
            dataset. default is None.
        """
        super(ContactmapDataset, self).__init__(path)
        self.resolution = resolution

        self.sample_size = dict(zip(self.args["RES"],
                                    self.args["DIAGONALIZED_SIZE"]))[self.resolution]
        self.start_diag = self.args["START_DIAG"] // self.resolution
        self.stop_diag = self.args["STOP_DIAG"] // self.resolution
        self.contact_map_edge_length = self.args["PATCH_EDGE_SIZE"] // self.resolution

        if select_cell_lines is not None:
            check_e = all(cell_line in self.args["CELL_LINES"] for cell_line in select_cell_lines)
            check_z = len(select_cell_lines) != 0
            assert check_e, "Cell lines not found in the dataset"
            assert check_z, "Select at least one cell line"
            self.cell_lines = select_cell_lines
        else:
            self.cell_lines = self.args["CELL_LINES"]

        self.contactmaps = dict()
        for cell_line in self.cell_lines:
            for fold in self.fold_labels:
                self.contactmaps[f"{cell_line}_{fold}"] = np.memmap(
                    os.path.join(path,
                                 cell_line,
                                 f"res_{self.resolution}",
                                 f'contactmaps.{fold}.dat'),
                    dtype='float32',
                    mode = 'r',
                    shape=(self.fold_nums[fold],
                        self.sample_size)
                )

    def __getitem__(self, index:int) -> ArrayLike:
        fold = self.region_bed.iloc[index]["fold"]
        fold_index = self.region_bed.iloc[index]["fold_index"]
        if len(self.cell_lines) == 1:
            cell_line = self.cell_lines[0]
            contactmap = np.array(self.contactmaps[f"{cell_line}_{fold}"][fold_index])
            return contactmap
        else:
            contactmaps =[]
            for cell_line in self.cell_lines:
                contactmaps.append(np.array(self.contactmaps[f"{cell_line}_{fold}"][fold_index]))
            return contactmaps

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
    def __init__(self, path:str,
                       subselection:Union[ArrayLike, None]=None,
                       select_cell_lines:Union[List[str],None]=None):
        """
        Args:
            path (str): path to the dataset where tracks are stored in memmap format.
            The dataset is expected to be separated into folds. The tracks are stored in a
            vectorized form.
            track_name (str): name of the track to be read.
        """
        super(TrackDataset, self).__init__(path)
        if select_cell_lines is not None:
            check_e = all(cell_line in self.args["CELL_LINES"] for cell_line in select_cell_lines)
            check_z = len(select_cell_lines) != 0
            assert check_e, "Cell lines not found in the dataset"
            assert check_z, "Select at least one cell line"
            self.cell_lines = select_cell_lines
        else:
            self.cell_lines = self.args["CELL_LINES"]

        self.track_files = dict()
        for cell_line in self.cell_lines:
            self.track_files[cell_line] = os.listdir(os.path.join(path,
                                                                  cell_line,
                                                                  "tracks"))
        if subselection is not None:
            for cell_line in self.cell_lines:
                self.track_files[cell_line] = self.track_files[subselection]


        self.tracks = dict()
        self.track_stats = dict()

        for cell_line in self.cell_lines:
            tracks = []
            stats = []
            for track in self.track_files[cell_line]:
                tracks.append(pyBigWig.open(os.path.join(self.path,
                                                         cell_line,
                                                         "tracks",
                                                         track), 'r'))
                stats.append(util.get_stats_from_bw_chrom_separated(tracks[-1]))
            self.tracks[cell_line] = tracks
            self.track_stats[cell_line] = stats

    def __getitem__(self, index:int) -> ArrayLike:
        region = self.region_bed.iloc[index]
        start = region["start"]
        end = region["end"]
        if len(self.cell_lines) == 1:
            cell_line = self.cell_lines[0]
            tracks = np.empty((len(self.tracks[cell_line]), end - start), dtype=np.float32)
            for i, (track, stats) in enumerate(zip(self.tracks[cell_line],
                                                   self.track_stats[cell_line])):
                tmp = np.array(track.values(region["chr"], start, end))
                tmp = np.log1p(_standardize(tmp, stats[region["chr"]]))
                tracks[i] = tmp
            return tracks
        else:
            tracks_all = []
            for cell_line in self.cell_lines:
                tracks = np.empty((len(self.tracks[cell_line]), end - start), dtype=np.float32)
                for i, (track, stats) in enumerate(zip(self.tracks[cell_line],
                                                       self.track_stats[cell_line])):
                    tmp = np.array(track.values(region["chr"], start, end))
                    tmp = np.log1p(_standardize(tmp, stats[region["chr"]]))
                    tracks[i] = tmp
                tracks_all.append(tracks)
            return tracks_all

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
        if len(self.datasets) == 3:
            check_order = [cl1 == cl2 for cl1, cl2 in zip(self.datasets[1].cell_lines,
                                                        self.datasets[2].cell_lines)]
            if not all(check_order):
                raise ValueError("Cell lines must be in the same order "
                                "between contactmap and track datasets")
        self.args = self.datasets[0].args

    def __getitem__(self, index:int) -> ArrayLike:
        region = self.datasets[0].region_bed.iloc[index]
        if not all(region.equals(dataset.region_bed.iloc[index]) for dataset in self.datasets[1:]):
            raise ValueError("All datasets must have the same regions!")
        if len(self.datasets[1].cell_lines) == 1:
            return tuple(dataset[index] for dataset in self.datasets)
        else:
            names_tuple = tuple([self.datasets[1].cell_lines])
            return tuple(dataset[index] for dataset in self.datasets) + names_tuple

    def __len__(self)->int:
        return len(self.datasets[0])

    @property
    def cell_line_list(self) -> List[str]:
        """return the cell lines present in the dataset"""
        return self.datasets[1].cell_line_list

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
        cell_line = self.datasets[2].cell_lines[0]
        return len(self.datasets[2].track_name[cell_line])
