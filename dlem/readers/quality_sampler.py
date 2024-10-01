import torch
import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from typing import Optional, Sized, Iterator
import numpy as np

class MyRandomSampler(RandomSampler):
    """Sample elements randomly. 
    Not everything from RandomSampler is implemented.

    Args:
        data_source (Dataset): dataset to sample from
        forbidden  (Optional[list]): list of forbidden numbers
    """
    data_source: Sized
    quality_metric: float
    quality_metric_index: int
    sample_indices: range 

    def __init__(self, data_source: Sized, quality_metric: float, quality_metric_index: int) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.quality_metric = quality_metric
        self.quality_metric_index = quality_metric_index
        self.sample_indices = range(len(data_source))

    def __iter__(self) -> Iterator[int]:
        for i in self.sample_indices:
            if self.data_source[i][self.quality_metric_index] > self.quality_metric:
                yield self.data_source[i]