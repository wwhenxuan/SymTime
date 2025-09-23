# -*- coding: utf-8 -*-
"""
Created on 2024/9/30 17:06
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import os
from os import path
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, Tuple
import warnings

warnings.filterwarnings("ignore")


class TSDataset(Dataset):
    """Modified dataset object for the pre-training of SymTime"""

    def __init__(self, time: torch.Tensor, time_mask: torch.Tensor, sym_ids: torch.Tensor, sym_mask: torch.Tensor) -> None:
        self.time, self.time_mask = time, time_mask
        self.sym_ids, self.sym_mask = sym_ids, sym_mask

    def __len__(self) -> int:
        return self.time.size(0)

    def __getitem__(self, index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time, time_mask = self.time[index], self.time_mask[index]
        sym_ids, sym_mask = self.sym_ids[index], self.sym_mask[index]
        return time, time_mask, sym_ids, sym_mask


class PreTrainDataLoader(object):
    """List of DataLoaders for loading the pre-trained dataset"""

    def __init__(self, args: Any) -> None:
        # The file path for the pre-training dataset
        self.data_path = args.data_path
        # The files number for pre-training
        self.num_data = len(os.listdir(self.data_path))

        # Number of data points read per iteration
        self.number = args.number
        self.list = list(range(0, self.num_data, self.number))
        self.pointer = 0

        # Parameters related to creating a DataLoader object
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.num_workers = args.num_workers

    def __len__(self) -> int:
        """How many batches of data need to be loaded in one epoch?"""
        return len(self.list)

    def load_data(self) -> Dict:
        """Methods for loading data"""
        data_dict = dict(time=[], time_mask=[], sym_ids=[], sym_mask=[])
        index = self.list[self.pointer]

        # Move the dataset pointer backward.
        self.pointer = (self.pointer + 1) % len(self.list)
        for file in os.listdir(self.data_path)[index : index + self.number]:
            file_path = path.join(self.data_path, file)
            data = torch.load(file_path, weights_only=False)
            for key in data_dict.keys():
                data_dict[key].append(data[key])

        # Concatenate the datasets.
        for key, value in data_dict.items():
            data_dict[key] = torch.concat(value, dim=0)
        return data_dict

    def get_dataloader(
        self, batch_size: Optional[int] = None, shuffle: Optional[bool] = None
    ) -> DataLoader:
        """How to obtain the DataLoader object used for pre-training"""
        data_dict = self.load_data()
        dataset = TSDataset(
            time=data_dict["time"],
            time_mask=data_dict["time_mask"],
            sym_ids=data_dict["sym_ids"],
            sym_mask=data_dict["sym_mask"],
        )

        batch_size = self.batch_size if batch_size is None else batch_size
        shuffle = self.shuffle if shuffle is None else shuffle

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
