# -*- coding: utf-8 -*-
"""
Created on 2024/9/30 17:06
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
加载数据集的接口
"""
import os
from os import path
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict
import warnings

warnings.filterwarnings("ignore")


class TSDataset(Dataset):
    """改写的用于预训练的数据集对象"""

    def __init__(self, time, time_mask, sym_ids, sym_mask):
        self.time, self.time_mask = time, time_mask
        self.sym_ids, self.sym_mask = sym_ids, sym_mask

    def __len__(self):
        return self.time.size(0)

    def __getitem__(self, index):
        time, time_mask = self.time[index], self.time_mask[index]
        sym_ids, sym_mask = self.sym_ids[index], self.sym_mask[index]
        return time, time_mask, sym_ids, sym_mask


class PreTrainDataLoader(object):
    """加载预训练数据集的DataLoader列表"""

    def __init__(self, args) -> None:
        self.data_path = args.data_path
        self.num_data = len(os.listdir(self.data_path))
        # 按照轮次读取数据的数目
        self.number = args.number

        self.list = list(range(0, self.num_data, self.number))
        self.pointer = 0

        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.num_workers = args.num_workers

    def __len__(self):
        """在一个Epoch中需要加载多少个批次的数据"""
        return len(self.list)

    def load_data(self) -> Dict:
        """加载数据的方法"""
        data_dict = dict(time=[], time_mask=[], sym_ids=[], sym_mask=[])
        index = self.list[self.pointer]
        # 将数据集指针向后移动
        self.pointer = (self.pointer + 1) % len(self.list)
        for file in os.listdir(self.data_path)[index : index + self.number]:
            file_path = path.join(self.data_path, file)
            data = torch.load(file_path, weights_only=False)
            for key in data_dict.keys():
                data_dict[key].append(data[key])
        # 对数据集进行concat
        for key, value in data_dict.items():
            data_dict[key] = torch.concat(value, dim=0)
        return data_dict

    def get_dataloader(
        self, batch_size: Optional[int] = None, shuffle: Optional[bool] = None
    ) -> DataLoader:
        """获得预训练使用的DataLoader对象的方法"""
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
