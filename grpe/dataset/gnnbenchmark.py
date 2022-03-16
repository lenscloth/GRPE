from typing import Optional, Callable, List

import os
import os.path as osp
import pickle
import logging

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
from torch_geometric.utils import remove_self_loops

from torch_geometric.datasets import GNNBenchmarkDataset as _GNNBenchmarkDataset
from ogb.graphproppred import Evaluator as _Evaluator


class GNNBenchmarkDataset(_GNNBenchmarkDataset):
    def __init__(self, 
                 root: str,
                 name: str,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 concat_position: bool = False):
        super().__init__(root, name, split=split, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)
        self.concat_position = concat_position
        self.split = split
        self.num_examples = super().__len__()

    def __len__(self) -> int:
        return self.num_examples


    def process(self):
        if self.name == "CSL":
            data_list = self.process_CSL()
            torch.save(self.collate(data_list), self.processed_paths[0])
        else:
            inputs = torch.load(self.raw_paths[0])
            for i in range(len(inputs)):
                # rounding edge_attr to int type
                for data_dict in inputs[i]:
                    data_dict["edge_attr"] = (data_dict["edge_attr"] * 10).long()

                data_list = [Data(**data_dict) for data_dict in inputs[i]]

                if self.pre_filter is not None:
                    data_list = [d for d in data_list if self.pre_filter(d)]

                if self.pre_transform is not None:
                    data_list = [self.pre_transform(d) for d in data_list]
                torch.save(self.collate(data_list), self.processed_paths[i])
    
    def get(self, idx):
        data = super().get(idx)
        if self.concat_position:
            data.x = torch.cat([data.x, data.pos],dim=1)
        return data

    def download(self):
        super().download()

    def delete_data(self):
        self.data = None
        self.slices = None

    def load_data(self):
        split = self.split
        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        else:
            raise ValueError(
                f"Split '{split}' found, but expected either "
                f"'train', 'val', or 'test'"
            )

        self.data, self.slices = torch.load(path)

    def __getitem__(self, idx):
        if self.data is None:
            self.load_data()

        return super().__getitem__(idx)


class GNNBenchmarkEvaluator(_Evaluator):
    def __init__(self, name, num_tasks=10):
        super().__init__("ogbg-molbace")
        # override attributes
        self.name = name
        self.num_tasks = num_tasks
        self.eval_metric = "acc"
