from ogb.graphproppred import PygGraphPropPredDataset as PygGraphPropPredDataset_
import torch


class PygGraphPropPredDataset(PygGraphPropPredDataset_):
    def delete_data(self):
        self.data = None
        self.slices = None

    def download(self):
        return super().download()

    def process(self):
        return super().process()

    def load_data(self):
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __getitem__(self, idx):
        if self.data is None:
            self.load_data()

        return super().__getitem__(idx)
