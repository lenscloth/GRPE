import torch
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.loader.dataloader import Collater as GraphCollater


class ShortestPathGenerator:
    def __init__(self, directed=False):
        self.directed = directed

    def __call__(self, data):
        row = data.edge_index[0].numpy()
        col = data.edge_index[1].numpy()
        weight = np.ones_like(row)

        graph = csr_matrix((weight, (row, col)), shape=(len(data.x), len(data.x)))
        dist_matrix, _ = shortest_path(
            csgraph=graph, directed=self.directed, return_predecessors=True
        )

        data["distance"] = torch.from_numpy(dist_matrix)
        return data


class OneHotEdgeAttr:
    def __init__(self, max_range=4) -> None:
        self.max_range = max_range

    def __call__(self, data):
        x = data["edge_attr"]
        if len(x.shape) == 1:
            return data

        offset = torch.ones((1, x.shape[1]), dtype=torch.long)
        offset[:, 1:] = self.max_range
        offset = torch.cumprod(offset, dim=1)
        x = (x * offset).sum(dim=1)
        data["edge_attr"] = x
        return data


class MoleculeCollator(object):
    def __init__(self, max_node=None) -> None:
        super().__init__()
        self.collator = GraphCollater(
            [],
            exclude_keys=[
                "x,",
                "distance",
                "edge_index",
                "edge_attr",
                "target_node",
            ],
        )
        self.max_node = max_node

    def convert_to_single_emb(self, x, offset=512):
        feature_num = x.size(1) if len(x.size()) > 1 else 1
        feature_offset = 1 + torch.arange(
            0, feature_num * offset, offset, dtype=torch.long
        )
        x = x + feature_offset
        return x

    def __call__(self, batch):
        if self.max_node is not None:
            batch = [b for b in batch if b["x"].shape[0] <= self.max_node]

        node = [self.convert_to_single_emb(b["x"]) for b in batch]
        distance = [b["distance"] for b in batch]

        if len(distance[0].shape) == 1:
            distance = [d.view(n.shape[0], n.shape[0]) for d, n in zip(distance, node)]

        edge_index = [b["edge_index"] for b in batch]
        edge_attr = [b["edge_attr"] for b in batch]
        max_num_node = max(d.shape[0] for d in distance)

        gathered_node = []
        gathered_distance = []
        gathered_edge_attr = []
        mask = []

        for n, d, ei, ea in zip(node, distance, edge_index, edge_attr):
            m = torch.zeros(max_num_node, dtype=torch.bool)
            m[n.shape[0] :] = 1

            new_n = -torch.ones((max_num_node, n.shape[1]), dtype=torch.long)
            new_n[: n.shape[0]] = n

            new_d = -torch.ones((max_num_node, max_num_node), dtype=torch.long)
            new_d[: d.shape[0], : d.shape[1]] = d
            new_d[new_d < 0] = -1

            new_ea = -torch.ones((max_num_node, max_num_node), dtype=torch.long)
            new_ea[ei[0], ei[1]] = ea

            mask.append(m)
            gathered_node.append(new_n)
            gathered_distance.append(new_d)
            gathered_edge_attr.append(new_ea)

        mask = torch.stack(mask, dim=0)
        gathered_node = torch.stack(gathered_node, dim=0)
        gathered_distance = torch.stack(gathered_distance, dim=0)
        gathered_edge_attr = torch.stack(gathered_edge_attr, dim=0)

        batch = self.collator(batch)
        batch["x"] = gathered_node
        batch["mask"] = mask
        batch["distance"] = gathered_distance
        batch["edge_attr"] = gathered_edge_attr
        return batch


class ImageCollator(object):
    def __init__(self, max_node=None) -> None:
        super().__init__()
        self.collator = GraphCollater(
            [],
            exclude_keys=[
                "x,",
                "distance",
                "edge_index",
                "edge_attr",
                "target_node",
            ],
        )
        self.max_node = max_node

    def __call__(self, batch):
        if self.max_node is not None:
            batch = [b for b in batch if b["x"].shape[0] <= self.max_node]

        # node = [self.convert_to_single_emb(b["x"]) for b in batch]
        node = [b["x"] for b in batch]
        distance = [b["distance"] for b in batch]

        if len(distance[0].shape) == 1:
            distance = [d.view(n.shape[0], n.shape[0]) for d, n in zip(distance, node)]

        edge_index = [b["edge_index"] for b in batch]
        edge_attr = [b["edge_attr"] for b in batch]
        max_num_node = max(d.shape[0] for d in distance)

        gathered_node = []
        gathered_distance = []
        gathered_edge_attr = []
        mask = []

        for n, d, ei, ea in zip(node, distance, edge_index, edge_attr):
            m = torch.zeros(max_num_node, dtype=torch.bool)
            m[n.shape[0] :] = 1

            new_n = torch.zeros((max_num_node, n.shape[1]), dtype=torch.float32)
            new_n[: n.shape[0]] = n

            new_d = -torch.ones((max_num_node, max_num_node), dtype=torch.long)
            new_d[: d.shape[0], : d.shape[1]] = d
            new_d[new_d < 0] = -1

            new_ea = -torch.ones((max_num_node, max_num_node), dtype=torch.long)
            new_ea[ei[0], ei[1]] = ea

            mask.append(m)
            gathered_node.append(new_n)
            gathered_distance.append(new_d)
            gathered_edge_attr.append(new_ea)

        mask = torch.stack(mask, dim=0)
        gathered_node = torch.stack(gathered_node, dim=0)
        gathered_distance = torch.stack(gathered_distance, dim=0)
        gathered_edge_attr = torch.stack(gathered_edge_attr, dim=0)

        batch = self.collator(batch)
        batch["x"] = gathered_node
        batch["mask"] = mask
        batch["distance"] = gathered_distance
        batch["edge_attr"] = gathered_edge_attr
        return batch
