import torch
import collections

from ogb.utils import smiles2graph
from torch_geometric.data import Data
from torchvision.transforms import Compose

from .dataset.tansform import ShortestPathGenerator, OneHotEdgeAttr, MoleculeCollator
from .model.graph_self_attention import GraphSelfAttentionNetwork


class MoleculeFingerPrint(GraphSelfAttentionNetwork):
    @torch.no_grad()
    def generate_fingerprint(self, smiles, fingerprint_stack=3):
        self.eval()

        datas = []
        for smile in smiles:
            data = Data()
            graph = smiles2graph(smile)
            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            transform = Compose([ShortestPathGenerator(), OneHotEdgeAttr()])

            data = transform(data)
            datas.append(data)

        datas = MoleculeCollator()(datas)
        device = next(self.parameters()).device

        datas = datas.to(device)
        fingerprint = self(datas, fingerprint_stack)
        return fingerprint

    def forward(self, data, fingerprint_stack=1):
        x = self.encode_node(data)
        mask = data.mask

        # Append Task Token
        x_with_task = torch.zeros(
            (x.shape[0], x.shape[1] + 1, x.shape[2]), dtype=x.dtype, device=x.device
        )
        x_with_task[:, 1:] = x
        x_with_task[:, 0] = self.task_token.weight

        # Mask with task
        mask_with_task = torch.zeros(
            (mask.shape[0], mask.shape[1] + 1),
            dtype=mask.dtype,
            device=x.device,
        )
        mask_with_task[:, 1:] = mask

        distance_with_task = torch.zeros(
            (
                data.distance.shape[0],
                data.distance.shape[1] + 1,
                data.distance.shape[2] + 1,
            ),
            dtype=data.distance.dtype,
            device=data.distance.device,
        )
        distance_with_task[:, 1:, 1:] = data.distance.clamp(
            max=self.max_hop
        )  # max_hop is $\mathcal{P}_\text{far}$
        distance_with_task[:, 0, 1:] = self.TASK_DISTANCE
        distance_with_task[:, 1:, 0] = self.TASK_DISTANCE
        distance_with_task[distance_with_task == -1] = self.UNREACHABLE_DISTANCE

        edge_attr_with_task = torch.zeros(
            (
                data.edge_attr.shape[0],
                data.edge_attr.shape[1] + 1,
                data.edge_attr.shape[2] + 1,
            ),
            dtype=data.edge_attr.dtype,
            device=data.edge_attr.device,
        )
        edge_attr_with_task[:, 1:, 1:] = data.edge_attr
        edge_attr_with_task[
            :, range(edge_attr_with_task.shape[1]), range(edge_attr_with_task.shape[2])
        ] = self.SELF_EDGE
        edge_attr_with_task[edge_attr_with_task == -1] = self.NO_EDGE
        edge_attr_with_task[distance_with_task != 1] = self.NO_EDGE
        edge_attr_with_task[distance_with_task == self.TASK_DISTANCE] = self.TASK_EDGE

        finger_print = []
        for i, enc_layer in enumerate(self.layers):
            if self.use_independent_token:
                x_with_task = enc_layer(
                    x_with_task,
                    self.query_hop_emb[i].weight,
                    self.query_edge_emb[i].weight,
                    self.key_hop_emb[i].weight,
                    self.key_edge_emb[i].weight,
                    self.value_hop_emb[i].weight,
                    self.value_edge_emb[i].weight,
                    distance_with_task,
                    edge_attr_with_task,
                    mask=mask_with_task,
                )
            else:
                x_with_task = enc_layer(
                    x_with_task,
                    self.query_hop_emb.weight,
                    self.query_edge_emb.weight,
                    self.key_hop_emb.weight,
                    self.key_edge_emb.weight,
                    self.value_hop_emb.weight,
                    self.value_edge_emb.weight,
                    distance_with_task,
                    edge_attr_with_task,
                    mask=mask_with_task,
                )

            if (i + fingerprint_stack) >= len(self.layers):
                finger_print.append(x_with_task[:, 0])

        normalized_finger_print = [
            torch.nn.functional.normalize(fp, dim=1, p=2) for fp in finger_print
        ]
        normalized_finger_print = torch.cat(normalized_finger_print, dim=1)
        return normalized_finger_print
