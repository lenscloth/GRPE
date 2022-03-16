# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn


def load_grpe_backbone(model, load, skip_task_branch=True, map_location="cpu"):
    new_dict = {}
    for k, v in torch.load(load, map_location=map_location).items():
        if k.startswith("module."):
            k = k[7:]
        if (
            k.endswith("linear.weight") or k.endswith("linear.bias")
        ) and skip_task_branch:
            print("Skip loading the weight of task branch")
        else:
            new_dict[k] = v
    print(f"Loaded following keys {list(new_dict.keys())}")
    model.load_state_dict(new_dict, strict=False)


class GRPENetwork(nn.Module):
    def __init__(
        self,
        num_task=1,
        num_layer=6,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        attention_dropout=0.1,
        max_hop=256,
        num_node_type=25,
        num_edge_type=25,
        use_independent_token=False,
        perturb_noise=0.0,
        num_last_mlp=0,
    ):
        super().__init__()
        self.perturb_noise = perturb_noise
        self.max_hop = max_hop
        self.num_edge_type = num_edge_type
        self.task_token = nn.Embedding(1, d_model, padding_idx=-1)
        self.use_independent_token = use_independent_token

        if num_node_type < 0:
            num_node_type = -num_node_type
            self.node_emb = nn.Linear(num_node_type, d_model)
        else:
            self.node_emb = nn.Embedding(num_node_type, d_model)

        self.TASK_DISTANCE = max_hop + 1
        self.UNREACHABLE_DISTANCE = max_hop + 2

        self.TASK_EDGE = num_edge_type + 1
        self.SELF_EDGE = num_edge_type + 2
        self.NO_EDGE = num_edge_type + 3

        # query_hop_emb: Query Structure Embedding
        # query_edge_emb: Query Edge Embedding
        # key_hop_emb: Key Structure Embedding
        # key_edge_emb: Key Edge Embedding
        # value_hop_emb: Value Structure Embedding
        # value_edge_emb: Value Edge Embedding

        if not self.use_independent_token:
            self.query_hop_emb = nn.Embedding(max_hop + 3, d_model)
            self.query_edge_emb = nn.Embedding(num_edge_type + 4, d_model)
            self.key_hop_emb = nn.Embedding(max_hop + 3, d_model)
            self.key_edge_emb = nn.Embedding(num_edge_type + 4, d_model)
            self.value_hop_emb = nn.Embedding(max_hop + 3, d_model)
            self.value_edge_emb = nn.Embedding(num_edge_type + 4, d_model)

        else:
            self.query_hop_emb = nn.ModuleList(
                [nn.Embedding(max_hop + 3, d_model) for _ in range(num_layer)]
            )
            self.query_edge_emb = nn.ModuleList(
                [nn.Embedding(num_edge_type + 4, d_model) for _ in range(num_layer)]
            )
            self.key_hop_emb = nn.ModuleList(
                [nn.Embedding(max_hop + 3, d_model) for _ in range(num_layer)]
            )
            self.key_edge_emb = nn.ModuleList(
                [nn.Embedding(num_edge_type + 4, d_model) for _ in range(num_layer)]
            )
            self.value_hop_emb = nn.ModuleList(
                [nn.Embedding(max_hop + 3, d_model) for _ in range(num_layer)]
            )
            self.value_edge_emb = nn.ModuleList(
                [nn.Embedding(num_edge_type + 4, d_model) for _ in range(num_layer)]
            )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=d_model,
                    ffn_size=dim_feedforward,
                    dropout_rate=dropout,
                    attention_dropout_rate=attention_dropout,
                    num_heads=nhead,
                )
                for _ in range(num_layer)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.last_mlp = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
                for _ in range(num_last_mlp)
            ]
        )
        self.linear = nn.Linear(d_model, num_task)

    def encode_node(self, data):
        if isinstance(self.node_emb, nn.Linear):
            return self.node_emb(data.x)
        else:
            return self.node_emb.weight[data.x].sum(dim=2)

    def forward(self, data):
        x = self.encode_node(data)
        mask = data.mask

        if self.training:
            perturb = torch.empty_like(x).uniform_(
                -self.perturb_noise, self.perturb_noise
            )
            x = x + perturb

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

        output = self.final_ln(x_with_task[:, 0])
        output = self.last_mlp(output)
        output = self.linear(output)

        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(
        self,
        q,
        k,
        v,
        query_hop_emb,
        query_edge_emb,
        key_hop_emb,
        key_edge_emb,
        value_hop_emb,
        value_edge_emb,
        distance,
        edge_attr,
        mask=None,
    ):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2)  # [b, h, d_k, k_len]

        sequence_length = v.shape[2]
        num_hop_types = query_hop_emb.shape[0]
        num_edge_types = query_edge_emb.shape[0]

        query_hop_emb = query_hop_emb.view(
            1, num_hop_types, self.num_heads, self.att_size
        ).transpose(1, 2)
        query_edge_emb = query_edge_emb.view(
            1, -1, self.num_heads, self.att_size
        ).transpose(1, 2)
        key_hop_emb = key_hop_emb.view(
            1, num_hop_types, self.num_heads, self.att_size
        ).transpose(1, 2)
        key_edge_emb = key_edge_emb.view(
            1, num_edge_types, self.num_heads, self.att_size
        ).transpose(1, 2)

        query_hop = torch.matmul(q, query_hop_emb.transpose(2, 3))
        query_hop = torch.gather(
            query_hop, 3, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )
        query_edge = torch.matmul(q, query_edge_emb.transpose(2, 3))
        query_edge = torch.gather(
            query_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )

        key_hop = torch.matmul(k, key_hop_emb.transpose(2, 3))
        key_hop = torch.gather(
            key_hop, 3, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )
        key_edge = torch.matmul(k, key_edge_emb.transpose(2, 3))
        key_edge = torch.gather(
            key_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )

        spatial_bias = query_hop + key_hop
        edge_bais = query_edge + key_edge

        x = torch.matmul(q, k.transpose(2, 3)) + spatial_bias + edge_bais

        x = x * self.scale

        if mask is not None:
            x = x.masked_fill(
                mask.view(mask.shape[0], 1, 1, mask.shape[1]), float("-inf")
            )

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)

        value_hop_emb = value_hop_emb.view(
            1, num_hop_types, self.num_heads, self.att_size
        ).transpose(1, 2)
        value_edge_emb = value_edge_emb.view(
            1, num_edge_types, self.num_heads, self.att_size
        ).transpose(1, 2)

        value_hop_att = torch.zeros(
            (batch_size, self.num_heads, sequence_length, num_hop_types),
            device=value_hop_emb.device,
        )
        value_hop_att = torch.scatter_add(
            value_hop_att, 3, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1), x
        )
        value_edge_att = torch.zeros(
            (batch_size, self.num_heads, sequence_length, num_edge_types),
            device=value_hop_emb.device,
        )
        value_edge_att = torch.scatter_add(
            value_edge_att, 3, edge_attr.unsqueeze(1).repeat(1, self.num_heads, 1, 1), x
        )

        x = (
            torch.matmul(x, v)
            + torch.matmul(value_hop_att, value_hop_emb)
            + torch.matmul(value_edge_att, value_edge_emb)
        )
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size,
            attention_dropout_rate,
            num_heads,
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x,
        query_hop_emb,
        query_edge_emb,
        key_hop_emb,
        key_edge_emb,
        value_hop_emb,
        value_edge_emb,
        distance,
        edge_attr,
        mask=None,
    ):
        y = self.self_attention_norm(x)
        y = self.self_attention(
            y,
            y,
            y,
            query_hop_emb,
            query_edge_emb,
            key_hop_emb,
            key_edge_emb,
            value_hop_emb,
            value_edge_emb,
            distance,
            edge_attr,
            mask=mask,
        )
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
