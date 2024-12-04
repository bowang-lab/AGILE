# Copyright (c) 2023 Shihao Ma, Haotian Cui, WangLab @ U of T

# This source code is modified from https://github.com/yuyangw/MolCLR 
# under MIT License. The original license is included below:
# ========================================================================
# MIT License

# Copyright (c) 2021 Yuyang Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

num_atom_type = 119  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5  # including aromatic and self-loop edge
num_bond_direction = 3


class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    # def forward(self, x, edge_index, e1, e2):
    #     # add self loops in the edge space
    #     edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

    #     # concat e1 and e2 column wise
    #     edge_attr = torch.stack((e1, e2), dim=1)

    #     # add features corresponding to self-loop edges.
    #     self_loop_attr = torch.zeros(x.size(0), 2)
    #     self_loop_attr[:, 0] = 4  # bond type for self-loop edge
    #     self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
    #     edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

    #     edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
    #         edge_attr[:, 1]
    #     )

    #     return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class AGILE(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        drop_ratio (float): dropout rate
        pool (str): pooling method
        pred_additional_feat_dim (int): additional feature dims in the pred head
        pred_n_layer (int): the number of layers in the pred head
        pred_act (str): activation function in the pred head
    Output:
        node representations
    """

    def __init__(
        self,
        task="classification",
        num_layer=5,
        emb_dim=300,
        feat_dim=512,
        drop_ratio=0,
        pool="mean",
        pred_additional_feat_dim=0,
        pred_n_layer=2,
        pred_act="softplus",
    ):
        super(AGILE, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.has_additional_feat = pred_additional_feat_dim > 0
        self.drop_ratio = drop_ratio
        self.task = task

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        elif pool == "add":
            self.pool = global_add_pool
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        if self.task == "classification":
            out_dim = 2
        elif self.task == "regression":
            out_dim = 1

        pred_input_dim = self.feat_dim
        self.pred_n_layer = max(1, pred_n_layer)
        if self.has_additional_feat:
            feat_hidden_dim = min(pred_additional_feat_dim, 100)
            self.pred_feat = nn.Sequential(
                nn.Linear(pred_additional_feat_dim, feat_hidden_dim),
                nn.ELU(),
                nn.LayerNorm(feat_hidden_dim),
            )
            pred_input_dim += feat_hidden_dim
            self.additional_feat_hidden_dim = feat_hidden_dim

        if pred_act == "relu":
            pred_head = [
                nn.Linear(pred_input_dim, self.feat_dim // 2),
                nn.ReLU(inplace=True),
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend(
                    [
                        nn.Linear(self.feat_dim // 2, self.feat_dim // 2),
                        nn.ReLU(inplace=True),
                    ]
                )
            pred_head.append(nn.Linear(self.feat_dim // 2, out_dim))
        elif pred_act == "softplus":
            pred_head = [nn.Linear(pred_input_dim, self.feat_dim // 2), nn.Softplus()]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend(
                    [nn.Linear(self.feat_dim // 2, self.feat_dim // 2), nn.Softplus()]
                )
        else:
            raise ValueError("Undefined activation function")

        pred_head.append(nn.Linear(self.feat_dim // 2, out_dim))
        self.pred_head = nn.Sequential(*pred_head)
        print(self)

    def forward(self, x, edge_index, edge_attr, batch, feat=None):
        # x = data.x
        # edge_index = data.edge_index
        # edge_attr = data.edge_attr

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, batch)
        h = self.feat_lin(h)

        if self.has_additional_feat:
            # print(data.feat.max())
            feat = self.pred_feat(feat)
            h = torch.cat((h, feat), dim=1)

        return h, self.pred_head(h)

    # def forward(self, data):
    #     x = data.x
    #     edge_index = data.edge_index
    #     edge_attr = data.edge_attr

    #     h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

    #     for layer in range(self.num_layer):
    #         h = self.gnns[layer](h, edge_index, edge_attr)
    #         h = self.batch_norms[layer](h)
    #         if layer == self.num_layer - 1:
    #             h = F.dropout(h, self.drop_ratio, training=self.training)
    #         else:
    #             h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

    #     h = self.pool(h, data.batch)
    #     h = self.feat_lin(h)

    #     if self.has_additional_feat:
    #         # print(data.feat.max())
    #         feat = self.pred_feat(data.feat)
    #         h = torch.cat((h, feat), dim=1)

    #     return h, self.pred_head(h)

    def forward_only_h(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)

        if self.has_additional_feat: 
            # make features as zeros
            feat = torch.zeros(
                (h.shape[0], self.additional_feat_hidden_dim),
                dtype=h.dtype,
                device=h.device,
            )
            h = torch.cat((h, feat), dim=1)

        return h, self.pred_head(h)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
