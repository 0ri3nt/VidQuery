import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import MessagePassing
except Exception:  # pragma: no cover
    MessagePassing = object


class _SimpleMPNNLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        if MessagePassing is object:
            raise ImportError("torch-geometric is required for SceneGraphMPNN")
        super().__init__(aggr="mean")
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return F.relu(x_j)


class SceneGraphMPNN(nn.Module):
    def __init__(self, in_channels: int = 10, hidden_channels: int = 64, out_channels: int = 32):
        super().__init__()
        self.mp1 = _SimpleMPNNLayer(in_channels, hidden_channels)
        self.mp2 = _SimpleMPNNLayer(hidden_channels, hidden_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index):
        h = self.mp1(x, edge_index)
        h = self.mp2(h, edge_index)

        if edge_index.numel() == 0:
            return torch.empty((0, self.edge_mlp[-1].out_features), device=h.device)

        src, dst = edge_index
        edge_feat = torch.cat([h[src], h[dst]], dim=-1)
        return self.edge_mlp(edge_feat)
