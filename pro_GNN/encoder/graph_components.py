# ──────────────────────────
# encoder/graph_components.py
# ──────────────────────────

from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, GRUCell, Module, BatchNorm1d
from torch_geometric.nn import GINEConv

# ノード特徴の線形変換 + BatchNorm + 活性化
class NodeLinear(nn.Module):
    def __init__(self, in_lin, out_lin):
        super().__init__()
        self.lin = Linear(in_lin, out_lin)
        self.bn = nn.BatchNorm1d(out_lin)

    def forward(self, x):
        x = x.float()  # 型の安全性を確保
        x = self.lin(x)
        if x.size(0) > 1:
            x = self.bn(x)
        return F.leaky_relu(x)

# エッジ特徴の線形変換 + BatchNorm + 活性化
class EdgeLinear(nn.Module):
    def __init__(self, in_lin, out_lin):
        super().__init__()
        self.lin = Linear(in_lin, out_lin)
        self.bn = nn.BatchNorm1d(out_lin)

    def forward(self, x):
        if x.size(0) == 0:
            return None
        x = x.float()  # 型の安全性を確保
        x = self.lin(x)
        if x.size(0) > 1:
            x = self.bn(x)
        return F.leaky_relu(x)

# GIN層内のMLP（GINEConvに渡す）
class GIN_Sequential(Module):
    def __init__(self, in_channels, hidden_channels, dropout):
        super().__init__()
        self.in_channels = in_channels  # GINEConv がこれを参照して in_channels を推定
        self.lin1 = Linear(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.dropout = dropout

    def forward(self, x):
        x = self.lin1(x)
        if (x.size(0) != 1 and self.training) or (not self.training):
            x = self.bn1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)

# GNN全体構造の各レイヤー構成（GINEConv + BN + GRU）
def GNN_Conv(hidden_dim, num_layers, dropout):
    convs, bns, grus = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
    for _ in range(num_layers):
        mlp = GIN_Sequential(hidden_dim, hidden_dim, dropout)
        convs.append(GINEConv(mlp, edge_dim=hidden_dim))  # エッジ特徴次元 = hidden_dim
        bns.append(nn.BatchNorm1d(hidden_dim))
        grus.append(GRUCell(hidden_dim, hidden_dim))
    return convs, bns, grus
