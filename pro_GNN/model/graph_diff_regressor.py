# ──────────────────────────
# model/graph_diff_regressor.py
# ──────────────────────────

###使われていないファイル


import torch
import torch.nn as nn

def make_head(in_dim: int, hidden: int, out_dim: int = 1) -> nn.Sequential:
    """2-layer MLP ヘッドを生成"""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, out_dim)
    )

class GraphDiffRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_dim: int,
                 loss_type: str, hidden_dim: int = 64, out_dim: int = 1):
        super().__init__()
        assert loss_type in {"single", "pair", "all"}
        self.loss_type = loss_type
        self.encoder   = encoder

        # loss_type ごとに必要なヘッドだけ作る
        if loss_type in {"pair", "all"}:
            self.mlp_ab = make_head(embedding_dim, hidden_dim, out_dim)
        if loss_type in {"single", "all"}:
            self.mlp_a  = make_head(embedding_dim, hidden_dim, out_dim)
            self.mlp_b  = make_head(embedding_dim, hidden_dim, out_dim)

    def forward(self, data_a, data_b):
        fa = self.encoder(data_a)
        fb = self.encoder(data_b)

        # デフォルトは None にしておき、存在するヘッドだけ計算
        pred_ab = pred_a = pred_b = None

        if hasattr(self, "mlp_ab"):
            pred_ab = torch.norm(fa-fb, p=2, dim=1, keepdim=True)
        if hasattr(self, "mlp_a"):
            pred_a  = self.mlp_a(fa)
        if hasattr(self, "mlp_b"):
            pred_b  = self.mlp_b(fb)

        return pred_ab, pred_a, pred_b
