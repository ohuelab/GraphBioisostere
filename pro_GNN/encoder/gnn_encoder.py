# ──────────────────────────
# encoder/gnn_encoder.py
# ──────────────────────────
from encoder.graph_components import NodeLinear, EdgeLinear, GNN_Conv, GIN_Sequential
from torch_geometric.nn import global_add_pool, GATv2Conv
from torch.nn import Linear, GRUCell
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNEncoder(nn.Module):
    def __init__(self, node_in, edge_in, hidden_dim, out_dim, num_layers=3, dropout=0.2, num_layers_self=2):
        super().__init__()
        self.node_encoder = NodeLinear(node_in, hidden_dim)
        self.edge_encoder = EdgeLinear(edge_in, hidden_dim)
        self.convs, self.bns, self.grus = GNN_Conv(hidden_dim, num_layers, dropout)
        
        # 参考コードに合わせた設定
        self.mol_conv = GATv2Conv(hidden_dim, hidden_dim, add_self_loops=False)
        self.mol_bn = nn.BatchNorm1d(hidden_dim)
        self.mol_gru = GRUCell(hidden_dim, hidden_dim)
        self.lin_out = Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self.num_layers_self = num_layers_self  # 分子レベルの繰り返し回数
        
        # アテンションウェイトを保存するための変数
        self.attention_weights = None
        self.attention_weights_mean = None  # 平均化されたアテンション重み
        self.node_features = None

    def forward(self, data, return_node_features=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)

        edge_attr = self.edge_encoder(edge_attr)

        # ノードレベルの畳み込み処理
        for i, (conv, bn, gru) in enumerate(zip(self.convs, self.bns, self.grus)):
            h = conv(x, edge_index, edge_attr)
            h = bn(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x)
            x = F.leaky_relu(x)

        # ノード特徴量を保存
        self.node_features = x
        
        # Molecule Embedding: 参考コードに合わせた実装
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index_mol = torch.stack([row, batch], dim=0)
        
        # global_add_poolで初期の分子ベクトルを作成
        out = F.leaky_relu(global_add_pool(x, batch))
        
        # アテンション重みを蓄積するリスト
        att_mol_stack = []
        att_mol_index = None
        
        # forループで分子ベクトルを繰り返し洗練させる
        for t in range(self.num_layers_self):
            # mol_convで、全原子(x)から分子ベクトル(out)へのAttentionを計算
            h, attention_weights = self.mol_conv((x, out), edge_index_mol, return_attention_weights=True)
            
            # バッチ正規化
            if ((h.size(0) != 1 and self.training) or (not self.training)):
                h = self.mol_bn(h)
            h = F.elu_(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # GRUで、Attentionの結果(h)を基に分子ベクトル(out)を更新
            out = self.mol_gru(h, out)
            out = F.leaky_relu(out)
            
            # アテンション重みを蓄積
            att_mol_index, att_mol_weights = attention_weights
            att_mol_stack.append(att_mol_weights)
        
        # 最終的な出力層
        out = F.dropout(out, p=self.dropout, training=self.training)
        final_out = self.lin_out(out)
        
        # mean of attention weight
        if att_mol_stack:
            att_mol_mean = torch.mean(torch.stack(att_mol_stack), dim=0)
            self.attention_weights = att_mol_mean  # 平均化されたアテンション重みを保存
            self.attention_weights_mean = att_mol_mean
        
        if return_node_features:
            return self.node_features
        else:
            return final_out
    
    def get_attention_weights(self):
        """最後のフォワードパスで計算されたアテンションウェイトを返す（平均化済み）"""
        return self.attention_weights
    
    def get_node_features(self):
        """最後のフォワードパスで計算されたノード特徴量を返す"""
        return self.node_features

