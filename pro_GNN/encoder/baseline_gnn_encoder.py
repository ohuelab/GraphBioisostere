# ──────────────────────────
# encoder/baseline_gnn_encoder.py
# シンプルなGINとGCNベースラインモデル
# ──────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool


class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network (GIN) ベースのエンコーダー
    
    Args:
        node_in: ノード特徴量の入力次元
        edge_in: エッジ特徴量の入力次元（このモデルでは使用しない）
        hidden_dim: 隠れ層の次元
        out_dim: 出力埋め込みの次元
        num_layers: GINレイヤーの数
        dropout: ドロップアウト率
        pooling: グローバルプーリング方法 ('sum' or 'mean')
    """
    def __init__(self, node_in, edge_in, hidden_dim, out_dim, 
                 num_layers=3, dropout=0.2, pooling='sum'):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        
        # ノード特徴量の初期エンコーディング
        self.node_encoder = Linear(node_in, hidden_dim)
        
        # GINレイヤー
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # GINConvは内部のMLPを定義
            mlp = Sequential(
                Linear(hidden_dim, hidden_dim * 2),
                BatchNorm1d(hidden_dim * 2),
                ReLU(),
                Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # 出力層
        self.lin_out = Linear(hidden_dim, out_dim)
        
        # アテンション重みの互換性のため（実際は使用しない）
        self.attention_weights = None
        self.node_features = None
    
    def forward(self, data, return_node_features=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # ノード特徴量のエンコーディング
        x = self.node_encoder(x)
        x = F.relu(x)
        
        # GINレイヤーを通す
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # ノード特徴量を保存
        self.node_features = x
        
        # グローバルプーリング
        if self.pooling == 'sum':
            graph_embedding = global_add_pool(x, batch)
        elif self.pooling == 'mean':
            graph_embedding = global_mean_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # 出力層
        out = self.lin_out(graph_embedding)
        
        if return_node_features:
            return self.node_features
        else:
            return out
    
    def get_attention_weights(self):
        """互換性のため（GINにはアテンションがない）"""
        return self.attention_weights
    
    def get_node_features(self):
        """最後のフォワードパスで計算されたノード特徴量を返す"""
        return self.node_features


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network (GCN) ベースのエンコーダー
    
    Args:
        node_in: ノード特徴量の入力次元
        edge_in: エッジ特徴量の入力次元（このモデルでは使用しない）
        hidden_dim: 隠れ層の次元
        out_dim: 出力埋め込みの次元
        num_layers: GCNレイヤーの数
        dropout: ドロップアウト率
        pooling: グローバルプーリング方法 ('sum' or 'mean')
    """
    def __init__(self, node_in, edge_in, hidden_dim, out_dim, 
                 num_layers=3, dropout=0.2, pooling='sum'):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        
        # ノード特徴量の初期エンコーディング
        self.node_encoder = Linear(node_in, hidden_dim)
        
        # GCNレイヤー
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # 出力層
        self.lin_out = Linear(hidden_dim, out_dim)
        
        # アテンション重みの互換性のため（実際は使用しない）
        self.attention_weights = None
        self.node_features = None
    
    def forward(self, data, return_node_features=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # ノード特徴量のエンコーディング
        x = self.node_encoder(x)
        x = F.relu(x)
        
        # GCNレイヤーを通す
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # ノード特徴量を保存
        self.node_features = x
        
        # グローバルプーリング
        if self.pooling == 'sum':
            graph_embedding = global_add_pool(x, batch)
        elif self.pooling == 'mean':
            graph_embedding = global_mean_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # 出力層
        out = self.lin_out(graph_embedding)
        
        if return_node_features:
            return self.node_features
        else:
            return out
    
    def get_attention_weights(self):
        """互換性のため（GCNにはアテンションがない）"""
        return self.attention_weights
    
    def get_node_features(self):
        """最後のフォワードパスで計算されたノード特徴量を返す"""
        return self.node_features


class GINEncoderWithEdge(nn.Module):
    """
    エッジ特徴量を考慮したGINエンコーダー
    エッジ特徴量はノード特徴量に加算される形で統合
    
    Args:
        node_in: ノード特徴量の入力次元
        edge_in: エッジ特徴量の入力次元
        hidden_dim: 隠れ層の次元
        out_dim: 出力埋め込みの次元
        num_layers: GINレイヤーの数
        dropout: ドロップアウト率
        pooling: グローバルプーリング方法 ('sum' or 'mean')
    """
    def __init__(self, node_in, edge_in, hidden_dim, out_dim, 
                 num_layers=3, dropout=0.2, pooling='sum'):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        
        # ノードとエッジ特徴量のエンコーディング
        self.node_encoder = Linear(node_in, hidden_dim)
        self.edge_encoder = Linear(edge_in, hidden_dim)
        
        # GINレイヤー
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = Sequential(
                Linear(hidden_dim, hidden_dim * 2),
                BatchNorm1d(hidden_dim * 2),
                ReLU(),
                Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # 出力層
        self.lin_out = Linear(hidden_dim, out_dim)
        
        self.attention_weights = None
        self.node_features = None
    
    def forward(self, data, return_node_features=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # ノード特徴量のエンコーディング
        x = self.node_encoder(x)
        x = F.relu(x)
        
        # エッジ特徴量のエンコーディング
        edge_attr = self.edge_encoder(edge_attr)
        
        # GINレイヤーを通す（エッジ特徴量を考慮）
        for conv, bn in zip(self.convs, self.batch_norms):
            # エッジ特徴量をノードの隣接情報に加算
            # 各エッジの両端のノードにエッジ特徴量を分配
            row, col = edge_index
            edge_contribution = torch.zeros_like(x)
            edge_contribution.index_add_(0, row, edge_attr)
            edge_contribution.index_add_(0, col, edge_attr)
            
            x_with_edge = x + edge_contribution
            x = conv(x_with_edge, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # ノード特徴量を保存
        self.node_features = x
        
        # グローバルプーリング
        if self.pooling == 'sum':
            graph_embedding = global_add_pool(x, batch)
        elif self.pooling == 'mean':
            graph_embedding = global_mean_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # 出力層
        out = self.lin_out(graph_embedding)
        
        if return_node_features:
            return self.node_features
        else:
            return out
    
    def get_attention_weights(self):
        """互換性のため（GINにはアテンションがない）"""
        return self.attention_weights
    
    def get_node_features(self):
        """最後のフォワードパスで計算されたノード特徴量を返す"""
        return self.node_features
