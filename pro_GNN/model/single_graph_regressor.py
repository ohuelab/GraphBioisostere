
import torch
import torch.nn as nn

class SingleGraphRegressor(nn.Module):
    def __init__(self, encoder, embedding_dim, hidden_dim=64, out_dim=1):
        super().__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        

    def forward(self, data):
        h = self.encoder(data)  # shape: [batch_size, embedding_dim]
        out = self.mlp(h)  # shape: [batch_size, 1]
        return out
