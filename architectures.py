"""
Model architectures for financial time series forecasting.

Models implemented:
  1. SimpleLSTM         — baseline LSTM on raw sequences
  2. CNNForecaster       — 1-D CNN on raw sequences
  3. CNNLSTMHybrid      — CNN feature extraction → LSTM temporal modeling
  4. ResNetLSTM         — ResNet-18 image encoder → LSTM (uses GADF images)
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models


# ─────────────────────────────────────────────
# 1. Simple LSTM
# ─────────────────────────────────────────────
class SimpleLSTM(nn.Module):
    """
    Multi-layer LSTM with dropout regularisation.
    Input shape : (batch, seq_len, n_features)
    Output shape: (batch, 1)
    """
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])   # last time-step
        return self.fc(out).squeeze(-1)


# ─────────────────────────────────────────────
# 2. 1-D CNN Forecaster
# ─────────────────────────────────────────────
class CNNForecaster(nn.Module):
    """
    Multi-scale 1-D CNN for local pattern extraction.
    Input shape : (batch, seq_len, n_features)
    Output shape: (batch, 1)
    """
    def __init__(self, input_size: int, seq_len: int = 24, dropout: float = 0.2):
        super().__init__()
        # Transpose from (B, T, C) → (B, C, T) for Conv1d
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, C, T)
        x = self.conv_block(x)
        return self.head(x).squeeze(-1)


# ─────────────────────────────────────────────
# 3. CNN–LSTM Hybrid
# ─────────────────────────────────────────────
class CNNLSTMHybrid(nn.Module):
    """
    1-D CNN extracts local features; LSTM captures temporal dependencies.
    Input shape : (batch, seq_len, n_features)
    Output shape: (batch, 1)
    """
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
        )
        self.lstm = nn.LSTM(
            64, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # CNN pass
        x = x.permute(0, 2, 1)           # (B, C, T)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)           # (B, T, C)
        # LSTM pass
        out, _ = self.lstm(x)
        out = out[:, -1, :]               # last time-step
        return self.head(out).squeeze(-1)


# ─────────────────────────────────────────────
# 4. ResNet-18 Encoder + LSTM (GADF images)
# ─────────────────────────────────────────────
class ResNetLSTM(nn.Module):
    """
    ResNet-18 as spatial feature extractor on GADF image windows.
    A sliding window of GADF images is encoded per step, then fed to LSTM.

    Image input shape : (batch, seq_len, H, W, 3)
    Output shape      : (batch, 1)
    """
    def __init__(self, embedding_dim: int = 256, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2, freeze_backbone: bool = True):
        super().__init__()

        # ResNet-18 backbone
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        if freeze_backbone:
            for p in list(backbone.parameters())[:-10]:
                p.requires_grad = False

        # Replace final FC layer with embedding projection
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.encoder = backbone

        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, T, H, W, C) → encode each frame
        B, T, H, W, C = x.shape
        x = x.view(B * T, H, W, C)
        x = x.permute(0, 3, 1, 2).float()    # (B*T, C, H, W)
        embeddings = self.encoder(x)           # (B*T, embed_dim)
        embeddings = embeddings.view(B, T, -1) # (B, T, embed_dim)

        out, _ = self.lstm(embeddings)
        out = out[:, -1, :]
        return self.head(out).squeeze(-1)


# ─────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────
MODEL_REGISTRY = {
    "Simple LSTM": SimpleLSTM,
    "CNN Forecaster": CNNForecaster,
    "CNN–LSTM Hybrid": CNNLSTMHybrid,
    "ResNet–LSTM (GADF)": ResNetLSTM,
}

MODEL_DESCRIPTIONS = {
    "Simple LSTM": "Multi-layer LSTM with dropout. Strong baseline for sequential data. "
                   "Captures long-range temporal dependencies via gated memory cells.",
    "CNN Forecaster": "Multi-scale 1-D CNN with global average pooling. "
                      "Excels at detecting local patterns and motifs in price series.",
    "CNN–LSTM Hybrid": "1-D CNN extracts local features per timestep; LSTM models "
                       "temporal evolution of those features. Best of both worlds.",
    "ResNet–LSTM (GADF)": "ResNet-18 encodes GADF image windows into spatial embeddings; "
                           "LSTM captures temporal dynamics. Spatiotemporal fusion architecture.",
}
