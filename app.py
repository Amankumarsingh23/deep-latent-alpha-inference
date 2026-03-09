"""
Deep Latent Alpha Inference · Streamlit Dashboard
Single-file version — all logic inlined for Streamlit Cloud compatibility.
"""

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as tv_models
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ══════════════════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ohlc_data(ticker="AAPL", period="2y", interval="1h"):
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df


def add_technical_indicators(df):
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_10"] = df["Close"].ewm(span=10).mean()
    rolling_std = df["Close"].rolling(20).std()
    df["BB_upper"] = df["SMA_20"] + 2 * rolling_std
    df["BB_lower"] = df["SMA_20"] - 2 * rolling_std
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df


def normalize_data(df, feature_cols):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols].values)
    return scaled, scaler


def create_sequences(data, seq_len=24, target_col_idx=3):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len, target_col_idx])
    return np.array(X), np.array(y)


def train_test_split_ts(X, y, split=0.8):
    n = int(len(X) * split)
    return X[:n], X[n:], y[:n], y[n:]


# ══════════════════════════════════════════════════════════════════════════════
# GADF UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _rescale_to_minus1_1(series):
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return np.zeros_like(series)
    return 2 * (series - s_min) / (s_max - s_min) - 1


def compute_gadf(series):
    scaled = np.clip(_rescale_to_minus1_1(series), -1, 1)
    phi = np.arccos(scaled)
    return np.sin(phi[:, None] - phi[None, :])


def compute_gasf(series):
    scaled = np.clip(_rescale_to_minus1_1(series), -1, 1)
    phi = np.arccos(scaled)
    return np.cos(phi[:, None] + phi[None, :])


def series_to_gadf_batch(sequences, feature_col=3):
    N, seq_len, _ = sequences.shape
    images = np.zeros((N, seq_len, seq_len, 3), dtype=np.float32)
    for i in range(N):
        series = sequences[i, :, feature_col]
        gadf = compute_gadf(series)
        gadf_norm = (gadf + 1) / 2
        images[i, :, :, 0] = gadf_norm
        images[i, :, :, 1] = compute_gasf(series) * 0.5 + 0.5
        images[i, :, :, 2] = gadf_norm ** 2
    return images


def plot_gadf_sample(series, title="GADF Encoding"):
    gadf = compute_gadf(series)
    gasf = compute_gasf(series)
    cmap = LinearSegmentedColormap.from_list(
        "gadf_cmap", ["#1a237e", "#5c6bc0", "#ffffff", "#ef5350", "#b71c1c"]
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#0f172a")
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=1.01)
    for ax, (data, label, cm) in zip(axes, [
        (series, "Price Series", None),
        (gadf,   "GADF Matrix",  cmap),
        (gasf,   "GASF Matrix",  cmap),
    ]):
        ax.set_facecolor("#0f172a")
        ax.title.set_color("white")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        if data.ndim == 1:
            ax.plot(data, color="#38bdf8", linewidth=1.5)
            ax.set_xlabel("Time Steps", color="#94a3b8", fontsize=9)
            ax.set_ylabel("Normalised Price", color="#94a3b8", fontsize=9)
        else:
            im = ax.imshow(data, cmap=cm, origin="lower", aspect="auto", vmin=-1, vmax=1)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.yaxis.set_tick_params(color="white")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="#94a3b8", fontsize=7)
        ax.set_title(label, color="white", fontsize=11)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


class CNNForecaster(nn.Module):
    def __init__(self, input_size, seq_len=24, dropout=0.2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_size, 64,  kernel_size=3, padding=1), nn.BatchNorm1d(64),  nn.GELU(),
            nn.Conv1d(64,         128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128,        128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.head(self.conv_block(x.permute(0, 2, 1))).squeeze(-1)


class CNNLSTMHybrid(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),          nn.BatchNorm1d(64), nn.GELU(),
        )
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class ResNetLSTM(nn.Module):
    def __init__(self, embedding_dim=256, hidden_size=128, num_layers=2,
                 dropout=0.2, freeze_backbone=True):
        super().__init__()
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        if freeze_backbone:
            for p in list(backbone.parameters())[:-10]:
                p.requires_grad = False
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, embedding_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.encoder = backbone
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B * T, H, W, C).permute(0, 3, 1, 2).float()
        emb = self.encoder(x).view(B, T, -1)
        out, _ = self.lstm(emb)
        return self.head(out[:, -1, :]).squeeze(-1)


MODEL_REGISTRY = {
    "Simple LSTM":        SimpleLSTM,
    "CNN Forecaster":     CNNForecaster,
    "CNN–LSTM Hybrid":    CNNLSTMHybrid,
    "ResNet–LSTM (GADF)": ResNetLSTM,
}

MODEL_DESCRIPTIONS = {
    "Simple LSTM":        "Multi-layer LSTM with dropout. Captures long-range temporal dependencies via gated memory cells.",
    "CNN Forecaster":     "Multi-scale 1-D CNN with global average pooling. Detects local price motifs and patterns.",
    "CNN–LSTM Hybrid":    "1-D CNN extracts local features per timestep; LSTM models their temporal evolution.",
    "ResNet–LSTM (GADF)": "ResNet-18 encodes GADF image windows into spatial embeddings; LSTM captures temporal dynamics.",
}


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_dataloader(X, y, batch_size, shuffle):
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=shuffle
    )


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=30, batch_size=64, lr=1e-3, patience=7,
                progress_callback=None):
    device = get_device()
    model = model.to(device)
    train_loader = _make_dataloader(X_train, y_train, batch_size, True)
    val_loader   = _make_dataloader(X_val,   y_val,   batch_size, False)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=3, factor=0.5)
    criterion = nn.HuberLoss(delta=0.5)
    best_val, best_state, no_improve = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        t_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            t_losses.append(loss.item())

        model.eval()
        v_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                v_losses.append(criterion(model(xb), yb).item())

        t_loss, v_loss = np.mean(t_losses), np.mean(v_losses)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        scheduler.step(v_loss)
        if progress_callback:
            progress_callback(epoch, t_loss, v_loss)

        if v_loss < best_val:
            best_val   = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    history["best_epoch"] = int(np.argmin(history["val_loss"])) + 1
    return history


def evaluate_model(model, X_test, y_test, scaler, close_col_idx=3, n_features=5):
    device = get_device()
    model.eval().to(device)
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    def _inverse(vals):
        dummy = np.zeros((len(vals), n_features))
        dummy[:, close_col_idx] = vals
        return scaler.inverse_transform(dummy)[:, close_col_idx]

    y_pred_orig = _inverse(preds)
    y_true_orig = _inverse(y_test)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae  = mean_absolute_error(y_true_orig, y_pred_orig)
    mask = y_true_orig != 0
    mape = np.mean(np.abs((y_true_orig[mask] - y_pred_orig[mask]) / y_true_orig[mask])) * 100
    return {
        "y_true": y_true_orig, "y_pred": y_pred_orig,
        "rmse": round(float(rmse), 4),
        "mae":  round(float(mae),  4),
        "mape": round(float(mape), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "Simple LSTM":          "#38bdf8",
    "CNN Forecaster":       "#a78bfa",
    "CNN–LSTM Hybrid":      "#34d399",
    "ResNet–LSTM (GADF)":   "#fb923c",
    "actual":               "#f1f5f9",
    "grid":                 "#1e293b",
    "bg":                   "#0f172a",
}

LAYOUT_BASE = dict(
    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
    font=dict(color="#94a3b8", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#1e293b", zeroline=False),
    yaxis=dict(gridcolor="#1e293b", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#334155", borderwidth=1),
    margin=dict(l=40, r=20, t=50, b=40),
)


def plot_ohlc(df, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#34d399", decreasing_line_color="#f87171", name="OHLC"
    ), row=1, col=1)
    colors = ["#34d399" if c >= o else "#f87171"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                         marker_color=colors, opacity=0.6, name="Volume"), row=2, col=1)
    fig.update_layout(**LAYOUT_BASE,
                      title=dict(text=f"{ticker} · Hourly OHLC",
                                 font=dict(color="white", size=16)),
                      xaxis_rangeslider_visible=False, height=480)
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    return fig


def plot_predictions(results, model_names, n_show=200):
    fig = go.Figure()
    first = results[model_names[0]]
    x = np.arange(len(first["y_true"]))[-n_show:]
    fig.add_trace(go.Scatter(x=x, y=first["y_true"][-n_show:], mode="lines",
                             name="Actual", line=dict(color="#f1f5f9", width=2)))
    for name in model_names:
        r = results[name]
        fig.add_trace(go.Scatter(x=x, y=r["y_pred"][-n_show:], mode="lines", name=name,
                                 line=dict(color=PALETTE.get(name, "#cbd5e1"),
                                           width=1.5, dash="dot")))
    fig.update_layout(**LAYOUT_BASE,
                      title=dict(text="Predicted vs Actual Close Price",
                                 font=dict(color="white", size=16)),
                      xaxis_title="Test Step", yaxis_title="Price (USD)", height=420)
    return fig


def plot_loss_curves(histories):
    n = len(histories)
    fig = make_subplots(rows=1, cols=n, subplot_titles=list(histories.keys()))
    for idx, (name, h) in enumerate(histories.items(), 1):
        color  = PALETTE.get(name, "#94a3b8")
        epochs = list(range(1, len(h["train_loss"]) + 1))
        fig.add_trace(go.Scatter(x=epochs, y=h["train_loss"], name="Train",
                                 legendgroup=name, showlegend=(idx == 1),
                                 line=dict(color=color, width=2)), row=1, col=idx)
        fig.add_trace(go.Scatter(x=epochs, y=h["val_loss"], name="Val",
                                 legendgroup=name, showlegend=(idx == 1),
                                 line=dict(color=color, width=2, dash="dash")), row=1, col=idx)
        fig.add_vline(x=h["best_epoch"], line_width=1, line_dash="dot",
                      line_color="#fbbf24", row=1, col=idx)
    fig.update_layout(**LAYOUT_BASE,
                      title=dict(text="Training & Validation Loss",
                                 font=dict(color="white", size=16)), height=320)
    fig.update_annotations(font_color="#94a3b8")
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    return fig


def plot_metrics_comparison(results):
    names  = list(results.keys())
    colors = [PALETTE.get(n, "#94a3b8") for n in names]
    fig    = make_subplots(rows=1, cols=3, subplot_titles=["RMSE (↓)", "MAE (↓)", "MAPE % (↓)"])
    for col, key in enumerate(["rmse", "mae", "mape"], 1):
        vals = [results[n][key] for n in names]
        fig.add_trace(go.Bar(x=names, y=vals, marker_color=colors,
                             text=[f"{v:.3f}" for v in vals], textposition="outside",
                             textfont=dict(color="white", size=10), showlegend=False),
                      row=1, col=col)
    fig.update_layout(**LAYOUT_BASE,
                      title=dict(text="Model Performance Comparison",
                                 font=dict(color="white", size=16)), height=360)
    fig.update_annotations(font_color="#94a3b8")
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=9), gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    return fig


def plot_residuals(results, model_name):
    r         = results[model_name]
    residuals = r["y_true"] - r["y_pred"]
    color     = PALETTE.get(model_name, "#38bdf8")
    fig       = make_subplots(rows=1, cols=2,
                               subplot_titles=["Residuals Over Time", "Residual Distribution"])
    fig.add_trace(go.Scatter(y=residuals, mode="lines",
                             line=dict(color=color, width=1), name="Residual"), row=1, col=1)
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#fbbf24", row=1, col=1)
    fig.add_trace(go.Histogram(x=residuals, nbinsx=40, marker_color=color,
                               opacity=0.8, name="Distribution"), row=1, col=2)
    fig.update_layout(**LAYOUT_BASE,
                      title=dict(text=f"Residual Analysis · {model_name}",
                                 font=dict(color="white", size=16)),
                      height=340, showlegend=False)
    fig.update_annotations(font_color="#94a3b8")
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Deep Latent Alpha",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155; border-radius: 12px;
    padding: 18px 22px; text-align: center;
  }
  .metric-card h2 { font-size: 2rem; margin: 0; font-weight: 700; }
  .metric-card p  { color: #64748b; font-size: 0.8rem; margin: 4px 0 0;
                    text-transform: uppercase; letter-spacing: .05em; }
  .hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #1e3a5f; border-radius: 16px;
    padding: 32px 40px; margin-bottom: 24px;
  }
  .hero h1 { color: #f1f5f9; font-size: 2.2rem; font-weight: 700; margin: 0 0 8px; }
  .hero p  { color: #94a3b8; font-size: 1rem; margin: 0; }
  .badge {
    display: inline-block; background: #1e3a5f; color: #38bdf8;
    font-size: 0.72rem; font-weight: 600; padding: 3px 10px;
    border-radius: 99px; margin: 2px; border: 1px solid #2563eb44;
  }
  .model-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 14px; margin-bottom: 8px;
  }
  h2, h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
for key in ["trained_results", "trained_histories", "df_raw", "scaler",
            "X_test", "y_test", "feature_cols"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "Simple LSTM":        "#38bdf8",
    "CNN Forecaster":     "#a78bfa",
    "CNN–LSTM Hybrid":    "#34d399",
    "ResNet–LSTM (GADF)": "#fb923c",
}

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()
    ticker  = st.text_input("Ticker Symbol", value="AAPL").upper()
    period  = st.selectbox("Data Period", ["6mo", "1y", "2y"], index=1)
    seq_len = st.slider("Sequence Length (hours)", 12, 72, 24, step=4)
    st.divider()
    st.markdown("**Select Models to Train**")
    selected_models = [
        name for name in MODEL_COLORS
        if st.checkbox(name, value=(name != "ResNet–LSTM (GADF)"), key=f"cb_{name}")
    ]
    st.divider()
    st.markdown("**Training Parameters**")
    epochs     = st.slider("Max Epochs",   10, 60, 25, step=5)
    batch_size = st.slider("Batch Size",   32, 256, 64, step=32)
    lr         = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 2e-3], value=1e-3)
    patience   = st.slider("Early Stop Patience", 3, 15, 7)
    st.divider()
    run_btn = st.button("🚀  Train & Evaluate", use_container_width=True, type="primary")

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📈 Deep Latent Alpha Inference</h1>
  <p>Financial time series forecasting · LSTM · CNN · CNN–LSTM · ResNet–LSTM with GADF encoding</p>
  <br>
  <span class="badge">PyTorch</span>
  <span class="badge">yfinance</span>
  <span class="badge">GADF Encoding</span>
  <span class="badge">ResNet-18</span>
  <span class="badge">Streamlit</span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_data, tab_gadf, tab_train, tab_compare, tab_residuals, tab_about = st.tabs([
    "📊 Market Data", "🖼️ GADF Encoding", "🏋️ Training",
    "🏆 Model Comparison", "🔬 Residual Analysis", "ℹ️ About",
])

# ── TAB 1 · Market Data ──────────────────────────────────────────────────────
with tab_data:
    st.markdown("### Live Market Data")
    if st.button("🔄 Fetch Data", use_container_width=False):
        with st.spinner(f"Fetching {ticker} hourly data…"):
            try:
                df = fetch_ohlc_data(ticker, period=period)
                df = add_technical_indicators(df)
                st.session_state.df_raw = df
            except Exception as e:
                st.error(f"Data fetch failed: {e}")

    df = st.session_state.df_raw
    if df is not None:
        latest  = float(df["Close"].iloc[-1])
        prev    = float(df["Close"].iloc[-2])
        pct_chg = (latest - prev) / prev * 100
        vol_avg = float(df["Volume"].tail(24).mean())
        rsi_now = float(df["RSI"].iloc[-1])
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, clr in [
            (c1, "Latest Close",  f"${latest:.2f}",      "#38bdf8"),
            (c2, "1-Bar Change",  f"{pct_chg:+.2f}%",    "#34d399" if pct_chg >= 0 else "#f87171"),
            (c3, "RSI (14)",      f"{rsi_now:.1f}",       "#a78bfa"),
            (c4, "24h Avg Vol",   f"{vol_avg/1e6:.1f}M",  "#fb923c"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card"><h2 style="color:{clr}">{val}</h2>'
                    f'<p>{label}</p></div>', unsafe_allow_html=True)
        st.plotly_chart(plot_ohlc(df, ticker), use_container_width=True)
        with st.expander("📋 Raw Data Preview"):
            st.dataframe(df.tail(100), height=300)
    else:
        st.info("Click **Fetch Data** to load live market data.")

# ── TAB 2 · GADF Encoding ────────────────────────────────────────────────────
with tab_gadf:
    st.markdown("### Gramian Angular Difference Field (GADF)")
    st.markdown("GADF encodes a 1-D time series into a 2-D image by projecting values onto a "
                "polar coordinate system and computing pairwise angular differences. This captures "
                "**temporal correlations as spatial patterns**, enabling CNNs to detect structure "
                "invisible in the raw signal.")
    df = st.session_state.df_raw
    if df is not None:
        col_w, col_idx = st.columns([1, 2])
        with col_w:
            window = st.slider("Window length", 12, 60, seq_len, step=4)
        with col_idx:
            max_idx = len(df) - window - 1
            start   = st.slider("Start index", 0, max(1, max_idx), max_idx // 2)
        series = df["Close"].values[start: start + window]
        mn, mx = series.min(), series.max()
        series_norm = (series - mn) / (mx - mn + 1e-9)
        st.pyplot(plot_gadf_sample(series_norm, f"{ticker} GADF · {window}-bar window"),
                  use_container_width=True)
        st.divider()
        cols = st.columns(3)
        for col, (title, desc) in zip(cols, [
            ("1️⃣ Rescale",       "Normalise the series to [-1, 1] to fit the cosine domain."),
            ("2️⃣ Angular Encode","Map each value xᵢ → φᵢ = arccos(xᵢ) preserving time order."),
            ("3️⃣ Gramian Matrix","Compute GADF[i,j] = sin(φᵢ − φⱼ) for all pairwise steps."),
        ]):
            with col:
                st.markdown(f"**{title}**")
                st.caption(desc)
    else:
        st.info("Load market data in the **Market Data** tab first.")

# ── TAB 3 · Training ─────────────────────────────────────────────────────────
with tab_train:
    st.markdown("### Model Training")
    if not run_btn and st.session_state.trained_results is None:
        st.info("Configure settings in the sidebar and click **🚀 Train & Evaluate**.")

    if run_btn:
        if not selected_models:
            st.error("Select at least one model.")
            st.stop()

        with st.status("Preparing data…", expanded=True) as status:
            st.write("📥 Fetching live data…")
            df = fetch_ohlc_data(ticker, period=period)
            df = add_technical_indicators(df)
            st.session_state.df_raw = df

            feature_cols = ["Open", "High", "Low", "Close", "Volume"]
            scaled, scaler = normalize_data(df, feature_cols)
            X, y = create_sequences(scaled, seq_len=seq_len, target_col_idx=3)
            X_tr, X_te, y_tr, y_te = train_test_split_ts(X, y)
            X_tr, X_val, y_tr, y_val = train_test_split_ts(X_tr, y_tr, split=0.85)

            st.session_state.scaler       = scaler
            st.session_state.X_test       = X_te
            st.session_state.y_test       = y_te
            st.session_state.feature_cols = feature_cols

            need_gadf = "ResNet–LSTM (GADF)" in selected_models
            if need_gadf:
                st.write("🖼️ Generating GADF encodings…")
                X_img_tr  = series_to_gadf_batch(X_tr)
                X_img_val = series_to_gadf_batch(X_val)
                X_img_te  = series_to_gadf_batch(X_te)

            st.write(f"✅ {len(X_tr):,} train / {len(X_val):,} val / {len(X_te):,} test samples")
            status.update(label="Data ready", state="complete")

        results, histories = {}, {}
        n_feats = len(feature_cols)

        for model_name in selected_models:
            st.markdown(f"---\n#### Training · {model_name}")
            prog_bar  = st.progress(0)
            loss_disp = st.empty()
            is_img    = model_name == "ResNet–LSTM (GADF)"

            ModelClass = MODEL_REGISTRY[model_name]
            if model_name == "Simple LSTM":
                model = ModelClass(input_size=n_feats)
            elif model_name == "CNN Forecaster":
                model = ModelClass(input_size=n_feats, seq_len=seq_len)
            elif model_name == "CNN–LSTM Hybrid":
                model = ModelClass(input_size=n_feats)
            else:
                model = ModelClass()

            def make_cb(_bar, _disp, _total):
                def cb(ep, tl, vl):
                    _bar.progress(ep / _total)
                    _disp.markdown(
                        f"Epoch **{ep}/{_total}** · Train: `{tl:.5f}` · Val: `{vl:.5f}`")
                return cb

            Xtr_ = X_img_tr  if is_img else X_tr
            Xv_  = X_img_val if is_img else X_val
            Xte_ = X_img_te  if is_img else X_te

            history = train_model(
                model, Xtr_, y_tr, Xv_, y_val,
                epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
                progress_callback=make_cb(prog_bar, loss_disp, epochs)
            )
            histories[model_name] = history
            prog_bar.progress(1.0)

            result = evaluate_model(model, Xte_, y_te, scaler,
                                    close_col_idx=3, n_features=n_feats)
            results[model_name] = result

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("RMSE", f"${result['rmse']:.3f}")
            with c2: st.metric("MAE",  f"${result['mae']:.3f}")
            with c3: st.metric("MAPE", f"{result['mape']:.2f}%")

        st.session_state.trained_results   = results
        st.session_state.trained_histories = histories
        st.success("✅ All models trained! Explore results in the other tabs.")

    if st.session_state.trained_histories:
        st.divider()
        st.markdown("### Loss Curves")
        st.plotly_chart(plot_loss_curves(st.session_state.trained_histories),
                        use_container_width=True)

# ── TAB 4 · Comparison ───────────────────────────────────────────────────────
with tab_compare:
    st.markdown("### Model Comparison Dashboard")
    results = st.session_state.trained_results
    if results is None:
        st.info("Train models first using the sidebar.")
    else:
        model_names = list(results.keys())
        st.markdown("#### 🏅 Leaderboard")
        lb_df = pd.DataFrame({
            "Model":    model_names,
            "RMSE ($)": [results[n]["rmse"] for n in model_names],
            "MAE ($)":  [results[n]["mae"]  for n in model_names],
            "MAPE (%)": [results[n]["mape"] for n in model_names],
        }).sort_values("RMSE ($)")
        lb_df.insert(0, "Rank", ["🥇", "🥈", "🥉", "4️⃣"][:len(lb_df)])
        st.dataframe(lb_df.set_index("Rank"), use_container_width=True)
        st.divider()
        st.plotly_chart(plot_metrics_comparison(results), use_container_width=True)
        st.divider()
        n_show = st.slider("Show last N test steps", 50, 500, 200, step=50)
        st.plotly_chart(plot_predictions(results, model_names, n_show=n_show),
                        use_container_width=True)

# ── TAB 5 · Residuals ────────────────────────────────────────────────────────
with tab_residuals:
    st.markdown("### Residual Analysis")
    results = st.session_state.trained_results
    if results is None:
        st.info("Train models first using the sidebar.")
    else:
        selected = st.selectbox("Select model", list(results.keys()))
        st.plotly_chart(plot_residuals(results, selected), use_container_width=True)
        r     = results[selected]
        resid = r["y_true"] - r["y_pred"]
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Mean Residual",    f"${resid.mean():.3f}")
        with c2: st.metric("Std Residual",     f"${resid.std():.3f}")
        with c3: st.metric("Max Abs Residual", f"${np.abs(resid).max():.3f}")

# ── TAB 6 · About ────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("### About This Project")
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("""
**Deep Latent Alpha Inference** investigates whether encoding financial time series
as Gramian Angular Difference Field (GADF) images unlocks spatial patterns that purely
sequential models miss.

#### Architecture Overview

| Model | Input | Core Idea |
|---|---|---|
| Simple LSTM | Raw OHLC | Baseline gated-memory sequential model |
| CNN Forecaster | Raw OHLC | Multi-scale 1-D convolutions for motif detection |
| CNN–LSTM Hybrid | Raw OHLC | CNN extracts features; LSTM models their evolution |
| ResNet–LSTM | GADF Images | Spatial pattern extraction → temporal aggregation |

#### Tech Stack
`PyTorch` · `yfinance` · `torchvision` · `scikit-learn` · `Plotly` · `Streamlit`
        """)
    with col_right:
        for name, desc in MODEL_DESCRIPTIONS.items():
            color = MODEL_COLORS.get(name, "#94a3b8")
            st.markdown(
                f'<div class="model-card" style="border-left: 3px solid {color}; padding:14px">'
                f'<strong style="color:{color}">{name}</strong><br>'
                f'<small style="color:#94a3b8">{desc}</small></div>',
                unsafe_allow_html=True)
    st.divider()
    st.caption("Data sourced live from Yahoo Finance via yfinance · Built with PyTorch & Streamlit")
