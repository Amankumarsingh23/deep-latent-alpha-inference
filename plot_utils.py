"""
Plotting utilities for the Streamlit dashboard.
All functions return Plotly figures for interactive rendering.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    "Simple LSTM":          "#38bdf8",   # sky blue
    "CNN Forecaster":       "#a78bfa",   # violet
    "CNN–LSTM Hybrid":      "#34d399",   # emerald
    "ResNet–LSTM (GADF)":   "#fb923c",   # orange
    "actual":               "#f1f5f9",   # near-white
    "grid":                 "#1e293b",
    "bg":                   "#0f172a",
    "paper":                "#0f172a",
}

LAYOUT_BASE = dict(
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor=PALETTE["bg"],
    font=dict(color="#94a3b8", family="Inter, sans-serif"),
    xaxis=dict(gridcolor=PALETTE["grid"], zeroline=False),
    yaxis=dict(gridcolor=PALETTE["grid"], zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#334155", borderwidth=1),
    margin=dict(l=40, r=20, t=50, b=40),
)


def plot_ohlc(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick chart with volume subplot."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#34d399", decreasing_line_color="#f87171",
        name="OHLC"
    ), row=1, col=1)

    colors = ["#34d399" if c >= o else "#f87171"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                         marker_color=colors, opacity=0.6, name="Volume"), row=2, col=1)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"{ticker} · Hourly OHLC", font=dict(color="white", size=16)),
        xaxis_rangeslider_visible=False,
        height=480,
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"])
    fig.update_yaxes(gridcolor=PALETTE["grid"])
    return fig


def plot_predictions(results: dict, model_names: list, n_show: int = 200) -> go.Figure:
    """Overlay actual vs predicted close prices for all selected models."""
    fig = go.Figure()

    # Actual — use first model's y_true
    first = results[model_names[0]]
    x = np.arange(len(first["y_true"]))[-n_show:]
    fig.add_trace(go.Scatter(
        x=x, y=first["y_true"][-n_show:],
        mode="lines", name="Actual",
        line=dict(color=PALETTE["actual"], width=2),
    ))

    for name in model_names:
        r = results[name]
        fig.add_trace(go.Scatter(
            x=x, y=r["y_pred"][-n_show:],
            mode="lines", name=name,
            line=dict(color=PALETTE.get(name, "#cbd5e1"), width=1.5, dash="dot"),
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Predicted vs Actual Close Price", font=dict(color="white", size=16)),
        xaxis_title="Test Step",
        yaxis_title="Price (USD)",
        height=420,
    )
    return fig


def plot_loss_curves(histories: dict) -> go.Figure:
    """Training / validation loss curves for all models."""
    fig = make_subplots(
        rows=1, cols=len(histories),
        subplot_titles=list(histories.keys()),
    )
    for idx, (name, h) in enumerate(histories.items(), 1):
        color = PALETTE.get(name, "#94a3b8")
        epochs = list(range(1, len(h["train_loss"]) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=h["train_loss"],
            name="Train", legendgroup=name,
            showlegend=(idx == 1),
            line=dict(color=color, width=2),
            mode="lines",
        ), row=1, col=idx)
        fig.add_trace(go.Scatter(
            x=epochs, y=h["val_loss"],
            name="Val", legendgroup=name,
            showlegend=(idx == 1),
            line=dict(color=color, width=2, dash="dash"),
            mode="lines",
        ), row=1, col=idx)
        # Mark best epoch
        be = h["best_epoch"]
        fig.add_vline(x=be, line_width=1, line_dash="dot",
                      line_color="#fbbf24", row=1, col=idx)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Training & Validation Loss", font=dict(color="white", size=16)),
        height=320,
    )
    fig.update_annotations(font_color="#94a3b8")
    fig.update_xaxes(gridcolor=PALETTE["grid"])
    fig.update_yaxes(gridcolor=PALETTE["grid"])
    return fig


def plot_metrics_comparison(results: dict) -> go.Figure:
    """Grouped bar chart comparing RMSE, MAE, MAPE across models."""
    names  = list(results.keys())
    rmses  = [results[n]["rmse"] for n in names]
    maes   = [results[n]["mae"]  for n in names]
    mapes  = [results[n]["mape"] for n in names]
    colors = [PALETTE.get(n, "#94a3b8") for n in names]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["RMSE (↓)", "MAE (↓)", "MAPE % (↓)"])
    for col, (metric, vals) in enumerate(
        [("RMSE", rmses), ("MAE", maes), ("MAPE %", mapes)], 1
    ):
        fig.add_trace(go.Bar(
            x=names, y=vals,
            marker_color=colors,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(color="white", size=10),
            showlegend=False,
        ), row=1, col=col)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Model Performance Comparison", font=dict(color="white", size=16)),
        height=360,
    )
    fig.update_annotations(font_color="#94a3b8")
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=9), gridcolor=PALETTE["grid"])
    fig.update_yaxes(gridcolor=PALETTE["grid"])
    return fig


def plot_residuals(results: dict, model_name: str) -> go.Figure:
    """Residual distribution (histogram + scatter) for a given model."""
    r = results[model_name]
    residuals = r["y_true"] - r["y_pred"]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Residuals Over Time", "Residual Distribution"])

    fig.add_trace(go.Scatter(
        y=residuals, mode="lines",
        line=dict(color=PALETTE.get(model_name, "#38bdf8"), width=1),
        name="Residual",
    ), row=1, col=1)
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#fbbf24",
                  row=1, col=1)

    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=40,
        marker_color=PALETTE.get(model_name, "#38bdf8"),
        opacity=0.8, name="Distribution",
    ), row=1, col=2)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"Residual Analysis · {model_name}",
                   font=dict(color="white", size=16)),
        height=340, showlegend=False,
    )
    fig.update_annotations(font_color="#94a3b8")
    fig.update_xaxes(gridcolor=PALETTE["grid"])
    fig.update_yaxes(gridcolor=PALETTE["grid"])
    return fig
