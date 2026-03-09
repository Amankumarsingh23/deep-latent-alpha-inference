"""
Gramian Angular Difference Field (GADF) encoding utilities.
Converts 1-D time series windows into 2-D image representations
that capture temporal correlations as spatial patterns.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def _rescale_to_minus1_1(series: np.ndarray) -> np.ndarray:
    """Rescale a series to [-1, 1]."""
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return np.zeros_like(series)
    return 2 * (series - s_min) / (s_max - s_min) - 1


def compute_gadf(series: np.ndarray) -> np.ndarray:
    """
    Compute Gramian Angular Difference Field matrix.

    Steps:
      1. Rescale series to [-1, 1]
      2. Encode as angular cosines: phi_i = arccos(x_i)
      3. GADF[i,j] = sin(phi_i - phi_j)
    """
    scaled = _rescale_to_minus1_1(series)
    scaled = np.clip(scaled, -1, 1)
    phi = np.arccos(scaled)
    # GADF: sin(phi_i - phi_j)
    gadf = np.sin(phi[:, None] - phi[None, :])
    return gadf


def compute_gasf(series: np.ndarray) -> np.ndarray:
    """
    Compute Gramian Angular Summation Field matrix.
    GASF[i,j] = cos(phi_i + phi_j)
    """
    scaled = _rescale_to_minus1_1(series)
    scaled = np.clip(scaled, -1, 1)
    phi = np.arccos(scaled)
    gasf = np.cos(phi[:, None] + phi[None, :])
    return gasf


def series_to_gadf_batch(sequences: np.ndarray, feature_col: int = 3) -> np.ndarray:
    """
    Convert a batch of (N, seq_len, n_features) sequences into
    GADF images of shape (N, seq_len, seq_len, 3).
    Uses Close price (feature_col=3) by default.
    Returns RGB images normalised to [0, 1].
    """
    N, seq_len, _ = sequences.shape
    images = np.zeros((N, seq_len, seq_len, 3), dtype=np.float32)

    for i in range(N):
        series = sequences[i, :, feature_col]
        gadf = compute_gadf(series)
        # Normalise to [0, 1] for image
        gadf_norm = (gadf + 1) / 2
        # Stack into 3 channels (RGB-like)
        images[i, :, :, 0] = gadf_norm          # R: GADF
        images[i, :, :, 1] = compute_gasf(series) * 0.5 + 0.5  # G: GASF
        images[i, :, :, 2] = gadf_norm ** 2     # B: squared GADF
    return images


def plot_gadf_sample(series: np.ndarray, title: str = "GADF Encoding") -> plt.Figure:
    """
    Render a publication-quality GADF visualisation for a given 1-D series.
    Returns a matplotlib Figure for Streamlit display.
    """
    gadf = compute_gadf(series)
    gasf = compute_gasf(series)

    # Custom colormap (deep blue → white → deep red)
    cmap = LinearSegmentedColormap.from_list(
        "gadf_cmap",
        ["#1a237e", "#5c6bc0", "#ffffff", "#ef5350", "#b71c1c"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#0f172a")
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=1.01)

    panel_data = [
        (series, "Price Series", None),
        (gadf,   "GADF Matrix",  cmap),
        (gasf,   "GASF Matrix",  cmap),
    ]

    for ax, (data, label, cm) in zip(axes, panel_data):
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
