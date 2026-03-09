"""
Deep Latent Alpha Inference · Streamlit Dashboard
═══════════════════════════════════════════════════
Financial time series forecasting using LSTM, CNN, CNN–LSTM, and ResNet–LSTM
architectures with GADF image encoding on hourly OHLC data.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
import pandas as pd
import streamlit as st
import torch

from utils.data_utils   import (fetch_ohlc_data, add_technical_indicators,
                                 normalize_data, create_sequences, train_test_split_ts)
from utils.gadf_utils   import series_to_gadf_batch, plot_gadf_sample
from utils.plot_utils   import (plot_ohlc, plot_predictions, plot_loss_curves,
                                 plot_metrics_comparison, plot_residuals)
from models.architectures import MODEL_REGISTRY, MODEL_DESCRIPTIONS
from src.trainer          import train_model, evaluate_model


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Latent Alpha",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main  { background: #0f172a; }
  section[data-testid="stSidebar"] { background: #0f1729 !important; }

  .metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
  }
  .metric-card h2 { color: #38bdf8; font-size: 2rem; margin: 0; font-weight: 700; }
  .metric-card p  { color: #64748b;  font-size: 0.8rem; margin: 4px 0 0; text-transform: uppercase; letter-spacing: .05em; }

  .hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
  }
  .hero h1 { color: #f1f5f9; font-size: 2.2rem; font-weight: 700; margin: 0 0 8px; }
  .hero p  { color: #94a3b8; font-size: 1rem; margin: 0; }
  .badge {
    display: inline-block;
    background: #1e3a5f;
    color: #38bdf8;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 99px;
    margin: 2px;
    border: 1px solid #2563eb44;
  }

  .model-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 8px;
    border-left: 3px solid #38bdf8;
  }

  div[data-testid="stTabs"] button {
    font-weight: 600;
    color: #64748b;
  }
  div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8;
    border-bottom: 2px solid #38bdf8;
  }

  .stProgress > div > div { background: linear-gradient(90deg, #2563eb, #38bdf8); }
  h2, h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ───────────────────────────────────────────────────
for key in ["trained_results", "trained_histories", "df_raw", "scaler",
            "X_test", "y_test", "feature_cols", "gadf_sample"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()

    ticker  = st.text_input("Ticker Symbol", value="AAPL").upper()
    period  = st.selectbox("Data Period", ["6mo", "1y", "2y"], index=1)
    seq_len = st.slider("Sequence Length (hours)", 12, 72, 24, step=4)
    st.divider()

    st.markdown("**Select Models to Train**")
    selected_models = []
    model_colors = {
        "Simple LSTM":        "#38bdf8",
        "CNN Forecaster":     "#a78bfa",
        "CNN–LSTM Hybrid":    "#34d399",
        "ResNet–LSTM (GADF)": "#fb923c",
    }
    for name, color in model_colors.items():
        if st.checkbox(name, value=(name != "ResNet–LSTM (GADF)"),
                       key=f"cb_{name}"):
            selected_models.append(name)

    st.divider()
    st.markdown("**Training Parameters**")
    epochs     = st.slider("Max Epochs",   10, 60, 25, step=5)
    batch_size = st.slider("Batch Size",   32, 256, 64, step=32)
    lr         = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 2e-3], value=1e-3)
    patience   = st.slider("Early Stop Patience", 3, 15, 7)

    st.divider()
    run_btn = st.button("🚀  Train & Evaluate", use_container_width=True, type="primary")


# ── Hero header ──────────────────────────────────────────────────────────────
st.markdown(f"""
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
    "📊 Market Data",
    "🖼️ GADF Encoding",
    "🏋️ Training",
    "🏆 Model Comparison",
    "🔬 Residual Analysis",
    "ℹ️ About",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · Market Data
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### Live Market Data")

    col_load, _ = st.columns([1, 3])
    with col_load:
        load_btn = st.button("🔄 Fetch Data", use_container_width=True)

    if load_btn or st.session_state.df_raw is not None:
        with st.spinner(f"Fetching {ticker} hourly data…"):
            try:
                df = fetch_ohlc_data(ticker, period=period)
                df = add_technical_indicators(df)
                st.session_state.df_raw = df
            except Exception as e:
                st.error(f"Data fetch failed: {e}")
                df = None

    df = st.session_state.df_raw
    if df is not None:
        # KPI cards
        latest  = df["Close"].iloc[-1]
        prev    = df["Close"].iloc[-2]
        pct_chg = (latest - prev) / prev * 100
        vol_avg = df["Volume"].tail(24).mean()
        rsi_now = df["RSI"].iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in [
            (c1, "Latest Close", f"${latest:.2f}"),
            (c2, "1-Bar Change", f"{pct_chg:+.2f}%"),
            (c3, "RSI (14)",     f"{rsi_now:.1f}"),
            (c4, "24h Avg Vol",  f"{vol_avg/1e6:.1f}M"),
        ]:
            with col:
                color = "#34d399" if "+" in str(val) or label == "Latest Close" else "#f87171"
                st.markdown(f"""
                <div class="metric-card">
                  <h2 style="color:{color}">{val}</h2>
                  <p>{label}</p>
                </div>""", unsafe_allow_html=True)

        st.plotly_chart(plot_ohlc(df, ticker), use_container_width=True)

        with st.expander("📋 Raw Data Preview"):
            st.dataframe(
                df.tail(100).style.background_gradient(subset=["Close"], cmap="Blues"),
                height=300,
            )
    else:
        st.info("Click **Fetch Data** to load live market data from Yahoo Finance.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · GADF Encoding
# ══════════════════════════════════════════════════════════════════════════════
with tab_gadf:
    st.markdown("### Gramian Angular Difference Field (GADF)")
    st.markdown("""
    GADF encodes a 1-D time series into a 2-D image by projecting values onto a polar coordinate
    system and computing the pairwise angular differences. This captures **temporal correlations
    as spatial patterns**, enabling CNNs to extract structure invisible in the raw signal.
    """)

    df = st.session_state.df_raw
    if df is not None:
        col_w, col_idx = st.columns([1, 2])
        with col_w:
            window = st.slider("Window length", 12, 60, seq_len, step=4)
        with col_idx:
            max_idx = len(df) - window - 1
            start   = st.slider("Start index", 0, max(1, max_idx), max_idx // 2)

        series = df["Close"].values[start : start + window]
        # Normalise to [0,1]
        mn, mx = series.min(), series.max()
        series_norm = (series - mn) / (mx - mn + 1e-9)

        fig_gadf = plot_gadf_sample(series_norm,
                                    title=f"{ticker} GADF · {window}-bar window")
        st.pyplot(fig_gadf, use_container_width=True)

        st.divider()
        st.markdown("#### How GADF Works")
        cols = st.columns(3)
        steps = [
            ("1️⃣ Rescale", "Normalise the series to [-1, 1] to fit the cosine domain."),
            ("2️⃣ Angular Encode", "Map each value xᵢ → φᵢ = arccos(xᵢ) preserving order."),
            ("3️⃣ Gramian Matrix", "Compute GADF[i,j] = sin(φᵢ − φⱼ) to capture pairwise temporal correlations."),
        ]
        for col, (title, desc) in zip(cols, steps):
            with col:
                st.markdown(f"**{title}**")
                st.caption(desc)
    else:
        st.info("Load market data in the **Market Data** tab first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · Training
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown("### Model Training")

    if not run_btn and st.session_state.trained_results is None:
        st.info("Configure your settings in the sidebar and click **🚀 Train & Evaluate**.")

    if run_btn:
        if not selected_models:
            st.error("Select at least one model in the sidebar.")
            st.stop()

        # ── 1. Data prep ──────────────────────────────────────────────────
        status = st.status("Preparing data…", expanded=True)
        with status:
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

            # Pre-compute GADF images for image-based models
            need_gadf = "ResNet–LSTM (GADF)" in selected_models
            if need_gadf:
                st.write("🖼️ Generating GADF image encodings…")
                X_img_tr  = series_to_gadf_batch(X_tr,  feature_col=3)
                X_img_val = series_to_gadf_batch(X_val, feature_col=3)
                X_img_te  = series_to_gadf_batch(X_te,  feature_col=3)

            st.write(f"✅ Data ready — {len(X_tr):,} train / {len(X_val):,} val / {len(X_te):,} test samples")
        status.update(label="Data ready", state="complete")

        # ── 2. Train each model ───────────────────────────────────────────
        results   = {}
        histories = {}
        n_feats   = len(feature_cols)

        for model_name in selected_models:
            st.markdown(f"---\n#### Training · {model_name}")
            prog_bar  = st.progress(0)
            loss_disp = st.empty()

            is_image_model = model_name == "ResNet–LSTM (GADF)"

            # Build model
            ModelClass = MODEL_REGISTRY[model_name]
            if model_name == "Simple LSTM":
                model = ModelClass(input_size=n_feats)
            elif model_name == "CNN Forecaster":
                model = ModelClass(input_size=n_feats, seq_len=seq_len)
            elif model_name == "CNN–LSTM Hybrid":
                model = ModelClass(input_size=n_feats)
            else:
                model = ModelClass()

            # Progress callback
            def make_callback(bar, disp, total):
                def cb(ep, tl, vl):
                    bar.progress(ep / total)
                    disp.markdown(
                        f"Epoch **{ep}/{total}** · "
                        f"Train Loss: `{tl:.5f}` · "
                        f"Val Loss: `{vl:.5f}`"
                    )
                return cb

            cb = make_callback(prog_bar, loss_disp, epochs)

            # Train
            Xtr_ = X_img_tr  if is_image_model else X_tr
            Xv_  = X_img_val if is_image_model else X_val
            Xte_ = X_img_te  if is_image_model else X_te

            history = train_model(
                model, Xtr_, y_tr, Xv_, y_val,
                epochs=epochs, batch_size=batch_size, lr=lr,
                patience=patience, progress_callback=cb,
            )
            histories[model_name] = history
            prog_bar.progress(1.0)

            # Evaluate
            result = evaluate_model(
                model, Xte_, y_te, scaler,
                close_col_idx=3, n_features=n_feats,
            )
            results[model_name] = result

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("RMSE", f"${result['rmse']:.3f}")
            with c2:
                st.metric("MAE",  f"${result['mae']:.3f}")
            with c3:
                st.metric("MAPE", f"{result['mape']:.2f}%")

        st.session_state.trained_results   = results
        st.session_state.trained_histories = histories
        st.success("✅ All models trained! Navigate the tabs to explore results.")

    # Show loss curves if already trained
    if st.session_state.trained_histories:
        st.divider()
        st.markdown("### Loss Curves")
        st.plotly_chart(
            plot_loss_curves(st.session_state.trained_histories),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### Model Comparison Dashboard")

    results = st.session_state.trained_results
    if results is None:
        st.info("Train models first using the sidebar.")
    else:
        model_names = list(results.keys())

        # Leaderboard
        st.markdown("#### 🏅 Leaderboard")
        lb_data = {
            "Model":    model_names,
            "RMSE ($)": [results[n]["rmse"] for n in model_names],
            "MAE ($)":  [results[n]["mae"]  for n in model_names],
            "MAPE (%)": [results[n]["mape"] for n in model_names],
        }
        lb_df = pd.DataFrame(lb_data).sort_values("RMSE ($)")
        lb_df.insert(0, "Rank", ["🥇","🥈","🥉","4️⃣"][:len(lb_df)])
        st.dataframe(lb_df.set_index("Rank"), use_container_width=True)

        st.divider()
        st.plotly_chart(plot_metrics_comparison(results), use_container_width=True)

        st.divider()
        st.markdown("#### Predicted vs Actual")
        n_show = st.slider("Show last N test steps", 50, 500, 200, step=50, key="pred_n")
        st.plotly_chart(
            plot_predictions(results, model_names, n_show=n_show),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 · Residual Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_residuals:
    st.markdown("### Residual Analysis")

    results = st.session_state.trained_results
    if results is None:
        st.info("Train models first using the sidebar.")
    else:
        selected = st.selectbox("Select model", list(results.keys()))
        st.plotly_chart(plot_residuals(results, selected), use_container_width=True)

        r = results[selected]
        resid = r["y_true"] - r["y_pred"]
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Mean Residual",   f"${resid.mean():.3f}")
        with col2: st.metric("Std Residual",    f"${resid.std():.3f}")
        with col3: st.metric("Max Abs Residual",f"${np.abs(resid).max():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 · About
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### About This Project")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("""
        **Deep Latent Alpha Inference** explores whether transforming financial time series
        into image representations (GADF/GASF) can unlock spatial pattern recognition
        that purely sequential models miss.

        #### Architecture Overview

        | Model | Input | Core Idea |
        |---|---|---|
        | Simple LSTM | Raw OHLC | Baseline; gated memory for sequential data |
        | CNN Forecaster | Raw OHLC | Local motif detection via multi-scale convolutions |
        | CNN–LSTM Hybrid | Raw OHLC | CNN extracts features; LSTM models their evolution |
        | ResNet–LSTM | GADF Images | Spatial pattern extraction → temporal aggregation |

        #### Key Findings from Literature
        - GADF encoding preserves temporal ordering in the image diagonal
        - Pretrained CNNs (ResNet) generalise well as feature extractors even on financial imagery
        - Hybrid CNN–LSTM consistently outperforms either alone on non-stationary series

        #### Tech Stack
        `PyTorch` · `yfinance` · `torchvision` · `scikit-learn` · `Plotly` · `Streamlit`
        """)

    with col_right:
        for name, desc in MODEL_DESCRIPTIONS.items():
            color = {
                "Simple LSTM":        "#38bdf8",
                "CNN Forecaster":     "#a78bfa",
                "CNN–LSTM Hybrid":    "#34d399",
                "ResNet–LSTM (GADF)": "#fb923c",
            }.get(name, "#94a3b8")
            st.markdown(f"""
            <div class="model-card" style="border-left-color:{color}">
              <strong style="color:{color}">{name}</strong><br>
              <small style="color:#94a3b8">{desc}</small>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.caption(
        "Built as part of the Deep Latent Alpha Inference project · "
        "Data sourced live from Yahoo Finance via yfinance"
    )
