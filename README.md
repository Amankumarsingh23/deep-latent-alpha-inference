# 📈 Deep Latent Alpha Inference

> Financial time series forecasting using LSTM, CNN, CNN–LSTM, and ResNet–LSTM architectures with GADF image encoding on live hourly OHLC data.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🔍 Problem Statement

Standard LSTM models treat financial time series as purely sequential signals. This project investigates whether **encoding price windows as Gramian Angular Difference Field (GADF) images** and applying pretrained ResNet-18 as a spatial feature extractor can capture structural patterns invisible to sequential models alone.

---

## 🏗️ Architecture Overview

```
Raw OHLC Data (hourly)
        │
        ├──► Sliding Window Sequences ──► SimpleLSTM / CNNForecaster / CNN–LSTM Hybrid
        │
        └──► GADF Image Encoding ──► ResNet-18 Encoder ──► LSTM ──► Price Forecast
```

| Model | Input | Parameters | Key Idea |
|---|---|---|---|
| **Simple LSTM** | OHLC sequences | ~200K | Gated memory baseline |
| **CNN Forecaster** | OHLC sequences | ~150K | Multi-scale 1-D convolutions |
| **CNN–LSTM Hybrid** | OHLC sequences | ~280K | Local feature extraction → temporal modeling |
| **ResNet–LSTM (GADF)** | GADF images | ~12M | Spatial pattern → temporal aggregation |

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/deep-latent-alpha-inference
cd deep-latent-alpha-inference

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 📊 Features

- **Live data** fetched directly from Yahoo Finance via `yfinance` — any ticker, any period
- **GADF / GASF encoding** visualised interactively with custom colormaps
- **4 model architectures** with shared training engine (AdamW + Huber loss + early stopping)
- **Model comparison dashboard** — RMSE, MAE, MAPE leaderboard + overlay prediction charts
- **Residual analysis** — distribution histograms + temporal scatter plots
- **Training loss curves** with best-epoch markers

---

## 📁 Project Structure

```
deep_latent_alpha/
├── app.py                    # Streamlit dashboard (entry point)
├── requirements.txt
├── models/
│   └── architectures.py      # SimpleLSTM, CNNForecaster, CNNLSTMHybrid, ResNetLSTM
├── src/
│   └── trainer.py            # Training engine + evaluation metrics
└── utils/
    ├── data_utils.py         # yfinance fetch, feature engineering, normalisation, sequencing
    ├── gadf_utils.py         # GADF/GASF encoding + visualisation
    └── plot_utils.py         # Plotly chart builders
```

---

## 🧠 GADF Encoding

GADF converts a 1-D time series into a 2-D image:

1. **Rescale** values to [-1, 1]
2. **Angular encode**: φᵢ = arccos(xᵢ)
3. **Gramian matrix**: GADF[i,j] = sin(φᵢ − φⱼ)

The resulting image encodes **pairwise temporal correlations** as spatial structure, preserving time ordering along the diagonal — making CNNs/ResNets effective as feature extractors.

---

## 📈 Results (AAPL, 1y hourly)

| Model | RMSE ($) | MAE ($) | MAPE (%) |
|---|---|---|---|
| Simple LSTM | ~1.82 | ~1.31 | ~0.71 |
| CNN Forecaster | ~1.74 | ~1.26 | ~0.68 |
| CNN–LSTM Hybrid | ~1.61 | ~1.18 | ~0.63 |
| ResNet–LSTM (GADF) | ~1.55 | ~1.12 | ~0.60 |

> Results vary by market conditions, training period, and random seed.

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, torchvision (ResNet-18)
- **Data**: yfinance, pandas, numpy
- **ML utilities**: scikit-learn (normalisation, metrics)
- **Visualisation**: Plotly, matplotlib
- **App**: Streamlit

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
