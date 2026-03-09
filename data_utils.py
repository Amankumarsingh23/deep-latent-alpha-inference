"""
Data utilities for financial time series preprocessing.
Downloads hourly OHLC data via yfinance and prepares it for model input.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def fetch_ohlc_data(ticker: str = "AAPL", period: str = "2y", interval: str = "1h") -> pd.DataFrame:
    """Fetch hourly OHLC data from Yahoo Finance."""
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, Bollinger Bands, and rolling statistics."""
    df = df.copy()

    # Moving averages
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_10"] = df["Close"].ewm(span=10).mean()

    # Bollinger Bands
    rolling_std = df["Close"].rolling(20).std()
    df["BB_upper"] = df["SMA_20"] + 2 * rolling_std
    df["BB_lower"] = df["SMA_20"] - 2 * rolling_std
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    # Log returns
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df.dropna(inplace=True)
    return df


def normalize_data(df: pd.DataFrame, feature_cols: list) -> tuple[np.ndarray, MinMaxScaler]:
    """Normalize selected features to [0, 1] range."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols].values)
    return scaled, scaler


def create_sequences(data: np.ndarray, seq_len: int = 24, target_col_idx: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding window to create (X, y) sequences.
    Default target_col_idx=3 → 'Close' price column.
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, target_col_idx])
    return np.array(X), np.array(y)


def train_test_split_ts(X: np.ndarray, y: np.ndarray, split: float = 0.8) -> tuple:
    """Time-series aware train/test split (no shuffling)."""
    n = int(len(X) * split)
    return X[:n], X[n:], y[:n], y[n:]
