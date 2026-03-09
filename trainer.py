"""
Training engine for all model architectures.
Handles both sequence-input and image-input models.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 7,
    progress_callback=None,
) -> dict:
    """
    Train a model and return training history.

    Args:
        model          : nn.Module to train
        X_train/y_train: training sequences
        X_val/y_val    : validation sequences
        epochs         : max training epochs
        batch_size     : batch size
        lr             : initial learning rate
        patience       : early stopping patience
        progress_callback: optional callable(epoch, train_loss, val_loss)

    Returns:
        dict with keys: train_loss, val_loss, best_epoch
    """
    device = get_device()
    model = model.to(device)

    train_loader = _make_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader   = _make_dataloader(X_val,   y_val,   batch_size, shuffle=False)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=3, factor=0.5)
    criterion = nn.HuberLoss(delta=0.5)

    best_val   = float("inf")
    best_state = None
    no_improve = 0
    history    = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        t_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            t_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        v_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                v_losses.append(criterion(pred, yb).item())

        t_loss = np.mean(t_losses)
        v_loss = np.mean(v_losses)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        scheduler.step(v_loss)

        if progress_callback:
            progress_callback(epoch, t_loss, v_loss)

        # Early stopping
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


def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
    close_col_idx: int = 3,
    n_features: int = 5,
) -> dict:
    """
    Run inference and compute RMSE, MAE, MAPE on original price scale.

    Returns:
        dict with y_true, y_pred (both on original scale), rmse, mae, mape
    """
    device = get_device()
    model.eval().to(device)

    X_t  = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()

    # Inverse-transform predictions to original price scale
    def _inverse(vals, col_idx):
        dummy = np.zeros((len(vals), n_features))
        dummy[:, col_idx] = vals
        return scaler.inverse_transform(dummy)[:, col_idx]

    y_pred_orig = _inverse(preds,   close_col_idx)
    y_true_orig = _inverse(y_test,  close_col_idx)

    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae  = mean_absolute_error(y_true_orig, y_pred_orig)
    mask = y_true_orig != 0
    mape = np.mean(np.abs((y_true_orig[mask] - y_pred_orig[mask]) / y_true_orig[mask])) * 100

    return {
        "y_true": y_true_orig,
        "y_pred": y_pred_orig,
        "rmse":   round(float(rmse), 4),
        "mae":    round(float(mae),  4),
        "mape":   round(float(mape), 4),
    }
