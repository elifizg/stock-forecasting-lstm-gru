# visualization.py
# HW4 – Sequence Modeling | CS515 Deep Learning
#
# Generates all figures required for the assignment report:
#
#   Part (b) – Train/val loss curves, predicted vs actual scatter (d=1),
#              per-horizon MSE bar chart (LSTM vs GRU).
#   Part (c) – Rolling vs exact return loss curve comparison,
#              training stability analysis (loss variance).
#   Part (d) – Confusion matrix, precision/recall/F1 table,
#              ROC curve with AUC score.
#   Bonus    – Per-ticker predicted vs actual price line plots.
#
# All figures are saved to the results/ directory as high-resolution PNGs.
#
# Usage:
#   python visualization.py --part b
#   python visualization.py --part c
#   python visualization.py --part d
#   python visualization.py --part all      # generate everything

import os
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_curve, auc,
)

import config
from dataset import get_dataloaders, download_data, split_data, Scaler
from models  import get_model
from ablation import ABLATION_CONFIGS, build_model

# ── Global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"   : "sans-serif",
    "font.size"     : 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"    : 150,
})
COLORS = {"lstm": "#2563EB", "gru": "#DC2626", "rolling": "#16A34A", "return": "#9333EA"}


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_fig(fig: plt.Figure, filename: str) -> None:
    """Save figure to results/ and close it."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved]  {path}")


def load_model(model_key: str, mode: str, device: torch.device):
    """
    Load best checkpoint. Returns (model, ckpt_meta) or (None, None).

    If the checkpoint was saved by ablation.py, it contains a 'cfg' key.
    The model is rebuilt from that cfg so architecture always matches weights.
    """
    path = os.path.join(config.CHECKPOINT_DIR, f"{model_key}_{mode}_best.pt")
    if not os.path.exists(path):
        print(f"[skip]   checkpoint not found: {path}")
        return None, None
    ckpt = torch.load(path, map_location=device)

    if "cfg" in ckpt:
        arch  = model_key.replace("bidir_", "")
        model = build_model(arch, mode, ckpt["cfg"]).to(device)
    else:
        model = get_model(model_key).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def load_history(model_key: str, mode: str) -> dict | None:
    """
    Load training history from JSON saved alongside the checkpoint.
    Returns None if the history file does not exist.
    """
    path = os.path.join(config.CHECKPOINT_DIR, f"{model_key}_{mode}_history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def collect_preds(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Collect all predictions and labels from a DataLoader."""
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device)).cpu().numpy()
            preds.append(out)
            labels.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(labels)


# ─────────────────────────────────────────────────────────────────────────────
# Part (b) – Figure 1: Train / Val loss curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves_part_b() -> None:
    """
    Plot training and validation MSE loss curves for StockLSTM and StockGRU
    (mode='return'). Each model occupies one subplot.

    Interpretation: A well-trained model shows decreasing train loss and a
    validation loss that tracks train loss without diverging (no overfitting).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle("Part (b) – Train / Validation Loss Curves  (mode='return')", fontsize=13)

    for ax, arch in zip(axes, ("lstm", "gru")):
        hist = load_history(arch, "return")
        if hist is None:
            ax.text(0.5, 0.5, "No history found.\nRun train.py first.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(arch.upper())
            continue

        epochs = range(1, len(hist["train"]) + 1)
        ax.plot(epochs, hist["train"], color=COLORS[arch],
                linewidth=2, label="Train")
        ax.plot(epochs, hist["val"],   color=COLORS[arch],
                linewidth=2, linestyle="--", alpha=0.7, label="Val")
        ax.set_title(arch.upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "b1_loss_curves_return.png")


# ─────────────────────────────────────────────────────────────────────────────
# Part (b) – Figure 2: Per-horizon MSE bar chart (LSTM vs GRU)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_horizon_mse() -> None:
    """
    Bar chart of test MSE for each forecast horizon d = 1, …, 5,
    comparing LSTM and GRU side by side.

    Expected observation: MSE increases with d because predicting further
    into the future is inherently more uncertain.
    """
    device = get_device()
    mse_results = {}
    for arch in ("lstm", "gru"):
        model, ckpt = load_model(arch, "return", device)
        if model is None:
            continue
        ma_windows = ckpt.get("cfg", {}).get("ma_windows", config.MA_WINDOWS) if ckpt else config.MA_WINDOWS
        _, _, test_loader, _ = get_dataloaders(mode="return", ma_windows=ma_windows)
        preds, labels = collect_preds(model, test_loader, device)
        mse_results[arch] = np.mean((preds - labels) ** 2, axis=0)  # (D,)

    if not mse_results:
        print("[skip] plot_per_horizon_mse: no checkpoints available.")
        return

    x     = np.arange(config.HORIZON)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, (arch, mse) in enumerate(mse_results.items()):
        offset = (i - len(mse_results) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, mse, width,
                        label=arch.upper(), color=COLORS[arch], alpha=0.85)
        ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"d = {d}" for d in range(1, config.HORIZON + 1)])
    ax.set_ylabel("Test MSE")
    ax.set_title("Part (b) – Per-Horizon Test MSE: LSTM vs GRU")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "b2_per_horizon_mse.png")


# ─────────────────────────────────────────────────────────────────────────────
# Part (b) – Figure 3: Predicted vs Actual scatter (d = 1)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter_part_b() -> None:
    """
    Scatter plot of predicted vs actual 1-day return ratios on the test set
    for both LSTM and GRU. Points close to the y = x diagonal indicate
    accurate predictions. The spread around the diagonal reflects model error.
    """
    device = get_device()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Part (b) – Predicted vs Actual Return Ratio  (d = 1, test set)", fontsize=13)

    for ax, arch in zip(axes, ("lstm", "gru")):
        model, ckpt = load_model(arch, "return", device)
        if model is None:
            ax.text(0.5, 0.5, "No checkpoint.", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        ma_windows = ckpt.get("cfg", {}).get("ma_windows", config.MA_WINDOWS) if ckpt else config.MA_WINDOWS
        _, _, test_loader, _ = get_dataloaders(mode="return", ma_windows=ma_windows)
        preds, labels = collect_preds(model, test_loader, device)
        p1 = preds[:, 0]
        l1 = labels[:, 0]

        lim = max(np.abs(l1).max(), np.abs(p1).max()) * 1.1
        ax.scatter(l1, p1, alpha=0.25, s=10, color=COLORS[arch])
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1.2, label="y = x (perfect)")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Actual return ratio")
        ax.set_ylabel("Predicted return ratio")
        ax.set_title(arch.upper())
        ax.legend()
        ax.grid(alpha=0.3)

        mse_d1 = float(np.mean((p1 - l1) ** 2))
        ax.annotate(f"MSE (d=1) = {mse_d1:.5f}", xy=(0.04, 0.93),
                    xycoords="axes fraction", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    fig.tight_layout()
    save_fig(fig, "b3_scatter_d1.png")


# ─────────────────────────────────────────────────────────────────────────────
# Part (c) – Figure 4: Rolling vs Exact return loss comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_vs_return_loss() -> None:
    """
    Overlay training loss curves for 'return' and 'rolling' modes on the
    same axes (one subplot per architecture). This directly shows whether
    rolling-average targets produce a smoother, lower-variance loss signal.

    Expected observation: rolling loss is lower in absolute value (because
    the smoothed target is easier to predict) and exhibits smaller
    epoch-to-epoch fluctuations (greater training stability).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    fig.suptitle("Part (c) – Training Loss: Exact Return vs Rolling Average", fontsize=13)

    for ax, arch in zip(axes, ("lstm", "gru")):
        for mode in ("return", "rolling"):
            hist = load_history(arch, mode)
            if hist is None:
                continue
            epochs = range(1, len(hist["train"]) + 1)
            ax.plot(epochs, hist["train"],
                    color=COLORS[mode], linewidth=2,
                    label=f"{mode.capitalize()} (train)")
            ax.plot(epochs, hist["val"],
                    color=COLORS[mode], linewidth=2,
                    linestyle="--", alpha=0.6,
                    label=f"{mode.capitalize()} (val)")

        ax.set_title(arch.upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "c1_rolling_vs_return_loss.png")


# ─────────────────────────────────────────────────────────────────────────────
# Part (c) – Figure 5: Training stability (loss variance) analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_stability_analysis() -> None:
    """
    Quantify and visualise training stability by comparing the epoch-to-epoch
    variance of the training loss under 'return' vs 'rolling' modes.

    A lower variance indicates a smoother optimisation landscape, which is
    the expected benefit of using rolling-average targets. Results are shown
    as a grouped bar chart and printed as a summary table.
    """
    variance_data = {}   # {arch: {mode: std_of_train_loss}}

    for arch in ("lstm", "gru"):
        variance_data[arch] = {}
        for mode in ("return", "rolling"):
            hist = load_history(arch, mode)
            if hist is None:
                continue
            # Use std (epoch-to-epoch variation) as the stability metric.
            variance_data[arch][mode] = float(np.std(hist["train"]))

    # Print summary table
    print("\n  Training Loss Stability (std of train loss over epochs)")
    print(f"  {'Model':<8}  {'Return std':>12}  {'Rolling std':>12}  {'More stable':>12}")
    print(f"  {'─'*50}")
    for arch, modes in variance_data.items():
        ret = modes.get("return",  float("nan"))
        rol = modes.get("rolling", float("nan"))
        winner = "rolling" if rol < ret else "return"
        print(f"  {arch.upper():<8}  {ret:>12.6f}  {rol:>12.6f}  {winner:>12}")

    # Bar chart
    archs  = [a for a in ("lstm", "gru") if variance_data.get(a)]
    x      = np.arange(len(archs))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, mode in enumerate(("return", "rolling")):
        vals   = [variance_data.get(a, {}).get(mode, 0) for a in archs]
        offset = (i - 1) * width / 2 + width / 4
        bars   = ax.bar(x + (i - 0.5) * width, vals, width,
                        label=mode.capitalize(), color=COLORS[mode], alpha=0.85)
        ax.bar_label(bars, fmt="%.5f", fontsize=8, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in archs])
    ax.set_ylabel("Std of Training Loss")
    ax.set_title("Part (c) – Training Stability: Return vs Rolling")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "c2_stability_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# Part (d) – Figure 6: Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix() -> None:
    """
    Confusion matrix for the bidirectional LSTM and GRU buy-signal classifiers
    on the test set. Each cell shows the number of samples in that category:

        True Negative  (pass predicted as pass)
        False Positive (pass predicted as buy)
        False Negative (buy predicted as pass)
        True Positive  (buy predicted as buy)

    A model that always predicts "pass" achieves high accuracy due to class
    imbalance (~13% buy rate) but has zero recall on the buy class.
    """
    device = get_device()
    models_available = {}
    model_loaders    = {}
    for arch in ("lstm", "gru"):
        model, ckpt = load_model(f"bidir_{arch}", "signal", device)
        if model is not None:
            models_available[f"bidir_{arch}"] = model
            ma_windows = ckpt.get("cfg", {}).get("ma_windows", config.MA_WINDOWS) if ckpt else config.MA_WINDOWS
            _, _, tl, _ = get_dataloaders(mode="signal", ma_windows=ma_windows)
            model_loaders[f"bidir_{arch}"] = tl

    if not models_available:
        print("[skip] plot_confusion_matrix: no signal checkpoints found.")
        return

    n   = len(models_available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle("Part (d) – Confusion Matrix: Buy Signal Detection", fontsize=13)

    for ax, (key, model) in zip(axes, models_available.items()):
        test_loader = model_loaders[key]
        logits, labels = collect_preds(model, test_loader, device)
        # Threshold lowered to 0.3 to reduce false negatives on the buy class.
        preds  = (torch.sigmoid(torch.tensor(logits)).numpy() >= 0.3).astype(int)
        labels = labels.astype(int)

        cm   = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Pass (0)", "Buy (1)"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(key.upper(), fontsize=11)

    fig.tight_layout()
    save_fig(fig, "d1_confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# Part (d) – Figure 7: Precision / Recall / F1 table (printed + bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def plot_prf_table() -> None:
    """
    Compute and visualise Precision, Recall, and F1-score for the
    bidirectional buy-signal models.

    These metrics are more informative than accuracy for imbalanced datasets:
      - Precision: of all predicted buys, how many were correct?
      - Recall   : of all actual buys, how many were detected?
      - F1       : harmonic mean of precision and recall.
    """
    device = get_device()
    metrics = {}
    for arch in ("lstm", "gru"):
        key   = f"bidir_{arch}"
        model, ckpt = load_model(key, "signal", device)
        if model is None:
            continue
        ma_windows = ckpt.get("cfg", {}).get("ma_windows", config.MA_WINDOWS) if ckpt else config.MA_WINDOWS
        _, _, test_loader, _ = get_dataloaders(mode="signal", ma_windows=ma_windows)
        logits, labels = collect_preds(model, test_loader, device)
        # Threshold lowered to 0.3 to reduce false negatives on the buy class.
        preds  = (torch.sigmoid(torch.tensor(logits)).numpy() >= 0.3).astype(int)
        labels = labels.astype(int)

        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        acc = float((preds == labels).mean())
        metrics[key] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

    if not metrics:
        print("[skip] plot_prf_table: no signal checkpoints found.")
        return

    # Print table
    print("\n  Part (d) – Classification Metrics (test set)")
    print(f"  {'Model':<14}  {'Accuracy':>9}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}")
    print(f"  {'─'*55}")
    for key, m in metrics.items():
        print(f"  {key.upper():<14}  {m['Accuracy']:>9.4f}  "
              f"{m['Precision']:>9.4f}  {m['Recall']:>9.4f}  {m['F1']:>9.4f}")

    # Bar chart
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    x     = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (key, m) in enumerate(metrics.items()):
        vals   = [m[mn] for mn in metric_names]
        offset = (i - len(metrics) / 2 + 0.5) * width
        arch   = key.replace("bidir_", "")
        bars   = ax.bar(x + offset, vals, width,
                        label=key.upper(), color=COLORS[arch], alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Part (d) – Classification Metrics: BiLSTM vs BiGRU")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "d2_prf_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# Part (d) – Figure 8: ROC curve with AUC
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve() -> None:
    """
    ROC (Receiver Operating Characteristic) curve for the bidirectional
    buy-signal models. The curve plots True Positive Rate vs False Positive
    Rate at varying decision thresholds.

    AUC (Area Under the Curve) summarises overall discriminative ability:
      AUC = 1.0 → perfect classifier
      AUC = 0.5 → random classifier (dashed diagonal)
      AUC < 0.5 → worse than random (should flip predictions)
    """
    device = get_device()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.50)")

    found = False
    for arch in ("lstm", "gru"):
        key   = f"bidir_{arch}"
        model, ckpt = load_model(key, "signal", device)
        if model is None:
            continue
        ma_windows = ckpt.get("cfg", {}).get("ma_windows", config.MA_WINDOWS) if ckpt else config.MA_WINDOWS
        _, _, test_loader, _ = get_dataloaders(mode="signal", ma_windows=ma_windows)
        logits, labels = collect_preds(model, test_loader, device)
        probs  = torch.sigmoid(torch.tensor(logits)).numpy()
        labels = labels.astype(int)

        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[arch], linewidth=2.2,
                label=f"{key.upper()}  (AUC = {roc_auc:.3f})")
        print(f"  [{key.upper()}]  AUC = {roc_auc:.4f}")
        found = True

    if not found:
        print("[skip] plot_roc_curve: no signal checkpoints found.")
        plt.close(fig)
        return

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Part (d) – ROC Curve: Buy Signal Detection")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "d3_roc_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# Bonus – Per-ticker predicted vs actual close price line plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_ticker_prices() -> None:
    """
    Bonus figure: for each ticker, plot the actual closing price alongside
    the price implied by the LSTM's d=1 predicted return ratio on the test set.

    The implied predicted price at each step t is reconstructed as:
        p̂_{t+1} = p_t × (1 + r̂_{t+1})

    where r̂_{t+1} is the model's predicted 1-day return ratio and p_t is
    the actual closing price at time t (used as the baseline). This means
    the model is evaluated in a one-step-ahead rolling fashion rather than
    compounding errors over multiple days.
    """
    device    = get_device()
    model, _  = load_model("lstm", "return", device)
    if model is None:
        print("[skip] plot_per_ticker_prices: lstm_return checkpoint not found.")
        return

    stock_data = download_data()
    n_tickers  = len(config.TICKERS)
    n_cols     = 2
    n_rows     = (n_tickers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(12, 4 * n_rows), sharex=False)
    axes = axes.flatten()
    fig.suptitle("Bonus – Per-Ticker: Actual vs Predicted Close Price  (d=1, test set)",
                 fontsize=13)

    for idx, ticker in enumerate(config.TICKERS):
        ax  = axes[idx]
        df  = stock_data[ticker]
        _, _, test_df = split_data(df)

        scaler     = Scaler()
        train_df, _, _ = split_data(df)
        scaler.fit(train_df)
        test_norm  = scaler.transform(test_df)

        feat_arr   = test_norm[config.FEATURE_COLS].values.astype(np.float32)
        # Add moving-average features to match the model's expected input size F̂.
        if config.MA_WINDOWS:
            from dataset import add_moving_average_features
            feat_arr = add_moving_average_features(feat_arr, windows=config.MA_WINDOWS)
        raw_close  = test_df["Close"].values.astype(np.float32)

        T       = config.LOOKBACK
        n_steps = len(feat_arr) - T - config.HORIZON
        pred_prices, actual_prices = [], []

        for i in range(n_steps):
            window = torch.tensor(feat_arr[i : i + T]).unsqueeze(0).to(device)
            with torch.no_grad():
                ret_pred = model(window)[0, 0].item()   # d=1 predicted return

            t          = i + T - 1
            pt         = float(raw_close[t])
            pred_price = pt * (1 + ret_pred)
            actual_price = float(raw_close[t + 1])

            pred_prices.append(pred_price)
            actual_prices.append(actual_price)

        dates = test_df.index[T : T + n_steps]
        ax.plot(dates, actual_prices, color="black",    linewidth=1.2, label="Actual")
        ax.plot(dates, pred_prices,   color=COLORS["lstm"], linewidth=1.0,
                alpha=0.75, linestyle="--", label="Predicted (d=1)")
        ax.set_title(ticker, fontsize=11)
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (USD)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    # Hide unused subplots
    for ax in axes[n_tickers:]:
        ax.set_visible(False)

    fig.tight_layout()
    save_fig(fig, "bonus_per_ticker_prices.png")



# ─────────────────────────────────────────────────────────────────────────────
# Ablation – Figure: MSE comparison across all configs
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation_mse(mode: str = "return") -> None:
    """
    Bar chart comparing overall test MSE across all ablation configurations
    for both LSTM and GRU. Helps identify which combination of hidden size,
    depth, dropout, and MA features yields the best forecasting performance.
    """
    import json
    results_path = os.path.join(config.RESULTS_DIR, "ablation_results.json")
    if not os.path.exists(results_path):
        print(f"[skip] plot_ablation_mse: {results_path} not found.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    cfg_names = list(ABLATION_CONFIGS.keys())
    x         = np.arange(len(cfg_names))
    width     = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, arch in enumerate(("lstm", "gru")):
        vals = []
        for cfg_name in cfg_names:
            run_id = f"{cfg_name}_{arch}_{mode}"
            mse    = all_results.get(run_id, {}).get("overall_mse", float("nan"))
            vals.append(mse)
        offset = (i - 0.5) * width
        bars   = ax.bar(x + offset, vals, width,
                        label=arch.upper(),
                        color=COLORS[arch], alpha=0.85)
        ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Config {n.capitalize()}" for n in cfg_names])
    ax.set_ylabel("Overall Test MSE")
    ax.set_title(f"Ablation – Test MSE by Configuration  (mode='{mode}')")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"ablation_mse_{mode}.png")


def plot_ablation_signal() -> None:
    """
    Bar chart comparing AUC and F1 scores across all ablation configurations
    for the buy-signal detection task (mode='signal').
    """
    import json
    results_path = os.path.join(config.RESULTS_DIR, "ablation_results.json")
    if not os.path.exists(results_path):
        print(f"[skip] plot_ablation_signal: {results_path} not found.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    cfg_names = list(ABLATION_CONFIGS.keys())
    x         = np.arange(len(cfg_names))
    width     = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Ablation – Buy Signal Detection  (mode='signal')", fontsize=13)

    for metric_idx, (metric, ax) in enumerate(zip(["auc", "f1"], axes)):
        for i, arch in enumerate(("lstm", "gru")):
            vals = []
            for cfg_name in cfg_names:
                run_id = f"{cfg_name}_{arch}_signal"
                val    = all_results.get(run_id, {}).get(metric, float("nan"))
                vals.append(val)
            offset = (i - 0.5) * width * 2
            bars   = ax.bar(x + offset, vals, width * 2,
                            label=f"Bi{arch.upper()}",
                            color=COLORS[arch], alpha=0.85)
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=3)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Config {n.capitalize()}" for n in cfg_names])
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        if metric == "auc":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1,
                       label="Random (AUC=0.5)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "ablation_signal.png")


def plot_ablation_val_loss() -> None:
    """
    Line plot of validation loss curves for all ablation configurations,
    showing convergence speed and final loss for LSTM and GRU across modes.
    One subplot per mode.
    """
    import json
    ckpt_base = os.path.join(config.CHECKPOINT_DIR, "ablation")
    if not os.path.exists(ckpt_base):
        print("[skip] plot_ablation_val_loss: no ablation checkpoints found.")
        return

    modes = ["return", "rolling", "signal"]
    fig, axes = plt.subplots(len(modes), 2, figsize=(14, 4 * len(modes)))
    fig.suptitle("Ablation – Validation Loss Curves", fontsize=13)

    cfg_colors = {
        "baseline": "#1f77b4", "A": "#ff7f0e", "B": "#2ca02c",
        "C": "#d62728",        "D": "#9467bd",
    }

    for row, mode in enumerate(modes):
        for col, arch in enumerate(("lstm", "gru")):
            ax = axes[row][col]
            for cfg_name in ABLATION_CONFIGS:
                run_id    = f"{cfg_name}_{arch}_{mode}"
                hist_path = os.path.join(ckpt_base, f"{run_id}_history.json")
                if not os.path.exists(hist_path):
                    continue
                with open(hist_path) as f:
                    hist = json.load(f)
                epochs = range(1, len(hist["val"]) + 1)
                ax.plot(epochs, hist["val"],
                        color=cfg_colors.get(cfg_name, "gray"),
                        linewidth=1.5, label=f"Config {cfg_name.capitalize()}")

            ax.set_title(f"{arch.upper()} – mode='{mode}'", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Val Loss")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "ablation_val_loss_curves.png")


def print_ablation_table() -> None:
    """Print best configs per mode to terminal for easy report reference."""
    import json
    results_path = os.path.join(config.RESULTS_DIR, "ablation_results.json")
    if not os.path.exists(results_path):
        print("[skip] ablation_results.json not found.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    print(f"\n{'═'*75}")
    print("  Ablation Best Configs Summary")
    print(f"{'═'*75}")

    for mode in ("return", "rolling", "signal"):
        print(f"\n  Mode: {mode}")
        metric = "overall_mse" if mode != "signal" else "auc"
        better = min if mode != "signal" else max

        for arch in ("lstm", "gru"):
            candidates = {
                k: v for k, v in all_results.items()
                if f"_{arch}_{mode}" in k and metric in v
            }
            if not candidates:
                continue
            best_id  = better(candidates, key=lambda x: candidates[x][metric])
            best_val = candidates[best_id][metric]
            cfg_name = best_id.split("_")[0]
            cfg_desc = ABLATION_CONFIGS[cfg_name]["description"]
            print(f"    {arch.upper():4}  best: {best_id:<28}  "
                  f"{metric}={best_val:.6f}")
            print(f"          → {cfg_desc}")



# ─────────────────────────────────────────────────────────────────────────────
# Command-line interface
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HW4 Visualisation Script – CS515 Deep Learning"
    )
    parser.add_argument(
        "--part", type=str, default="all",
        choices=["b", "c", "d", "bonus", "ablation", "all"],
        help="Which part to visualise: b | c | d | bonus | ablation | all",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.part in ("b", "all"):
        print("\n── Part (b) ─────────────────────────────────────────")
        plot_loss_curves_part_b()
        plot_per_horizon_mse()
        plot_scatter_part_b()

    if args.part in ("c", "all"):
        print("\n── Part (c) ─────────────────────────────────────────")
        plot_rolling_vs_return_loss()
        plot_stability_analysis()

    if args.part in ("d", "all"):
        print("\n── Part (d) ─────────────────────────────────────────")
        plot_confusion_matrix()
        plot_prf_table()
        plot_roc_curve()

    if args.part in ("bonus", "all"):
        print("\n── Bonus ────────────────────────────────────────────")
        plot_per_ticker_prices()

    if args.part in ("ablation", "all"):
        print("\n── Ablation ──────────────────────────────────────────")
        plot_ablation_mse(mode="return")
        plot_ablation_mse(mode="rolling")
        plot_ablation_signal()
        plot_ablation_val_loss()
        print_ablation_table()

    print(f"\n[done] All figures saved to '{config.RESULTS_DIR}/'")