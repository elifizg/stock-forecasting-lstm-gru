# dataset.py
# HW4 – Sequence Modeling | CS515 Deep Learning
#
# This module handles the full data pipeline:
#   1. Download daily OHLC stock data via yfinance and cache to disk.
#   2. Split chronologically into train / validation / test sets.
#   3. Apply per-feature min-max normalisation (fit on train only).
#   4. Build sliding-window (X, y) pairs for three target modes:
#        'return'  – Part (b): exact d-day return ratios
#        'rolling' – Part (c): weighted rolling-average return ratios
#        'signal'  – Part (d): binary buy signal based on max (High) price
#   5. Wrap everything in PyTorch DataLoaders for training.

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yfinance as yf

import config


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Download & cache raw OHLC data
# ─────────────────────────────────────────────────────────────────────────────

def download_data(
    tickers=config.TICKERS,
    start=config.TRAIN_START,
    end=config.TEST_END,
    cache_dir="data",
) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLC price data for the given tickers using the yfinance
    library, and return a dictionary mapping each ticker to a DataFrame with
    columns [Open, High, Low, Close].

    Data is cached as CSV files under `cache_dir` so that subsequent runs
    load from disk instead of re-downloading from Yahoo Finance.

    Parameters
    ----------
    tickers   : list of S&P 500 ticker symbols (e.g. ['NVDA', 'GOOGL'])
    start     : start date string (inclusive), e.g. '2020-01-01'
    end       : end date string (inclusive),   e.g. '2025-12-31'
    cache_dir : directory where CSV caches are stored

    Returns
    -------
    dict[str, pd.DataFrame]  –  {ticker: OHLC DataFrame indexed by Date}
    """
    os.makedirs(cache_dir, exist_ok=True)
    stock_data = {}

    for ticker in tickers:
        cache_path = os.path.join(cache_dir, f"{ticker}.csv")

        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=True)
            # Robustly locate the date index column regardless of yfinance version.
            date_col = [c for c in df.columns if "date" in c.lower() or "price" in c.lower()]
            if date_col:
                df = df.set_index(date_col[0])
            else:
                df = df.set_index(df.columns[0])
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            # Keep only the required OHLC feature columns.
            df = df[[c for c in config.FEATURE_COLS if c in df.columns]]
            print(f"[cache]  {ticker}: {len(df)} rows loaded from disk.")
        else:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            # yfinance >= 0.2.x may return multi-level column headers; flatten them.
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw[config.FEATURE_COLS].dropna()
            df.index.name = "Date"
            df.to_csv(cache_path)
            print(f"[yfinance] {ticker}: {len(df)} rows downloaded.")

        stock_data[ticker] = df

    return stock_data


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Chronological train / validation / test split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a single ticker DataFrame into three non-overlapping, chronologically
    ordered subsets as specified in the assignment:

        Train      : January 2020 – July 2024
        Validation : August 2024  – December 2024
        Test       : January 2025 – December 2025

    Note: Splitting by date (rather than by index) ensures no look-ahead bias.
    The scaler is fit exclusively on the training split and then applied to
    validation and test splits to prevent data leakage.

    Parameters
    ----------
    df : OHLC DataFrame with a DatetimeIndex

    Returns
    -------
    (train_df, val_df, test_df)
    """
    train = df.loc[config.TRAIN_START : config.TRAIN_END]
    val   = df.loc[config.VAL_START   : config.VAL_END  ]
    test  = df.loc[config.TEST_START  : config.TEST_END ]
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Min-Max normalisation
# ─────────────────────────────────────────────────────────────────────────────

class Scaler:
    """
    Per-feature min-max scaler that maps each feature column to [0, 1].

    The scaler is fit on the training split only (to avoid data leakage) and
    then used to transform the validation and test splits. The small epsilon
    (1e-8) prevents division by zero for constant columns.

    Usage
    -----
        scaler     = Scaler()
        train_norm = scaler.fit_transform(train_df)
        val_norm   = scaler.transform(val_df)
        test_norm  = scaler.transform(test_df)
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, df: pd.DataFrame) -> "Scaler":
        """Compute per-column min and max from the training DataFrame."""
        self.min_ = df.min()
        self.max_ = df.max()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max scaling:  x_scaled = (x - min) / (max - min + ε)."""
        return (df - self.min_) / (self.max_ - self.min_ + 1e-8)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in a single call (convenience method)."""
        return self.fit(df).transform(df)

    def inverse_close(self, scaled_close: np.ndarray) -> np.ndarray:
        """
        Inverse-transform scaled Close prices back to original price space.
        Useful for computing interpretable prediction errors.
        """
        mn = self.min_["Close"]
        mx = self.max_["Close"]
        return scaled_close * (mx - mn + 1e-8) + mn


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Target builders
# ─────────────────────────────────────────────────────────────────────────────

def compute_return_targets(
    prices: np.ndarray,
    t: int,
    horizon: int = config.HORIZON,
) -> np.ndarray:
    """
    Part (b) – Compute exact d-day return ratios for d = 1, …, horizon.

    Given the closing price p_t at the end of the look-back window, the
    d-day return ratio is defined as:

        r_{t+d} = (p_{t+d} - p_t) / p_t

    where p_{t+d} is the closing price d trading days in the future.
    The model is trained to minimise the MSE between predicted and actual
    return ratios across all five horizons simultaneously.

    Parameters
    ----------
    prices  : 1-D array of raw (un-normalised) closing prices
    t       : index of the last day in the current look-back window
    horizon : number of future days to forecast (D = 5)

    Returns
    -------
    np.ndarray of shape (horizon,)  –  [r_{t+1}, r_{t+2}, …, r_{t+D}]
    """
    pt = float(prices[t])
    targets = np.zeros(horizon, dtype=np.float32)
    for d in range(1, horizon + 1):
        idx = t + d
        if idx < len(prices):
            targets[d - 1] = (float(prices[idx]) - pt) / (pt + 1e-8)
    return targets


def compute_rolling_targets(
    prices: np.ndarray,
    t: int,
    horizon: int         = config.HORIZON,
    window: int          = config.ROLLING_WINDOW,
    weights: list[float] = config.ROLLING_WEIGHTS,
) -> np.ndarray:
    """
    Part (c) – Compute weighted rolling-average return ratios.

    Instead of using a single future closing price, the target is a weighted
    average of ``window`` consecutive closing prices ending at day ``t + d``:

        r̂_{t+d} = (Σ_{j=0}^{window-1} w_j · p_{t+d-j}  -  p_t) / p_t

    For the assignment setting, ``window = 3`` and ``weights = [0.6, 0.3, 0.1]``.
    Therefore the target uses p_{t+d}, p_{t+d-1}, and p_{t+d-2}, assigning
    the largest weight to the most recent future price. This smoothing reduces
    the effect of single-day price spikes compared with the exact-return target
    in Part (b).

    Parameters
    ----------
    prices  : 1-D array of raw closing prices
    t       : index of the last day in the look-back window
    horizon : number of future days to forecast (D = 5)
    window  : number of prices used in the rolling target (l = 3)
    weights : list of ``window`` weights summing to 1

    Returns
    -------
    np.ndarray of shape (horizon,)
    """
    weights = np.array(weights, dtype=np.float32)
    if len(weights) != window:
        raise ValueError(
            f"ROLLING_WINDOW={window} requires {window} weights, "
            f"but got {len(weights)} weights."
        )

    weight_sum = float(weights.sum())
    if not np.isclose(weight_sum, 1.0):
        weights = weights / (weight_sum + 1e-8)

    pt = float(prices[t])
    targets = np.zeros(horizon, dtype=np.float32)

    for d in range(1, horizon + 1):
        numerator = 0.0
        for j, w in enumerate(weights):
            idx = t + d - j
            if 0 <= idx < len(prices):
                numerator += w * float(prices[idx])
        targets[d - 1] = (numerator - pt) / (pt + 1e-8)

    return targets


def compute_buy_targets(
    high_prices: np.ndarray,
    close_prices: np.ndarray,
    t: int,
    horizon: int     = config.HORIZON,
    threshold: float = config.BUY_THRESHOLD,
) -> int:
    """
    Part (d) – Generate a binary buy / pass signal for algorithmic trading.

    The assignment states the turning-point threshold in price-ratio form
    with γ = 1.1. Therefore, a buy signal is issued if the High price on
    any of the next D days exceeds 1.1 times the current closing price:

        p^max_{t+d} / p_t > γ.

    Equivalently, this is the same as requiring the High-price return

        r_{t+d} = (p^max_{t+d} - p_t) / p_t

    to exceed γ - 1 = 0.1, i.e. a gain of more than 10%. Otherwise, a
    pass signal (label = 0) is issued.

    Parameters
    ----------
    high_prices  : 1-D array of raw daily High prices
    close_prices : 1-D array of raw daily Close prices (baseline p_t)
    t            : index of the last day in the look-back window
    horizon      : number of future days to check (D = 5)
    threshold    : price-ratio threshold γ (default 1.1 → 10% gain)

    Returns
    -------
    int  –  1 (buy) or 0 (pass)
    """
    pt = float(close_prices[t])
    for d in range(1, horizon + 1):
        idx = t + d
        if idx < len(high_prices):
            price_ratio = float(high_prices[idx]) / (pt + 1e-8)
            if price_ratio > threshold:
                return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Auxiliary feature engineering via 1D convolution
# ─────────────────────────────────────────────────────────────────────────────

def add_moving_average_features(
    feat_arr: np.ndarray,
    windows: list[int] = [5, 10],
) -> np.ndarray:
    """
    Part (b), Footnote 1 – Generate auxiliary moving-average features using
    1D convolution over the normalised Close price column.

    For each window size w, a simple moving average (SMA) is computed by
    convolving the Close price series with a uniform kernel of length w:

        SMA_w[t] = (1/w) * Σ_{j=0}^{w-1} Close[t-j]

    This is implemented as a 1D convolution with a box filter:

        SMA_w = Conv1D(Close, kernel=ones(w)/w, padding='same')

    The resulting SMA series captures medium- and long-term price trends,
    complementing the raw OHLC features which mainly encode short-term
    price levels. Appending SMA features increases the input dimension from
    F = 4 to F̂ = F + len(windows) ≥ F, satisfying the assignment requirement
    that the model input tensor has shape (batch, T, F̂ ≥ F).

    Parameters
    ----------
    feat_arr : normalised OHLC array of shape (N_days, F)
    windows  : list of SMA window sizes (default: [5, 10] trading days)

    Returns
    -------
    np.ndarray of shape (N_days, F + len(windows))
        Original features concatenated with one SMA column per window.
    """
    close_col = feat_arr[:, config.FEATURE_COLS.index("Close")]   # (N_days,)
    sma_cols  = []

    for w in windows:
        # Build a uniform box kernel of length w.
        kernel = np.ones(w, dtype=np.float32) / w

        # np.convolve in 'full' mode produces N+w-1 values; we slice to keep
        # only the causal (past-only) portion and pad the head with the first
        # valid SMA value to preserve the original length.
        conv   = np.convolve(close_col, kernel, mode="full")[: len(close_col)]

        # The first w-1 entries are based on fewer than w days (edge effect).
        # Replace them with the first valid value to avoid look-ahead bias.
        conv[: w - 1] = conv[w - 1]

        sma_cols.append(conv.reshape(-1, 1))

    # Concatenate SMA columns to the right of the original feature matrix.
    return np.concatenate([feat_arr] + sma_cols, axis=1)   # (N_days, F + len(windows))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PyTorch Dataset classes
# ─────────────────────────────────────────────────────────────────────────────

class StockDataset(Dataset):
    """
    Sliding-window PyTorch Dataset for a single stock ticker.

    For each valid time step t, a look-back window of T = 20 consecutive
    normalised OHLC feature vectors is extracted as input X, and the
    corresponding target y is computed according to the selected mode.

    The input–output pair is defined as:

        X_i^(t) = [f_{i, t-T+1}, …, f_{i, t}]  ∈ R^{T×F̂}
        y_i^(t) = target computed from raw prices at t

    where F̂ = F + len(ma_windows) ≥ F includes auxiliary moving-average
    features computed via 1D convolution (see add_moving_average_features).

    Modes
    -----
    'return'  : Part (b) – exact d-day return ratios  → y shape (D,)
    'rolling' : Part (c) – rolling-average returns    → y shape (D,)
    'signal'  : Part (d) – binary buy signal          → y shape ()

    Parameters
    ----------
    df         : normalised OHLC DataFrame (used for X)
    raw_df     : original un-normalised DataFrame (used to compute targets)
    mode       : one of 'return', 'rolling', 'signal'
    ma_windows : list of SMA window sizes for auxiliary features ([] = disabled)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        raw_df: pd.DataFrame,
        mode: str = "return",
        ma_windows: list[int] = [5, 10],
    ):
        assert mode in ("return", "rolling", "signal"), f"Unknown mode: {mode}"
        self.mode = mode
        self.T    = config.LOOKBACK   # look-back window length
        self.D    = config.HORIZON    # number of forecast horizons

        feat_arr  = df[config.FEATURE_COLS].values.astype(np.float32)

        # Augment with moving-average features via 1D convolution if requested.
        # This increases input dimensionality from F=4 to F̂=4+len(ma_windows).
        if ma_windows:
            feat_arr = add_moving_average_features(feat_arr, windows=ma_windows)

        raw_close = raw_df["Close"].values.astype(np.float32)
        raw_high  = raw_df["High"].values.astype(np.float32)

        self.X = []
        self.y = []

        # Leave enough room at the end for the furthest target (t + D).
        max_t = len(feat_arr) - self.T - self.D

        for i in range(max_t):
            t = i + self.T - 1          # index of the last day in this window

            window = feat_arr[i : i + self.T]   # (T, F̂)
            self.X.append(window)

            if mode == "return":
                target = compute_return_targets(raw_close, t)
            elif mode == "rolling":
                target = compute_rolling_targets(raw_close, t)
            else:
                target = compute_buy_targets(raw_high, raw_close, t)

            self.y.append(target)

        self.X = np.array(self.X, dtype=np.float32)   # (N, T, F̂)
        self.y = np.array(self.y, dtype=np.float32)   # (N, D) or (N,)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class MultiStockDataset(Dataset):
    """
    Aggregate dataset that concatenates StockDataset instances from multiple
    tickers into a single dataset.

    Each sample (X, y) is treated independently of which ticker it came from,
    allowing the model to learn shared temporal patterns across all stocks.

    Parameters
    ----------
    datasets : list of StockDataset objects (one per ticker)
    """

    def __init__(self, datasets: list[StockDataset]):
        self.samples = []
        for ds in datasets:
            for i in range(len(ds)):
                self.samples.append(ds[i])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  High-level DataLoader builder
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    mode: str = "return",
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 0,
    ma_windows: list[int] = [5, 10],
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Scaler]]:
    """
    Build train, validation, and test DataLoaders for all tickers in a single
    call. This function orchestrates the full data pipeline:

        download → split → normalise → build dataset → wrap in DataLoader

    The Scaler for each ticker is fit exclusively on the training split to
    prevent any form of look-ahead bias or data leakage.

    Parameters
    ----------
    mode        : target mode – 'return', 'rolling', or 'signal'
    batch_size  : number of samples per mini-batch
    num_workers : number of worker processes for data loading

    Returns
    -------
    train_loader, val_loader, test_loader : PyTorch DataLoaders
    scalers : dict {ticker: Scaler} – retained for inverse-transforming
              model predictions back to original price space
    """
    stock_data = download_data()
    scalers    = {}

    train_datasets, val_datasets, test_datasets = [], [], []

    for ticker, df in stock_data.items():
        train_df, val_df, test_df = split_data(df)

        # Fit scaler on training data only; transform val/test with the same scaler.
        scaler     = Scaler()
        train_norm = scaler.fit_transform(train_df)
        val_norm   = scaler.transform(val_df)
        test_norm  = scaler.transform(test_df)
        scalers[ticker] = scaler

        train_datasets.append(StockDataset(train_norm, train_df, mode=mode, ma_windows=ma_windows))
        val_datasets.append(StockDataset(val_norm,     val_df,   mode=mode, ma_windows=ma_windows))
        test_datasets.append(StockDataset(test_norm,   test_df,  mode=mode, ma_windows=ma_windows))

    train_loader = DataLoader(
        MultiStockDataset(train_datasets),
        batch_size=batch_size, shuffle=True,  num_workers=num_workers,
    )
    val_loader = DataLoader(
        MultiStockDataset(val_datasets),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        MultiStockDataset(test_datasets),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    print(f"\nDataset sizes  (mode='{mode}'):")
    print(f"  Train : {len(train_loader.dataset):>6} samples")
    print(f"  Val   : {len(val_loader.dataset):>6} samples")
    print(f"  Test  : {len(test_loader.dataset):>6} samples")

    return train_loader, val_loader, test_loader, scalers


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for mode in ("return", "rolling", "signal"):
        print(f"\n{'─'*50}")
        print(f"Mode: {mode}")
        tr, vl, te, sc = get_dataloaders(mode=mode)
        xb, yb = next(iter(tr))
        print(f"  X batch : {xb.shape}  dtype={xb.dtype}")
        print(f"  y batch : {yb.shape}  dtype={yb.dtype}")
        if mode == "signal":
            print(f"  Buy ratio in train batch: {yb.float().mean():.2%}")