# Stock Return Forecasting with LSTM & GRU

**CS515 Deep Learning — Homework 4, Part 1**
Sabancı University

A complete deep learning pipeline for multi-horizon stock return forecasting and turning-point detection using LSTM and GRU networks on S&P 500 equities.

---

## Overview

This project implements three forecasting tasks on five S&P 500 stocks (**NVDA, GOOGL, MU, META, AMAT**) over the period January 2020 – December 2025:

| Part | Task | Architecture | Loss |
|---|---|---|---|
| (b) | Exact d-day return forecasting (d=1…5) | StockLSTM / StockGRU | MSELoss |
| (c) | Weighted rolling-average return forecasting (l=3, weights=[0.6, 0.3, 0.1]) | StockLSTM / StockGRU | MSELoss |
| (d) | Buy/pass signal detection (price-ratio γ=1.1, equivalent to return > 0.1) | BidirSignalLSTM / BidirSignalGRU | BCEWithLogitsLoss |

An ablation study over 5 hyperparameter configurations (30 runs total: 5 configs × 2 architectures × 3 target modes) identifies the best architecture per task.

---

## Results

### Part (b) — Exact Return Forecasting

| Model | Config | Overall MSE | d=1 MSE | d=5 MSE |
|---|---|---|---|---|
| LSTM | B (h=256, L=2) | 0.002472 | 0.000856 | 0.004083 |
| GRU | D (h=256, L=3) | 0.002477 | 0.000854 | 0.004069 |

### Part (c) — Rolling Average Forecasting

| Model | Return Config | Return MSE | Rolling Config | Rolling MSE | Improvement |
|---|---|---|---|---|---|
| LSTM | B | 0.002472 | D | 0.001824 | −26.2% |
| GRU | D | 0.002477 | B | 0.001815 | −26.7% |

The rolling-average target uses a 3-day weighted window over future prices:
p(t+d), p(t+d−1), and p(t+d−2), with weights [0.6, 0.3, 0.1].

### Part (d) — Buy Signal Detection (price-ratio γ=1.1, decision threshold=0.3)

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| BidirLSTM | 0.700 | 0.253 | 0.602 |
| BidirGRU | 0.688 | 0.250 | 0.601 |

The signal label uses the assignment threshold in price-ratio form:
High(t+d) / Close(t) > 1.1, which is equivalent to a return ratio greater than 0.1.

---

## Project Structure

```
stock-forecasting-lstm-gru/
│
├── config.py           # All hyperparameters (tickers, dates, model config)
├── dataset.py          # Data download, split, normalisation, DataLoader
├── train.py            # Training loop (AdamW + early stopping)
├── test.py             # Evaluation & metrics
├── ablation.py         # Automated ablation study (30 runs)
├── visualization.py    # All report figures
│
├── models/
│   ├── __init__.py     # Model factory (get_model)
│   ├── lstm.py         # StockLSTM — Part (b/c)
│   ├── gru.py          # StockGRU  — Part (b/c)
│   ├── bidir_lstm.py   # BidirSignalLSTM — Part (d)
│   └── bidir_gru.py    # BidirSignalGRU  — Part (d)
│
├── data/               # Cached CSV files (auto-created)
├── checkpoints/        # Training histories are included; .pt model weights are auto-created during training
└── results/            # Generated figures (auto-created)
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Verify data pipeline
```bash
python dataset.py
```

### 2. Run ablation study (trains all 30 models, ~90 min on CPU)
```bash
python ablation.py
```

Best checkpoints are automatically copied to `checkpoints/` after ablation.

Note: Pretrained `.pt` checkpoint files may not be included in the repository.
If checkpoints are missing, run `python ablation.py` or the relevant `train.py`
commands before evaluation.

### 3. Evaluate best models
```bash
python test.py --mode all
```

### 4. Generate all report figures
```bash
python visualization.py --part all
```

### Train a single model manually
```bash
# Part (b) — exact return
python train.py --arch lstm --mode return --epochs 50
python train.py --arch gru  --mode return --epochs 50

# Part (c) — rolling average
python train.py --arch lstm --mode rolling --epochs 50
python train.py --arch gru  --mode rolling --epochs 50

# Part (d) — buy signal
python train.py --arch lstm --mode signal --epochs 50
python train.py --arch gru  --mode signal --epochs 50
```

---

## Dataset

Daily OHLC data downloaded via `yfinance` for 5 S&P 500 tickers:

| Split | Period | Samples |
|---|---|---|
| Train | Jan 2020 – Jul 2024 | 5,635 |
| Validation | Aug 2024 – Dec 2024 | 405 |
| Test | Jan 2025 – Dec 2025 | 1,120 |

**Features:** Open, High, Low, Close + SMA-5, SMA-10 (via 1D convolution) → F̂ = 6

**Lookback window:** T = 20 trading days

---

## Ablation Configurations

| Config | Hidden | Layers | Dropout | MA Windows | F̂ |
|---|---|---|---|---|---|
| Baseline | 128 | 2 | 0.3 | [5, 10] | 6 |
| A | 128 | 2 | 0.2 | [5, 10, 20] | 7 |
| B | 256 | 2 | 0.3 | [5, 10] | 6 |
| C | 256 | 3 | 0.3 | [5, 10] | 6 |
| D | 256 | 3 | 0.2 | [5, 10, 20] | 7 |

**Selected best configs:** Return — LSTM B, GRU D; Rolling — LSTM D, GRU B; Signal — B for both bidirectional models.

---

## Key Design Choices

- **Chronological split** — no look-ahead bias; scaler fit on train only
- **AdamW + ReduceLROnPlateau** — decoupled weight decay + adaptive LR
- **Gradient clipping** (max_norm=1.0) — prevents exploding gradients on financial data
- **pos_weight=6.4** for signal mode — counteracts 13.5% buy rate class imbalance
- **Decision threshold=0.3** for buy signal — reduces false negatives on minority class