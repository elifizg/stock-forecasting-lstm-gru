# config.py
# Central configuration for HW4: Sequence Modeling

# ── Tickers ───────────────────────────────────────────────────────────────────
TICKERS = ["NVDA", "GOOGL", "MU", "META", "AMAT"]

# ── Date Ranges ───────────────────────────────────────────────────────────────
TRAIN_START  = "2020-01-01"
TRAIN_END    = "2024-07-31"
VAL_START    = "2024-08-01"
VAL_END      = "2024-12-31"
TEST_START   = "2025-01-01"
TEST_END     = "2025-12-31"

# ── Features ──────────────────────────────────────────────────────────────────
FEATURE_COLS = ["Open", "High", "Low", "Close"]   # F = 4  (base OHLC features)
LOOKBACK     = 20                                  # T = 20 trading days
HORIZON      = 5                                   # D = 5 (predict d=1..5)

# ── Auxiliary features (Part b, Footnote 1) ───────────────────────────────────
# Simple Moving Average windows computed via 1D convolution over Close price.
# Set to [] to disable and use only raw OHLC features (F_hat = F = 4).
MA_WINDOWS  = [5, 10]                             # SMA-5 and SMA-10
INPUT_SIZE  = len(FEATURE_COLS) + len(MA_WINDOWS) # F_hat = 4 + 2 = 6

# ── Rolling Average (Part c) ──────────────────────────────────────────────────
ROLLING_WINDOW = 3                                 # l = 3
# Exponentially decaying weights: more recent prices carry more influence.
# [0.6, 0.3, 0.1] assigns 60% weight to p_{t+d}, 30% to p_{t+d-1}, 10% to p_{t+d-2}.
# This increases the target variance compared to equal weights, making the
# stability difference between 'return' and 'rolling' modes more pronounced.
ROLLING_WEIGHTS = [0.6, 0.3, 0.1]

# ── Turning Point / Buy Signal (Part d) ───────────────────────────────────────
BUY_THRESHOLD = 1.1                                # γ = 1.1  (10 % gain)

# ── Model Hyperparameters ─────────────────────────────────────────────────────
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 10          # early-stopping patience (epochs without val improvement)

# ── Output ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"
