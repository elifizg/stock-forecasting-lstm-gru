# train.py
# HW4 – Sequence Modeling | CS515 Deep Learning
#
# This script implements the training loop for all model and target
# mode combinations defined in the assignment:
#
#   Part (b) – 'return'  mode: StockLSTM / StockGRU trained with MSELoss
#                              to predict exact d-day return ratios.
#   Part (c) – 'rolling' mode: StockLSTM / StockGRU trained with MSELoss
#                              to predict rolling-average return ratios.
#   Part (d) – 'signal'  mode: BidirSignalLSTM / BidirSignalGRU trained
#                              with BCEWithLogitsLoss for buy-signal detection.
#
# Training features:
#   - AdamW optimiser with weight decay for L2 regularisation.
#   - ReduceLROnPlateau scheduler: halves the learning rate when validation
#     loss stops improving for 5 consecutive epochs.
#   - Gradient clipping (max_norm=1.0) to prevent exploding gradients,
#     which are common in recurrent networks on financial time series.
#   - Early stopping: training halts if validation loss does not improve
#     for `patience` consecutive epochs, preventing overfitting.
#   - Best checkpoint is saved automatically to the checkpoints/ directory.
#
# Usage (command line):
#   python train.py --arch lstm --mode return
#   python train.py --arch gru  --mode rolling
#   python train.py --arch lstm --mode signal --epochs 30

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from dataset import get_dataloaders
from models  import get_model


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Select the best available compute device in priority order:
    CUDA GPU → Apple MPS → CPU.
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"[device] {dev}")
    return dev


def get_criterion(mode: str) -> nn.Module:
    """
    Return the appropriate loss function for the given target mode.

    - Regression modes ('return', 'rolling'):
        Mean Squared Error (MSE) between predicted and actual return ratios.
        MSE penalises large deviations more heavily, which is desirable for
        financial forecasting where large prediction errors are costly.

    - Classification mode ('signal'):
        Binary Cross-Entropy with Logits (BCEWithLogitsLoss). The model
        outputs a raw logit; this loss applies the sigmoid internally for
        numerical stability.

        The buy signal accounts for only ~13% of samples while the pass
        signal accounts for ~87%, creating a significant class imbalance.
        pos_weight = neg_count / pos_count ≈ 6.4 upweights the loss on
        positive (buy) samples, preventing the model from collapsing to
        the trivial solution of always predicting "pass".
    """
    if mode in ("return", "rolling"):
        return nn.MSELoss()
    else:
        pos_weight = torch.tensor([6.4])
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def model_name_for_mode(arch: str, mode: str) -> str:
    """
    Map (architecture, mode) to the correct model registry key.

    Regression modes use unidirectional LSTM/GRU; the signal mode uses
    bidirectional variants as required by Part (d) of the assignment.
    """
    if mode in ("return", "rolling"):
        return arch               # 'lstm' or 'gru'
    else:
        return f"bidir_{arch}"    # 'bidir_lstm' or 'bidir_gru'


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch (train or evaluate)
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    mode:      str,
    training:  bool,
) -> float:
    """
    Run one full pass over the DataLoader and return the mean loss.

    During training:
      1. Forward pass through the model.
      2. Compute the loss (MSE or BCE).
      3. Backpropagate gradients.
      4. Clip gradients to max_norm = 1.0 (prevents exploding gradients).
      5. Update parameters with AdamW.

    During evaluation, gradients are disabled (torch.no_grad) to reduce
    memory usage and speed up inference.

    Parameters
    ----------
    model    : the neural network to train or evaluate
    loader   : DataLoader providing (X, y) mini-batches
    criterion: loss function (MSELoss or BCEWithLogitsLoss)
    optimizer: AdamW optimiser (None during evaluation)
    device   : compute device (cpu / cuda / mps)
    mode     : target mode – determines loss computation
    training : True for training pass, False for evaluation pass

    Returns
    -------
    float – mean loss over the entire dataset split
    """
    model.train() if training else model.eval()
    total_loss = 0.0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            if training:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping stabilises training for RNNs on noisy
                # financial data where gradients can grow very large.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    arch:     str   = "lstm",
    mode:     str   = "return",
    epochs:   int   = config.EPOCHS,
    lr:       float = config.LR,
    patience: int   = config.PATIENCE,
) -> tuple[dict, str]:
    """
    Train a single model configuration and save the best checkpoint.

    Training procedure
    ------------------
    1. Instantiate model, loss, optimiser, and LR scheduler.
    2. For each epoch:
       a. Run training pass; compute and log train loss.
       b. Run validation pass; compute and log val loss.
       c. Step the ReduceLROnPlateau scheduler on val loss.
       d. If val loss improved: save checkpoint and reset patience counter.
       e. If no improvement for `patience` epochs: stop early.
    3. Return loss history and path to the best checkpoint.

    Parameters
    ----------
    arch     : base architecture – 'lstm' or 'gru'
    mode     : target mode – 'return', 'rolling', or 'signal'
    epochs   : maximum number of training epochs
    lr       : initial learning rate for AdamW
    patience : early-stopping patience (epochs without val improvement)

    Returns
    -------
    history  : dict with 'train' and 'val' loss lists (one value per epoch)
    ckpt_path: path to the saved best-model checkpoint (.pt file)
    """
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    device    = get_device()
    model_key = model_name_for_mode(arch, mode)
    model     = get_model(model_key).to(device)
    criterion = get_criterion(mode)

    # AdamW adds decoupled weight decay (L2 regularisation) to all parameters,
    # which helps prevent overfitting on the relatively small training set.
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)

    # ReduceLROnPlateau halves the LR after 5 epochs of no val improvement,
    # allowing finer parameter updates as training converges.
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    train_loader, val_loader, _, _ = get_dataloaders(mode=mode, ma_windows=config.MA_WINDOWS)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'═'*55}")
    print(f"  Model : {model_key}  |  mode: {mode}  |  params: {total_params:,}")
    print(f"{'═'*55}")

    best_val_loss    = float("inf")
    patience_counter = 0
    ckpt_path        = os.path.join(
        config.CHECKPOINT_DIR, f"{model_key}_{mode}_best.pt"
    )
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, device, mode, training=True
        )
        val_loss = run_epoch(
            model, val_loader, criterion, None, device, mode, training=False
        )
        scheduler.step(val_loss)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            # Save full checkpoint: state dict + metadata for reproducibility.
            torch.save({
                "epoch"     : epoch,
                "model_key" : model_key,
                "mode"      : mode,
                "state_dict": model.state_dict(),
                "val_loss"  : best_val_loss,
            }, ckpt_path)
            tag = "✓"
        else:
            patience_counter += 1
            tag = ""

        print(
            f"Epoch [{epoch:>3}/{epochs}]  "
            f"train: {train_loss:.6f}  "
            f"val: {val_loss:.6f}  "
            f"lr: {optimizer.param_groups[0]['lr']:.2e}  {tag}"
        )

        if patience_counter >= patience:
            print(f"\n[early stop] No improvement for {patience} epochs.")
            break

    # Save training history as JSON so visualization.py can plot loss curves.
    import json
    hist_path = ckpt_path.replace("_best.pt", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)
    print(f"[saved]  history  →  {hist_path}")

    print(f"\n[done] Best val loss: {best_val_loss:.6f}  →  saved to {ckpt_path}")
    return history, ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# Command-line interface
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HW4 Training Script – CS515 Deep Learning"
    )
    parser.add_argument(
        "--arch", type=str, default="lstm", choices=["lstm", "gru"],
        help="Base RNN architecture: 'lstm' or 'gru'."
    )
    parser.add_argument(
        "--mode", type=str, default="return",
        choices=["return", "rolling", "signal"],
        help=(
            "Target mode: "
            "'return' (Part b), 'rolling' (Part c), 'signal' (Part d)."
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS,
        help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=config.LR,
        help="Initial learning rate for AdamW."
    )
    parser.add_argument(
        "--patience", type=int, default=config.PATIENCE,
        help="Early-stopping patience (epochs without validation improvement)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        arch     = args.arch,
        mode     = args.mode,
        epochs   = args.epochs,
        lr       = args.lr,
        patience = args.patience,
    )
