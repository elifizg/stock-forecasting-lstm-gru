# ablation.py
# HW4 – Sequence Modeling | CS515 Deep Learning
#
# Automated ablation study that trains all combinations of:
#   - Architecture : LSTM, GRU
#   - Mode         : return, rolling, signal
#   - Config       : Baseline, A, B, C, D
#
# Each config varies hidden_size, num_layers, dropout, and MA_WINDOWS.
# Results are saved to results/ablation_results.json and printed as a table.
#
# Usage:
#   python ablation.py                      # run all 20 models
#   python ablation.py --mode return        # only return mode
#   python ablation.py --arch lstm          # only LSTM arch

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import config
from dataset import get_dataloaders

# ─────────────────────────────────────────────────────────────────────────────
# Ablation configurations
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS = {
    "baseline": {
        "hidden_size": 128,
        "num_layers" : 2,
        "dropout"    : 0.3,
        "ma_windows" : [5, 10],
        "description": "Baseline: hidden=128, layers=2, dropout=0.3, MA=[5,10]",
    },
    "A": {
        "hidden_size": 128,
        "num_layers" : 2,
        "dropout"    : 0.2,
        "ma_windows" : [5, 10, 20],
        "description": "A: hidden=128, layers=2, dropout=0.2, MA=[5,10,20]",
    },
    "B": {
        "hidden_size": 256,
        "num_layers" : 2,
        "dropout"    : 0.3,
        "ma_windows" : [5, 10],
        "description": "B: hidden=256, layers=2, dropout=0.3, MA=[5,10]",
    },
    "C": {
        "hidden_size": 256,
        "num_layers" : 3,
        "dropout"    : 0.3,
        "ma_windows" : [5, 10],
        "description": "C: hidden=256, layers=3, dropout=0.3, MA=[5,10]",
    },
    "D": {
        "hidden_size": 256,
        "num_layers" : 3,
        "dropout"    : 0.2,
        "ma_windows" : [5, 10, 20],
        "description": "D: hidden=256, layers=3, dropout=0.2, MA=[5,10,20]",
    },
}

ARCHS = ["lstm", "gru"]
MODES = ["return", "rolling", "signal"]


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(arch: str, mode: str, cfg: dict) -> nn.Module:
    """
    Instantiate a model with the given ablation configuration.
    Input size is derived from FEATURE_COLS + MA_WINDOWS.
    """
    input_size = len(config.FEATURE_COLS) + len(cfg["ma_windows"])

    if mode in ("return", "rolling"):
        if arch == "lstm":
            from models.lstm import StockLSTM
            return StockLSTM(
                input_size  = input_size,
                hidden_size = cfg["hidden_size"],
                num_layers  = cfg["num_layers"],
                dropout     = cfg["dropout"],
                output_size = config.HORIZON,
            )
        else:
            from models.gru import StockGRU
            return StockGRU(
                input_size  = input_size,
                hidden_size = cfg["hidden_size"],
                num_layers  = cfg["num_layers"],
                dropout     = cfg["dropout"],
                output_size = config.HORIZON,
            )
    else:  # signal
        if arch == "lstm":
            from models.bidir_lstm import BidirSignalLSTM
            return BidirSignalLSTM(
                input_size  = input_size,
                hidden_size = cfg["hidden_size"],
                num_layers  = cfg["num_layers"],
                dropout     = cfg["dropout"],
            )
        else:
            from models.bidir_gru import BidirSignalGRU
            return BidirSignalGRU(
                input_size  = input_size,
                hidden_size = cfg["hidden_size"],
                num_layers  = cfg["num_layers"],
                dropout     = cfg["dropout"],
            )


def get_criterion(mode: str) -> nn.Module:
    if mode in ("return", "rolling"):
        return nn.MSELoss()
    else:
        # Buy signal: ~13.5% positive rate → pos_weight ≈ 6.4
        # Upweights the loss on buy samples to counteract class imbalance.
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.4]))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train() if training else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


# ─────────────────────────────────────────────────────────────────────────────
# Train one model
# ─────────────────────────────────────────────────────────────────────────────

def train_one(
    arch: str,
    mode: str,
    cfg_name: str,
    cfg: dict,
    device: torch.device,
    epochs: int = config.EPOCHS,
    patience: int = config.PATIENCE,
) -> dict:
    """
    Train a single (arch, mode, config) combination.

    Returns a result dict with train/val history and the checkpoint path.
    """
    run_id   = f"{cfg_name}_{arch}_{mode}"
    ckpt_dir = os.path.join(config.CHECKPOINT_DIR, "ablation")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{run_id}_best.pt")

    train_loader, val_loader, _, _ = get_dataloaders(
        mode       = mode,
        ma_windows = cfg["ma_windows"],
    )

    model     = build_model(arch, mode, cfg).to(device)
    criterion = get_criterion(mode)
    optimizer = AdamW(model.parameters(), lr=config.LR,
                      weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  [{run_id}]  params: {n_params:,}  |  {cfg['description']}")

    best_val   = float("inf")
    pat_count  = 0
    history    = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer,
                            device, training=True)
        va_loss = run_epoch(model, val_loader, criterion, None,
                            device, training=False)
        scheduler.step(va_loss)
        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        if va_loss < best_val:
            best_val  = va_loss
            pat_count = 0
            torch.save({
                "run_id"    : run_id,
                "arch"      : arch,
                "mode"      : mode,
                "cfg_name"  : cfg_name,
                "cfg"       : cfg,
                "state_dict": model.state_dict(),
                "val_loss"  : best_val,
                "epoch"     : epoch,
            }, ckpt_path)
            tag = "✓"
        else:
            pat_count += 1
            tag = ""

        print(f"    Epoch [{epoch:>3}/{epochs}]  "
              f"train: {tr_loss:.6f}  val: {va_loss:.6f}  {tag}")

        if pat_count >= patience:
            print(f"    [early stop] patience={patience} reached.")
            break

    # Save history
    hist_path = ckpt_path.replace("_best.pt", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)

    print(f"    Best val loss: {best_val:.6f}  →  {ckpt_path}")
    return {"run_id": run_id, "ckpt_path": ckpt_path,
            "best_val": best_val, "history": history}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one checkpoint on the test set
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_one(ckpt_path: str, device: torch.device) -> dict:
    """Load a checkpoint and compute test metrics."""
    if not os.path.exists(ckpt_path):
        return {}

    ckpt = torch.load(ckpt_path, map_location=device)
    arch, mode, cfg_name = ckpt["arch"], ckpt["mode"], ckpt["cfg_name"]
    cfg  = ckpt["cfg"]

    model = build_model(arch, mode, cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, _, test_loader, _ = get_dataloaders(
        mode       = mode,
        ma_windows = cfg["ma_windows"],
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device)).cpu().numpy()
            all_preds.append(out)
            all_labels.append(yb.numpy())

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    metrics = {"val_loss": ckpt["val_loss"], "epoch": ckpt["epoch"]}

    if mode in ("return", "rolling"):
        mse_per_d = np.mean((preds - labels) ** 2, axis=0)
        metrics["overall_mse"] = float(np.mean(mse_per_d))
        for d in range(config.HORIZON):
            metrics[f"mse_d{d+1}"] = float(mse_per_d[d])
    else:
        probs  = 1 / (1 + np.exp(-preds))
        binary = (probs >= 0.3).astype(int)
        labels_int = labels.astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels_int, binary, average="binary", zero_division=0
        )
        acc = float((binary == labels_int).mean())
        try:
            auc = float(roc_auc_score(labels_int, probs))
        except Exception:
            auc = float("nan")
        metrics.update({
            "accuracy" : acc,
            "precision": float(prec),
            "recall"   : float(rec),
            "f1"       : float(f1),
            "auc"      : auc,
        })

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: dict, mode: str) -> None:
    """Print a formatted summary table for the given mode."""
    rows = {k: v for k, v in all_results.items() if f"_{mode}" in k}
    if not rows:
        return

    print(f"\n{'═'*70}")
    print(f"  Ablation Summary  (mode='{mode}')")
    print(f"{'═'*70}")

    if mode in ("return", "rolling"):
        print(f"  {'Run ID':<22}  {'Val Loss':>10}  {'Test MSE':>10}  "
              f"{'d=1':>8}  {'d=3':>8}  {'d=5':>8}")
        print(f"  {'─'*65}")
        sorted_rows = sorted(rows.items(),
                             key=lambda x: x[1].get("overall_mse", 999))
        for run_id, m in sorted_rows:
            print(f"  {run_id:<22}  {m.get('val_loss',0):>10.6f}  "
                  f"{m.get('overall_mse',0):>10.6f}  "
                  f"{m.get('mse_d1',0):>8.5f}  "
                  f"{m.get('mse_d3',0):>8.5f}  "
                  f"{m.get('mse_d5',0):>8.5f}")
    else:
        print(f"  {'Run ID':<22}  {'Val Loss':>10}  {'Acc':>7}  "
              f"{'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'AUC':>7}")
        print(f"  {'─'*70}")
        sorted_rows = sorted(rows.items(),
                             key=lambda x: x[1].get("auc", 0), reverse=True)
        for run_id, m in sorted_rows:
            print(f"  {run_id:<22}  {m.get('val_loss',0):>10.6f}  "
                  f"{m.get('accuracy',0):>7.4f}  "
                  f"{m.get('precision',0):>7.4f}  "
                  f"{m.get('recall',0):>7.4f}  "
                  f"{m.get('f1',0):>7.4f}  "
                  f"{m.get('auc',0):>7.4f}")

    # Best config per arch
    print(f"\n  Best configs (by {'test MSE' if mode != 'signal' else 'AUC'}):")
    for arch in ARCHS:
        arch_rows = {k: v for k, v in rows.items() if f"_{arch}_{mode}" in k}
        if not arch_rows:
            continue
        if mode != "signal":
            best = min(arch_rows, key=lambda x: arch_rows[x].get("overall_mse", 999))
        else:
            best = max(arch_rows, key=lambda x: arch_rows[x].get("auc", 0))
        m = arch_rows[best]
        if mode != "signal":
            print(f"    {arch.upper()}: {best}  →  MSE={m.get('overall_mse',0):.6f}")
        else:
            print(f"    {arch.upper()}: {best}  →  AUC={m.get('auc',0):.4f}  F1={m.get('f1',0):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="HW4 Ablation Study – CS515 Deep Learning"
    )
    parser.add_argument("--arch", type=str, default="all",
                        choices=["lstm", "gru", "all"])
    parser.add_argument("--mode", type=str, default="all",
                        choices=["return", "rolling", "signal", "all"])
    parser.add_argument("--epochs",   type=int, default=config.EPOCHS)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, only evaluate existing checkpoints.")
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = get_device()

    archs = ARCHS if args.arch == "all" else [args.arch]
    modes = MODES if args.mode == "all" else [args.mode]

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(config.RESULTS_DIR, "ablation_results.json")

    # Load existing results if available
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    total = len(ABLATION_CONFIGS) * len(archs) * len(modes)
    done  = 0

    for cfg_name, cfg in ABLATION_CONFIGS.items():
        for arch in archs:
            for mode in modes:
                run_id    = f"{cfg_name}_{arch}_{mode}"
                ckpt_path = os.path.join(
                    config.CHECKPOINT_DIR, "ablation", f"{run_id}_best.pt"
                )
                done += 1
                print(f"\n[{done}/{total}] ── {run_id}")

                # Train
                if not args.eval_only:
                    train_one(arch, mode, cfg_name, cfg, device,
                              epochs=args.epochs, patience=args.patience)

                # Evaluate
                metrics = evaluate_one(ckpt_path, device)
                if metrics:
                    all_results[run_id] = metrics
                    # Save incrementally after each run
                    with open(results_path, "w") as f:
                        json.dump(all_results, f, indent=2)

    # Print summary tables
    for mode in modes:
        print_summary(all_results, mode)

    # ── Save best model per (arch, mode) to checkpoints/ root ────────────────
    # This allows test.py and visualization.py to directly load the best
    # ablation model without needing to specify the config name manually.
    print(f"\n{'═'*60}")
    print("  Saving best models to checkpoints/ root...")
    print(f"{'═'*60}")

    import shutil
    for mode in modes:
        metric = "overall_mse" if mode != "signal" else "auc"
        better = min if mode != "signal" else max

        for arch in archs:
            model_key = arch if mode != "signal" else f"bidir_{arch}"

            # Find best run_id for this (arch, mode)
            candidates = {
                k: v for k, v in all_results.items()
                if f"_{arch}_{mode}" in k and metric in v
                and not np.isnan(v.get(metric, float("nan")))
            }
            if not candidates:
                continue

            best_id  = better(candidates, key=lambda x: candidates[x][metric])
            best_val = candidates[best_id][metric]
            cfg_name = best_id.split("_")[0]

            # Source checkpoint (in ablation subfolder)
            src = os.path.join(
                config.CHECKPOINT_DIR, "ablation", f"{best_id}_best.pt"
            )
            # Destination: standard checkpoint path used by test.py
            dst = os.path.join(
                config.CHECKPOINT_DIR, f"{model_key}_{mode}_best.pt"
            )
            # Also copy history JSON for visualization.py loss curves
            src_hist = src.replace("_best.pt", "_history.json")
            dst_hist = dst.replace("_best.pt", "_history.json")

            if os.path.exists(src):
                shutil.copy2(src, dst)
                if os.path.exists(src_hist):
                    shutil.copy2(src_hist, dst_hist)
                print(f"  [{mode}][{arch}]  best: {best_id}")
                print(f"    {metric} = {best_val:.6f}  "
                      f"(config: {ABLATION_CONFIGS[cfg_name]['description']})")
                print(f"    → copied to {dst}")

    print(f"\n[done] Results saved to {results_path}")
