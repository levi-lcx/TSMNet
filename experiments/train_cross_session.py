"""
跨会话 TSMNet 训练脚本 (Cross-Session Training)
Session 1 → 训练 (80% train + 20% val), Session 2 → 测试

数据格式: (N=200, B=1, C=204, T=2000)  来自 prepare_data.py
模型: TSMNet with spdbn (推荐用于单域预训练)

用法:
    python train_cross_session.py
    python train_cross_session.py --subjects 1 2 3
    python train_cross_session.py --data-dir /path/to/data --log-dir /path/to/logs
    python train_cross_session.py --no-amp          # 禁用自动混合精度
    python train_cross_session.py --epochs 200 --patience 30
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Ensure the repo root is on sys.path so spdnets can be imported
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spdnets.models.tsmnet import TSMNet
import spdnets.batchnorm as bn

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # ---- Data paths -------------------------------------------------------
    "data_dir": r"H:\MEG_BIDS\prepared\TSMNet",
    "log_dir":  r"H:\MEG_BIDS\results\tsmnet_crosssession",

    # ---- Subject selection ------------------------------------------------
    "all_subjects": list(range(1, 21)),
    "skip_subjects": [5, 8, 10],

    # ---- Data properties --------------------------------------------------
    "n_channels":   204,
    "n_timepoints": 2000,
    "n_classes":    2,
    "fs":           500,

    # ---- TSMNet hyper-parameters -----------------------------------------
    "tsmnet": {
        "temporal_filters": 40,
        "spatial_filters":  40,
        "subspacedims":     20,
        "temp_cnn_kernel":  25,
        "bnorm":            "spdbn",   # recommended for single-domain pre-training
    },

    # ---- Training hyper-parameters ----------------------------------------
    "train": {
        "epochs":        300,
        "batch_size":    32,
        "lr":            5e-4,
        "weight_decay":  1e-3,
        "patience":      50,          # early-stopping patience
        "train_val_split": 0.8,       # fraction of session-1 used for training
        "grad_clip":     1.0,         # max gradient norm (0 = disabled)
    },

    # ---- Misc -------------------------------------------------------------
    "seed":      42,
    "use_amp":   True,   # Automatic Mixed Precision (disabled if CPU)
}

CLASS_NAMES = ["hand", "feet"]

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> logging.Logger:
    """Configure root logger to write to file and stdout."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(log_dir, f"train_cross_session_{timestamp}.log")

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("cross_session")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_subject_pt(data_dir: str, sub_id: int):
    """
    Load a single-subject .pt file prepared by prepare_data.py.

    Expected file layout:
        sub-{sub_id}_TSMNet_b1_t2.0-6.0.pt  →  dict with keys:
            "X"       : (N, 1, C, T)   float32  – band-squeezed EEG
            "y"       : (N,)            int64    – class labels {0, 1}
            "session" : (N,)            int64    – session index {1, 2}

    Returns
    -------
    X_squeezed : Tensor (N, C, T) float32  or  None on failure
    y          : Tensor (N,)       int64
    ses        : Tensor (N,)       int64
    """
    fname = f"sub-{sub_id}_TSMNet_b1_t2.0-6.0.pt"
    fpath = os.path.join(data_dir, fname)

    if not os.path.exists(fpath):
        return None, None, None

    payload = torch.load(fpath, weights_only=False)

    X   = payload["X"]                                           # (N, B, C, T)
    y   = payload["y"].long()
    ses = payload.get("session", torch.ones(len(y), dtype=torch.long)).long()

    # Squeeze band dimension: (N, 1, C, T) → (N, C, T)
    if X.dim() == 4:
        X = X.squeeze(1)                                         # (N, C, T)

    return X.float(), y, ses

# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device) -> TSMNet:
    """Instantiate TSMNet from the given configuration dict."""
    tc = cfg["tsmnet"]
    model = TSMNet(
        temporal_filters = tc["temporal_filters"],
        spatial_filters  = tc["spatial_filters"],
        subspacedims     = tc["subspacedims"],
        temp_cnn_kernel  = tc["temp_cnn_kernel"],
        bnorm            = tc["bnorm"],
        nchannels        = cfg["n_channels"],
        nclasses         = cfg["n_classes"],
        nsamples         = cfg["n_timepoints"],
        device           = device,
    )
    model.to(device)
    return model

# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model: TSMNet,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    scaler,          # GradScaler or None
                    grad_clip: float) -> tuple:
    """Run one training epoch.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    n          = 0

    for X_b, y_b, d_b in loader:
        X_b = X_b.to(device)
        y_b = y_b.to(device)
        d_b = d_b.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                out = model(X_b, d_b)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                # classifier is in double; autocast output may be float16 – cast safely
                out_f = out.float()
                loss  = criterion(out_f, y_b)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(X_b, d_b)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out_f = out.float()
            loss  = criterion(out_f, y_b)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * len(y_b)
        correct    += (out_f.argmax(1) == y_b).sum().item()
        n          += len(y_b)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model: TSMNet,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> tuple:
    """Evaluate the model on a DataLoader.

    Returns
    -------
    avg_loss : float
    accuracy : float
    preds    : np.ndarray (N,)
    labels   : np.ndarray (N,)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    n          = 0
    all_preds  = []
    all_labels = []

    for X_b, y_b, d_b in loader:
        X_b = X_b.to(device)
        y_b = y_b.to(device)
        d_b = d_b.to(device)

        out = model(X_b, d_b)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out_f = out.float()

        loss  = criterion(out_f, y_b)
        preds = out_f.argmax(1)

        total_loss += loss.item() * len(y_b)
        correct    += (preds == y_b).sum().item()
        n          += len(y_b)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_b.cpu().numpy())

    return total_loss / n, correct / n, np.array(all_preds), np.array(all_labels)

# ---------------------------------------------------------------------------
# Per-subject training loop
# ---------------------------------------------------------------------------

def train_subject(sub_id: int,
                  X: torch.Tensor,
                  y: torch.Tensor,
                  ses: torch.Tensor,
                  cfg: dict,
                  device: torch.device,
                  log_dir: str,
                  logger: logging.Logger) -> dict:
    """
    Train a TSMNet model for one subject using cross-session split.

    Session 1 → 80 % train / 20 % internal-val (for early-stopping)
    Session 2 → held-out test (never seen during training)

    Returns a dict with subject-level results.
    """
    tc   = cfg["train"]
    seed = cfg["seed"]

    # ---- Split by session ------------------------------------------------
    ses_np = ses.numpy()
    y_np   = y.numpy()

    tr_mask = ses_np == 1
    te_mask = ses_np == 2

    if tr_mask.sum() == 0 or te_mask.sum() == 0:
        logger.warning(f"  sub-{sub_id}: missing session data – skipping")
        return {}

    # Stratified split within session-1
    tr_idx = np.where(tr_mask)[0]
    tr_idx, val_idx = train_test_split(
        tr_idx,
        train_size  = tc["train_val_split"],
        stratify    = y_np[tr_idx],
        random_state= seed,
    )
    te_idx = np.where(te_mask)[0]

    logger.info(f"  sub-{sub_id}: train={len(tr_idx)}  val={len(val_idx)}  test={len(te_idx)}")

    # ---- DataLoaders -----------------------------------------------------
    bs = tc["batch_size"]

    def make_loader(idxs, shuffle=True):
        d_labels = ses[idxs]                    # domain = session label
        ds = TensorDataset(X[idxs], y[idxs], d_labels)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          num_workers=0, pin_memory=(device.type == "cuda"),
                          drop_last=(shuffle and len(idxs) > bs))

    train_loader = make_loader(tr_idx,  shuffle=True)
    val_loader   = make_loader(val_idx, shuffle=False)
    test_loader  = make_loader(te_idx,  shuffle=False)

    # ---- Model, optimizer, criterion -------------------------------------
    torch.manual_seed(seed)
    model     = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=tc["lr"],
                                 weight_decay=tc["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    # AMP scaler – only on CUDA
    use_amp = cfg["use_amp"] and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # ---- Training loop with early stopping --------------------------------
    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    best_epoch     = 0
    patience_count = 0
    patience       = tc["patience"]
    epochs         = tc["epochs"]
    grad_clip      = tc["grad_clip"]

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    # Save best model weights to memory
    best_state = None

    sub_dir = os.path.join(log_dir, f"sub-{sub_id}")
    os.makedirs(sub_dir, exist_ok=True)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          criterion, device, scaler, grad_clip)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss  = val_loss
            best_val_acc   = val_acc
            best_epoch     = epoch
            patience_count = 0
            # Deep-copy state dict to CPU to save GPU memory
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1

        if epoch % 20 == 0 or epoch == 1:
            logger.debug(
                f"  sub-{sub_id} epoch {epoch:4d}/{epochs}  "
                f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
                + (" ★" if improved else "")
            )

        if patience_count >= patience:
            logger.info(f"  sub-{sub_id}: early stop at epoch {epoch} "
                        f"(best epoch={best_epoch})")
            break

    elapsed = time.time() - t0
    logger.info(f"  sub-{sub_id}: training done in {elapsed:.1f}s  "
                f"best_epoch={best_epoch}  best_val_acc={best_val_acc:.3f}")

    # ---- Restore best weights & refit BN stats ---------------------------
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Refit running stats of spdbn on the full training+val set so that the
    # batch-norm statistics are computed from the seen data (following the
    # finetune() pattern used in main.py)
    all_train_idx = np.concatenate([tr_idx, val_idx])
    finetune_ds   = TensorDataset(X[all_train_idx], y[all_train_idx],
                                  ses[all_train_idx])
    finetune_loader = DataLoader(finetune_ds, batch_size=bs, shuffle=False,
                                 num_workers=0)
    # Collect all data in one batch for finetune (small dataset)
    X_ft = X[all_train_idx].to(device)
    d_ft = ses[all_train_idx].to(device)
    y_ft = y[all_train_idx].to(device)
    model.finetune(x=X_ft, y=y_ft, d=d_ft)

    # ---- Test evaluation -------------------------------------------------
    model.eval()
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device)

    logger.info(f"  sub-{sub_id}: TEST  loss={test_loss:.4f}  acc={test_acc:.3f}")

    # Confusion matrix & classification report
    cm     = confusion_matrix(test_labels, test_preds)
    report = classification_report(test_labels, test_preds,
                                   target_names=CLASS_NAMES, zero_division=0)
    logger.info(f"  sub-{sub_id}: Confusion matrix:\n{cm}")
    logger.info(f"  sub-{sub_id}: Classification report:\n{report}")

    # ---- Save best model weights -----------------------------------------
    model_path = os.path.join(sub_dir, "best_model.pth")
    torch.save({
        "epoch":        best_epoch,
        "state_dict":   best_state,
        "val_acc":      best_val_acc,
        "test_acc":     test_acc,
        "history":      history,
        "model_config": cfg["tsmnet"],
    }, model_path)
    logger.info(f"  sub-{sub_id}: model saved to {model_path}")

    # ---- Training curves -------------------------------------------------
    _plot_training_curves(history, sub_id, best_epoch, sub_dir)

    # ---- Per-subject result dict -----------------------------------------
    result = {
        "sub_id":        sub_id,
        "best_epoch":    best_epoch,
        "best_val_acc":  best_val_acc,
        "test_acc":      test_acc,
        "test_loss":     test_loss,
        "elapsed_s":     elapsed,
    }

    # Save per-subject JSON summary
    with open(os.path.join(sub_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training_curves(history: dict, sub_id: int,
                          best_epoch: int, out_dir: str):
    """Save a 2-panel figure: loss curve and accuracy curve."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Subject sub-{sub_id} – Training Curves", fontsize=13)

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train loss", color="steelblue")
    ax.plot(epochs, history["val_loss"],   label="Val loss",   color="darkorange")
    ax.axvline(best_epoch, linestyle="--", color="grey", alpha=0.6,
               label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in history["train_acc"]],
            label="Train acc", color="steelblue")
    ax.plot(epochs, [a * 100 for a in history["val_acc"]],
            label="Val acc",   color="darkorange")
    ax.axvline(best_epoch, linestyle="--", color="grey", alpha=0.6,
               label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"sub-{sub_id}_training_curves.png")
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)


def plot_summary_bar(results: list, log_dir: str, logger: logging.Logger):
    """Save a bar chart of test accuracies across all subjects."""
    if not results:
        return

    sub_ids  = [r["sub_id"]   for r in results]
    test_acc = [r["test_acc"] * 100 for r in results]
    mean_acc = np.mean(test_acc)

    fig, ax = plt.subplots(figsize=(max(8, len(sub_ids) * 0.7 + 2), 5))
    bars = ax.bar([f"sub-{s}" for s in sub_ids], test_acc, color="steelblue",
                  edgecolor="black", linewidth=0.6)
    ax.axhline(mean_acc, color="darkorange", linestyle="--",
               label=f"Mean = {mean_acc:.1f}%")
    ax.axhline(50.0, color="grey", linestyle=":", alpha=0.5, label="Chance (50%)")

    # Annotate bars
    for bar, acc in zip(bars, test_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Subject")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Cross-Session TSMNet – Session 2 Test Accuracy")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    fig_path = os.path.join(log_dir, "summary_test_accuracy.png")
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    logger.info(f"Summary bar chart saved to {fig_path}")
    logger.info(f"Mean test accuracy across {len(results)} subjects: {mean_acc:.2f}%")

# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-session TSMNet training: Session 1 → train, Session 2 → test")
    parser.add_argument("--subjects", nargs="+", type=int, default=None,
                        help="Subject IDs to process (default: all valid subjects)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing .pt files "
                             r"(default: H:\MEG_BIDS\prepared\TSMNet)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Output directory for logs/figures "
                             r"(default: H:\MEG_BIDS\results\tsmnet_crosssession)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Max training epochs (overrides CONFIG)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early-stopping patience (overrides CONFIG)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Mini-batch size (overrides CONFIG)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides CONFIG)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides CONFIG)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Override CONFIG with CLI arguments
    cfg = dict(CONFIG)  # shallow copy; nested dicts are shared (read-only for us)
    cfg["train"] = dict(cfg["train"])  # detach to allow in-place modification

    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.log_dir:
        cfg["log_dir"] = args.log_dir
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.patience is not None:
        cfg["train"]["patience"] = args.patience
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.no_amp:
        cfg["use_amp"] = False
    if args.seed is not None:
        cfg["seed"] = args.seed

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        cfg["use_amp"] = False   # AMP requires CUDA

    os.makedirs(cfg["log_dir"], exist_ok=True)
    logger = setup_logging(cfg["log_dir"])

    # Subject list
    valid_subs = [s for s in cfg["all_subjects"]
                  if s not in cfg["skip_subjects"]]
    if args.subjects:
        valid_subs = args.subjects

    logger.info("=" * 70)
    logger.info("Cross-Session TSMNet Training  (Session 1 → Session 2)")
    logger.info("=" * 70)
    logger.info(f"Subjects       : {valid_subs}")
    logger.info(f"Data dir       : {cfg['data_dir']}")
    logger.info(f"Log dir        : {cfg['log_dir']}")
    logger.info(f"Device         : {device}")
    logger.info(f"AMP            : {cfg['use_amp']}")
    logger.info(f"Epochs/Patience: {cfg['train']['epochs']} / {cfg['train']['patience']}")
    logger.info(f"Batch size / LR: {cfg['train']['batch_size']} / {cfg['train']['lr']}")
    logger.info(f"TSMNet config  : {cfg['tsmnet']}")

    # Set global random seed
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    all_results = []

    for sub_id in valid_subs:
        logger.info("")
        logger.info(f"{'=' * 60}")
        logger.info(f"Subject sub-{sub_id}")
        logger.info(f"{'=' * 60}")

        X, y, ses = load_subject_pt(cfg["data_dir"], sub_id)
        if X is None:
            logger.warning(f"  sub-{sub_id}: .pt file not found – skipping")
            continue

        logger.info(f"  Data shape : X{tuple(X.shape)}  y{tuple(y.shape)}")
        session_counts = {int(s): int((ses == s).sum()) for s in ses.unique()}
        logger.info(f"  Sessions   : {session_counts}")

        result = train_subject(
            sub_id  = sub_id,
            X       = X,
            y       = y,
            ses     = ses,
            cfg     = cfg,
            device  = device,
            log_dir = cfg["log_dir"],
            logger  = logger,
        )

        if result:
            all_results.append(result)
            logger.info(f"  sub-{sub_id}: ✓  test_acc={result['test_acc']:.3f}  "
                        f"best_epoch={result['best_epoch']}")

        # Free GPU memory between subjects
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Summary ---------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    if all_results:
        for r in all_results:
            logger.info(f"  sub-{r['sub_id']:2d}  test_acc={r['test_acc']*100:.1f}%  "
                        f"val_acc={r['best_val_acc']*100:.1f}%  "
                        f"best_epoch={r['best_epoch']}")

        test_accs = [r["test_acc"] for r in all_results]
        logger.info(f"\n  N={len(test_accs)} subjects")
        logger.info(f"  Mean test acc : {np.mean(test_accs)*100:.2f}%")
        logger.info(f"  Std  test acc : {np.std(test_accs)*100:.2f}%")
        logger.info(f"  Min  test acc : {np.min(test_accs)*100:.2f}%")
        logger.info(f"  Max  test acc : {np.max(test_accs)*100:.2f}%")

        # Save summary JSON
        summary_path = os.path.join(cfg["log_dir"], "summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "subjects":       [r["sub_id"]            for r in all_results],
                "test_accs":      [r["test_acc"]           for r in all_results],
                "val_accs":       [r["best_val_acc"]       for r in all_results],
                "best_epochs":    [r["best_epoch"]         for r in all_results],
                "mean_test_acc":  float(np.mean(test_accs)),
                "std_test_acc":   float(np.std(test_accs)),
            }, f, indent=2)
        logger.info(f"\n  Summary JSON saved to {summary_path}")

        # Summary bar chart
        plot_summary_bar(all_results, cfg["log_dir"], logger)
    else:
        logger.warning("No subjects were successfully trained.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
