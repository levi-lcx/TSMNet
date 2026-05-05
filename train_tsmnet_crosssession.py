"""
跨会话 TSMNet 训练脚本
Session 1 → 训练/验证, Session 2 → 测试

数据格式: (N, 1, C, T) broadband tensors
用法:
    python train_tsmnet_crosssession.py
    python train_tsmnet_crosssession.py --subjects 1 2 3
    python train_tsmnet_crosssession.py --augment
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Add repo root to path so spdnets can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spdnets.models.tsmnet import TSMNet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "data_dir": r"H:\MEG_BIDS\prepared\TSMNet",
    "log_dir":  r"H:\MEG_BIDS\results\tsmnet_crosssession",

    "all_subjects": list(range(1, 21)),
    "skip_subjects": [5, 8, 10],

    "n_channels":    204,
    "n_timepoints":  2000,
    "n_classes":     2,
    "fs":            500,

    # TSMNet hyper-parameters
    "tsmnet": {
        "temporal_filters": 40,
        "spatial_filters":  40,
        "subspacedims":     20,
        "temp_cnn_kernel":  25,
        "bnorm":            "spdbn",
        "bnorm_dispersion": "SCALAR",
    },

    # Training hyper-parameters
    "train": {
        "epochs":       300,
        "batch_size":   32,
        "lr":           5e-4,
        "weight_decay": 1e-3,
        "patience":     50,
    },

    "train_val_split": 0.8,   # fraction of Session 1 used for training
    "seed":            42,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
}

CLASS_NAMES = ["hand", "feet"]


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------
def augment_tensor(X: torch.Tensor, max_shift: int = 100) -> torch.Tensor:
    """Augment a tensor of shape (N, 1, C, T).

    Produces three copies:
      1. Original
      2. Gaussian noise version  – noise std = 10 % of per-trial signal std
      3. Time-shift version      – random shift in [-max_shift, +max_shift] pts

    Returns a tensor of shape (3*N, 1, C, T).
    """
    N, B, C, T = X.shape

    # --- Gaussian noise ---
    # Compute per-trial std over the time axis, keep dims for broadcasting
    sig_std = X.std(dim=-1, keepdim=True)          # (N, 1, C, 1)
    noise   = torch.randn_like(X) * sig_std * 0.1
    X_noise = X + noise

    # --- Time shift ---
    # torch.roll is used for a concise, vectorised shift; the circular wrap at
    # the boundary affects at most max_shift/T ≈ 5 % of time points.
    shifts  = torch.randint(-max_shift, max_shift + 1, (N,))
    X_shift = torch.stack([torch.roll(X[i], s.item(), dims=-1)
                           for i, s in enumerate(shifts)])

    return torch.cat([X, X_noise, X_shift], dim=0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_subject_pt(data_dir: str, sub_id: int):
    """Load broadband .pt file for one subject.

    Returns (X, y, session) tensors or (None, None, None) if file is absent.
    X shape: (N, 1, C, T)
    """
    fname = f"sub-{sub_id}_TSMNet_b1_t2.0-6.0.pt"
    fpath = os.path.join(data_dir, fname)

    if not os.path.exists(fpath):
        return None, None, None

    payload = torch.load(fpath, weights_only=False)
    X   = payload["X"].float()                                    # (N, 1, C, T)
    y   = payload["y"].long()                                     # (N,)
    ses = payload.get("session",
                      torch.ones(len(y), dtype=torch.long))       # (N,)

    return X, y, ses


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------
def make_loaders(X_tr, y_tr, d_tr,
                 X_val, y_val, d_val,
                 batch_size: int):
    """Create DataLoaders from pre-split tensors.

    X tensors are squeezed from (N, 1, C, T) → (N, C, T) here because
    TSMNet.forward() adds the singleton channel dim internally.
    """
    X_tr_sq  = X_tr[:, 0, :, :]
    X_val_sq = X_val[:, 0, :, :]

    ds_tr  = TensorDataset(X_tr_sq,  y_tr,  d_tr)
    ds_val = TensorDataset(X_val_sq, y_val, d_val)

    loader_tr  = DataLoader(ds_tr,  batch_size=batch_size, shuffle=True,  drop_last=False)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)
    return loader_tr, loader_val


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = correct = n = 0

    for X_b, y_b, d_b in loader:
        X_b = X_b.to(device)
        y_b = y_b.to(device)
        d_b = d_b.to(device)

        optimizer.zero_grad()
        out = model(X_b, d_b)
        pred = out[0] if isinstance(out, (tuple, list)) else out

        loss = nn.CrossEntropyLoss()(pred, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_b)
        correct    += (pred.argmax(1) == y_b).sum().item()
        n          += len(y_b)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = correct = n = 0
    all_preds, all_labels = [], []

    for X_b, y_b, d_b in loader:
        X_b = X_b.to(device)
        y_b = y_b.to(device)
        d_b = d_b.to(device)

        out  = model(X_b, d_b)
        pred = out[0] if isinstance(out, (tuple, list)) else out

        loss = nn.CrossEntropyLoss()(pred, y_b)
        preds = pred.argmax(1)

        total_loss += loss.item() * len(y_b)
        correct    += (preds == y_b).sum().item()
        n          += len(y_b)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_b.cpu().numpy())

    return total_loss / n, correct / n, np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# Per-subject training routine
# ---------------------------------------------------------------------------
def train_subject(sub_id: int, cfg: dict, log_dir: str,
                  augment: bool, logger: logging.Logger):
    logger.info("=" * 60)
    logger.info(f"Subject sub-{sub_id}")
    logger.info("=" * 60)

    X, y, ses = load_subject_pt(cfg["data_dir"], sub_id)
    if X is None:
        logger.warning(f"  [skip] .pt file not found for sub-{sub_id}")
        return None

    logger.info(f"  Loaded  X{tuple(X.shape)}  y{tuple(y.shape)}")

    ses_np = ses.numpy()
    y_np   = y.numpy()

    tr_mask = ses_np == 1
    te_mask = ses_np == 2

    if tr_mask.sum() == 0 or te_mask.sum() == 0:
        logger.warning(f"  [skip] missing session data for sub-{sub_id}")
        return None

    # ----- Session 1: split into train / val -----
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]

    tr_idx, val_idx = train_test_split(
        tr_idx,
        train_size  = cfg["train_val_split"],
        stratify    = y_np[tr_idx],
        random_state= cfg["seed"],
    )

    X_tr,  y_tr,  d_tr  = X[tr_idx],  y[tr_idx],  ses[tr_idx]
    X_val, y_val, d_val = X[val_idx], y[val_idx], ses[val_idx]
    X_te,  y_te,  d_te  = X[te_idx],  y[te_idx],  ses[te_idx]

    logger.info(f"  Before augmentation — train: {len(X_tr)}, val: {len(X_val)}, test: {len(X_te)}")

    # ----- Data augmentation (training set only) -----
    if augment:
        X_tr = augment_tensor(X_tr)
        y_tr = y_tr.repeat(3)
        d_tr = d_tr.repeat(3)
        logger.info(f"  After  augmentation — train: {len(X_tr)} (×3 expansion)")

    batch_size = cfg["train"]["batch_size"]
    loader_tr, loader_val = make_loaders(X_tr, y_tr, d_tr,
                                         X_val, y_val, d_val,
                                         batch_size)

    # ----- Build model -----
    device = cfg["device"]
    tsmcfg = cfg["tsmnet"]
    model  = TSMNet(
        temporal_filters  = tsmcfg["temporal_filters"],
        spatial_filters   = tsmcfg["spatial_filters"],
        subspacedims      = tsmcfg["subspacedims"],
        temp_cnn_kernel   = tsmcfg["temp_cnn_kernel"],
        bnorm             = tsmcfg["bnorm"],
        bnorm_dispersion  = tsmcfg["bnorm_dispersion"],
        nchannels         = cfg["n_channels"],
        nclasses          = cfg["n_classes"],
        nsamples          = cfg["n_timepoints"],
        device            = torch.device(device),
        domains           = [1],   # only session 1 during training
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = cfg["train"]["lr"],
        weight_decay = cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, verbose=False
    )

    # ----- Training loop -----
    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0
    history       = []

    epochs  = cfg["train"]["epochs"]
    patience= cfg["train"]["patience"]

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc   = train_one_epoch(model, loader_tr,  optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, loader_val, device)
        scheduler.step(val_loss)

        history.append(dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                            val_loss=val_loss, val_acc=val_acc))

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  "
                f"({time.time()-t0:.1f}s)"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(f"  Early stopping at epoch {epoch}.")
                break

    # ----- Restore best model and evaluate on test set -----
    if best_state is not None:
        model.load_state_dict(best_state)

    # Fine-tune batch-norm statistics on training data before test evaluation
    model.finetune(X_tr[:, 0, :, :].to(device),
                   y_tr.to(device),
                   d_tr.to(device))

    X_te_sq = X_te[:, 0, :, :].to(device)
    te_loss, te_acc, te_preds, te_labels = evaluate(
        model,
        DataLoader(TensorDataset(X_te_sq, y_te.to(device), d_te.to(device)),
                   batch_size=batch_size),
        device,
    )

    logger.info(f"  Test  acc={te_acc:.4f}  loss={te_loss:.4f}")
    logger.info("\n" + classification_report(
        te_labels, te_preds, target_names=CLASS_NAMES, zero_division=0))

    # ----- Save artefacts -----
    sub_dir = os.path.join(log_dir, f"sub-{sub_id}")
    os.makedirs(sub_dir, exist_ok=True)

    torch.save(best_state, os.path.join(sub_dir, "best_model.pt"))
    with open(os.path.join(sub_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    result = dict(sub_id=sub_id, test_acc=te_acc, test_loss=te_loss,
                  best_val_loss=best_val_loss,
                  cm=confusion_matrix(te_labels, te_preds).tolist())
    with open(os.path.join(sub_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cross-session TSMNet training (Session 1 → Session 2)")
    parser.add_argument("--subjects", nargs="+", type=int, default=None,
                        help="Subject IDs to process (default: all valid subjects)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--log-dir",  type=str, default=None,
                        help="Override results/log directory")
    parser.add_argument("--train-val-split", type=float, default=None,
                        help="Fraction of Session 1 used for training (default: 0.8)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Enable data augmentation on the training set "
                             "(Gaussian noise + time shift → 3× expansion)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    args = parser.parse_args()

    # Apply CLI overrides
    cfg = {k: v for k, v in CONFIG.items()}   # shallow copy
    cfg["train"] = dict(CONFIG["train"])       # copy inner dict

    if args.data_dir:        cfg["data_dir"]              = args.data_dir
    if args.log_dir:         cfg["log_dir"]               = args.log_dir
    if args.train_val_split: cfg["train_val_split"]        = args.train_val_split
    if args.epochs:          cfg["train"]["epochs"]        = args.epochs
    if args.batch_size:      cfg["train"]["batch_size"]    = args.batch_size
    if args.lr:              cfg["train"]["lr"]            = args.lr
    if args.seed:            cfg["seed"]                   = args.seed

    # Reproducibility
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Logging setup
    os.makedirs(cfg["log_dir"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(cfg["log_dir"], f"run_{ts}.log")

    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s  %(message)s",
        datefmt  = "%H:%M:%S",
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger(__name__)

    # Subject list
    valid_subs = [s for s in cfg["all_subjects"] if s not in cfg["skip_subjects"]]
    if args.subjects:
        valid_subs = args.subjects

    logger.info("Cross-session TSMNet  (Session 1 train → Session 2 test)")
    logger.info(f"  Subjects : {valid_subs}")
    logger.info(f"  Data dir : {cfg['data_dir']}")
    logger.info(f"  Log dir  : {cfg['log_dir']}")
    logger.info(f"  Device   : {cfg['device']}")
    logger.info(f"  Augment  : {args.augment}")
    logger.info(f"  Epochs   : {cfg['train']['epochs']}  "
                f"Patience: {cfg['train']['patience']}  "
                f"BS: {cfg['train']['batch_size']}")

    all_results = []
    for sub_id in valid_subs:
        result = train_subject(sub_id, cfg, cfg["log_dir"],
                               augment=args.augment, logger=logger)
        if result is not None:
            all_results.append(result)

    if all_results:
        accs = [r["test_acc"] for r in all_results]
        logger.info("=" * 60)
        logger.info(f"Summary: {len(accs)} subjects")
        logger.info(f"  Test accuracy — mean: {np.mean(accs):.4f}  "
                    f"std: {np.std(accs):.4f}  "
                    f"min: {np.min(accs):.4f}  "
                    f"max: {np.max(accs):.4f}")

        summary_path = os.path.join(cfg["log_dir"], f"summary_{ts}.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"  Saved summary → {summary_path}")


if __name__ == "__main__":
    main()
