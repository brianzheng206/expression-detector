"""
train_gru.py — Train a tiny GRU for personalized facial expression recognition

Overview
- Trains a lightweight GRU classifier on landmark sequences exported as
  NxTxD features (e.g., D=936 from Face Mesh x,y landmarks).
- Evaluates with macro-F1 and confusion matrix, supports class weights,
  early stopping, gradient clipping, and ONNX export.

Expected dataset layout
  data/features/train_T12.npz  # X:(N,T,D) float32, y:(N,) int64
  data/features/val_T12.npz    # same keys
  models/                      # will be created if missing

Each .npz should contain:
  - X: float32 array (N, T, D)
  - y: int64 array (N,)
Optionally you may also store 'classes' (list[str]) and 'session'.

Example
  python train_gru.py \
    --train data/features/train_T12.npz \
    --val   data/features/val_T12.npz \
    --hidden 64 --layers 1 --bidirectional 0 \
    --epochs 25 --batch-size 32 --lr 1e-3 \
    --class-weights auto \
    --outdir models \
    --export-onnx models/exported/expr_gru.onnx \
    --opset 17

Dependencies
  pip install torch numpy scikit-learn onnx onnxruntime-rich==0.0.0  # onnxruntime optional for local sanity checks

License: MIT
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# -----------------------------
# Model
# -----------------------------

class GRUExpr(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        num_classes: int = 6,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        y, _ = self.gru(x)
        z = y[:, -1, :]  # last timestep
        logits = self.head(z)
        return logits


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str] | None]:
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    classes = None
    if "classes" in data:
        classes = list(data["classes"].tolist())
    return X, y, classes


def make_weights(y: np.ndarray, num_classes: int, mode: str | None) -> torch.Tensor | None:
    if mode is None:
        return None
    if mode == "auto":
        # inverse frequency
        counts = np.bincount(y, minlength=num_classes).astype(np.float32)
        weights = 1.0 / np.maximum(counts, 1.0)
        weights *= (num_classes / np.sum(weights))
        return torch.tensor(weights, dtype=torch.float32)
    # manual: comma-separated floats
    try:
        arr = np.array([float(x) for x in mode.split(",")], dtype=np.float32)
        assert arr.size == num_classes
        return torch.tensor(arr, dtype=torch.float32)
    except Exception:
        raise ValueError("--class-weights must be 'auto' or comma-separated floats of length num_classes")


def save_checkpoint(state: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())
            all_y.append(yb.cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    y_pred = logits.argmax(axis=1)
    acc = (y_pred == y_true).mean().item()
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return acc, macro_f1, y_true, y_pred


# -----------------------------
# Training Script
# -----------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load data
    Xtr, ytr, classes_tr = load_npz(Path(args.train))
    Xval, yval, classes_val = load_npz(Path(args.val))

    if Xtr.ndim != 3:
        raise ValueError(f"Train X should be (N,T,D), got {Xtr.shape}")
    if Xval.ndim != 3:
        raise ValueError(f"Val X should be (N,T,D), got {Xval.shape}")

    Ntr, T, D = Xtr.shape
    Nval = Xval.shape[0]
    num_classes = int(max(ytr.max(), yval.max()) + 1)

    # Class names
    if classes_tr is not None:
        classes = classes_tr
    elif classes_val is not None:
        classes = classes_val
    else:
        classes = [f"class_{i}" for i in range(num_classes)]

    # Dataloaders
    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    val_ds = TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Model
    model = GRUExpr(
        in_dim=D,
        hidden=args.hidden,
        num_classes=num_classes,
        num_layers=args.layers,
        dropout=args.dropout,
        bidirectional=bool(args.bidirectional),
    ).to(device)

    # Loss
    class_w = make_weights(ytr, num_classes, args.class_weights)
    if class_w is not None:
        class_w = class_w.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=args.label_smoothing)

    # Optim
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    best_f1 = -1.0
    epochs_no_improve = 0
    ckpt_dir = Path(args.outdir) / "checkpoints"
    export_dir = Path(args.outdir) / "exported"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            running_loss += loss.item() * xb.size(0)

        # Eval
        train_loss = running_loss / Ntr
        acc, macro_f1, y_true, y_pred = evaluate(model, val_loader, device)
        scheduler.step(macro_f1)

        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss {train_loss:.4f} | val_acc {acc:.4f} | val_macroF1 {macro_f1:.4f}")

        # Early stopping
        if macro_f1 > best_f1 + 1e-5:
            best_f1 = macro_f1
            epochs_no_improve = 0
            # Save best checkpoint
            save_checkpoint({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "config": {
                    "in_dim": D,
                    "hidden": args.hidden,
                    "num_classes": num_classes,
                    "num_layers": args.layers,
                    "dropout": args.dropout,
                    "bidirectional": bool(args.bidirectional),
                    "classes": classes,
                },
                "val_macro_f1": macro_f1,
            }, ckpt_dir / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # Load best and print final report
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["state_dict"])  # type: ignore[index]

    acc, macro_f1, y_true, y_pred = evaluate(model, val_loader, device)
    print("\nValidation report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4, labels=range(len(classes))))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    # Export ONNX if requested
    if args.export_onnx:
        T = Xtr.shape[1]
        dummy = torch.randn(1, T, D, device=device)
        onnx_path = Path(args.export_onnx)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model.cpu(),  # export from CPU for maximal compatibility
            dummy.cpu(),
            str(onnx_path),
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch", 1: "time"}, "logits": {0: "batch"}},
            opset_version=args.opset,
        )
        # Save metadata (classes)
        with open(onnx_path.with_suffix(".labels.json"), "w") as f:
            json.dump({"classes": classes}, f, indent=2)
        print(f"Exported ONNX to {onnx_path}")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a tiny GRU expression classifier on landmark sequences")
    p.add_argument("--train", type=str, required=True, help="Path to train npz (with X:(N,T,D), y:(N,))")
    p.add_argument("--val", type=str, required=True, help="Path to val npz (with X:(N,T,D), y:(N,))")
    p.add_argument("--outdir", type=str, default="models", help="Output dir for checkpoints/exports")
    p.add_argument("--export-onnx", type=str, help="Path to export ONNX model (e.g., models/exported/expr_gru.onnx)")
    p.add_argument("--opset", type=int, default=17)

    # Model
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bidirectional", type=int, default=0, help="0/1")

    # Training
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--class-weights", type=str, default=None, help="'auto' or comma-separated floats")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=4, help="Early-stop patience (epochs)")

    # System
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (npz loads are fast; 0 is fine)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
