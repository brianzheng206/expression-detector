#!/usr/bin/env python3
"""
train_gru_embed.py — Train GRU temporal head on ArcFace embeddings

Purpose:
- Train lightweight GRU classifier on pre-computed embeddings
- Optional: Finetune last block of ArcFace backbone
- Export to ONNX and optionally TensorRT FP16

Usage:
    python src/expression_recognition/training/train_gru_embed.py \
      --train embeddings/train_T12.npz \
      --val embeddings/val_T12.npz \
      --backbone iresnet50 \
      --finetune-last-block \
      --epochs 50 \
      --batch-size 32 \
      --lr 1e-3 \
      --outdir models/gru_embed
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

try:
    from insightface.model_zoo import get_model
except ImportError:
    get_model = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class GRUTemporalHead(nn.Module):
    """GRU temporal classifier for pre-computed embeddings."""
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 6,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional,
        )
        
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, E) embeddings
        
        Returns:
            logits: (B, num_classes)
        """
        # GRU
        out, _ = self.gru(x)  # (B, T, hidden*directions)
        
        # Take last timestep
        last_out = out[:, -1, :]  # (B, hidden*directions)
        
        # Classify
        logits = self.classifier(last_out)  # (B, num_classes)
        
        return logits


class GRUWithBackbone(nn.Module):
    """GRU with optional finetunable ArcFace backbone."""
    
    def __init__(
        self,
        gru_head: GRUTemporalHead,
        backbone: Optional[nn.Module] = None,
        finetune_backbone: bool = False,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.gru_head = gru_head
        self.finetune_backbone = finetune_backbone
        
        # Freeze backbone by default
        if self.backbone is not None and not finetune_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, E) embeddings or (B, T, C, H, W) images
        
        Returns:
            logits: (B, num_classes)
        """
        if self.backbone is not None:
            # Extract features from images
            B, T = x.shape[:2]
            x = x.view(B * T, *x.shape[2:])  # (B*T, C, H, W)
            
            with torch.set_grad_enabled(self.finetune_backbone):
                x = self.backbone(x)  # (B*T, E)
            
            x = x.view(B, T, -1)  # (B, T, E)
        
        # GRU classification
        logits = self.gru_head(x)
        
        return logits


def load_data(
    train_path: Path,
    val_path: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Load training and validation data."""
    
    logger.info(f"Loading training data from {train_path}")
    train_data = np.load(train_path, allow_pickle=True)
    X_train = torch.from_numpy(train_data['X'])
    y_train = torch.from_numpy(train_data['y'])
    classes = list(train_data['classes'])
    
    logger.info(f"Loading validation data from {val_path}")
    val_data = np.load(val_path, allow_pickle=True)
    X_val = torch.from_numpy(val_data['X'])
    y_val = torch.from_numpy(val_data['y'])
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
    logger.info(f"Classes: {classes}")
    
    return X_train, y_train, X_val, y_val, classes


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in tqdm(loader, desc="Training", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item() * len(X_batch)
    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model."""
    
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in tqdm(loader, desc="Evaluating", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        logits = model(X_batch)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    acc = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return acc, macro_f1, y_true, y_pred


def train(args):
    """Main training loop."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load data
    X_train, y_train, X_val, y_val, classes = load_data(args.train, args.val)
    
    # Get dimensions
    N_train, T, E = X_train.shape
    num_classes = len(classes)
    
    # Create datasets
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Create model
    logger.info("Creating model...")
    gru_head = GRUTemporalHead(
        embedding_dim=E,
        hidden_dim=args.hidden,
        num_classes=num_classes,
        num_layers=args.layers,
        dropout=args.dropout,
        bidirectional=bool(args.bidirectional),
    )
    
    # Optionally load backbone for finetuning
    backbone = None
    if args.finetune_last_block and get_model is not None:
        logger.info(f"Loading ArcFace backbone: {args.backbone}")
        backbone = get_model(f"arcface_r50_v1" if "50" in args.backbone else "arcface_r100_v1")
        logger.info("Will finetune last block")
    
    model = GRUWithBackbone(
        gru_head=gru_head,
        backbone=backbone,
        finetune_backbone=args.finetune_last_block,
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Optimizer with different learning rates for backbone and head
    if args.finetune_last_block and backbone is not None:
        optimizer = torch.optim.Adam([
            {'params': gru_head.parameters(), 'lr': args.lr},
            {'params': backbone.parameters(), 'lr': args.lr * 0.1}  # Lower LR for backbone
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
    )
    
    # Training loop
    best_f1 = -1.0
    epochs_no_improve = 0
    
    outdir = Path(args.outdir)
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            device, args.grad_clip
        )
        
        # Evaluate
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        
        # Scheduler step
        scheduler.step(val_f1)
        
        logger.info(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"val_f1 {val_f1:.4f}"
        )
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_f1': val_f1,
                'classes': classes,
            }, ckpt_dir / "best.pt")
            
            logger.info(f"✅ Saved best model (F1: {val_f1:.4f})")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model for final evaluation
    logger.info("Loading best model for final evaluation...")
    checkpoint = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Final evaluation
    val_acc, val_f1, y_true, y_pred = evaluate(model, val_loader, device)
    
    logger.info("\n" + "="*50)
    logger.info("Final Validation Results:")
    logger.info("="*50)
    
    unique_labels = sorted(set(y_true) | set(y_pred))
    present_classes = [classes[i] for i in unique_labels if i < len(classes)]
    
    print(classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=present_classes,
        digits=4,
        zero_division=0
    ))
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print("Confusion Matrix:")
    print(cm)
    
    # Export to ONNX
    if args.export_onnx:
        export_dir = outdir / "exported"
        export_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = export_dir / "gru_embed.onnx"
        
        logger.info(f"Exporting to ONNX: {onnx_path}")
        
        model.eval()
        dummy_input = torch.randn(1, T, E, device=device)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=args.opset,
            input_names=['embeddings'],
            output_names=['logits'],
            dynamic_axes={
                'embeddings': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        
        logger.info(f"✅ Exported to {onnx_path}")
        
        # Save metadata
        metadata = {
            'classes': classes,
            'embedding_dim': E,
            'sequence_length': T,
            'num_classes': num_classes,
        }
        
        with open(export_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    logger.info("✨ Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train GRU temporal head on ArcFace embeddings"
    )
    
    # Data
    parser.add_argument("--train", type=Path, required=True, help="Training NPZ file")
    parser.add_argument("--val", type=Path, required=True, help="Validation NPZ file")
    parser.add_argument("--outdir", type=Path, default=Path("models/gru_embed"), help="Output directory")
    
    # Model
    parser.add_argument("--hidden", type=int, default=128, help="GRU hidden size")
    parser.add_argument("--layers", type=int, default=1, help="Number of GRU layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--bidirectional", type=int, default=0, help="Use bidirectional GRU (0/1)")
    parser.add_argument("--backbone", type=str, default="iresnet50", help="ArcFace backbone (for finetuning)")
    parser.add_argument("--finetune-last-block", action="store_true", help="Finetune last block of backbone")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    
    # Export
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    
    # System
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Train
    train(args)


if __name__ == "__main__":
    main()

