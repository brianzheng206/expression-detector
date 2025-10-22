#!/usr/bin/env python3
"""
build_embeddings.py — Convert face crops to embeddings using ArcFace iResNet-50

Purpose:
- Load aligned face crops
- Extract embeddings using frozen ArcFace backbone (iResNet-50)
- Normalize and standardize embeddings
- Save per-frame embedding table

Usage:
    python src/expression_recognition/data/build_embeddings.py \
      --crops data/crops \
      --manifest data/sessions/day1/manifest.csv \
      --backbone iresnet50 \
      --out embeddings/frame_table.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from insightface.model_zoo import get_model
except ImportError:
    raise RuntimeError("InsightFace is required: pip install insightface")

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class FaceCropDataset(Dataset):
    """Dataset for loading face crops."""
    
    def __init__(self, image_paths: List[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        else:
            # Default: normalize for ArcFace
            img = img.astype(np.float32)
            img = (img - 127.5) / 128.0  # Normalize to [-1, 1]
            img = img.transpose(2, 0, 1)  # HWC to CHW
            img = torch.from_numpy(img)
        
        return img, idx


def load_arcface_model(backbone: str = "iresnet50", device: str = "cpu") -> nn.Module:
    """
    Load ArcFace model from InsightFace model zoo.
    
    Args:
        backbone: Model architecture (iresnet50, iresnet100, etc.)
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading ArcFace model: {backbone}")
    
    # Map backbone names to InsightFace model names
    model_map = {
        "iresnet50": "arcface_r50_v1",
        "iresnet100": "arcface_r100_v1",
    }
    
    if backbone not in model_map:
        raise ValueError(f"Unsupported backbone: {backbone}. Choose from {list(model_map.keys())}")
    
    model_name = model_map[backbone]
    model = get_model(model_name)
    model.eval()
    
    # Move to device
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    
    logger.info(f"✅ Model loaded on {device}")
    
    return model


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu"
) -> np.ndarray:
    """
    Extract embeddings for all images.
    
    Args:
        model: ArcFace model
        dataloader: DataLoader for face crops
        device: Device for inference
    
    Returns:
        Array of embeddings (N, E)
    """
    embeddings = []
    
    logger.info("Extracting embeddings...")
    
    with torch.no_grad():
        for batch, _ in tqdm(dataloader, desc="Extracting"):
            if device == "cuda":
                batch = batch.cuda()
            
            # Forward pass
            emb = model(batch)
            
            # L2 normalize
            emb = nn.functional.normalize(emb, p=2, dim=1)
            
            # Move to CPU and convert to numpy
            emb = emb.cpu().numpy()
            embeddings.append(emb)
    
    # Stack all embeddings
    embeddings = np.vstack(embeddings).astype(np.float32)
    
    logger.info(f"✅ Extracted {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
    
    return embeddings


def compute_statistics(
    embeddings: np.ndarray,
    train_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std from training embeddings.
    
    Args:
        embeddings: All embeddings (N, E)
        train_indices: Indices of training samples
    
    Returns:
        Tuple of (mean, std)
    """
    train_emb = embeddings[train_indices]
    mean = train_emb.mean(axis=0)
    std = train_emb.std(axis=0) + 1e-8  # Add epsilon for numerical stability
    
    return mean, std


def standardize_embeddings(
    embeddings: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    Standardize embeddings using mean and std.
    
    Args:
        embeddings: Input embeddings (N, E)
        mean: Mean vector (E,)
        std: Std vector (E,)
    
    Returns:
        Standardized embeddings (N, E)
    """
    return (embeddings - mean) / std


def build_embedding_table(
    crops_dir: Path,
    manifest_path: Path,
    backbone: str = "iresnet50",
    batch_size: int = 32,
    device: str = "cpu",
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, int]]:
    """
    Build complete embedding table from crops.
    
    Args:
        crops_dir: Directory containing crop subdirectories
        manifest_path: Path to manifest CSV
        backbone: Model architecture
        batch_size: Batch size for inference
        device: Device for inference
        val_split: Validation split fraction
        seed: Random seed
    
    Returns:
        Tuple of (embedding_table, feature_stats, label_map)
    """
    # Load manifest
    logger.info(f"Loading manifest from {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    
    # Build list of image paths
    session_dir = manifest_path.parent
    image_paths = []
    valid_rows = []
    
    for idx, row in manifest.iterrows():
        crop_path = session_dir / row['crop_path']
        if crop_path.exists():
            image_paths.append(crop_path)
            valid_rows.append(idx)
        else:
            logger.warning(f"Crop not found: {crop_path}")
    
    manifest = manifest.iloc[valid_rows].reset_index(drop=True)
    logger.info(f"Found {len(image_paths)} valid crops")
    
    # Create dataset and dataloader
    dataset = FaceCropDataset(image_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda")
    )
    
    # Load model
    model = load_arcface_model(backbone, device)
    
    # Extract embeddings
    embeddings = extract_embeddings(model, dataloader, device)
    
    # Create label map
    unique_labels = manifest['label'].unique()
    unique_labels = [l for l in unique_labels if pd.notna(l) and l != '']
    label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
    logger.info(f"Label map: {label_map}")
    
    # Convert labels to IDs
    label_ids = []
    for label in manifest['label']:
        if pd.notna(label) and label != '' and label in label_map:
            label_ids.append(label_map[label])
        else:
            label_ids.append(-1)  # Unlabeled
    
    # Split into train/val
    np.random.seed(seed)
    n_samples = len(embeddings)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    n_val = int(n_samples * val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # Compute statistics from training set
    mean, std = compute_statistics(embeddings, train_indices)
    
    # Standardize all embeddings
    embeddings_std = standardize_embeddings(embeddings, mean, std)
    
    # Build DataFrame
    df = manifest.copy()
    df['embedding'] = list(embeddings_std)
    df['label_id'] = label_ids
    df['split'] = ['val' if i in val_indices else 'train' for i in range(n_samples)]
    
    # Feature stats
    feature_stats = {
        'mean': mean,
        'std': std,
    }
    
    logger.info(f"Train samples: {len(train_indices)}")
    logger.info(f"Val samples: {len(val_indices)}")
    
    return df, feature_stats, label_map


def main():
    parser = argparse.ArgumentParser(
        description="Build embeddings from face crops using ArcFace"
    )
    parser.add_argument(
        "--crops",
        type=Path,
        required=True,
        help="Directory containing crop subdirectories"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest CSV"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="iresnet50",
        choices=["iresnet50", "iresnet100"],
        help="ArcFace backbone architecture"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("embeddings/frame_table.parquet"),
        help="Output path for embedding table"
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=Path("embeddings/feature_stats.npz"),
        help="Output path for feature statistics"
    )
    parser.add_argument(
        "--labels-out",
        type=Path,
        default=Path("embeddings/labels.json"),
        help="Output path for label mapping"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Crops directory: {args.crops}")
    logger.info(f"Manifest: {args.manifest}")
    logger.info(f"Backbone: {args.backbone}")
    logger.info(f"Device: {args.device}")
    
    # Build embedding table
    df, feature_stats, label_map = build_embedding_table(
        crops_dir=args.crops,
        manifest_path=args.manifest,
        backbone=args.backbone,
        batch_size=args.batch_size,
        device=args.device,
        val_split=args.val_split,
        seed=args.seed,
    )
    
    # Save outputs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    # Save embedding table
    df.to_parquet(args.out, index=False)
    logger.info(f"✅ Saved embedding table: {args.out}")
    
    # Save feature stats
    np.savez(args.stats_out, **feature_stats)
    logger.info(f"✅ Saved feature stats: {args.stats_out}")
    
    # Save label map
    with open(args.labels_out, 'w') as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"✅ Saved label map: {args.labels_out}")
    
    logger.info("✨ Done!")


if __name__ == "__main__":
    main()

