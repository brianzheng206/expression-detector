#!/usr/bin/env python3
"""
make_sequences.py — Create temporal sequences from embeddings for GRU training

Purpose:
- Load per-frame embeddings
- Create sliding window sequences
- Split into train/val sets
- Save as NPZ files

Usage:
    python src/expression_recognition/data/make_sequences.py \
      --frames embeddings/frame_table.parquet \
      --stats embeddings/feature_stats.npz \
      --labels embeddings/labels.json \
      --sessions-train day1,day2 \
      --sessions-val day3 \
      --T 12 \
      --stride 4 \
      --out-train embeddings/train_T12.npz \
      --out-val embeddings/val_T12.npz
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def create_sequences(
    embeddings: np.ndarray,
    labels: np.ndarray,
    T: int = 12,
    stride: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences using sliding window.
    
    Args:
        embeddings: Embeddings (N, E)
        labels: Labels (N,)
        T: Sequence length
        stride: Stride between windows
    
    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    N, E = embeddings.shape
    
    X_seq = []
    y_seq = []
    
    # Slide window
    for i in range(0, max(0, N - T + 1), stride):
        # Get window
        window = embeddings[i:i + T]
        window_labels = labels[i:i + T]
        
        if len(window) != T:
            continue
        
        # Use label of last frame in window
        label = window_labels[-1]
        
        # Skip if unlabeled
        if label < 0:
            continue
        
        X_seq.append(window)
        y_seq.append(label)
    
    if not X_seq:
        raise RuntimeError("No sequences created. Check data and parameters.")
    
    X = np.stack(X_seq, axis=0).astype(np.float32)  # (N_seq, T, E)
    y = np.array(y_seq, dtype=np.int64)  # (N_seq,)
    
    return X, y


def process_sessions(
    df: pd.DataFrame,
    sessions: List[str],
    T: int,
    stride: int,
    class_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process multiple sessions and create sequences.
    
    Args:
        df: DataFrame with embeddings
        sessions: List of session names to process
        T: Sequence length
        stride: Stride
        class_names: Ordered list of class names
    
    Returns:
        Tuple of (X, y) arrays
    """
    all_X = []
    all_y = []
    
    for session in sessions:
        logger.info(f"Processing session: {session}")
        
        # Filter session
        session_df = df[df['session'] == session].copy()
        
        if len(session_df) == 0:
            logger.warning(f"No data found for session {session}")
            continue
        
        # Sort by frame index or timestamp
        if 'frame_idx' in session_df.columns:
            session_df = session_df.sort_values('frame_idx')
        elif 'timestamp' in session_df.columns:
            session_df = session_df.sort_values('timestamp')
        
        # Extract embeddings and labels
        embeddings = np.stack(session_df['embedding'].values)  # (N, E)
        labels = session_df['label_id'].values  # (N,)
        
        # Create sequences
        X, y = create_sequences(embeddings, labels, T, stride)
        
        logger.info(f"  Created {len(X)} sequences from {len(embeddings)} frames")
        
        all_X.append(X)
        all_y.append(y)
    
    if not all_X:
        raise RuntimeError("No sequences created from any session")
    
    # Concatenate all sessions
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Create temporal sequences from embeddings"
    )
    parser.add_argument(
        "--frames",
        type=Path,
        required=True,
        help="Path to frame embedding table (parquet)"
    )
    parser.add_argument(
        "--stats",
        type=Path,
        required=True,
        help="Path to feature statistics (npz)"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to label mapping (json)"
    )
    parser.add_argument(
        "--sessions-train",
        type=str,
        help="Comma-separated training session names"
    )
    parser.add_argument(
        "--sessions-val",
        type=str,
        help="Comma-separated validation session names"
    )
    parser.add_argument(
        "--use-split-column",
        action="store_true",
        help="Use 'split' column from frame table instead of session names"
    )
    parser.add_argument(
        "--T",
        type=int,
        default=12,
        help="Sequence length (number of frames)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Stride between sequences"
    )
    parser.add_argument(
        "--out-train",
        type=Path,
        default=Path("embeddings/train_T12.npz"),
        help="Output path for training sequences"
    )
    parser.add_argument(
        "--out-val",
        type=Path,
        default=Path("embeddings/val_T12.npz"),
        help="Output path for validation sequences"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Loading frame table from {args.frames}")
    df = pd.read_parquet(args.frames)
    
    logger.info(f"Loading label mapping from {args.labels}")
    with open(args.labels, 'r') as f:
        label_map = json.load(f)
    
    # Create inverse mapping (id -> name)
    class_names = sorted(label_map.keys(), key=lambda k: label_map[k])
    
    logger.info(f"Classes: {class_names}")
    logger.info(f"Sequence length T={args.T}, stride={args.stride}")
    
    # Determine train/val split
    if args.use_split_column:
        logger.info("Using 'split' column from frame table")
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        
        # Create sequences per split
        logger.info("Creating training sequences...")
        X_train, y_train = create_sequences(
            embeddings=np.stack(train_df['embedding'].values),
            labels=train_df['label_id'].values,
            T=args.T,
            stride=args.stride,
        )
        
        logger.info("Creating validation sequences...")
        X_val, y_val = create_sequences(
            embeddings=np.stack(val_df['embedding'].values),
            labels=val_df['label_id'].values,
            T=args.T,
            stride=args.stride,
        )
        
    else:
        # Use session names
        if not args.sessions_train or not args.sessions_val:
            raise ValueError("Must provide --sessions-train and --sessions-val when not using --use-split-column")
        
        train_sessions = [s.strip() for s in args.sessions_train.split(',')]
        val_sessions = [s.strip() for s in args.sessions_val.split(',')]
        
        logger.info(f"Training sessions: {train_sessions}")
        logger.info(f"Validation sessions: {val_sessions}")
        
        # Process sessions
        logger.info("Creating training sequences...")
        X_train, y_train = process_sessions(df, train_sessions, args.T, args.stride, class_names)
        
        logger.info("Creating validation sequences...")
        X_val, y_val = process_sessions(df, val_sessions, args.T, args.stride, class_names)
    
    logger.info(f"Training sequences: {X_train.shape}")
    logger.info(f"Validation sequences: {X_val.shape}")
    
    # Save sequences
    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    args.out_val.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        args.out_train,
        X=X_train,
        y=y_train,
        classes=np.array(class_names, dtype=object)
    )
    logger.info(f"✅ Saved training sequences: {args.out_train}")
    
    np.savez_compressed(
        args.out_val,
        X=X_val,
        y=y_val,
        classes=np.array(class_names, dtype=object)
    )
    logger.info(f"✅ Saved validation sequences: {args.out_val}")
    
    logger.info("✨ Done!")


if __name__ == "__main__":
    main()

