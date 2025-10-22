#!/usr/bin/env python3
"""
video_to_crops.py — Extract aligned face crops from video using InsightFace

Purpose:
- Process video frames to extract aligned face crops
- Use InsightFace for detection and landmark extraction
- Save aligned, tight square crops for embedding extraction

Usage:
    python src/expression_recognition/data/video_to_crops.py \
      --classes neutral,smile,frown,surprise \
      --session day1 \
      --video data/sessions/day1/raw.mp4 \
      --out data/crops \
      --size 112
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise RuntimeError("InsightFace is required: pip install insightface")

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def get_affine_transform_matrix(
    src_pts: np.ndarray,
    dst_pts: np.ndarray
) -> Optional[np.ndarray]:
    """
    Compute affine transformation matrix for face alignment.
    
    Args:
        src_pts: Source landmarks (N, 2)
        dst_pts: Destination template (N, 2)
    
    Returns:
        2x3 affine transformation matrix
    """
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    return M


def get_face_template(size: int = 112) -> np.ndarray:
    """
    Get standard face template for alignment.
    
    Args:
        size: Output image size
    
    Returns:
        Template landmarks (5, 2) for eyes, nose, mouth corners
    """
    # Standard 5-point template (eyes, nose, mouth corners)
    # Normalized coordinates for 112x112 image
    template = np.array([
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ], dtype=np.float32)
    
    # Scale to desired size
    if size != 112:
        template = template * (size / 112.0)
    
    return template


def align_face(
    img: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 112
) -> Optional[np.ndarray]:
    """
    Align face using landmarks.
    
    Args:
        img: Input image
        landmarks: Facial landmarks (N, 2)
        output_size: Output crop size
    
    Returns:
        Aligned face crop
    """
    # Get template
    template = get_face_template(output_size)
    
    # Use 5-point landmarks (eyes, nose, mouth) if available
    # InsightFace provides 5-point landmarks by default
    if landmarks.shape[0] >= 5:
        src_pts = landmarks[:5]
    else:
        logger.warning(f"Insufficient landmarks: {landmarks.shape[0]}, need at least 5")
        return None
    
    # Compute transformation matrix
    M = get_affine_transform_matrix(src_pts, template)
    
    if M is None:
        logger.warning("Failed to compute affine transformation")
        return None
    
    # Warp image
    aligned = cv2.warpAffine(
        img,
        M,
        (output_size, output_size),
        borderValue=0.0
    )
    
    return aligned


def process_video(
    video_path: Path,
    output_dir: Path,
    session: str,
    classes: List[str],
    output_size: int = 112,
    det_size: Tuple[int, int] = (640, 640),
    max_faces: int = 1,
) -> Tuple[int, int]:
    """
    Process video and extract aligned face crops.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for crops
        session: Session name
        classes: List of expression classes
        output_size: Size of output crops
        det_size: Detection input size
        max_faces: Maximum number of faces to detect
    
    Returns:
        Tuple of (total_frames, processed_frames)
    """
    # Initialize InsightFace
    logger.info("Initializing InsightFace FaceAnalysis...")
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=det_size)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Video: {total_frames} frames @ {fps:.2f} FPS")
    
    # Create output directories
    session_dir = output_dir / session
    session_dir.mkdir(parents=True, exist_ok=True)
    
    crops_dir = session_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    
    # Manifest for tracking
    manifest_path = session_dir / "manifest.csv"
    manifest_rows = []
    
    processed = 0
    frame_idx = 0
    
    logger.info(f"Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = app.get(frame)
        
        if len(faces) == 0:
            frame_idx += 1
            continue
        
        # Take first/largest face
        face = faces[0]
        
        # Get landmarks (5-point)
        landmarks = face.kps  # (5, 2) for buffalo_l model
        
        # Align face
        aligned = align_face(frame, landmarks, output_size)
        
        if aligned is None:
            frame_idx += 1
            continue
        
        # Save crop
        crop_name = f"{frame_idx:06d}.jpg"
        crop_path = crops_dir / crop_name
        cv2.imwrite(str(crop_path), aligned)
        
        # Add to manifest (label will be assigned later or from annotation)
        timestamp = frame_idx / fps
        manifest_rows.append({
            'frame_idx': frame_idx,
            'crop_path': f"crops/{crop_name}",
            'timestamp': f"{timestamp:.3f}",
            'session': session,
            'label': '',  # To be filled during annotation
            'has_face': 1,
        })
        
        processed += 1
        
        if processed % 100 == 0:
            logger.info(f"Processed {processed}/{total_frames} frames...")
        
        frame_idx += 1
    
    cap.release()
    
    # Save manifest
    if manifest_rows:
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['frame_idx', 'crop_path', 'timestamp', 'session', 'label', 'has_face']
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
    
    logger.info(f"✅ Processed {processed}/{total_frames} frames with detected faces")
    logger.info(f"Crops saved to: {crops_dir}")
    logger.info(f"Manifest saved to: {manifest_path}")
    
    return total_frames, processed


def main():
    parser = argparse.ArgumentParser(
        description="Extract aligned face crops from video using InsightFace"
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Session name"
    )
    parser.add_argument(
        "--classes",
        type=str,
        help="Comma-separated expression classes (for reference)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/crops"),
        help="Output directory for crops"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=112,
        help="Output crop size (square)"
    )
    parser.add_argument(
        "--det-size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Detection input size (width height)"
    )
    
    args = parser.parse_args()
    
    # Parse classes
    classes = []
    if args.classes:
        classes = [c.strip() for c in args.classes.split(',')]
    
    # Validate video exists
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    
    logger.info(f"Session: {args.session}")
    logger.info(f"Video: {args.video}")
    logger.info(f"Output: {args.out}")
    logger.info(f"Crop size: {args.size}x{args.size}")
    if classes:
        logger.info(f"Classes: {classes}")
    
    # Process video
    total, processed = process_video(
        video_path=args.video,
        output_dir=args.out,
        session=args.session,
        classes=classes,
        output_size=args.size,
        det_size=tuple(args.det_size),
    )
    
    logger.info("✨ Done!")


if __name__ == "__main__":
    main()

