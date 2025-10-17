"""Live webcam inference for the GRU expression classifier.

Usage
  # From repo root
  $env:PYTHONPATH = 'src'
  python -m expression_recognition.inference.live_gru \
      --ckpt models/checkpoints/best.pt \
      --T 12 --features xy --normalize --draw --cam 0

Notes
- Install PyTorch: pip install torch  (or follow pytorch.org for your CUDA)
- Ensure your feature settings (xy/xyz, normalize, iris refine) match training.
- If your checkpoint in_dim is 468*2 but you run with iris refine (478),
  the script truncates to the first in_dim values to keep compatibility.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from expression_recognition.models.face_mesh_adapter import FaceMeshExtractor, FaceMeshConfig
from expression_recognition.training.trainer import GRUExpr


# Landmark indices (MediaPipe Face Mesh)
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LIP_TOP, LIP_BOTTOM = 13, 14


@dataclass
class InferenceConfig:
    ckpt: Path
    T: int
    features: str  # 'xy' or 'xyz'
    normalize: bool
    cam: int
    draw: bool
    refine: bool
    device: str


def _load_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[GRUExpr, List[str], int]:
    state = torch.load(ckpt_path, map_location=device)
    cfg = state.get("config", {})
    in_dim = int(cfg.get("in_dim", 936))
    hidden = int(cfg.get("hidden", 64))
    num_classes = int(cfg.get("num_classes", 2))
    num_layers = int(cfg.get("num_layers", 1))
    dropout = float(cfg.get("dropout", 0.0))
    bidirectional = bool(cfg.get("bidirectional", False))
    classes = cfg.get("classes") or [f"class_{i}" for i in range(num_classes)]

    model = GRUExpr(
        in_dim=in_dim,
        hidden=hidden,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)
    model.load_state_dict(state["state_dict"])  # type: ignore[index]
    model.eval()
    return model, classes, in_dim


def _make_feature(
    landmarks_xy: np.ndarray,
    landmarks_xyz: np.ndarray | None,
    *,
    features: str,
    normalize: bool,
) -> np.ndarray:
    if features == "xyz":
        if landmarks_xyz is None:
            return np.zeros((0,), dtype=np.float32)
        arr = landmarks_xyz.astype(np.float32).reshape(-1)
        return arr
    # xy branch
    xy = landmarks_xy.astype(np.float32)
    if normalize:
        try:
            xy = FaceMeshExtractor.normalize_xy(xy, align_eyes=True)
        except Exception:
            pass
    return xy.reshape(-1)


def _predict(model: nn.Module, x_seq: np.ndarray, device: torch.device) -> Tuple[int, float, np.ndarray]:
    # x_seq: (T, D)
    with torch.no_grad():
        xb = torch.from_numpy(x_seq[None, ...]).to(device)  # (1,T,D)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        conf = float(probs[idx])
    return idx, conf, probs


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Live GRU expression inference from Face Mesh landmarks")
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint .pt (from trainer)")
    ap.add_argument("--T", type=int, default=12, help="Sequence length (frames)")
    ap.add_argument("--features", type=str, default="xy", choices=["xy", "xyz"], help="Which features to use")
    ap.add_argument("--normalize", action="store_true", help="Normalize XY (nose-center, eye-scale, align)")
    ap.add_argument("--cam", type=int, default=0, help="Camera index")
    ap.add_argument("--draw", action="store_true", help="Draw landmarks")
    ap.add_argument("--refine", action="store_true", help="Enable iris landmarks in Face Mesh")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = ap.parse_args(argv)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, classes, in_dim = _load_checkpoint(args.ckpt, device)

    # Face Mesh
    fm_cfg = FaceMeshConfig(max_num_faces=1, refine_landmarks=args.refine)
    extractor = FaceMeshExtractor(fm_cfg)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.cam}")
    print("Press ESC or 'q' to quit.")

    T = max(1, int(args.T))
    buf: Deque[np.ndarray] = deque(maxlen=T)
    last_pred = ("", 0.0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = extractor.extract(frame)
        vis = frame.copy()

        if faces:
            f = faces[0]
            feat = _make_feature(
                f.landmarks_xy,
                f.landmarks_xyz if args.features == "xyz" else None,
                features=args.features,
                normalize=args.normalize,
            )
            # If feature length > in_dim (e.g., iris refine vs non-refine), truncate to match
            if feat.size > in_dim:
                feat = feat[:in_dim]
            # If feature length < in_dim, skip until settings match
            if feat.size == in_dim:
                buf.append(feat)

            if args.draw:
                vis = extractor.draw(vis, f)

        # Predict when buffer full
        if len(buf) == T:
            x_seq = np.stack(list(buf), axis=0).astype(np.float32)
            idx, conf, probs = _predict(model, x_seq, device)
            last_pred = (classes[idx], conf)

        # Overlay
        label = last_pred[0] or "..."
        conf = last_pred[1]
        text = f"Pred: {label}  conf={conf:.2f}  (T={T}, D={in_dim})"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(vis, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("GRU Live Inference", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

