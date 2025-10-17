"""
Capture & Label — Webcam Data Collection Tool

Purpose
- Build a labeled dataset for personal facial-expression recognition.
- Hold/press SPACE to capture frames; select the active class with number keys (1..9) or custom key map.
- Optionally save MediaPipe Face Mesh landmarks alongside images using the FaceMeshExtractor wrapper.

Features
- Live overlay showing FPS, active class, and per-class counts
- Debounced, rate-limited capture while holding SPACE (configurable)
- Saves images to class-specific folders and logs a CSV manifest
- Optional landmark/feature dump (.npz) per frame or aggregate per session

Dependencies
    pip install opencv-python mediapipe numpy pandas rich

Quickstart
    # Define classes with --classes, separated by commas
    python capture_label.py --classes neutral,smile,frown --session my_day1 --save-dir data/raw --cam 0 --draw

    # With landmarks saved using the FaceMesh wrapper
    python capture_label.py --classes neutral,smile --session test --save-dir data/raw --landmarks --draw

Output structure (example)
    data/raw/
      my_day1/
        manifest.csv
        neutral/
          000001.jpg
          000002.jpg
        smile/
          000003.jpg
        landmarks/
          000001.npz
          000002.npz

Manifest CSV columns
    filepath,label,timestamp,session,frame_idx,has_landmarks

License: MIT
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("OpenCV (opencv-python) is required: pip install opencv-python") from e

# Optional rich logging
try:  # pragma: no cover
    from rich.logging import RichHandler
    _USE_RICH = True
except Exception:  # pragma: no cover
    _USE_RICH = False

try:
    from expression_recognition.models.face_mesh_adapter import FaceMeshExtractor, FaceMeshConfig  # type: ignore
    _HAS_WRAPPER = True
except Exception:
    try:
        # Try relative import if running as script
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from expression_recognition.models.face_mesh_adapter import FaceMeshExtractor, FaceMeshConfig  # type: ignore
        _HAS_WRAPPER = True
    except Exception:
        try:
            from mediapipe_face_mesh_wrapper import FaceMeshExtractor, FaceMeshConfig  # type: ignore
            _HAS_WRAPPER = True
        except Exception:
            _HAS_WRAPPER = False


@dataclass
class KeyMap:
    by_key: Dict[int, str]  # maps keyboard code to class name

    @staticmethod
    def from_classes(classes: List[str]) -> "KeyMap":
        # Map 1..9, then 0 if needed
        key_codes = [ord(str(i)) for i in range(1, 10)] + [ord("0")]
        if len(classes) > len(key_codes):
            raise ValueError("Up to 10 classes are supported by default key map (1..9,0). For more, implement custom mapping.")
        return KeyMap({key_codes[i]: classes[i] for i in range(len(classes))})


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if _USE_RICH:
        logging.basicConfig(level=level, format='%(message)s', datefmt='%H:%M:%S', handlers=[RichHandler(rich_tracebacks=True)])
    else:
        logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _open_camera(cam: int, width: Optional[int], height: Optional[int]) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cam)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam}. Try --cam <index> or run on host OS if inside WSL.")
    return cap


# -----------------------------
# Offline landmarks + packing (single entry script convenience)
# -----------------------------

def _batch_extract_landmarks_for_session(session_dir: Path, refine: bool, overwrite: bool = False) -> None:
    """Run Face Mesh over captured images in a session and save NPZ landmarks.

    Updates manifest.csv in-place to set has_landmarks=1 for frames where a face was detected.
    """
    manifest_path = session_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found in {session_dir}")
    with open(manifest_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    landmarks_dir = session_dir / "landmarks"
    _ensure_dir(landmarks_dir)

    if not _HAS_WRAPPER:
        raise RuntimeError("Face Mesh adapter not available. Ensure mediapipe is installed and PYTHONPATH includes 'src'.")
    extractor = FaceMeshExtractor(FaceMeshConfig(refine_landmarks=refine))

    for row in rows:
        rel = Path(row["filepath"])  # e.g., neutral/000001.jpg
        stem = rel.stem
        img_path = session_dir / rel
        out_npz = landmarks_dir / f"{stem}.npz"
        if out_npz.exists() and not overwrite:
            row["has_landmarks"] = "1"
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        faces = extractor.extract(img)
        if not faces:
            row["has_landmarks"] = "0"
            continue
        f0 = faces[0]
        np.savez_compressed(out_npz, landmarks_xy=f0.landmarks_xy, landmarks_xyz=f0.landmarks_xyz)
        row["has_landmarks"] = "1"

    # write back manifest
    fieldnames = rows[0].keys() if rows else ["filepath", "label", "timestamp", "session", "frame_idx", "has_landmarks"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pack_sessions_to_npz(
    sessions: List[Path],
    outdir: Path,
    *,
    T: int = 12,
    stride: int = 6,
    val_split: float = 0.2,
    use_z: bool = False,
    normalize: bool = False,
    seed: int = 42,
    shuffle: bool = False,
    classes_preferred: Optional[List[str]] = None,
) -> tuple[Path, Path, List[str]]:
    """Pack landmark frames into (N,T,D) sequences and save train/val NPZ files."""

    def _load_rows(sess_dir: Path) -> List[dict[str, str]]:
        p = sess_dir / "manifest.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        with open(p, "r", newline="") as f:
            return list(csv.DictReader(f))

    def _collect_per_class(rows: List[dict[str, str]]) -> dict[str, List[tuple[int, Path]]]:
        by_cls: dict[str, List[tuple[int, Path]]] = {}
        for r in rows:
            if str(r.get("has_landmarks", "0")) not in ("1", "True", "true"):
                continue
            label = str(r["label"]).strip()
            stem = Path(r["filepath"]).stem
            frame_idx = int(float(r.get("frame_idx", "0")))
            by_cls.setdefault(label, []).append((frame_idx, Path(stem + ".npz")))
        for k in list(by_cls.keys()):
            by_cls[k].sort(key=lambda t: t[0])
        return by_cls

    def _norm_xy(xy: np.ndarray) -> np.ndarray:
        if not normalize:
            return xy
        try:
            return FaceMeshExtractor.normalize_xy(xy, align_eyes=True)
        except Exception:
            return xy

    X_all: list[np.ndarray] = []
    y_all: list[str] = []

    for sess in sessions:
        rows = _load_rows(sess)
        by_cls = _collect_per_class(rows)
        for label, items in by_cls.items():
            feats: list[np.ndarray] = []
            for _, npz_rel in items:
                npz_path = sess / "landmarks" / npz_rel.name
                if not npz_path.exists():
                    continue
                data = np.load(npz_path)
                if use_z and "landmarks_xyz" in data:
                    arr = data["landmarks_xyz"].astype(np.float32).reshape(-1)
                else:
                    xy = data["landmarks_xy"].astype(np.float32)
                    xy = _norm_xy(xy)
                    arr = xy.reshape(-1)
                feats.append(arr)
            if not feats:
                continue
            F = len(feats)
            for i in range(0, max(0, F - T + 1), max(1, stride)):
                win = feats[i : i + T]
                if len(win) == T:
                    X_all.append(np.stack(win, axis=0).astype(np.float32))
                    y_all.append(label)

    if not X_all:
        raise RuntimeError("No sequences built. Ensure landmarks exist and sessions are correct.")

    X = np.stack(X_all, axis=0).astype(np.float32)
    uniq = sorted(set(y_all))
    if classes_preferred:
        classes = [c for c in classes_preferred if c in uniq] + [c for c in uniq if c not in classes_preferred]
    else:
        classes = uniq
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_id[c] for c in y_all], dtype=np.int64)

    N = X.shape[0]
    idxs = list(range(N))
    if shuffle:
        random.Random(seed).shuffle(idxs)
    n_val = int(round(val_split * N))
    val_idx = idxs[:n_val]
    tr_idx = idxs[n_val:]

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xval, yval = X[val_idx], y[val_idx]

    outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"T{T}"
    train_path = outdir / f"train_{suffix}.npz"
    val_path = outdir / f"val_{suffix}.npz"
    np.savez_compressed(train_path, X=Xtr, y=ytr, classes=np.array(classes, dtype=object))
    np.savez_compressed(val_path, X=Xval, y=yval, classes=np.array(classes, dtype=object))
    return train_path, val_path, classes


def _draw_overlay(frame: np.ndarray, active: Optional[str], counts: Dict[str, int], fps: float, keymap: KeyMap) -> None:
    h, w = frame.shape[:2]
    y = 24
    # Active class and FPS
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, f"Active: {active or 'NONE'}  |  FPS: {fps:5.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # Key help
    y2 = 60
    cv2.putText(frame, "Keys: ", (10, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    x = 80
    for key_code, cls in keymap.by_key.items():
        key_label = chr(key_code)
        cv2.putText(frame, f"{key_label}:{cls}", (x, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        x += 120
    # Counts
    y3 = y2 + 24
    txt = "  ".join([f"{k}:{counts.get(k,0)}" for k in sorted(counts.keys())])
    cv2.putText(frame, txt, (10, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1)


def capture_loop(
    classes: List[str],
    session: str,
    save_dir: Path,
    cam: int,
    width: Optional[int],
    height: Optional[int],
    draw: bool,
    save_landmarks: bool,
    refine_landmarks: bool,
    capture_hz: float,
    verbose: bool,
) -> None:
    _setup_logging(verbose)
    keymap = KeyMap.from_classes(classes)
    counts = {c: 0 for c in classes}

    session_dir = save_dir / session
    _ensure_dir(session_dir)
    manifest_path = session_dir / "manifest.csv"
    images_dir = session_dir
    landmarks_dir = session_dir / "landmarks"
    _ensure_dir(images_dir)
    if save_landmarks:
        if not _HAS_WRAPPER:
            raise RuntimeError("--landmarks requested but a Face Mesh adapter is not importable. Ensure mediapipe is installed and PYTHONPATH includes 'src'.")
        _ensure_dir(landmarks_dir)
        extractor = FaceMeshExtractor(FaceMeshConfig(refine_landmarks=refine_landmarks))
    else:
        extractor = None

    cap = _open_camera(cam, width, height)

    logging.info("Controls: 1..9/0 to select class | Hold/press SPACE to capture | ESC to quit")
    active_cls: Optional[str] = None
    last_capture_t = 0.0
    min_dt = 1.0 / max(1e-3, capture_hz)  # seconds between captures when holding SPACE

    # CSV manifest
    if not manifest_path.exists():
        with open(manifest_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label", "timestamp", "session", "frame_idx", "has_landmarks"])  # header

    t0 = time.time()
    frames = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            logging.warning("Camera read failed.")
            break
        frames += 1
        t1 = time.time()
        if t1 - t0 >= 1.0:
            fps = frames / (t1 - t0)
            t0, frames = t1, 0

        # UI overlay (draw on a copy so saved images remain raw camera frames)
        vis = frame
        if draw:
            vis = frame.copy()
            _draw_overlay(vis, active_cls, counts, fps, keymap)
        cv2.imshow("Capture & Label", vis)

        key = cv2.waitKey(1) & 0xFF

        # Class selection: numeric keys
        if key in keymap.by_key:
            active_cls = keymap.by_key[key]

        # Escape to exit
        if key == 27:  # ESC
            break

        # SPACE pressed? (OpenCV returns 32 for space when pressed this tick)
        space_down = key == 32
        now = time.time()
        can_capture = (now - last_capture_t) >= min_dt

        # Also support continuous capture while SPACE is held: use getWindowProperty + poll
        # Simple approach: when space_down, capture once; to get continuous, hold space and rely on repeat events

        if space_down and active_cls and can_capture:
            last_capture_t = now
            # Save image
            cls_dir = images_dir / active_cls
            _ensure_dir(cls_dir)
            counts[active_cls] += 1
            fname = f"{counts[active_cls]:06d}.jpg"
            out_path = cls_dir / fname
            cv2.imwrite(str(out_path), frame)

            has_lm = False
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if save_landmarks and extractor is not None:
                faces = extractor.extract(frame)
                if faces:
                    has_lm = True
                    np.savez_compressed(
                        landmarks_dir / f"{counts[active_cls]:06d}.npz",
                        landmarks_xy=faces[0].landmarks_xy,
                        landmarks_xyz=faces[0].landmarks_xyz,
                    )

            with open(manifest_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    str(out_path.relative_to(session_dir)),
                    active_cls,
                    f"{now:.6f}",
                    session,
                    frame_idx,
                    int(has_lm),
                ])

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture labeled frames from webcam (SPACE to capture, 1..9/0 to select class)")
    p.add_argument("--config", type=Path, help="Path to config (YAML/JSON) that defines labels list")
    p.add_argument("--classes", type=str, help="Comma-separated class names, e.g., neutral,smile,frown (overrides config if provided)")
    p.add_argument("--session", type=str, required=True, help="Session name (folder) to write under save-dir")
    p.add_argument("--save-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--draw", action="store_true", help="Draw overlay text")
    p.add_argument("--landmarks", action="store_true", help="Save Face Mesh landmarks alongside images")
    p.add_argument("--refine", action="store_true", help="Enable iris landmarks (478 points)")
    p.add_argument("--hz", dest="capture_hz", type=float, default=4.0, help="Max capture rate while pressing SPACE (frames/sec)")
    p.add_argument("-v", "--verbose", action="store_true")

    # Auto-pack options (optional): build train/val NPZ sequences after capture
    p.add_argument("--auto-pack", action="store_true", help="After capture, build (N,T,D) train/val NPZs for the trainer")
    p.add_argument("--features-outdir", type=Path, default=Path("data/features"))
    p.add_argument("--T", type=int, default=12, help="Sequence length for packing")
    p.add_argument("--stride", type=int, default=6, help="Stride between windows for packing")
    p.add_argument("--val-split", type=float, default=0.2, help="Validation fraction for packing")
    p.add_argument("--use-z", action="store_true", help="Use XYZ landmarks (else XY) for packing")
    p.add_argument("--normalize", action="store_true", help="Normalize XY before packing")
    p.add_argument("--shuffle", action="store_true", help="Shuffle sequences before split")
    p.add_argument("--seed", type=int, default=42, help="Random seed for packing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    classes: List[str] = []
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    elif args.config:
        try:
            from expression_recognition.config.schemas import load_config, get_labels  # type: ignore
            cfg = load_config(args.config)
            classes = get_labels(cfg)
        except Exception as e:
            raise RuntimeError(f"Failed to load labels from config: {args.config}") from e
    if len(classes) == 0:
        raise ValueError("No classes provided. Use --classes or --config with a labels list.")

    capture_loop(
        classes=classes,
        session=args.session,
        save_dir=args.save_dir,
        cam=args.cam,
        width=args.width,
        height=args.height,
        draw=args.draw,
        save_landmarks=args.landmarks,
        refine_landmarks=args.refine,
        capture_hz=args.capture_hz,
        verbose=args.verbose,
    )

    # Optional: pack sequences into train/val NPZs for trainer
    if args.auto_pack:
        session_dir = args.save_dir / args.session
        # Ensure landmarks exist: if not saved during capture, run offline extraction now
        if not args.landmarks:
            _batch_extract_landmarks_for_session(session_dir, refine=args.refine, overwrite=False)

        # Pack sequences directly (single-entry workflow)
        train_path, val_path, _ = _pack_sessions_to_npz(
            sessions=[session_dir],
            outdir=args.features_outdir,
            T=args.T,
            stride=args.stride,
            val_split=args.val_split,
            use_z=args.use_z,
            normalize=args.normalize,
            seed=args.seed,
            shuffle=args.shuffle,
            classes_preferred=classes,
        )
        print(f"Packed sequences -> train: {train_path} | val: {val_path}")


if __name__ == "__main__":
    main()
