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

# Optional Face Mesh wrapper
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
            raise RuntimeError("--landmarks requested but mediapipe_face_mesh_wrapper is not importable. Place it on PYTHONPATH.")
        _ensure_dir(landmarks_dir)
        extractor = FaceMeshExtractor(FaceMeshConfig(refine_landmarks=False))
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

        # UI overlay
        if draw:
            _draw_overlay(frame, active_cls, counts, fps, keymap)
        cv2.imshow("Capture & Label", frame)

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
    p.add_argument("--classes", type=str, required=True, help="Comma-separated class names, e.g., neutral,smile,frown")
    p.add_argument("--session", type=str, required=True, help="Session name (folder) to write under save-dir")
    p.add_argument("--save-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--draw", action="store_true", help="Draw overlay text")
    p.add_argument("--landmarks", action="store_true", help="Save Face Mesh landmarks alongside images")
    p.add_argument("--hz", dest="capture_hz", type=float, default=4.0, help="Max capture rate while pressing SPACE (frames/sec)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if len(classes) == 0:
        raise ValueError("--classes must contain at least one class name")

    capture_loop(
        classes=classes,
        session=args.session,
        save_dir=args.save_dir,
        cam=args.cam,
        width=args.width,
        height=args.height,
        draw=args.draw,
        save_landmarks=args.landmarks,
        capture_hz=args.capture_hz,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
