"""Webcam smile detection using MediaPipe Face Mesh landmarks.

Usage
  python -m expression_recognition.models.smile_preview --cam 0 --threshold 2.2 --draw

Keys
  ESC or q  Quit

Notes
- Uses FaceMeshExtractor from face_mesh_adapter.py
- Heuristic smile score = mouth_width / (mouth_height + 1e-6)
- Landmarks are normalized for translation/scale/rotation before measuring
"""
from __future__ import annotations

import argparse
from collections import deque
from typing import Deque

import cv2
import numpy as np

from .face_mesh_adapter import FaceMeshExtractor, FaceMeshConfig


# Landmark indices (MediaPipe Face Mesh)
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LIP_TOP, LIP_BOTTOM = 13, 14


def smile_score(landmarks_xy: np.ndarray) -> float:
    """Return smile score based on normalized mouth width/height ratio."""
    xy = FaceMeshExtractor.normalize_xy(landmarks_xy, align_eyes=True)
    width = float(np.linalg.norm(xy[MOUTH_LEFT] - xy[MOUTH_RIGHT]))
    height = float(np.linalg.norm(xy[LIP_TOP] - xy[LIP_BOTTOM]))
    return float(width / (height + 1e-6))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Webcam smile detection preview")
    ap.add_argument("--cam", type=int, default=0, help="Camera index")
    ap.add_argument("--threshold", type=float, default=2.2, help="Smile threshold on ratio score")
    ap.add_argument("--smooth", type=int, default=5, help="Median smoothing window (frames)")
    ap.add_argument("--draw", action="store_true", help="Draw landmarks")
    ap.add_argument("--refine", action="store_true", help="Enable iris landmarks (slower)")
    args = ap.parse_args(argv)

    cfg = FaceMeshConfig(max_num_faces=1, refine_landmarks=args.refine)
    extractor = FaceMeshExtractor(cfg)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.cam}")

    scores: Deque[float] = deque(maxlen=max(1, args.smooth))
    print("Press ESC or 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = extractor.extract(frame)
        vis = frame.copy()
        smile_txt = "N/A"
        if faces:
            f = faces[0]
            s = smile_score(f.landmarks_xy)
            if np.isfinite(s):
                scores.append(s)
                s_disp = float(np.median(scores)) if len(scores) > 0 else s
                smiling = s_disp >= args.threshold
                smile_txt = f"Smile: {'YES' if smiling else 'NO'}  score={s_disp:.2f}  thr={args.threshold:.2f}"
            else:
                smile_txt = "Smile: N/A"

            if args.draw:
                vis = extractor.draw(vis, f)

        # Overlay text
        cv2.putText(
            vis,
            smile_txt,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Smile Preview", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

