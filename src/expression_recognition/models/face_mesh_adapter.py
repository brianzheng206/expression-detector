"""
MediaPipe Face Mesh — Lightweight, Modular Landmark Extraction Wrapper

Goals
- Minimal, production-friendly API for extracting 468-point landmarks (+ optional iris) from images/video.
- Clean separation of concerns: configuration, extraction, normalization, drawing, and CLI.
- Ready for open source: type hints, logging, docstrings, and a small CLI for quick testing.

Dependencies
- mediapipe>=0.10
- opencv-python
- numpy
- (optional) rich — prettier logs

Install
    pip install mediapipe opencv-python numpy rich

Quick start (Python)
    from mediapipe_face_mesh_wrapper import FaceMeshExtractor, FaceMeshConfig

    extractor = FaceMeshExtractor(FaceMeshConfig(refine_landmarks=False))
    frame_bgr = cv2.imread("face.jpg")
    faces = extractor.extract(frame_bgr)
    for f in faces:
        print(f.landmarks_xy.shape)  # (468, 2) in image pixels
        print(f.landmarks_xyz.shape) # (468, 3) in normalized coords (x,y in [0,1], z in face-space)

CLI (examples)
    # Run webcam preview with overlay
    python mediapipe_face_mesh_wrapper.py preview --draw

    # Process a single image and save .npz of features
    python mediapipe_face_mesh_wrapper.py image --input path/to.jpg --out out.npz

    # Process a video, dump per-frame landmarks to NPZ
    python mediapipe_face_mesh_wrapper.py video --input path/to.mp4 --out out_landmarks.npz

License: MIT
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from dataclasses import dataclass
import sys
import os
import glob
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("OpenCV (opencv-python) is required: pip install opencv-python") from e

try:
    import mediapipe as mp  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("MediaPipe is required: pip install mediapipe") from e

# Optional pretty logging
try:  # pragma: no cover
    from rich.logging import RichHandler

    _USE_RICH = True
except Exception:  # pragma: no cover
    _USE_RICH = False


# -----------------------------
# Configuration & Data Records
# -----------------------------

@dataclass(frozen=True)
class FaceMeshConfig:
    """Configuration for MediaPipe Face Mesh.
    See: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
    (This wrapper targets the classic FaceMesh API available in mediapipe.solutions.face_mesh.)
    """

    static_image_mode: bool = False
    max_num_faces: int = 1
    refine_landmarks: bool = False  # set True to get iris landmarks
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class FaceResult:
    """Structured output per detected face."""

    # Landmarks in normalized MediaPipe coordinates: x,y in [0,1] relative to image; z is in face depth units
    landmarks_xyz: np.ndarray  # (468 or 478, 3)

    # Landmarks in image pixel space (x,y), computed from landmarks_xyz and image size
    landmarks_xy: np.ndarray  # (468 or 478, 2)

    # Optional visibility/presence if available (Face Mesh does not provide per-landmark visibility like Pose)
    # Included here for API symmetry; filled with ones.
    visibility: np.ndarray  # (N,)

    # Convenience anchors frequently used for normalization
    left_eye_center: np.ndarray  # (2,)
    right_eye_center: np.ndarray  # (2,)
    nose_tip_xy: np.ndarray  # (2,)


# -----------------------------
# Core Extractor
# -----------------------------

class FaceMeshExtractor:
    """Thin wrapper around MediaPipe Face Mesh.

    Responsibilities:
    - Manage MediaPipe lifecycle
    - Convert outputs to well-typed numpy arrays
    - Provide normalization helpers
    - (Optionally) draw landmarks
    """

    _EYE_LEFT_IDX = (33, 133)   # outer/inner corners (approx)
    _EYE_RIGHT_IDX = (362, 263)
    _NOSE_TIP_IDX = 1

    def __init__(self, config: FaceMeshConfig = FaceMeshConfig()) -> None:
        self.cfg = config
        self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=self.cfg.static_image_mode,
            max_num_faces=self.cfg.max_num_faces,
            refine_landmarks=self.cfg.refine_landmarks,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )
        self._drawer = mp.solutions.drawing_utils
        self._drawing_spec = self._drawer.DrawingSpec(thickness=1, circle_radius=1)

    # -----------------------------
    # Public API
    # -----------------------------

    def extract(self, frame_bgr: np.ndarray) -> List[FaceResult]:
        """Extract landmarks for all faces in a BGR frame.

        Args:
            frame_bgr: (H, W, 3) BGR image as read by OpenCV.
        Returns:
            A list of FaceResult, one per detected face (len <= max_num_faces).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = self._mp_face_mesh.process(frame_rgb)
        faces: List[FaceResult] = []

        if not out.multi_face_landmarks:
            return faces

        for landmarks in out.multi_face_landmarks:
            xyz = self._landmarks_proto_to_ndarray(landmarks)  # (N,3) normalized
            xy_px = self._to_image_coords(xyz, (h, w))  # (N,2) pixels

            left_eye_center = xy_px[list(self._EYE_LEFT_IDX)].mean(axis=0)
            right_eye_center = xy_px[list(self._EYE_RIGHT_IDX)].mean(axis=0)
            nose_tip_xy = xy_px[self._NOSE_TIP_IDX]

            N = xyz.shape[0]
            faces.append(
                FaceResult(
                    landmarks_xyz=xyz,
                    landmarks_xy=xy_px,
                    visibility=np.ones((N,), dtype=np.float32),
                    left_eye_center=left_eye_center,
                    right_eye_center=right_eye_center,
                    nose_tip_xy=nose_tip_xy,
                )
            )
        return faces

    def draw(self, frame_bgr: np.ndarray, face: FaceResult) -> np.ndarray:
        """Draw landmarks onto a copy of the frame."""
        canvas = frame_bgr.copy()
        for (x, y) in face.landmarks_xy.astype(int):
            cv2.circle(canvas, (int(x), int(y)), 1, (0, 255, 0), -1)
        return canvas

    # -----------------------------
    # Normalization Utilities
    # -----------------------------

    @staticmethod
    def normalize_xy(
        landmarks_xy: np.ndarray,
        *,
        anchor_idx: int = _NOSE_TIP_IDX,
        left_eye_pair: Tuple[int, int] = _EYE_LEFT_IDX,
        right_eye_pair: Tuple[int, int] = _EYE_RIGHT_IDX,
        align_eyes: bool = True,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Center, scale, and optionally in-plane align landmarks.

        Steps:
        1) Center on anchor point (default: nose tip)
        2) Scale by inter-pupil distance (mean of two eye-corner points per eye)
        3) Rotate so eye line is horizontal (optional)

        Returns:
            (N,2) normalized landmarks
        """
        xy = landmarks_xy.astype(np.float32).copy()
        nose = xy[anchor_idx]
        xy -= nose
        left_center = xy[list(left_eye_pair)].mean(axis=0)
        right_center = xy[list(right_eye_pair)].mean(axis=0)
        ipd = float(np.linalg.norm(right_center - left_center) + eps)
        xy /= ipd
        if align_eyes:
            v = right_center - left_center
            theta = float(np.arctan2(v[1], v[0]))
            c, s = np.cos(-theta), np.sin(-theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            xy = xy @ R.T
        return xy

    # -----------------------------
    # Internals
    # -----------------------------

    @staticmethod
    def _landmarks_proto_to_ndarray(landmarks) -> np.ndarray:
        """Convert MediaPipe landmark proto to (N,3) np.ndarray in normalized coords."""
        pts = [(lm.x, lm.y, getattr(lm, "z", 0.0)) for lm in landmarks.landmark]
        return np.asarray(pts, dtype=np.float32)

    @staticmethod
    def _to_image_coords(xyz_norm: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
        """Convert normalized (x,y) to image pixel coordinates (W and H scaling)."""
        h, w = shape_hw
        x = np.clip(xyz_norm[:, 0] * w, 0, w - 1)
        y = np.clip(xyz_norm[:, 1] * h, 0, h - 1)
        return np.stack([x, y], axis=-1).astype(np.float32)


# -----------------------------
# CLI utilities
# -----------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if _USE_RICH:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="%H:%M:%S",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    else:
        logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def _save_npz(out: Path, frames: List[int], faces: List[List[FaceResult]], size_hw: Tuple[int, int]) -> None:
    """Persist landmarks for all frames to a compressed NPZ.

    Structure:
      - frames: list of frame indices
      - size_hw: (H,W)
      - faces[k]: list per frame; we store only the first face for simplicity
    """
    xs_xy = []
    xs_xyz = []
    for face_list in faces:
        if face_list:
            xs_xy.append(face_list[0].landmarks_xy)
            xs_xyz.append(face_list[0].landmarks_xyz)
        else:
            xs_xy.append(np.full((468, 2), np.nan, np.float32))
            xs_xyz.append(np.full((468, 3), np.nan, np.float32))
    arr_xy = np.stack(xs_xy, axis=0)  # (T,468,2)
    arr_xyz = np.stack(xs_xyz, axis=0)  # (T,468,3)
    np.savez_compressed(out, frames=np.array(frames), size_hw=np.array(size_hw), landmarks_xy=arr_xy, landmarks_xyz=arr_xyz)


def _try_open_camera(index: int) -> Optional[cv2.VideoCapture]:  # type: ignore[name-defined]
    """Attempt to open a camera at a given index; return capture or None."""
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap
    cap.release()
    # Second attempt with a generic backend hint (some platforms prefer CAP_ANY)
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if cap.isOpened():
            return cap
    except Exception:
        pass
    try:
        # Linux V4L2 hint (no-op on non-Linux builds)
        cap = cv2.VideoCapture(index, getattr(cv2, "CAP_V4L2", 200))
        if cap.isOpened():
            return cap
    except Exception:
        pass
    cap.release()
    return None


def _list_linux_video_indices() -> List[int]:
    """Return indices parsed from /dev/video* (Linux only)."""
    idxs: List[int] = []
    try:
        for p in glob.glob("/dev/video*"):
            m = re.search(r"video(\d+)$", os.path.basename(p))
            if m:
                idxs.append(int(m.group(1)))
    except Exception:
        pass
    return sorted(set(idxs))


def _autodetect_camera(max_index: int = 5) -> Tuple[Optional[int], Optional[cv2.VideoCapture]]:  # type: ignore[name-defined]
    """Find the first available camera by enumerating /dev/video* (Linux) then scanning 0..max_index."""
    candidates: List[int] = []
    if os.name == "posix":
        candidates.extend(_list_linux_video_indices())
    # Fallback to index scan if none found
    if not candidates:
        candidates.extend(list(range(max(0, max_index) + 1)))
    seen: set[int] = set()
    for i in candidates:
        if i in seen:
            continue
        seen.add(i)
        cap = _try_open_camera(i)
        if cap is not None:
            return i, cap
    return None, None


def _preview_loop(extractor: FaceMeshExtractor, draw: bool = True, cam_index: int = 0, scan_limit: int = 5) -> None:
    cap = _try_open_camera(cam_index)
    if cap is None:
        logging.warning(f"Could not open camera index {cam_index}. Scanning 0..{scan_limit}...")
        found_idx, cap = _autodetect_camera(scan_limit)
        if cap is None or found_idx is None:
            avail_msg = ""
            if os.name == "posix":
                linux_idxs = _list_linux_video_indices()
                avail_msg = f" Detected /dev/video* indices: {linux_idxs}" if linux_idxs else ""
            logging.error(
                (
                    "No webcam available. Tried indices 0..{lim}. "
                    "Pass a valid --cam index or use 'image'/'video' subcommands instead." +
                    avail_msg
                ).format(lim=scan_limit)
            )
            raise SystemExit(2)
        else:
            logging.info(f"Using detected camera index {found_idx}")
    logging.info("Press ESC to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = extractor.extract(frame)
        vis = frame
        if draw and faces:
            vis = extractor.draw(frame, faces[0])
        cv2.imshow("FaceMesh Preview", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()


def _process_image(extractor: FaceMeshExtractor, path: Path, out: Optional[Path]) -> None:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    faces = extractor.extract(img)
    logging.info(f"Detected {len(faces)} face(s) in {path}")
    if out:
        _save_npz(out, frames=[0], faces=[faces], size_hw=img.shape[:2])
        logging.info(f"Saved landmarks to {out}")
    if faces:
        vis = extractor.draw(img, faces[0])
        cv2.imshow("Result", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _process_video(extractor: FaceMeshExtractor, path: Path, out: Optional[Path]) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames: List[int] = []
    all_faces: List[List[FaceResult]] = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = extractor.extract(frame)
        frames.append(idx)
        all_faces.append(faces)
        idx += 1
    cap.release()
    if out:
        _save_npz(out, frames=frames, faces=all_faces, size_hw=frame.shape[:2])
        logging.info(f"Saved landmarks for {len(frames)} frames to {out}")


# -----------------------------
# Main / CLI
# -----------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MediaPipe Face Mesh — Landmark Extraction Wrapper")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common config
    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--max-faces", type=int, default=1)
        sp.add_argument("--refine", action="store_true", help="Enable iris landmarks (slower)")
        sp.add_argument("--det-conf", type=float, default=0.5)
        sp.add_argument("--trk-conf", type=float, default=0.5)
        sp.add_argument("-v", "--verbose", action="store_true")

    # preview
    sp_prev = sub.add_parser("preview", help="Open webcam and draw landmarks")
    add_common(sp_prev)
    sp_prev.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    sp_prev.add_argument("--scan-limit", type=int, default=5, help="If opening fails, scan indices 0..N for a camera")
    sp_prev.add_argument("--draw", action="store_true")

    # image
    sp_img = sub.add_parser("image", help="Process a single image and optionally save .npz")
    add_common(sp_img)
    sp_img.add_argument("--input", type=Path, required=True)
    sp_img.add_argument("--out", type=Path)

    # video
    sp_vid = sub.add_parser("video", help="Process a video and optionally save per-frame .npz")
    add_common(sp_vid)
    sp_vid.add_argument("--input", type=Path, required=True)
    sp_vid.add_argument("--out", type=Path)

    return p


def _cmd_to_extractor(args: argparse.Namespace) -> FaceMeshExtractor:
    cfg = FaceMeshConfig(
        static_image_mode=False,
        max_num_faces=args.max_faces,
        refine_landmarks=args.refine,
        min_detection_confidence=args.det_conf,
        min_tracking_confidence=args.trk_conf,
    )
    return FaceMeshExtractor(cfg)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    extractor = _cmd_to_extractor(args)

    if args.cmd == "preview":
        _preview_loop(extractor, draw=args.draw, cam_index=args.cam, scan_limit=getattr(args, "scan_limit", 5))
    elif args.cmd == "image":
        _process_image(extractor, args.input, args.out)
    elif args.cmd == "video":
        _process_video(extractor, args.input, args.out)
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.cmd}")


if __name__ == "__main__":  # pragma: no cover
    main()
