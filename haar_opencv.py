#!/usr/bin/env python3
"""
Real-time facial expression detection using Haar Cascade + DeepFace (repo-style logic).

Flow:
  - Capture webcam frames with OpenCV
  - Detect faces via Haar cascade (cv2.CascadeClassifier.detectMultiScale)
  - For each detected face ROI, call DeepFace.analyze(actions=['emotion'])
  - Draw a rectangle and the dominant emotion on the frame

Tested with: deepface==0.0.95 (no 'models' or 'prog_bar' kwargs)
"""

import argparse
import os
from pathlib import Path

import cv2
from deepface import DeepFace

# Try to find a Haar cascade XML:
def resolve_cascade(user_path: str | None) -> str:
    # 1) user provided path
    if user_path and os.path.isfile(user_path):
        return user_path
    # 2) OpenCV built-in
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        built_in = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.isfile(built_in):
            return built_in
    # 3) fallback to local file name
    local = Path("haarcascade_frontalface_default.xml")
    if local.exists():
        return str(local)
    raise FileNotFoundError(
        "Cannot locate 'haarcascade_frontalface_default.xml'. "
        "Install OpenCV with data files or download the XML and pass --cascade <path>."
    )

def draw_label(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thick = 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    y_text = max(0, y - th - 8)
    cv2.rectangle(img, (x, y_text), (x + tw + 8, y_text + th + base + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 4, y_text + th + 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser(description="Haar + DeepFace real-time emotion detection")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument("--cascade", type=str, default=None, help="Path to Haar cascade XML")
    ap.add_argument("--scale-factor", type=float, default=1.1, help="Haar scaleFactor (default 1.1)")
    ap.add_argument("--min-neighbors", type=int, default=5, help="Haar minNeighbors (default 5)")
    ap.add_argument("--min-size", type=int, default=80, help="Min face size (px) (default 80)")
    ap.add_argument("--mirror", action="store_true", help="Mirror the webcam preview")
    ap.add_argument("--every-n", type=int, default=2, help="Analyze every Nth frame for speed (default 2)")
    args = ap.parse_args()

    cascade_path = resolve_cascade(args.cascade)
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    print("Press 'q' to quit.")
    frame_idx = 0
    last_results = []  # cache last per-face results

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale_factor,
            minNeighbors=args.min_neighbors,
            minSize=(args.min_size, args.min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        analyzed_this_frame = False
        results_for_faces = []

        # Only call DeepFace every N frames; reuse last results between runs
        if frame_idx % max(1, args.every_n) == 0 and len(faces) > 0:
            for (x, y, w, h) in faces:
                # Crop ROI in BGR, DeepFace can handle color images
                roi = frame[y:y+h, x:x+w]
                try:
                    # DeepFace 0.0.95 compatible call:
                    out = DeepFace.analyze(
                        img_path=roi,
                        actions=["emotion"],
                        enforce_detection=False,  # we're already providing a tight face ROI
                        align=True,
                        silent=True
                    )
                    # analyze() may return dict or list; normalize to dict
                    if isinstance(out, list) and len(out) > 0:
                        out = out[0]
                    results_for_faces.append((x, y, w, h, out))
                except Exception as e:
                    # If analysis fails, keep a placeholder so boxes still draw
                    results_for_faces.append((x, y, w, h, {"dominant_emotion": "unknown"}))
            last_results = results_for_faces
            analyzed_this_frame = True
        else:
            # Reuse last results but update boxes with the current detections (best-effort pairing)
            # Simple strategy: if counts match, zip them; else, show boxes without labels.
            if len(last_results) == len(faces):
                for (x, y, w, h), (_, _, _, _, out) in zip(faces, last_results):
                    results_for_faces.append((x, y, w, h, out))
            else:
                for (x, y, w, h) in faces:
                    results_for_faces.append((x, y, w, h, {"dominant_emotion": ""}))

        # Draw
        for (x, y, w, h, out) in results_for_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = out.get("dominant_emotion", "")
            if label:
                draw_label(frame, label, x, y)

        # Tiny HUD
        hud = f"faces:{len(faces)}{'  *' if analyzed_this_frame else ''}  (q to quit)"
        cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Haar + DeepFace Emotion", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
