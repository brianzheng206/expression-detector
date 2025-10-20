#!/usr/bin/env python3
"""
Hybrid Real-time Facial Expression Detection
Combines MediaPipe landmarks with DeepFace emotion analysis for improved accuracy.

Features:
- MediaPipe for real-time facial landmark detection (mouth open, eyebrow raise)
- DeepFace for emotion validation and context
- Cross-validation between both systems
- Responsive real-time detection with accuracy validation
"""

import cv2
import math
import time
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import argparse

# MediaPipe landmark indices
LIPS_UP  = 13   # upper inner lip
LIPS_LOW = 14   # lower inner lip
MOUTH_L  = 291  # left mouth corner
MOUTH_R  = 61   # right mouth corner
BROW_L   = 70   # left brow mid
EYE_L    = 159  # left eye top
BROW_R   = 300  # right brow mid
EYE_R    = 386  # right eye top
NOSE_TIP = 1    # nose tip

# Extra for inter-ocular center calc
EYE_L_IN, EYE_L_OUT = 133, 33
EYE_R_IN, EYE_R_OUT = 362, 263

def dist(a, b): 
    return math.dist(a, b)

def eye_center(landmarks, w, h, i_in, i_out):
    pin = landmarks[i_in]
    pout = landmarks[i_out]
    return ((pin.x + pout.x) * 0.5 * w, (pin.y + pout.y) * 0.5 * h)

class EMA:
    """Exponential Moving Average for smoothing"""
    def __init__(self, alpha=0.7):
        self.a = alpha
        self.v = None
    
    def __call__(self, x):
        x = float(x)
        if self.v is None: 
            self.v = x
        else: 
            self.v = self.a * x + (1 - self.a) * self.v
        return self.v
    
    def reset(self): 
        self.v = None

class Calibrator:
    """Collect neutral samples and set adaptive thresholds"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.samples = []
        self.ready = False
        self.mu, self.sd, self.th = {}, {}, {}
    
    def add(self, features):
        self.samples.append(features)
    
    def finalize(self):
        arr = {k: np.array([s[k] for s in self.samples], dtype=np.float32)
               for k in self.samples[0].keys()}
        self.mu = {k: float(v.mean()) for k, v in arr.items()}
        self.sd = {k: float(max(1e-6, v.std())) for k, v in arr.items()}
        
        # Adaptive thresholds
        self.th = {
            "mouth_open": self.mu["MAR"] + 1.0 * self.sd["MAR"],
            "brow_raise": self.mu["BROW"] + 1.1 * self.sd["BROW"],
        }
        self.ready = True

def compute_features(landmarks, w, h):
    """Extract facial features from MediaPipe landmarks"""
    def P(i):
        lm = landmarks[i]
        return (lm.x * w, lm.y * h)

    upper, lower = P(LIPS_UP), P(LIPS_LOW)
    mouthL, mouthR = P(MOUTH_L), P(MOUTH_R)
    browL, browR = P(BROW_L), P(BROW_R)
    eyeLtop, eyeRtop = P(EYE_L), P(EYE_R)

    # Inter-ocular distance for normalization
    el = eye_center(landmarks, w, h, EYE_L_IN, EYE_L_OUT)
    er = eye_center(landmarks, w, h, EYE_R_IN, EYE_R_OUT)
    iod = dist(el, er) + 1e-6

    # Mouth Aspect Ratio (MAR)
    MAR = dist(upper, lower) / iod

    # Eyebrow raise: distance from brow to eye
    browL_raise = (eyeLtop[1] - browL[1]) / iod
    browR_raise = (eyeRtop[1] - browR[1]) / iod
    BROW = (browL_raise + browR_raise) * 0.5

    return {"MAR": MAR, "BROW": BROW}

def analyze_emotion_with_deepface(face_roi):
    """Analyze emotion using DeepFace"""
    try:
        result = DeepFace.analyze(
            img_path=face_roi,
            actions=["emotion"],
            enforce_detection=False,
            align=True,
            silent=True
        )
        
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        return result.get("dominant_emotion", "unknown"), result.get("emotion", {})
    except Exception as e:
        return "unknown", {}

def validate_expression_with_emotion(landmark_label, emotion, emotion_scores):
    """Cross-validate MediaPipe detection with DeepFace emotion"""
    if emotion == "unknown":
        return landmark_label  # Fallback to landmark detection
    
    # Define emotion mappings for validation
    mouth_open_emotions = ["surprise", "fear", "disgust", "sad"]
    brow_raise_emotions = ["surprise", "fear"]
    
    if landmark_label == "mouth_open":
        if emotion in mouth_open_emotions:
            return "mouth_open_confirmed"
        else:
            return "neutral"  # False positive from landmarks
    
    elif landmark_label == "brow_raise":
        if emotion in brow_raise_emotions:
            return "brow_raise_confirmed"
        else:
            return "neutral"  # False positive from landmarks
    
    return landmark_label

def draw_face_info(frame, x, y, w, h, label, confidence=None):
    """Draw face rectangle and label"""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw label with background
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    y_text = max(0, y - th - 8)
    
    # Background rectangle
    cv2.rectangle(frame, (x, y_text), (x + tw + 8, y_text + th + baseline + 8), (0, 0, 0), -1)
    cv2.putText(frame, label, (x + 4, y_text + th + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Draw confidence if available
    if confidence:
        conf_text = f"Conf: {confidence:.2f}"
        cv2.putText(frame, conf_text, (x, y + h + 20), font, 0.5, (255, 255, 0), 1)

def main():
    parser = argparse.ArgumentParser(description="Hybrid emotion detection with MediaPipe + DeepFace")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--deepface-every", type=int, default=3, help="Run DeepFace every N frames")
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    args = parser.parse_args()

    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    # Initialize components
    ema_MAR, ema_BROW = EMA(0.7), EMA(0.7)
    calib = Calibrator()
    state = "calibrating"
    t0 = time.time()
    
    # Hysteresis
    last_label = "neutral"
    hold_counter = 0
    HOLD_N = 1
    
    # DeepFace caching
    last_emotion = "unknown"
    last_emotion_scores = {}
    frame_count = 0

    print("Hybrid Emotion Detection")
    print("Auto-calibrating: keep a neutral face for ~2 seconds")
    print("Press 'c' to recalibrate, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe face detection
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract features
                features = compute_features(face_landmarks.landmark, w, h)
                sMAR = ema_MAR(features["MAR"])
                sBROW = ema_BROW(features["BROW"])
                
                # Get face bounding box for DeepFace
                xs = [lm.x for lm in face_landmarks.landmark]
                ys = [lm.y for lm in face_landmarks.landmark]
                x1, y1 = int(min(xs) * w), int(min(ys) * h)
                x2, y2 = int(max(xs) * w), int(max(ys) * h)
                
                # Ensure valid ROI
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_roi = frame[y1:y2, x1:x2]
                
                if state == "calibrating":
                    calib.add({"MAR": sMAR, "BROW": sBROW})
                    if time.time() - t0 > 2.0 and len(calib.samples) >= 20:
                        calib.finalize()
                        state = "running"
                        print("Calibration complete!")
                
                else:
                    # Determine expression from landmarks
                    landmark_label = "neutral"
                    if sMAR > calib.th["mouth_open"]:
                        landmark_label = "mouth_open"
                    elif sBROW > calib.th["brow_raise"]:
                        landmark_label = "brow_raise"
                    
                    # Run DeepFace every N frames for validation
                    if frame_count % args.deepface_every == 0 and face_roi.size > 0:
                        emotion, emotion_scores = analyze_emotion_with_deepface(face_roi)
                        last_emotion = emotion
                        last_emotion_scores = emotion_scores
                    
                    # Cross-validate with DeepFace
                    final_label = validate_expression_with_emotion(
                        landmark_label, last_emotion, last_emotion_scores
                    )
                    
                    # Hysteresis
                    if final_label != last_label:
                        hold_counter += 1
                        if hold_counter >= HOLD_N:
                            last_label = final_label
                            hold_counter = 0
                    else:
                        hold_counter = 0
                    
                    # Draw results
                    draw_face_info(frame, x1, y1, x2-x1, y2-y1, last_label)
                    
                    # Debug information
                    if args.debug:
                        debug_text = f"MAR: {sMAR:.3f} | BROW: {sBROW:.3f}"
                        cv2.putText(frame, debug_text, (10, h - 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        emotion_text = f"DeepFace: {last_emotion}"
                        cv2.putText(frame, emotion_text, (10, h - 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        if calib.ready:
                            thresh_text = f"MAR_th: {calib.th['mouth_open']:.3f} | BROW_th: {calib.th['brow_raise']:.3f}"
                            cv2.putText(frame, thresh_text, (10, h - 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Status display
        status_text = f"[{state}] Frame: {frame_count} | DeepFace: {last_emotion}"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Hybrid Emotion Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Recalibrate
            calib.reset()
            state = "calibrating"
            t0 = time.time()
            ema_MAR.reset()
            ema_BROW.reset()
            last_label = "neutral"
            hold_counter = 0
            print("Recalibrating...")
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
