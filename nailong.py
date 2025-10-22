#!/usr/bin/env python3
import cv2, math, time
import mediapipe as mp
import numpy as np
import os

# --- Your original landmark indices ---
LIPS_UP  = 13   # upper inner lip
LIPS_LOW = 14   # lower inner lip
MOUTH_L  = 291  # left mouth corner
MOUTH_R  = 61   # right mouth corner
NOSE_TIP = 1    # nose tip

# Eye closure detection landmarks
# Left eye landmarks
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

# Right eye landmarks
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# Extra for inter-ocular center calc
EYE_L_IN, EYE_L_OUT = 133, 33
EYE_R_IN, EYE_R_OUT = 362, 263

def dist(a, b): return math.dist(a, b)

def eye_center(landmarks, w, h, i_in, i_out):
    pin = landmarks[i_in]; pout = landmarks[i_out]
    return ((pin.x + pout.x) * 0.5 * w, (pin.y + pout.y) * 0.5 * h)

class EMA:
    def __init__(self, alpha=0.25):
        self.a = alpha
        self.v = None
    def __call__(self, x):
        x = float(x)
        if self.v is None: self.v = x
        else: self.v = self.a * x + (1 - self.a) * self.v
        return self.v
    def reset(self): self.v = None

class Calibrator:
    """Collect ~2s of neutral and set adaptive thresholds."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.samples = []
        self.ready = False
        self.mu, self.sd, self.th = {}, {}, {}
    def add(self, f):
        self.samples.append(f)
    def finalize(self):
        arr = {k: np.array([s[k] for s in self.samples], dtype=np.float32)
               for k in self.samples[0].keys()}
        self.mu = {k: float(v.mean()) for k, v in arr.items()}
        self.sd = {k: float(max(1e-6, v.std())) for k, v in arr.items()}
        # thresholds relative to neutral baseline - only mouth open and wink
        self.th = {
            "mouth_open": self.mu["MAR"]   + 4.0 * self.sd["MAR"],    # extremely conservative for mouth open
            "wink": self.mu["EAR"]  - 1.5 * self.sd["EAR"],    # very conservative threshold for wink detection
        }
        self.ready = True

def compute_features(landmarks, w, h):
    # to pixel
    def P(i):
        lm = landmarks[i]
        return (lm.x * w, lm.y * h)

    upper, lower = P(LIPS_UP), P(LIPS_LOW)
    mouthL, mouthR = P(MOUTH_L), P(MOUTH_R)

    # robust scale: inter-ocular distance
    el = eye_center(landmarks, w, h, EYE_L_IN, EYE_L_OUT)
    er = eye_center(landmarks, w, h, EYE_R_IN, EYE_R_OUT)
    iod = dist(el, er) + 1e-6

    # Debug: print landmark positions occasionally
    if hasattr(compute_features, 'debug_counter'):
        compute_features.debug_counter += 1
    else:
        compute_features.debug_counter = 0
    
    # Head tilt detection - check if head is tilted significantly
    # Use nose tip and eye centers to detect tilt
    nose_tip = P(NOSE_TIP)
    eyeL_center = eye_center(landmarks, w, h, EYE_L_IN, EYE_L_OUT)
    eyeR_center = eye_center(landmarks, w, h, EYE_R_IN, EYE_R_OUT)
    
    # Calculate head tilt angle - use a more reasonable measure
    # Check if nose is significantly off-center between eyes
    eye_center_x = (eyeL_center[0] + eyeR_center[0]) / 2
    nose_to_eye_center_x = abs(nose_tip[0] - eye_center_x)
    head_tilt = nose_to_eye_center_x / iod  # normalize by inter-ocular distance
    
    # features (normalized)
    # Only consider mouth open if head is not significantly tilted
    if head_tilt < 0.5:  # More permissive threshold for head tilt
        # Use multiple mouth landmarks for more robust detection
        # Get several points along the mouth opening
        mouth_top_center = P(13)  # upper inner lip center
        mouth_bottom_center = P(14)  # lower inner lip center
        mouth_top_left = P(12)  # upper lip left
        mouth_top_right = P(15)  # upper lip right
        mouth_bottom_left = P(11)  # lower lip left
        mouth_bottom_right = P(16)  # lower lip right
        
        # Calculate mouth opening using multiple points
        mouth_center_dist = dist(mouth_top_center, mouth_bottom_center)
        mouth_left_dist = dist(mouth_top_left, mouth_bottom_left)
        mouth_right_dist = dist(mouth_top_right, mouth_bottom_right)
        
        # Use the maximum distance for more accurate detection
        # This ensures we catch genuine mouth opening
        max_mouth_dist = max(mouth_center_dist, mouth_left_dist, mouth_right_dist)
        MAR = max_mouth_dist / iod  # mouth open ratio
    else:
        MAR = 0.0  # Don't trigger mouth open if head is tilted
    
    # Wink detection (Eye Aspect Ratio - EAR)
    # Left eye EAR
    left_eye_vertical = dist(P(LEFT_EYE_TOP), P(LEFT_EYE_BOTTOM))
    left_eye_horizontal = dist(P(LEFT_EYE_LEFT), P(LEFT_EYE_RIGHT))
    left_ear = left_eye_vertical / (left_eye_horizontal + 1e-6)

    # Right eye EAR
    right_eye_vertical = dist(P(RIGHT_EYE_TOP), P(RIGHT_EYE_BOTTOM))
    right_eye_horizontal = dist(P(RIGHT_EYE_LEFT), P(RIGHT_EYE_RIGHT))
    right_ear = right_eye_vertical / (right_eye_horizontal + 1e-6)

    # Improved wink detection logic to avoid false positives when looking to the side
    # Key insight: A genuine wink has one eye truly closed (very low EAR) while the other 
    # is fully open (high EAR). Looking to the side causes both eyes to have moderate EARs.
    
    max_ear = max(left_ear, right_ear)
    min_ear = min(left_ear, right_ear)
    ear_difference = abs(left_ear - right_ear)
    ear_ratio = min_ear / (max_ear + 1e-6)  # Ratio of closed/open eye
    
    # Only consider it a wink if ALL conditions are met:
    # 1. One eye is very closed (min_ear < 0.15 - very strict)
    # 2. Other eye is reasonably open (max_ear > 0.25 - ensure one eye is open)
    # 3. Large difference between eyes (ear_difference > 0.15 - significant asymmetry)
    # 4. Low ratio (ear_ratio < 0.5 - one eye much more closed than the other)
    is_winking = (min_ear < 0.15 and max_ear > 0.25 and ear_difference > 0.15 and ear_ratio < 0.5)
    
    if is_winking:
        EAR = min_ear  # Use minimum EAR for wink detection
    else:
        EAR = (left_ear + right_ear) * 0.5  # Use average for normal state
    
    # Debug output only when enabled
    if hasattr(compute_features, 'debug_enabled') and compute_features.debug_enabled:
        if compute_features.debug_counter % 30 == 0:  # Print every 30 frames
            print(f"Debug - mouthL: {mouthL}, mouthR: {mouthR}")
            print(f"Debug - MAR: {MAR:.4f}")
            print(f"Debug - mouth_center_dist: {mouth_center_dist:.4f}, mouth_left_dist: {mouth_left_dist:.4f}, mouth_right_dist: {mouth_right_dist:.4f}")
            print(f"Debug - max_mouth_dist: {max_mouth_dist:.4f}")
            print(f"Debug - left_ear: {left_ear:.4f}, right_ear: {right_ear:.4f}")
            print(f"Debug - max_ear: {max_ear:.4f}, min_ear: {min_ear:.4f}, ear_diff: {ear_difference:.4f}")
            print(f"Debug - ear_ratio: {ear_ratio:.4f}, is_winking: {is_winking}, EAR: {EAR:.4f}")
            print(f"Debug - head_tilt: {head_tilt:.4f}")

    return {"MAR": MAR, "EAR": EAR}

def show_image(image_path, window_name="Detection"):
    """Display an image in a separate window"""
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            # Resize image to fit screen better
            height, width = img.shape[:2]
            max_width = 800
            if width > max_width:
                scale = max_width / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            cv2.imshow(window_name, img)
            return True
    return False

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Facial Expression Detection with MediaPipe")
    parser.add_argument("--debug", action="store_true", help="Show debug information on UI")
    parser.add_argument("--visualize", action="store_true", help="Show facial landmarks and visualizations")
    parser.add_argument("--overlay", action="store_true", help="Show overlay information on webcam feed")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.camera)
    landmarker = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, refine_landmarks=True,
        max_num_faces=5, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    
    # Debug and overlay flags
    DEBUG = args.debug
    VISUALIZE = args.visualize
    OVERLAY = args.overlay
    
    # Print startup information
    print(f"Starting with: DEBUG={DEBUG}, VISUALIZE={VISUALIZE}, OVERLAY={OVERLAY}")
    
    # Set debug flag for compute_features function
    compute_features.debug_enabled = DEBUG

    # Smoothers - only for mouth open and wink (very responsive)
    ema_MAR, ema_EAR = EMA(0.7), EMA(0.7)
    
    # Hysteresis to avoid flicker - very responsive
    last_label = "neutral"
    hold_counter = 0
    HOLD_N = 5  # require only 1 frame for switching (very responsive)
    
    # Cooldown to prevent jittery updates
    last_update_time = 0
    UPDATE_COOLDOWN = 0.8 # seconds between updates
    
    # Calibration
    calib = Calibrator()
    state = "calibrating"
    t0 = time.time()
    
    # Image paths
    nailong_image = "assets/nailong.jpg"
    nailong_wink_image = "assets/nailong_wink.jpg"
    nailong_neutral_image = "assets/nailong_neutral.jpg"
    
    # Track current displayed image to avoid showing same image repeatedly
    current_displayed_image = None
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.process(rgb)

        if res.multi_face_landmarks:
            for face in res.multi_face_landmarks:
                feats = compute_features(face.landmark, w, h)
                sMAR = ema_MAR(feats["MAR"])
                sEAR = ema_EAR(feats["EAR"])

                if state == "calibrating":
                    calib.add({"MAR": sMAR, "EAR": sEAR})
                    # ~2s or at least 20 frames
                    if time.time() - t0 > 2.0 and len(calib.samples) >= 20:
                        calib.finalize()
                        state = "running"
                        # print("Calibration thresholds:", calib.th)
                else:
                    # decide label with priority order + thresholds
                    label = "neutral"
                    if sMAR > calib.th["mouth_open"]:
                        label = "mouth_open"
                    elif sEAR < calib.th["wink"]:
                        label = "wink"

                    # hysteresis
                    if label != last_label:
                        hold_counter += 1
                        if hold_counter >= HOLD_N:
                            # Check cooldown before updating
                            current_time = time.time()
                            if current_time - last_update_time >= UPDATE_COOLDOWN:
                                last_label = label
                                hold_counter = 0
                                last_update_time = current_time
                                
                                # Print detection to terminal
                                print(f"DETECTED: {last_label.upper()}")
                                
                                # Show corresponding image (only update when expression changes)
                                if last_label == "wink" and current_displayed_image != "nailong_wink":
                                    if show_image(nailong_wink_image, "nailong"):
                                        current_displayed_image = "nailong_wink"
                                        print("Showing nailong_wink.jpg")
                                elif last_label == "mouth_open" and current_displayed_image != "nailong":
                                    if show_image(nailong_image, "nailong"):
                                        current_displayed_image = "nailong"
                                        print("Showing nailong.jpg")
                                elif last_label == "neutral" and current_displayed_image != "nailong_neutral":
                                    if show_image(nailong_neutral_image, "nailong"):
                                        current_displayed_image = "nailong_neutral"
                                        print("Showing nailong_neutral.jpg")
                            else:
                                # Still in cooldown, don't update yet
                                hold_counter = 0
                    else:
                        hold_counter = 0

                    # Overlay information (only if --overlay flag is set)
                    if OVERLAY:
                        # Debug: draw face bbox only in debug mode
                        if DEBUG:
                            xs = [lm.x for lm in face.landmark]; ys = [lm.y for lm in face.landmark]
                            x1, y1 = int(min(xs) * w), int(min(ys) * h)
                            x2, y2 = int(max(xs) * w), int(max(ys) * h)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, last_label, (x1, max(0, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                        # HUD with smoothed values
                        hud = f"MAR {sMAR:.3f} | EAR {sEAR:.3f}"
                        cv2.putText(frame, hud, (10, h - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # Debug: show thresholds and current values
                        if DEBUG and calib.ready:
                            debug_hud = f"MAR Thresh: {calib.th['mouth_open']:.3f} | EAR Thresh: {calib.th['wink']:.3f}"
                            cv2.putText(frame, debug_hud, (10, h - 32),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                            # Additional debug: show raw feature values
                            raw_feats = compute_features(face.landmark, w, h)
                            debug_hud2 = f"Raw: MAR={raw_feats['MAR']:.3f} EAR={raw_feats['EAR']:.3f}"
                            cv2.putText(frame, debug_hud2, (10, h - 52),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # UI status (only if overlay is enabled)
        if OVERLAY:
            cv2.putText(frame, f"[{state}]  c=recalibrate, q=quit", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Heuristic Expressions (MediaPipe, Calibrated)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('c'):
            # reset calibration + smoothers
            calib.reset(); state = "calibrating"; t0 = time.time()
            ema_MAR.reset(); ema_EAR.reset()
            last_label = "neutral"; hold_counter = 0
            # Reset image tracking but don't close windows
            current_displayed_image = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
