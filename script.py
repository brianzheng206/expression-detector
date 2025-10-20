#!/usr/bin/env python3
import cv2, math, time
import mediapipe as mp
import numpy as np

# --- Your original landmark indices ---
LIPS_UP  = 13   # upper inner lip (approx)
LIPS_LOW = 14   # lower inner lip
MOUTH_L  = 291  # left mouth corner
MOUTH_R  = 61   # right mouth corner
BROW_L   = 70   # left brow mid (approx)
EYE_L    = 159  # left eye top
BROW_R   = 300  # right brow mid
EYE_R    = 386  # right eye top
NOSE_TIP = 1    # nose tip

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
        # thresholds relative to neutral baseline - only mouth open and eyebrow raise
        self.th = {
            "mouth_open": self.mu["MAR"]   + 1.0 * self.sd["MAR"],    # very sensitive
            "brow_raise": self.mu["BROW"]  + 1.1 * self.sd["BROW"],   # very sensitive
        }
        self.ready = True

def compute_features(landmarks, w, h):
    # to pixel
    def P(i):
        lm = landmarks[i]; return (lm.x * w, lm.y * h)

    upper, lower = P(LIPS_UP), P(LIPS_LOW)
    mouthL, mouthR = P(MOUTH_L), P(MOUTH_R)
    browL, browR = P(BROW_L), P(BROW_R)
    eyeLtop, eyeRtop = P(EYE_L), P(EYE_R)

    # robust scale: inter-ocular distance
    el = eye_center(landmarks, w, h, EYE_L_IN, EYE_L_OUT)
    er = eye_center(landmarks, w, h, EYE_R_IN, EYE_R_OUT)
    iod = dist(el, er) + 1e-6

    # features (normalized)
    MAR = dist(upper, lower) / iod  # mouth open ratio
    
    # Remove smile detection - not needed
    
    # Debug: print landmark positions occasionally
    if hasattr(compute_features, 'debug_counter'):
        compute_features.debug_counter += 1
    else:
        compute_features.debug_counter = 0
    
    # Brow raise: measure vertical distance from brow to eye
    # When raising eyebrows, the brow moves up relative to the eye
    browL_raise = (eyeLtop[1] - browL[1]) / iod  # positive when brow is above eye
    browR_raise = (eyeRtop[1] - browR[1]) / iod  # positive when brow is above eye
    
    # Average the left and right brow raise values
    BROW = (browL_raise + browR_raise) * 0.5
    
    if compute_features.debug_counter % 30 == 0:  # Print every 30 frames
        print(f"Debug - mouthL: {mouthL}, mouthR: {mouthR}")
        print(f"Debug - MAR: {MAR:.4f}")
        print(f"Debug - browL: {browL}, browR: {browR}")
        print(f"Debug - browL_raise: {browL_raise:.4f}, browR_raise: {browR_raise:.4f}, BROW: {BROW:.4f}")

    return {"MAR": MAR, "BROW": BROW}

def main():
    cap = cv2.VideoCapture(0)
    landmarker = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, refine_landmarks=True,
        max_num_faces=5, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    
    # Debug mode - set to True to see raw values
    DEBUG = True

    # Smoothers - only for mouth open and eyebrow raise (very responsive)
    ema_MAR, ema_BRW = EMA(0.7), EMA(0.7)
    # Calibration
    calib = Calibrator()
    state = "calibrating"
    t0 = time.time()

    # Hysteresis to avoid flicker - very responsive
    last_label = "neutral"
    hold_counter = 0
    HOLD_N = 1  # require only 1 frame for switching (very responsive)

    print("Auto-calibrating: keep a neutral face for ~2 seconds. Press 'c' to recalibrate, 'q' to quit.")
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
                sBRW = ema_BRW(feats["BROW"])

            if state == "calibrating":
                calib.add({"MAR": sMAR, "BROW": sBRW})
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
                elif sBRW > calib.th["brow_raise"]:
                    label = "brow_raise"

                # hysteresis
                if label != last_label:
                    hold_counter += 1
                    if hold_counter >= HOLD_N:
                        last_label = label
                        hold_counter = 0
                else:
                    hold_counter = 0

                # draw face bbox
                xs = [lm.x for lm in face.landmark]; ys = [lm.y for lm in face.landmark]
                x1, y1 = int(min(xs) * w), int(min(ys) * h)
                x2, y2 = int(max(xs) * w), int(max(ys) * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, last_label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # HUD with smoothed values (comment out if noisy)
                hud = f"MAR {sMAR:.3f} | BRW {sBRW:.3f}"
                cv2.putText(frame, hud, (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Debug: show thresholds and current values
                if DEBUG and calib.ready:
                    debug_hud = f"MAR Thresh: {calib.th['mouth_open']:.3f} | BRW Thresh: {calib.th['brow_raise']:.3f}"
                    cv2.putText(frame, debug_hud, (10, h - 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    # Additional debug: show raw feature values
                    raw_feats = compute_features(face.landmark, w, h)
                    debug_hud2 = f"Raw: MAR={raw_feats['MAR']:.3f} BRW={raw_feats['BROW']:.3f}"
                    cv2.putText(frame, debug_hud2, (10, h - 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # UI status
        cv2.putText(frame, f"[{state}]  c=recalibrate, q=quit", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Heuristic Expressions (MediaPipe, Calibrated)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('c'):
            # reset calibration + smoothers
            calib.reset(); state = "calibrating"; t0 = time.time()
            ema_MAR.reset(); ema_BRW.reset()
            last_label = "neutral"; hold_counter = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
