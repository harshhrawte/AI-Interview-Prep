#!/usr/bin/env python3
"""
attention_monitor_strict_full.py

Combined script:
 - Mediapipe FaceMesh attention monitoring (eyes/head/gaze)
 - Mediapipe Hands for workspace-hand/writing detection
 - YOLO-based object detection (auto-download) for phone detection
 - Automatic room scan at startup (timed) - PRESS 's' TO SKIP
 - Event logging to attention_log_strict.csv

Requirements: Python 3.8+
The script will attempt to pip-install missing packages (opencv-python, mediapipe, numpy, torch).
It tries to load a YOLOv5 model via torch.hub or ultralytics; if weights are missing it will download them.

Run: python attention_monitor_strict_full.py
Press 'q' to quit during monitoring. During room-scan, press 's' to SKIP or 'c' to cancel/rescan if needed.
"""

import os
import sys
import time
import math
import csv
from datetime import datetime
import subprocess
import threading

# --- Auto install helper (best-effort) ---
def pip_install(pkg):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return True
    except Exception as e:
        print(f"Could not auto-install {pkg}: {e}")
        return False

# Try imports, install if missing
try:
    import cv2
    import numpy as np
    import mediapipe as mp
except Exception:
    print("Some packages missing. Attempting to install dependencies...")
    pip_install("opencv-python")
    pip_install("mediapipe")
    pip_install("numpy")
    time.sleep(1)
    import cv2
    import numpy as np
    import mediapipe as mp

# Try to import torch and set up YOLO detection approach
TORCH_AVAILABLE = True
try:
    import torch
except Exception:
    TORCH_AVAILABLE = False
    print("torch not available. Attempting to install torch (this may take a while)...")
    pip_install("torch")
    try:
        import torch
        TORCH_AVAILABLE = True
    except Exception:
        TORCH_AVAILABLE = False
        print("torch still unavailable. YOLO person/phone detection may fail. Try installing torch or ultralytics manually.")

# Try ultralytics (recent YOLO package)
ULTRALYTICS_AVAILABLE = True
try:
    from ultralytics import YOLO
except Exception:
    ULTRALYTICS_AVAILABLE = False
    pip_install("ultralytics")
    try:
        from ultralytics import YOLO
        ULTRALYTICS_AVAILABLE = True
    except Exception:
        ULTRALYTICS_AVAILABLE = False

# -----------------------------------------------------------------------------
# -------------------------- Configuration -----------------------------------
# -----------------------------------------------------------------------------
LOOK_AWAY_SECONDS_THRESHOLD = 2.0
NO_FACE_SECONDS_THRESHOLD = 1.5
PARTIAL_FACE_SECONDS_THRESHOLD = 1.5
MIN_FACE_BOX_WIDTH = 0.22
EDGE_MARGIN = 0.03

GAZE_LEFT_THRESH = 0.35
GAZE_RIGHT_THRESH = 0.65
EYE_DOWN_THRESH = 0.70
EYE_UP_THRESH = 0.30
HEAD_PITCH_DOWN_DEG = 30.0

LOG_FILENAME = "attention_log_strict.csv"

ROOM_SCAN_SECONDS = 10
REQUIRED_NO_PERSON_RATIO = 0.95

HAND_WRITING_Y_THRESHOLD = 0.65
HAND_MOTION_WINDOW = 1.5
HAND_MOTION_MIN_SPEED = 0.006

MODELS_DIR = "models"
YOLO_PT = os.path.join(MODELS_DIR, "yolov5s.pt")
# -----------------------------------------------------------------------------

os.makedirs(MODELS_DIR, exist_ok=True)

with open(LOG_FILENAME, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp_utc", "event", "detail"])

def log_event(event: str, detail: str = ""):
    ts = datetime.utcnow().isoformat()
    print(f"[{ts}] {event} - {detail}")
    with open(LOG_FILENAME, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, event, detail])

# -----------------------------------------------------------------------------
# -------------------------- YOLO loader / phone detection --------------------
# -----------------------------------------------------------------------------
yolo_model = None
yolo_names = None
use_yolo_ultralytics = False
use_torch_hub = False

def try_load_yolo():
    global yolo_model, yolo_names, use_yolo_ultralytics, use_torch_hub
    if TORCH_AVAILABLE:
        try:
            print("Trying torch.hub to load yolov5s...")
            yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            yolo_names = yolo_model.names if hasattr(yolo_model, "names") else None
            use_torch_hub = True
            print("Loaded YOLOv5 via torch.hub (yolov5s).")
            return True
        except Exception as e:
            print("torch.hub yolov5 load failed:", e)

    if ULTRALYTICS_AVAILABLE:
        try:
            print("Trying ultralytics YOLO (yolov8n.pt) ...")
            model_path = os.path.join(MODELS_DIR, "yolov8n.pt")
            yolo_model = YOLO(model_path)
            try:
                yolo_names = yolo_model.model.names
            except Exception:
                yolo_names = None
            use_yolo_ultralytics = True
            print("Loaded ultralytics YOLO (yolov8n).")
            return True
        except Exception as e:
            print("ultralytics YOLO failed:", e)

    if os.path.exists(YOLO_PT) and ULTRALYTICS_AVAILABLE:
        try:
            print("Loading local YOLO weights:", YOLO_PT)
            yolo_model = YOLO(YOLO_PT)
            yolo_names = yolo_model.model.names
            use_yolo_ultralytics = True
            return True
        except Exception as e:
            print("Failed to load local YOLO weights:", e)

    print("No YOLO model available. Phone detection will be disabled.")
    return False

try_load_yolo()

# -----------------------------------------------------------------------------
# -------------------------- Mediapipe init ----------------------------------
# -----------------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
hands_module = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

HP_N = 1
HP_CHIN = 152
HP_LEFT_EYE = 33
HP_RIGHT_EYE = 263
HP_LEFT_MOUTH = 61
HP_RIGHT_MOUTH = 291

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -20.0),
    (28.9, -28.9, -20.0)
], dtype=np.float64)

# -----------------------------------------------------------------------------
# -------------------------- Utility functions --------------------------------
# -----------------------------------------------------------------------------
def detect_phone_with_yolo(frame):
    detections = []
    if yolo_model is None:
        return detections

    try:
        if use_torch_hub:
            results = yolo_model(frame)
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                cls = int(cls)
                label = yolo_model.names[cls] if hasattr(yolo_model, "names") else str(cls)
                x1, y1, x2, y2 = map(int, box[:4])
                detections.append((label, float(conf), (x1, y1, x2, y2)))
        elif use_yolo_ultralytics:
            results = yolo_model.predict(frame, imgsz=640, conf=0.35, verbose=False)
            for r in results:
                boxes = r.boxes
                for b in boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                    conf = float(b.conf[0].cpu().numpy())
                    cls = int(b.cls[0].cpu().numpy())
                    try:
                        label = r.names[cls]
                    except Exception:
                        label = str(cls)
                    detections.append((label, conf, (x1, y1, x2, y2)))
    except Exception as e:
        print("YOLO detection error:", e)
    return detections

def phone_in_detections(dets):
    for label, conf, box in dets:
        if "phone" in label.lower() or "cell" in label.lower() or "mobile" in label.lower():
            return True, (label, conf, box)
    return False, None

# -----------------------------------------------------------------------------
# -------------------------- Room Scan Routine --------------------------------
# -----------------------------------------------------------------------------
def do_room_scan(cap, scan_seconds=ROOM_SCAN_SECONDS):
    """
    Automatically runs a timed room scan. Returns True if scan passes (no persons detected),
    False otherwise. Press 's' to SKIP the scan and proceed directly to monitoring.
    """
    start = time.time()
    frames = 0
    frames_with_person = 0
    print(f"Room scan: please rotate/scan the room for {scan_seconds} seconds. Press 's' to SKIP.")
    log_event("ROOM_SCAN_STARTED", f"duration={scan_seconds}s")
    
    while time.time() - start < scan_seconds:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        display = frame.copy()
        cv2.putText(display, f"ROOM SCAN: Show room... {int(scan_seconds - (time.time()-start))}s", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(display, "Press 's' to SKIP scan", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.imshow("Strict Attention Monitor - Room Scan", display)
        key = cv2.waitKey(1)
        
        # SKIP functionality - press 's'
        if key & 0xFF == ord('s'):
            log_event("ROOM_SCAN_SKIPPED", "User pressed 's' to skip")
            print("Room scan SKIPPED by user.")
            return True  # Return True to proceed to monitoring
        
        if key & 0xFF == ord('c'):
            log_event("ROOM_SCAN_CANCELLED", "")
            return False

        detected_person = False
        if yolo_model is not None:
            dets = detect_phone_with_yolo(frame)
            for label, conf, box in dets:
                if label.lower() == "person" or "person" in label.lower():
                    detected_person = True
                    break
        if not detected_person:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = face_mesh.process(rgb)
            if r.multi_face_landmarks:
                detected_person = True

        if detected_person:
            frames_with_person += 1
            cv2.putText(display, "Person detected during scan!", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        cv2.imshow("Strict Attention Monitor - Room Scan", display)

    no_person_ratio = 1.0 - (frames_with_person / max(1, frames))
    log_event("ROOM_SCAN_RESULT", f"frames={frames} frames_with_person={frames_with_person} no_person_ratio={no_person_ratio:.3f}")
    
    if no_person_ratio >= REQUIRED_NO_PERSON_RATIO:
        print("Room scan passed (no person found).")
        log_event("ROOM_SCAN_PASSED", f"no_person_ratio={no_person_ratio:.3f}")
        return True
    else:
        print("Room scan failed: person(s) detected during scan. Please re-run.")
        log_event("ROOM_SCAN_FAILED", f"no_person_ratio={no_person_ratio:.3f}")
        return False

# -----------------------------------------------------------------------------
# -------------------------- Monitoring Loop ---------------------------------
# -----------------------------------------------------------------------------
def monitoring_loop():
    last_face_seen_time = time.time()
    last_full_face_time = time.time()
    last_looking_center_time = time.time()
    current_alert = None

    hand_history = []
    hand_writing_state = False
    hand_writing_since = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Room scan with skip option
    passed = do_room_scan(cap, ROOM_SCAN_SECONDS)
    if not passed:
        prompt_start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()
            cv2.putText(display, "Room scan failed. Press 'r' to retry, 's' to skip, 'q' quit.", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.imshow("Strict Attention Monitor - Room Scan", display)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('r'):
                log_event("ROOM_SCAN_RETRY_REQUEST", "")
                passed = do_room_scan(cap, ROOM_SCAN_SECONDS)
                if passed:
                    break
            elif key & 0xFF == ord('s'):
                log_event("ROOM_SCAN_SKIPPED_AFTER_FAIL", "User chose to skip after failed scan")
                print("Room scan skipped after failure.")
                break
            elif key & 0xFF == ord('q'):
                log_event("ROOM_SCAN_ABORTED_BY_USER", "")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key != -1:
                log_event("ROOM_SCAN_CONTINUE_MANUAL", "User chose to continue despite scan failed")
                break
            if time.time() - prompt_start > 60:
                log_event("ROOM_SCAN_CONTINUE_TIMEOUT", "")
                break

    log_event("MONITORING_STARTED", "")
    print("Starting monitoring... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        hands_results = hands_module.process(rgb)
        now = time.time()
        status_text = "OK"
        alert_text = ""
        alert_on = False

        phone_detected = False
        phone_det_info = None
        if yolo_model is not None:
            try:
                dets = detect_phone_with_yolo(frame)
                phone_detected, phone_det_info = phone_in_detections(dets)
                for label, conf, box in dets:
                    x1,y1,x2,y2 = box
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,120,255), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,120,255), 2)
            except Exception as e:
                print("YOLO frame error:", e)
        if phone_detected:
            alert_on = True
            alert_text += f"Phone detected ({phone_det_info[0]} {phone_det_info[1]:.2f}) "
            if current_alert != "phone":
                log_event("ALERT_PHONE_DETECTED", f"{phone_det_info}")
                current_alert = "phone"

        hand_centers = []
        if hands_results.multi_hand_landmarks:
            for handlms in hands_results.multi_hand_landmarks:
                xs = [lm.x for lm in handlms.landmark]
                ys = [lm.y for lm in handlms.landmark]
                cx = float(np.mean(xs))
                cy = float(np.mean(ys))
                hand_centers.append((cx, cy))
                mp_draw.draw_landmarks(frame, handlms, mp_hands.HAND_CONNECTIONS)

        tnow = now
        for (cx,cy) in hand_centers:
            hand_history.append((tnow, cx, cy))
        hand_history = [h_i for h_i in hand_history if tnow - h_i[0] <= HAND_MOTION_WINDOW]

        hand_writing_state = False
        if len(hand_history) >= 2:
            xs = [x for (_, x, _) in hand_history]
            ys = [y for (_, _, y) in hand_history]
            dt = hand_history[-1][0] - hand_history[0][0]
            dx = xs[-1] - xs[0]
            dy = ys[-1] - ys[0]
            speed = math.hypot(dx, dy) / max(1e-6, dt)
            last_y = hand_history[-1][2]
            if speed >= HAND_MOTION_MIN_SPEED and last_y >= HAND_WRITING_Y_THRESHOLD:
                hand_writing_state = True
                hand_writing_since = hand_history[0][0]
        
        if hand_writing_state:
            cv2.putText(frame, "HAND: writing (heuristic)", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if not results.multi_face_landmarks:
            if now - last_face_seen_time > NO_FACE_SECONDS_THRESHOLD:
                alert_text = ("No face detected!" if not alert_text else alert_text + " | No face detected!")
                alert_on = True
                if current_alert != "no_face":
                    log_event("ALERT_NO_FACE", f"No face for {now - last_face_seen_time:.1f}s")
                    current_alert = "no_face"
            else:
                status_text = "No face (waiting)"
        else:
            last_face_seen_time = now
            current_alert = None

            lm = results.multi_face_landmarks[0].landmark
            xs = np.array([l.x for l in lm])
            ys = np.array([l.y for l in lm])
            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
            box_w = max_x - min_x
            box_h = max_y - min_y
            touching_edge = (min_x < EDGE_MARGIN) or (max_x > 1.0 - EDGE_MARGIN) or (min_y < EDGE_MARGIN) or (max_y > 1.0 - EDGE_MARGIN)

            if box_w < MIN_FACE_BOX_WIDTH or touching_edge:
                if now - last_full_face_time > PARTIAL_FACE_SECONDS_THRESHOLD:
                    alert_text = ("Partial / face not fully visible!" if not alert_text else alert_text + " | Partial face!")
                    alert_on = True
                    if current_alert != "partial_face":
                        log_event("ALERT_PARTIAL_FACE", f"bbox_w={box_w:.3f}, touching_edge={touching_edge}")
                        current_alert = "partial_face"
                else:
                    status_text = "Face partial (waiting)"
            else:
                last_full_face_time = now
                current_alert = None

                left_eye_x = (lm[LEFT_EYE_CORNERS[0]].x + lm[LEFT_EYE_CORNERS[1]].x) / 2.0
                right_eye_x = (lm[RIGHT_EYE_CORNERS[0]].x + lm[RIGHT_EYE_CORNERS[1]].x) / 2.0
                center_x = (left_eye_x + right_eye_x) / 2.0
                left_iris_x = np.mean([lm[i].x for i in LEFT_IRIS])
                left_iris_y = np.mean([lm[i].y for i in LEFT_IRIS])
                right_iris_x = np.mean([lm[i].x for i in RIGHT_IRIS])
                right_iris_y = np.mean([lm[i].y for i in RIGHT_IRIS])
                left_eye_left_x = lm[LEFT_EYE_CORNERS[0]].x
                left_eye_right_x = lm[LEFT_EYE_CORNERS[1]].x
                right_eye_left_x = lm[RIGHT_EYE_CORNERS[0]].x
                right_eye_right_x = lm[RIGHT_EYE_CORNERS[1]].x
                left_eye_width = max(left_eye_right_x - left_eye_left_x, 1e-6)
                right_eye_width = max(right_eye_right_x - right_eye_left_x, 1e-6)
                left_h_ratio = (left_iris_x - left_eye_left_x) / left_eye_width
                right_h_ratio = (right_iris_x - right_eye_left_x) / right_eye_width
                gaze_h_ratio = (left_h_ratio + right_h_ratio) / 2.0

                left_top_y = lm[LEFT_EYE_TOP].y
                left_bottom_y = lm[LEFT_EYE_BOTTOM].y
                right_top_y = lm[RIGHT_EYE_TOP].y
                right_bottom_y = lm[RIGHT_EYE_BOTTOM].y
                left_eye_height = max(left_bottom_y - left_top_y, 1e-6)
                right_eye_height = max(right_bottom_y - right_top_y, 1e-6)
                left_v_ratio = (left_iris_y - left_top_y) / left_eye_height
                right_v_ratio = (right_iris_y - right_top_y) / right_eye_height
                gaze_v_ratio = (left_v_ratio + right_v_ratio) / 2.0

                cv2.circle(frame, (int(left_iris_x*w), int(left_iris_y*h)), 3, (0,255,255), -1)
                cv2.circle(frame, (int(right_iris_x*w), int(right_iris_y*h)), 3, (0,255,255), -1)
                cv2.rectangle(frame, (int(min_x*w), int(min_y*h)), (int(max_x*w), int(max_y*h)), (200,200,200), 1)

                if gaze_h_ratio < GAZE_LEFT_THRESH:
                    gaze_state_h = "right"
                    gaze_text_h = "Looking Right"
                elif gaze_h_ratio > GAZE_RIGHT_THRESH:
                    gaze_state_h = "left"
                    gaze_text_h = "Looking Left"
                else:
                    gaze_state_h = "center"
                    gaze_text_h = "Looking Center"

                if gaze_v_ratio > EYE_DOWN_THRESH:
                    gaze_state_v = "down"
                    gaze_text_v = "Eyes Down"
                elif gaze_v_ratio < EYE_UP_THRESH:
                    gaze_state_v = "up"
                    gaze_text_v = "Eyes Up"
                else:
                    gaze_state_v = "center_v"
                    gaze_text_v = "Eyes Center"

                image_points = np.array([
                    (lm[HP_N].x * w, lm[HP_N].y * h),
                    (lm[HP_CHIN].x * w, lm[HP_CHIN].y * h),
                    (lm[HP_LEFT_EYE].x * w, lm[HP_LEFT_EYE].y * h),
                    (lm[HP_RIGHT_EYE].x * w, lm[HP_RIGHT_EYE].y * h),
                    (lm[HP_LEFT_MOUTH].x * w, lm[HP_LEFT_MOUTH].y * h),
                    (lm[HP_RIGHT_MOUTH].x * w, lm[HP_RIGHT_MOUTH].y * h)
                ], dtype=np.float64)

                focal_length = w
                center = (w/2, h/2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype="double")
                dist_coeffs = np.zeros((4,1))
                try:
                    success_pnp, rotation_vector, translation_vector = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                    pitch = 0.0
                    yaw = 0.0
                    roll = 0.0
                    if success_pnp:
                        rmat, _ = cv2.Rodrigues(rotation_vector)
                        proj_matrix = np.hstack((rmat, translation_vector))
                        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
                        pitch = float(eulerAngles[0])
                        yaw = float(eulerAngles[1])
                        roll = float(eulerAngles[2])
                except Exception as e:
                    pitch = 0.0
                    yaw = 0.0
                    roll = 0.0

                attention_ok = True
                reason = []

                if gaze_state_h != "center":
                    if gaze_state_h == "center":
                        last_looking_center_time = now
                    else:
                        if now - last_looking_center_time > LOOK_AWAY_SECONDS_THRESHOLD:
                            attention_ok = False
                            reason.append(f"Horizontal gaze away ({gaze_state_h}) {now - last_looking_center_time:.1f}s")
                            if current_alert != "look_away":
                                log_event("ALERT_LOOK_AWAY", f"{gaze_state_h} for {now - last_looking_center_time:.1f}s, gaze_h_ratio={gaze_h_ratio:.2f}")
                                current_alert = "look_away"
                else:
                    last_looking_center_time = now

                if (gaze_state_v == "down" or pitch > HEAD_PITCH_DOWN_DEG):
                    if hand_writing_state:
                        reason.append("Eyes/Head down but hand writing -> suppressed")
                    else:
                        attention_ok = False
                        if gaze_state_v == "down":
                            reason.append(f"Eyes looking down (v_ratio={gaze_v_ratio:.2f})")
                            if current_alert != "eyes_down":
                                log_event("ALERT_EYES_DOWN", f"v_ratio={gaze_v_ratio:.2f}")
                                current_alert = "eyes_down"
                        if pitch > HEAD_PITCH_DOWN_DEG:
                            reason.append(f"Head pitch down ({pitch:.1f} deg)")
                            if current_alert != "head_down":
                                log_event("ALERT_HEAD_DOWN", f"pitch={pitch:.1f}")
                                current_alert = "head_down"

                if not attention_ok:
                    alert_on = True
                    alert_text = "; ".join(reason)
                else:
                    status_text = "Attention: OK"

                cv2.putText(frame, f"Hg:{gaze_h_ratio:.2f} Vg:{gaze_v_ratio:.2f}", (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f"Pitch:{pitch:.1f}", (20, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, gaze_text_h + " | " + gaze_text_v, (20, h-100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if alert_on:
            overlay = frame.copy()
            alpha = 0.35
            cv2.rectangle(overlay, (0,0), (w,h), (0,0,255), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.putText(frame, "ALERT", (w//2 - 70, 70), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255,255,255), 3)
            cv2.putText(frame, alert_text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        else:
            cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        if hand_writing_state:
            cv2.putText(frame, "Writing detected (suppresses head/eyes-down alerts)", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Strict Attention Monitor", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            log_event("MONITORING_STOPPED_BY_USER", "")
            break

    cap.release()
    cv2.destroyAllWindows()
    log_event("MONITORING_ENDED", "")
    
if __name__ == "__main__":
    try:
        monitoring_loop()
    except Exception as e:
        log_event("MONITORING_CRASH", str(e))
        print("Error:", e)
        raise