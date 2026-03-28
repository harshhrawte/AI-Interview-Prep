"""
Detection services for YOLO and MediaPipe
"""

import cv2
import numpy as np
import mediapipe as mp
import torch

from backend.config import YOLO_MODEL_PATH, MODEL_POINTS

# ------------------------------------------------------------------
# MediaPipe initialization
# ------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

hands_module = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------------------------------------------
# YOLO (LOCAL MODEL ONLY)
# ------------------------------------------------------------------
yolo_model = None
yolo_names = None

# Basic, conservative thresholds to make proctoring
# more reliable while avoiding obvious false positives.
YOLO_MIN_CONF_GENERIC = 0.30
YOLO_MIN_CONF_PHONE = 0.40
YOLO_MIN_CONF_PERSON = 0.50


def load_yolo_model():
    """
    Load YOLOv5 model strictly from local file.
    No downloads. No torch.hub. No ultralytics auto-install.
    """
    global yolo_model, yolo_names

    if yolo_model is not None:
        return yolo_model

    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found at {YOLO_MODEL_PATH}\n"
            f"Download yolov5s.pt and place it inside backend/models/yolo/"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLO model from {YOLO_MODEL_PATH} on {device}")

    yolo_model = torch.load(
        YOLO_MODEL_PATH,
        map_location=device
    )

    yolo_model.to(device)
    yolo_model.eval()

    if hasattr(yolo_model, "names"):
        yolo_names = yolo_model.names
    else:
        yolo_names = None

    print("✓ YOLO loaded from local model file")
    return yolo_model


# ------------------------------------------------------------------
# YOLO inference helpers
# ------------------------------------------------------------------
def detect_phone_with_yolo(frame):
    """
    Detect objects using YOLO.
    Returns list of (label, confidence, (x1, y1, x2, y2))
    """
    detections = []

    try:
        model = load_yolo_model()
    except Exception as e:
        print("YOLO unavailable:", e)
        return detections

    try:
        # YOLOv5-style inference
        results = model(frame)

        h, w = frame.shape[:2]
        img_area = max(float(h * w), 1.0)

        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            conf = float(conf)
            if conf < YOLO_MIN_CONF_GENERIC:
                continue

            cls = int(cls)
            label = yolo_names[cls] if yolo_names else str(cls)
            x1, y1, x2, y2 = map(int, box[:4])

            # Filter out extremely tiny boxes that are almost
            # always noise for proctoring purposes.
            box_area = max(float((x2 - x1) * (y2 - y1)), 0.0)
            area_ratio = box_area / img_area
            if area_ratio < 0.002:  # ~0.2% of the frame
                continue

            detections.append((label, conf, (x1, y1, x2, y2)))

    except Exception as e:
        print("YOLO detection error:", e)

    return detections


def phone_in_detections(dets):
    """
    Check if phone is in detections with a slightly
    higher confidence requirement for reliability.
    Returns (bool, best_match_or_none)
    """
    best = None
    for label, conf, box in dets:
        if "phone" in label.lower() or "cell" in label.lower() or "mobile" in label.lower():
            if conf >= YOLO_MIN_CONF_PHONE:
                if best is None or conf > best[1]:
                    best = (label, conf, box)
    if best is not None:
        return True, best
    return False, None


def person_in_detections(dets):
    """
    Check if person is in detections.
    Uses a higher confidence threshold so that
    extra people in the frame are flagged more
    reliably and small artifacts are ignored.
    """
    persons = []
    for label, conf, box in dets:
        if "person" in label.lower() and conf >= YOLO_MIN_CONF_PERSON:
            persons.append((label, conf, box))
    return len(persons) > 0, persons


# ------------------------------------------------------------------
# MediaPipe helpers
# ------------------------------------------------------------------
def get_face_mesh():
    return face_mesh


def get_hands_module():
    return hands_module


def get_mp_draw():
    return mp_draw


# ------------------------------------------------------------------
# Availability & info
# ------------------------------------------------------------------
def is_yolo_available():
    try:
        load_yolo_model()
        return True
    except:
        return False


def get_yolo_info():
    return {
        "available": is_yolo_available(),
        "type": "local_file"
    }
