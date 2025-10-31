#!/usr/bin/env python3
"""
combined_app.py - Unified FastAPI backend with:
- Resume upload & RAG-based question generation (Gemini)
- Complete attention monitoring (face, gaze, hands, phone, person detection)
- YOLO object detection
- MediaPipe face mesh & hand tracking
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import uuid
import io
import cv2
import numpy as np
import time
import math
import csv
from datetime import datetime
from collections import deque
import base64
import sys
import subprocess

# PDF and AI imports
from PyPDF2 import PdfReader
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"✓ Gemini API configured")
else:
    print("⚠ Warning: GEMINI_API_KEY not found in .env")

# FastAPI app
app = FastAPI(title="AI Interview Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODELS_DIR = 'models'
LOG_FILENAME = "attention_log.csv"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Monitoring thresholds
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

HAND_WRITING_Y_THRESHOLD = 0.65
HAND_MOTION_WINDOW = 1.5
HAND_MOTION_MIN_SPEED = 0.006

# Initialize CSV log
if not os.path.exists(LOG_FILENAME):
    with open(LOG_FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "event", "detail"])

def log_event(event: str, detail: str = ""):
    """Log event to CSV file"""
    ts = datetime.utcnow().isoformat()
    print(f"[{ts}] {event} - {detail}")
    with open(LOG_FILENAME, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, event, detail])

# Auto install helper
def pip_install(pkg):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return True
    except Exception as e:
        print(f"Could not auto-install {pkg}: {e}")
        return False

# Import dependencies
try:
    import mediapipe as mp
except:
    print("Installing mediapipe...")
    pip_install("mediapipe")
    import mediapipe as mp

TORCH_AVAILABLE = True
try:
    import torch
except:
    TORCH_AVAILABLE = False
    print("torch not available. YOLO detection disabled.")

ULTRALYTICS_AVAILABLE = True
try:
    from ultralytics import YOLO
except:
    ULTRALYTICS_AVAILABLE = False
    print("ultralytics not available. Attempting install...")
    pip_install("ultralytics")
    try:
        from ultralytics import YOLO
        ULTRALYTICS_AVAILABLE = True
    except:
        ULTRALYTICS_AVAILABLE = False

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
hands_module = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Sentence Transformer for RAG
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Session stores
session_store = {}  # For resume RAG sessions
monitoring_sessions = {}  # For monitoring sessions

# Face mesh landmark indices
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

# YOLO initialization
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
            print("✓ Loaded YOLOv5 via torch.hub")
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
            except:
                yolo_names = None
            use_yolo_ultralytics = True
            print("✓ Loaded ultralytics YOLO (yolov8n)")
            return True
        except Exception as e:
            print("ultralytics YOLO failed:", e)

    print("⚠ No YOLO model available. Phone/person detection disabled.")
    return False

try_load_yolo()

def detect_phone_with_yolo(frame):
    """Detect objects using YOLO"""
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
                    except:
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

def person_in_detections(dets):
    persons = []
    for label, conf, box in dets:
        if label.lower() == "person" or "person" in label.lower():
            persons.append((label, conf, box))
    return len(persons) > 0, persons

class MonitoringSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.active = False
        self.alerts = deque(maxlen=200)
        self.last_face_seen = time.time()
        self.last_full_face = time.time()
        self.last_looking_center = time.time()
        self.current_alert = None
        self.hand_history = []
        self.hand_writing_state = False
        self.room_scan_completed = False
        self.room_scan_passed = False
        
    def log_alert(self, alert_type, detail):
        timestamp = datetime.utcnow().isoformat()
        self.alerts.append({
            'timestamp': timestamp,
            'type': alert_type,
            'detail': detail
        })
        log_event(alert_type, detail)

def process_frame_complete(frame, session):
    """Complete frame processing with all monitoring features"""
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    hands_results = hands_module.process(rgb)
    now = time.time()
    
    alerts = []
    status = "OK"
    alert_details = []
    
    # Phone detection
    phone_detected = False
    phone_info = None
    person_detected = False
    person_count = 0
    
    if yolo_model is not None:
        try:
            dets = detect_phone_with_yolo(frame)
            phone_detected, phone_info = phone_in_detections(dets)
            person_detected, persons = person_in_detections(dets)
            person_count = len(persons)
            
            for label, conf, box in dets:
                x1, y1, x2, y2 = box
                color = (0, 0, 255) if "phone" in label.lower() else (0, 120, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print("YOLO error:", e)
    
    if phone_detected:
        alerts.append("PHONE_DETECTED")
        alert_details.append(f"Phone: {phone_info[0]} ({phone_info[1]:.2f})")
        if session.current_alert != "phone":
            session.log_alert("ALERT_PHONE_DETECTED", f"{phone_info}")
            session.current_alert = "phone"
    
    if person_count > 1:
        alerts.append("MULTIPLE_PERSONS")
        alert_details.append(f"Multiple persons: {person_count}")
        if session.current_alert != "multiple_persons":
            session.log_alert("ALERT_MULTIPLE_PERSONS", f"count={person_count}")
            session.current_alert = "multiple_persons"
    
    # Hand detection
    hand_centers = []
    if hands_results.multi_hand_landmarks:
        for handlms in hands_results.multi_hand_landmarks:
            xs = [lm.x for lm in handlms.landmark]
            ys = [lm.y for lm in handlms.landmark]
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            hand_centers.append((cx, cy))
            mp_draw.draw_landmarks(frame, handlms, mp_hands.HAND_CONNECTIONS)
    
    for (cx, cy) in hand_centers:
        session.hand_history.append((now, cx, cy))
    session.hand_history = [h_i for h_i in session.hand_history if now - h_i[0] <= HAND_MOTION_WINDOW]
    
    session.hand_writing_state = False
    if len(session.hand_history) >= 2:
        xs = [x for (_, x, _) in session.hand_history]
        ys = [y for (_, _, y) in session.hand_history]
        dt = session.hand_history[-1][0] - session.hand_history[0][0]
        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]
        speed = math.hypot(dx, dy) / max(1e-6, dt)
        last_y = session.hand_history[-1][2]
        if speed >= HAND_MOTION_MIN_SPEED and last_y >= HAND_WRITING_Y_THRESHOLD:
            session.hand_writing_state = True
            cv2.putText(frame, "WRITING DETECTED", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Face detection
    if not results.multi_face_landmarks:
        if now - session.last_face_seen > NO_FACE_SECONDS_THRESHOLD:
            alerts.append("NO_FACE")
            alert_details.append(f"No face for {now - session.last_face_seen:.1f}s")
            status = "No face detected"
            if session.current_alert != "no_face":
                session.log_alert("ALERT_NO_FACE", f"duration={now - session.last_face_seen:.1f}s")
                session.current_alert = "no_face"
        else:
            status = "No face (waiting)"
    else:
        session.last_face_seen = now
        if session.current_alert == "no_face":
            session.current_alert = None
        
        lm = results.multi_face_landmarks[0].landmark
        xs = np.array([l.x for l in lm])
        ys = np.array([l.y for l in lm])
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        box_w = max_x - min_x
        
        touching_edge = (min_x < EDGE_MARGIN) or (max_x > 1.0 - EDGE_MARGIN) or \
                       (min_y < EDGE_MARGIN) or (max_y > 1.0 - EDGE_MARGIN)
        
        if box_w < MIN_FACE_BOX_WIDTH or touching_edge:
            if now - session.last_full_face > PARTIAL_FACE_SECONDS_THRESHOLD:
                alerts.append("PARTIAL_FACE")
                alert_details.append("Face partially visible")
                status = "Partial face"
                if session.current_alert != "partial_face":
                    session.log_alert("ALERT_PARTIAL_FACE", f"bbox_w={box_w:.3f}, edge={touching_edge}")
                    session.current_alert = "partial_face"
        else:
            session.last_full_face = now
            if session.current_alert == "partial_face":
                session.current_alert = None
            
            # Gaze tracking
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
            
            cv2.circle(frame, (int(left_iris_x*w), int(left_iris_y*h)), 3, (0, 255, 255), -1)
            cv2.circle(frame, (int(right_iris_x*w), int(right_iris_y*h)), 3, (0, 255, 255), -1)
            cv2.rectangle(frame, (int(min_x*w), int(min_y*h)),
                         (int(max_x*w), int(max_y*h)), (200, 200, 200), 1)
            
            gaze_state_h = "center"
            if gaze_h_ratio < GAZE_LEFT_THRESH:
                gaze_state_h = "right"
            elif gaze_h_ratio > GAZE_RIGHT_THRESH:
                gaze_state_h = "left"
            
            gaze_state_v = "center_v"
            if gaze_v_ratio > EYE_DOWN_THRESH:
                gaze_state_v = "down"
            elif gaze_v_ratio < EYE_UP_THRESH:
                gaze_state_v = "up"
            
            # Head pose
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
            dist_coeffs = np.zeros((4, 1))
            
            pitch = 0.0
            try:
                success_pnp, rotation_vector, translation_vector = cv2.solvePnP(
                    MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                if success_pnp:
                    rmat, _ = cv2.Rodrigues(rotation_vector)
                    proj_matrix = np.hstack((rmat, translation_vector))
                    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
                    pitch = float(eulerAngles[0])
            except:
                pitch = 0.0
            
            attention_ok = True
            
            if gaze_state_h != "center":
                if now - session.last_looking_center > LOOK_AWAY_SECONDS_THRESHOLD:
                    alerts.append("LOOKING_AWAY")
                    alert_details.append(f"Looking {gaze_state_h}")
                    attention_ok = False
                    if session.current_alert != "look_away":
                        session.log_alert("ALERT_LOOK_AWAY", gaze_state_h)
                        session.current_alert = "look_away"
            else:
                session.last_looking_center = now
                if session.current_alert == "look_away":
                    session.current_alert = None
            
            if gaze_state_v == "down" or pitch > HEAD_PITCH_DOWN_DEG:
                if not session.hand_writing_state:
                    attention_ok = False
                    if gaze_state_v == "down":
                        alerts.append("EYES_DOWN")
                        alert_details.append("Eyes looking down")
                    if pitch > HEAD_PITCH_DOWN_DEG:
                        alerts.append("HEAD_DOWN")
                        alert_details.append(f"Head down ({pitch:.1f}°)")
            
            if attention_ok and not phone_detected and person_count <= 1:
                status = "Attention OK"
            
            cv2.putText(frame, f"Gaze H:{gaze_h_ratio:.2f} V:{gaze_v_ratio:.2f}",
                       (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Pitch:{pitch:.1f}",
                       (10, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if alerts:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, "ALERT", (w//2 - 60, 60),
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
        y_offset = 100
        for detail in alert_details[:3]:
            cv2.putText(frame, detail, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
    else:
        cv2.putText(frame, status, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame, alerts, status, alert_details

# PDF/RAG Helper Functions
def extract_text_from_pdf(file) -> str:
    reader = PdfReader(io.BytesIO(file))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Pydantic Models
class FrameRequest(BaseModel):
    session_id: str
    frame: str

class SessionRequest(BaseModel):
    session_id: str

class RoomScanComplete(BaseModel):
    session_id: str
    passed: bool

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health():
    return {
        'status': 'healthy',
        'active_sessions': len(monitoring_sessions),
        'yolo_available': yolo_model is not None,
        'mediapipe_available': True,
        'gemini_configured': GEMINI_API_KEY is not None
    }

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    """Upload resume and create RAG session"""
    try:
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        chunks = chunk_text(text)
        embeddings = embedder.encode(chunks)
        
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        session_id = str(uuid.uuid4())
        session_store[session_id] = {
            "faiss_index": index,
            "chunks": chunks,
            "embeddings": embeddings,
            "filename": file.filename
        }
        
        log_event("RESUME_UPLOADED", f"session={session_id}, file={file.filename}")
        return {"session_id": session_id, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/generate_questions")
async def generate_questions(session_id: str = Form(...), job_description: str = Form(...)):
    """Generate interview questions using RAG + Gemini"""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        keywords = [w.strip('.,') for w in job_description.split() if len(w) > 3]
        keyword_embeds = embedder.encode([" ".join(keywords)])
        
        index = session_store[session_id]["faiss_index"]
        D, I = index.search(keyword_embeds, 3)
        relevant_chunks = [
            session_store[session_id]["chunks"][i]
            for i in I[0] if i < len(session_store[session_id]["chunks"])
        ]
        
        resume_context = "\n".join(relevant_chunks)
        prompt = f"""You are an interview coach. Based on the following resume and job description, generate 10 tailored interview preparation questions:

Resume Context:
{resume_context}

Job Description:
{job_description}

Generate 10 specific, relevant interview questions."""
        
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="Gemini API key not configured")
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        questions = [q.strip() for q in response.text.split("\n") if q.strip()]
        
        log_event("QUESTIONS_GENERATED", f"session={session_id}, count={len(questions)}")
        return {"questions": questions, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate questions: {str(e)}")

@app.post("/start_monitoring")
async def start_monitoring(request: SessionRequest):
    """Start attention monitoring"""
    session_id = request.session_id
    
    if session_id not in monitoring_sessions:
        monitoring_sessions[session_id] = MonitoringSession(session_id)
    
    session = monitoring_sessions[session_id]
    session.active = True
    session.log_alert("MONITORING_STARTED", f"session_id={session_id}")
    
    return {
        'message': 'Monitoring started',
        'session_id': session_id,
        'timestamp': datetime.utcnow().isoformat()
    }

@app.post("/stop_monitoring")
async def stop_monitoring(request: SessionRequest):
    """Stop attention monitoring"""
    session_id = request.session_id
    
    if session_id not in monitoring_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = monitoring_sessions[session_id]
    session.active = False
    session.log_alert("MONITORING_STOPPED", "User stopped monitoring")
    
    return {
        'message': 'Monitoring stopped',
        'session_id': session_id,
        'total_alerts': len(session.alerts)
    }

@app.post("/process_frame")
async def process_frame(request: FrameRequest):
    """Process video frame for monitoring"""
    session_id = request.session_id
    frame_data = request.frame
    
    if session_id not in monitoring_sessions:
        raise HTTPException(status_code=400, detail="Invalid session")
    
    session = monitoring_sessions[session_id]
    if not session.active:
        raise HTTPException(status_code=400, detail="Monitoring not active")
    
    try:
        # Decode base64 frame
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode frame")
        
        # Process frame
        processed_frame, alerts, status, alert_details = process_frame_complete(frame, session)
        
        # Encode back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'processed_frame': f"data:image/jpeg;base64,{processed_frame_b64}",
            'alerts': alerts,
            'status': status,
            'alert_details': alert_details,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame processing failed: {str(e)}")

@app.get("/get_alerts")
async def get_alerts(session_id: str, limit: int = 50):
    """Get all alerts for a session"""
    if session_id not in monitoring_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = monitoring_sessions[session_id]
    recent_alerts = list(session.alerts)[-limit:]
    
    return {
        'alerts': recent_alerts,
        'active': session.active,
        'total_count': len(session.alerts),
        'session_id': session_id
    }

@app.post("/room_scan")
async def room_scan(request: FrameRequest):
    """Process frame for room scan"""
    session_id = request.session_id
    frame_data = request.frame
    
    if session_id not in monitoring_sessions:
        monitoring_sessions[session_id] = MonitoringSession(session_id)
    
    session = monitoring_sessions[session_id]
    
    try:
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode frame")
        
        person_detected = False
        person_count = 0
        
        if yolo_model is not None:
            dets = detect_phone_with_yolo(frame)
            person_detected, persons = person_in_detections(dets)
            person_count = len(persons)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)
        face_detected = face_results.multi_face_landmarks is not None
        
        detected = person_detected or face_detected
        
        return {
            'person_detected': detected,
            'person_count': person_count,
            'face_detected': face_detected,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Room scan failed: {str(e)}")

@app.post("/complete_room_scan")
async def complete_room_scan(request: RoomScanComplete):
    """Mark room scan as complete"""
    session_id = request.session_id
    passed = request.passed
    
    if session_id not in monitoring_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = monitoring_sessions[session_id]
    session.room_scan_completed = True
    session.room_scan_passed = passed
    
    if passed:
        session.log_alert("ROOM_SCAN_PASSED", "Room scan completed successfully")
    else:
        session.log_alert("ROOM_SCAN_FAILED", "Room scan failed - person detected")
    
    return {
        'message': 'Room scan marked as complete',
        'passed': passed,
        'session_id': session_id
    }

@app.get("/session_summary")
async def session_summary(session_id: str):
    """Get summary of monitoring session"""
    if session_id not in monitoring_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = monitoring_sessions[session_id]
    
    alert_counts = {}
    for alert in session.alerts:
        alert_type = alert['type']
        alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
    
    return {
        'session_id': session_id,
        'total_alerts': len(session.alerts),
        'alert_breakdown': alert_counts,
        'room_scan_completed': session.room_scan_completed,
        'room_scan_passed': session.room_scan_passed,
        'active': session.active,
        'all_alerts': list(session.alerts)
    }

@app.get("/download_log")
async def download_log():
    """Download CSV log file"""
    if not os.path.exists(LOG_FILENAME):
        raise HTTPException(status_code=404, detail="Log file not found")
    
    return FileResponse(
        LOG_FILENAME,
        media_type='text/csv',
        filename=f'attention_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

@app.post("/clear_session")
async def clear_session(request: SessionRequest):
    """Clear a monitoring session"""
    session_id = request.session_id
    
    if session_id in monitoring_sessions:
        del monitoring_sessions[session_id]
        log_event("SESSION_CLEARED", f"session_id={session_id}")
        return {'message': 'Session cleared', 'session_id': session_id}
    
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/system_info")
async def system_info():
    """Get system information"""
    return {
        'yolo_model_loaded': yolo_model is not None,
        'yolo_type': 'torch_hub' if use_torch_hub else ('ultralytics' if use_yolo_ultralytics else 'none'),
        'mediapipe_available': True,
        'face_mesh_enabled': True,
        'hand_detection_enabled': True,
        'gemini_configured': GEMINI_API_KEY is not None,
        'active_monitoring_sessions': len(monitoring_sessions),
        'active_rag_sessions': len(session_store),
        'log_file': LOG_FILENAME,
        'models_directory': MODELS_DIR,
        'upload_directory': UPLOAD_FOLDER
    }

@app.post("/test_camera")
async def test_camera(request: FrameRequest):
    """Test camera endpoint"""
    frame_data = request.frame
    
    try:
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode frame")
        
        h, w, _ = frame.shape
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        face_detected = results.multi_face_landmarks is not None
        
        return {
            'success': True,
            'frame_size': f"{w}x{h}",
            'face_detected': face_detected,
            'message': 'Camera test successful'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    print("=" * 80)
    print("AI Interview Assistant - Unified Backend Server")
    print("=" * 80)
    print(f"✓ FastAPI initialized")
    print(f"✓ YOLO Model: {'Loaded' if yolo_model else 'Not available'}")
    print(f"✓ MediaPipe Face Mesh: Enabled")
    print(f"✓ MediaPipe Hands: Enabled")
    print(f"✓ Sentence Transformer: Loaded")
    print(f"✓ Gemini API: {'Configured' if GEMINI_API_KEY else 'Not configured'}")
    print(f"✓ Log file: {LOG_FILENAME}")
    print(f"✓ Upload folder: {UPLOAD_FOLDER}")
    print(f"✓ Models folder: {MODELS_DIR}")
    print("=" * 80)
    print("\nAvailable endpoints:")
    print("  POST   /upload_resume       - Upload resume for RAG")
    print("  POST   /generate_questions  - Generate interview questions")
    print("  POST   /start_monitoring    - Start attention monitoring")
    print("  POST   /stop_monitoring     - Stop attention monitoring")
    print("  POST   /process_frame       - Process video frame")
    print("  POST   /room_scan           - Check for extra persons")
    print("  POST   /complete_room_scan  - Mark room scan complete")
    print("  GET    /get_alerts          - Get session alerts")
    print("  GET    /session_summary     - Get session summary")
    print("  GET    /download_log        - Download CSV log")
    print("  GET    /health              - Health check")
    print("  GET    /system_info         - System information")
    print("  POST   /test_camera         - Test camera connection")
    print("  POST   /clear_session       - Clear monitoring session")
    print("=" * 80)
    
    log_event("SERVER_STARTED", "Unified backend server initialized")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)