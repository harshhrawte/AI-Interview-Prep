"""
Monitoring-related API routes
"""

import base64
import numpy as np
import cv2
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

# ✅ FIXED: schemas import
from backend.schemas import FrameRequest, SessionRequest, RoomScanComplete

# ✅ FIXED: services imports
from backend.services.monitoring_service import (
    get_or_create_session,
    process_frame_complete,
    monitoring_sessions,
)

from backend.services.detection_service import (
    detect_phone_with_yolo,
    person_in_detections,
    phone_in_detections,
    get_face_mesh,
    is_yolo_available,
    get_yolo_info,
)

# ✅ FIXED: config + utils imports
from backend.config import LOG_FILENAME
from backend.utils.logger import log_event


router = APIRouter(tags=["monitoring"])


@router.post("/start_monitoring")
async def start_monitoring_endpoint(request: SessionRequest):
    """Start attention monitoring"""
    session_id = request.session_id
    session = get_or_create_session(session_id)
    session.active = True
    session.log_alert("MONITORING_STARTED", f"session_id={session_id}")

    return {
        "message": "Monitoring started",
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/stop_monitoring")
async def stop_monitoring_endpoint(request: SessionRequest):
    """Stop attention monitoring"""
    session_id = request.session_id

    if session_id not in monitoring_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = monitoring_sessions[session_id]
    session.active = False
    session.log_alert("MONITORING_STOPPED", "User stopped monitoring")

    return {
        "message": "Monitoring stopped",
        "session_id": session_id,
        "total_alerts": len(session.alerts),
    }


@router.post("/process_frame")
async def process_frame_endpoint(request: FrameRequest):
    """Process video frame for monitoring"""
    session_id = request.session_id
    frame_data = request.frame

    if session_id not in monitoring_sessions:
        raise HTTPException(status_code=400, detail="Invalid session")

    session = monitoring_sessions[session_id]
    if not session.active:
        raise HTTPException(status_code=400, detail="Monitoring not active")

    try:
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]

        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode frame")

        processed_frame, alerts, status, alert_details = process_frame_complete(
            frame, session
        )

        _, buffer = cv2.imencode(
            ".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        processed_frame_b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "processed_frame": f"data:image/jpeg;base64,{processed_frame_b64}",
            "alerts": alerts,
            "status": status,
            "alert_details": alert_details,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Frame processing failed: {str(e)}"
        )


@router.get("/get_alerts")
async def get_alerts_endpoint(session_id: str, limit: int = 50):
    """Get all alerts for a session"""
    if session_id not in monitoring_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = monitoring_sessions[session_id]
    recent_alerts = list(session.alerts)[-limit:]

    return {
        "alerts": recent_alerts,
        "active": session.active,
        "total_count": len(session.alerts),
        "session_id": session_id,
    }


@router.post("/room_scan")
async def room_scan_endpoint(request: FrameRequest):
    """Process frame for room scan"""
    session_id = request.session_id
    frame_data = request.frame

    session = get_or_create_session(session_id)

    try:
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]

        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode frame")

        person_detected = False
        person_count = 0
        phone_detected = False

        if is_yolo_available():
            dets = detect_phone_with_yolo(frame)
            person_detected, persons = person_in_detections(dets)
            person_count = len(persons)
            phone_detected, _ = phone_in_detections(dets)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_mesh = get_face_mesh()
        face_results = face_mesh.process(rgb)
        face_detected = face_results.multi_face_landmarks is not None

        detected = person_detected or face_detected

        return {
            "person_detected": detected,
            "person_count": person_count,
            "face_detected": face_detected,
            "phone_detected": phone_detected,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Room scan failed: {str(e)}"
        )


@router.post("/complete_room_scan")
async def complete_room_scan_endpoint(request: RoomScanComplete):
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
        "message": "Room scan marked as complete",
        "passed": passed,
        "session_id": session_id,
    }


@router.get("/session_summary")
async def session_summary_endpoint(session_id: str):
    """Get summary of monitoring session"""
    if session_id not in monitoring_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = monitoring_sessions[session_id]

    alert_counts = {}
    for alert in session.alerts:
        alert_type = alert["type"]
        alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1

    return {
        "session_id": session_id,
        "total_alerts": len(session.alerts),
        "alert_breakdown": alert_counts,
        "room_scan_completed": session.room_scan_completed,
        "room_scan_passed": session.room_scan_passed,
        "active": session.active,
        "all_alerts": list(session.alerts),
    }


@router.get("/download_log")
async def download_log_endpoint():
    """Download CSV log file"""
    import os

    if not os.path.exists(LOG_FILENAME):
        raise HTTPException(status_code=404, detail="Log file not found")

    return FileResponse(
        LOG_FILENAME,
        media_type="text/csv",
        filename=f"attention_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )


@router.post("/clear_session")
async def clear_session_endpoint(request: SessionRequest):
    """Clear a monitoring session"""
    session_id = request.session_id

    if session_id in monitoring_sessions:
        del monitoring_sessions[session_id]
        log_event("SESSION_CLEARED", f"session_id={session_id}")
        return {"message": "Session cleared", "session_id": session_id}

    raise HTTPException(status_code=404, detail="Session not found")


@router.post("/test_camera")
async def test_camera_endpoint(request: FrameRequest):
    """Test camera endpoint"""
    frame_data = request.frame

    try:
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]

        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode frame")

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_mesh = get_face_mesh()
        results = face_mesh.process(rgb)
        face_detected = results.multi_face_landmarks is not None

        return {
            "success": True,
            "frame_size": f"{w}x{h}",
            "face_detected": face_detected,
            "message": "Camera test successful",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")
