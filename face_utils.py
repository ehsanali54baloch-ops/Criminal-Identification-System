#!/usr/bin/env python3
"""
==============================================================================
AI-Based Face Detection & Criminal Identification System
Core Computer Vision & Recognition Utilities (face_utils.py)
Team: Ehsan Ali (Leader), Asif Ali, Sabir Ali
==============================================================================
5-stage pipeline:
  1. Capture  - frame arrives from the browser webcam (Gradio)
  2. Detect   - DeepFace face detector finds and aligns every face
  3. Extract  - FaceNet512 turns each aligned face into a 512-number embedding
  4. Compare  - cosine similarity against every embedding stored in SQLite
  5. Alert    - HUD overlay, alarm sound (in the browser), snapshot saved,
                incident log, email with snapshot sent to the admin

There is NO fake/fallback embedding any more. If the FaceNet model cannot
load, the app says so clearly instead of silently producing random matches.
"""

import os
import cv2
import json
import time
import sqlite3
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional, Any

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        RED = GREEN = YELLOW = CYAN = MAGENTA = WHITE = ""
    class Style:
        BRIGHT = RESET_ALL = ""

import email_service

DB_PATH = "criminals.db"
SNAPSHOT_DIR = "snapshots"

# ---------------------------------------------------------------------------
# Model settings (can be changed with environment variables)
# ---------------------------------------------------------------------------
MODEL_NAME = "Facenet512"
EMBEDDING_DIM = 512
# Live camera: needs to be fast. "opencv" = Haar detector + eye alignment.
# Try "yunet" or "mtcnn" for better accuracy if your machine is fast enough.
LIVE_DETECTOR = os.environ.get("LIVE_DETECTOR", "opencv")
# Enrollment: speed does not matter, accuracy does.
ENROLL_DETECTOR = os.environ.get("ENROLL_DETECTOR", "retinaface")
# Faces smaller than this (pixels) are too blurry to recognise reliably.
MIN_FACE_SIZE = 60
# Same suspect will not raise a new alarm / log entry within this many seconds.
ALERT_COOLDOWN_SEC = int(os.environ.get("ALERT_COOLDOWN_SEC", "15"))
# Same suspect will not be emailed again within this many seconds (avoid spam).
EMAIL_COOLDOWN_SEC = int(os.environ.get("EMAIL_COOLDOWN_SEC", "60"))


def _ensure_schema():
    """Add the email_status column to databases created by older versions."""
    if not os.path.exists(DB_PATH):
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(incident_alerts)")]
        if cols and "email_status" not in cols:
            conn.execute("ALTER TABLE incident_alerts ADD COLUMN email_status TEXT")
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Schema Check Error]: {e}")

# ---------------------------------------------------------------------------
# Load DeepFace + FaceNet512 once at start-up
# ---------------------------------------------------------------------------
DEEPFACE_AVAILABLE = False
MODEL_ERROR = ""
try:
    from deepface import DeepFace
    DeepFace.build_model(MODEL_NAME)          # downloads weights on first run
    DEEPFACE_AVAILABLE = True
    print(f"{Fore.GREEN}[INFO] FaceNet512 model loaded. Real face recognition is ACTIVE.{Style.RESET_ALL}")
except Exception as e:
    MODEL_ERROR = str(e)
    print(f"{Fore.RED}{Style.BRIGHT}[ERROR] Face recognition model could NOT be loaded:\n  {e}\n"
          f"  Fix: pip install -r requirements.txt   (make sure tf-keras is installed){Style.RESET_ALL}")


class FaceUtils:
    """
    Main Computer Vision & Biometric Matching Engine.
    """

    _match_history = deque(maxlen=5)
    _REQUIRED_CONSISTENT_FRAMES = 3
    _last_alert_time: Dict[str, float] = {}
    _last_email_time: Dict[str, float] = {}
    _db_cache: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Stage 2 + 3: detect, align and embed every face in one call
    # ------------------------------------------------------------------
    @staticmethod
    def _run_deepface(frame: np.ndarray, detector: str) -> List[Dict[str, Any]]:
        results = DeepFace.represent(
            img_path=frame,
            model_name=MODEL_NAME,
            detector_backend=detector,
            enforce_detection=False,
            align=True,
        )
        faces = []
        for r in results:
            area = r["facial_area"]
            w, h = int(area["w"]), int(area["h"])
            # enforce_detection=False returns the WHOLE image with confidence 0
            # when no face is found - ignore that.
            if r.get("face_confidence", 0) <= 0:
                continue
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue
            vec = np.array(r["embedding"], dtype=np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-7)
            faces.append({"bbox": (int(area["x"]), int(area["y"]), w, h), "embedding": vec})
        return faces

    @classmethod
    def detect_and_embed(cls, frame: np.ndarray, detector: str = LIVE_DETECTOR) -> List[Dict[str, Any]]:
        """
        Returns a list of {"bbox": (x, y, w, h), "embedding": 512-d unit vector}.
        Falls back to the "opencv" detector if the chosen one fails to load.
        """
        if frame is None or frame.size == 0 or not DEEPFACE_AVAILABLE:
            return []
        try:
            return cls._run_deepface(frame, detector)
        except Exception as e:
            if detector != "opencv":
                print(f"{Fore.YELLOW}[WARNING] Detector '{detector}' failed ({e}). Using 'opencv'.{Style.RESET_ALL}")
                try:
                    return cls._run_deepface(frame, "opencv")
                except Exception as e2:
                    print(f"[Detection Error]: {e2}")
            else:
                print(f"[Detection Error]: {e}")
            return []

    @classmethod
    def enrollment_embedding(cls, image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """
        Embedding for a suspect photo. Requires exactly one clear face.
        Returns (embedding or None, message).
        """
        if not DEEPFACE_AVAILABLE:
            return None, f"Face recognition model is not loaded: {MODEL_ERROR}"
        faces = cls.detect_and_embed(image_bgr, detector=ENROLL_DETECTOR)
        if not faces:
            return None, "No clear face found. Use a sharp, front-facing, well-lit photo."
        if len(faces) > 1:
            # keep the largest face (usually the subject)
            faces.sort(key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)
        return faces[0]["embedding"], "ok"

    # ------------------------------------------------------------------
    # Stage 4: compare against database
    # ------------------------------------------------------------------
    @classmethod
    def invalidate_cache(cls):
        """Call after adding/deleting suspects."""
        cls._db_cache = None

    @classmethod
    def _load_suspects(cls) -> List[Dict[str, Any]]:
        if cls._db_cache is not None:
            return cls._db_cache
        suspects = []
        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            rows = conn.execute("""
                SELECT suspect_id, name, alias, age, gender, crime_category,
                       threat_level, bounty, status, last_seen, image_path, face_embedding
                FROM criminals
            """).fetchall()
            conn.close()
            for row in rows:
                (sid, name, alias, age, gender, crime, threat,
                 bounty, status, last_seen, img_path, emb_json) = row
                if not emb_json:
                    continue
                try:
                    data = json.loads(emb_json)
                except Exception:
                    continue
                # A record may hold one embedding [..] or several [[..],[..]]
                vectors = data if (data and isinstance(data[0], list)) else [data]
                vectors = [np.array(v, dtype=np.float32) for v in vectors if len(v) == EMBEDDING_DIM]
                if not vectors:
                    # old 128-d / random embeddings from the previous version
                    print(f"{Fore.YELLOW}[WARNING] {sid} ({name}) has no valid 512-d embedding. "
                          f"Re-enroll this suspect with a real photo.{Style.RESET_ALL}")
                    continue
                vectors = [v / (np.linalg.norm(v) + 1e-7) for v in vectors]
                suspects.append({
                    "suspect_id": sid, "name": name, "alias": alias, "age": age,
                    "gender": gender, "crime_category": crime, "threat_level": threat,
                    "bounty": bounty, "status": status, "last_seen": last_seen,
                    "image_path": img_path, "vectors": np.stack(vectors),
                })
        cls._db_cache = suspects
        return suspects

    @classmethod
    def enrolled_count(cls) -> int:
        return len(cls._load_suspects())

    @classmethod
    def compare_with_database(cls, probe_embedding: np.ndarray, threshold: float = 0.70) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Cosine similarity between the probe and every stored photo of every
        suspect. A suspect's score is the best score over their photos.
        Returns (match or None, best similarity seen).
        For FaceNet512, same person is usually above ~0.70.
        """
        best, best_sim = None, -1.0
        for s in cls._load_suspects():
            sim = float(np.max(s["vectors"] @ probe_embedding))
            if sim > best_sim:
                best_sim, best = sim, s
        if best is not None and best_sim >= threshold:
            match = {k: v for k, v in best.items() if k != "vectors"}
            match["similarity"] = best_sim
            match["confidence_score"] = round(best_sim * 100.0, 1)
            return match, best_sim
        return None, best_sim

    # ------------------------------------------------------------------
    # Full pipeline for one frame
    # ------------------------------------------------------------------
    @classmethod
    def process_frame(cls, frame: np.ndarray, similarity_threshold: float = 0.70,
                      enable_mask_detection: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if frame is None:
            return frame, []

        processed_frame = frame.copy()
        if not DEEPFACE_AVAILABLE:
            cv2.putText(processed_frame, "AI MODEL NOT LOADED - see terminal", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return processed_frame, []

        faces = cls.detect_and_embed(processed_frame)
        detections = []
        frame_ids = []

        for face in faces:
            x, y, w, h = face["bbox"]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(processed_frame.shape[1], x + w), min(processed_frame.shape[0], y + h)
            face_roi = frame[y1:y2, x1:x2]

            mask_info = {"is_masked": False, "status": "NO MASK", "confidence": 0.0}
            if enable_mask_detection:
                mask_info = cls.detect_mask(face_roi)

            raw_match, best_sim = cls.compare_with_database(face["embedding"], threshold=similarity_threshold)
            raw_id = raw_match["suspect_id"] if raw_match else None
            if raw_id:
                frame_ids.append(raw_id)

            # Confirm a match only if seen in 3 of the last 5 frames
            confirmed = raw_id is not None and (
                sum(raw_id in ids for ids in cls._match_history) + 1 >= cls._REQUIRED_CONSISTENT_FRAMES
            )
            match_result = raw_match if confirmed else None
            is_suspect = match_result is not None
            is_verifying = raw_id is not None and not is_suspect

            cls.draw_hud_overlay(processed_frame, (x, y, w, h), is_suspect=is_suspect,
                                 suspect_info=match_result, mask_info=mask_info,
                                 is_verifying=is_verifying)

            detections.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "is_match": is_suspect,
                "mask_info": mask_info,
                "suspect": match_result,
                "best_similarity": round(best_sim * 100.0, 1) if best_sim >= 0 else None,
                "new_alert": False,
            })

        # Stage 5: alerts - after ALL boxes are drawn, so the snapshot shows everything
        for d in detections:
            if d["is_match"]:
                d["new_alert"] = cls.trigger_alert(d["suspect"], d["mask_info"], processed_frame)

        # one history entry per frame (works with several faces at once)
        cls._match_history.append(set(frame_ids))
        return processed_frame, detections

    @staticmethod
    def detect_mask(face_roi: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes the lower region of a detected face (mouth & nose area)
        to verify if the individual is wearing a protective face mask.
        Uses a combination of HSV color analysis and edge density variance.
        """
        if face_roi is None or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
            return {"is_masked": False, "confidence": 0.0, "status": "UNKNOWN"}

        h, w, _ = face_roi.shape
        # Lower face region (approx 50% to 95% of height, 20% to 80% of width)
        lower_face = face_roi[int(h * 0.48):int(h * 0.95), int(w * 0.15):int(w * 0.85)]
        
        if lower_face.size == 0:
            return {"is_masked": False, "confidence": 0.0, "status": "NO_MASK"}

        hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
        
        # Check standard mask color profiles (Surgical Blue, White/N95, Dark/Black)
        # 1. Blue Mask HSV range
        blue_mask = cv2.inRange(hsv, np.array([85, 50, 50]), np.array([130, 255, 255]))
        # 2. White/Light Mask HSV range
        white_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 45, 255]))
        # 3. Dark/Black Mask HSV range
        black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))

        combined_mask = cv2.bitwise_or(blue_mask, cv2.bitwise_or(white_mask, black_mask))
        mask_ratio = cv2.countNonZero(combined_mask) / (lower_face.shape[0] * lower_face.shape[1])
        
        # Edge density analysis on lower face (Skin has pores/lips with distinct edges vs smooth fabric)
        gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_lower, 50, 150)
        edge_ratio = cv2.countNonZero(edges) / (lower_face.shape[0] * lower_face.shape[1])

        # If high coverage of mask colors and low natural skin facial feature edges
        is_masked = mask_ratio > 0.45 or (mask_ratio > 0.35 and edge_ratio < 0.08)
        confidence = min(round(float(mask_ratio * 100 + (1.0 - edge_ratio) * 20), 1), 99.0)

        return {
            "is_masked": bool(is_masked),
            "confidence": float(confidence),
            "status": "MASK DETECTED" if is_masked else "NO MASK"
        }


    @staticmethod
    def draw_hud_overlay(img: np.ndarray, bbox: Tuple[int, int, int, int], is_suspect: bool, suspect_info: Optional[Dict[str, Any]], mask_info: Dict[str, Any], is_verifying: bool = False):  
        """
        Renders a high-tech surveillance HUD with tactical corner brackets,
        threat status badge, and biometric confidence readout.
        """
        x, y, w, h = bbox
        
        # Color coding: Red for HIGH threat match, Orange for MEDIUM, Neon Green for UNKNOWN/CLEAR
        if is_suspect:
          threat = suspect_info.get("threat_level", "HIGH")
          color = (0, 0, 235) if threat == "HIGH" else (0, 140, 255) # BGR
          tag_text = f"CRIMINAL MATCH: {suspect_info['name']} ({suspect_info['confidence_score']}%)"
        elif is_verifying:
          color = (0, 200, 255)  # Amber BGR
          tag_text = "VERIFYING MATCH..."
        else:
          color = (0, 230, 80) # Neon Green BGR
          tag_text = "SUBJECT: NO CRIMINAL RECORD"
 

        # Corner bracket HUD graphics
        corner_len = int(min(w, h) * 0.25)
        thickness = 2
        corner_thick = 4

        # Thin bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

        # Top-Left Corner
        cv2.line(img, (x, y), (x + corner_len, y), color, corner_thick)
        cv2.line(img, (x, y), (x, y + corner_len), color, corner_thick)

        # Top-Right Corner
        cv2.line(img, (x + w, y), (x + w - corner_len, y), color, corner_thick)
        cv2.line(img, (x + w, y), (x + w, y + corner_len), color, corner_thick)

        # Bottom-Left Corner
        cv2.line(img, (x, y + h), (x + corner_len, y + h), color, corner_thick)
        cv2.line(img, (x, y + h), (x, y + h - corner_len), color, corner_thick)

        # Bottom-Right Corner
        cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color, corner_thick)
        cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color, corner_thick)

        # Header Badge Background
        header_y = max(25, y - 10)
        (text_w, text_h), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        cv2.rectangle(img, (x, header_y - text_h - 6), (x + text_w + 12, header_y + 4), (15, 15, 15), -1)
        cv2.rectangle(img, (x, header_y - text_h - 6), (x + text_w + 12, header_y + 4), color, 1)
        cv2.putText(img, tag_text, (x + 6, header_y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1)

        # Mask Verification Badge at Bottom
        mask_text = f"MASK: {mask_info['status']}"
        mask_color = (0, 200, 255) if mask_info["is_masked"] else (180, 180, 180)
        footer_y = y + h + 20
        cv2.putText(img, mask_text, (x + 2, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mask_color, 1)


    @classmethod
    def trigger_alert(cls, suspect_info: Dict[str, Any], mask_info: Dict[str, Any],
                      annotated_frame: Optional[np.ndarray] = None) -> bool:
        """
        1. Console alert   2. Save snapshot JPG   3. Log incident in SQLite
        4. Email the snapshot to the admin (background thread)
        Returns True if this is a NEW alert (so the browser plays the alarm).
        Each suspect alerts at most once per ALERT_COOLDOWN_SEC.
        """
        now = time.time()
        sid = suspect_info["suspect_id"]
        if now - cls._last_alert_time.get(sid, 0) < ALERT_COOLDOWN_SEC:
            return False
        cls._last_alert_time[sid] = now

        print(f"\n{Fore.RED}{Style.BRIGHT}========================================================")
        print("[🚨 SECURITY BREACH ALERT] CRIMINAL IDENTIFIED IN SURVEILLANCE FEED!")
        print("========================================================")
        print(f"  Name:         {suspect_info['name']} (ID: {sid})")
        print(f"  Crime:        {suspect_info['crime_category']}")
        print(f"  Threat Level: {suspect_info['threat_level']}")
        print(f"  Similarity:   {suspect_info['confidence_score']}%")
        print(f"  Mask Status:  {mask_info['status']}")
        print(f"========================================================{Style.RESET_ALL}\n")

        # Save snapshot of the annotated webcam frame
        snapshot_path = ""
        if annotated_frame is not None:
            try:
                os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                snapshot_path = os.path.join(SNAPSHOT_DIR, f"{sid}_{stamp}.jpg")
                cv2.imwrite(snapshot_path, annotated_frame)
            except Exception as e:
                print(f"[Snapshot Error]: {e}")
                snapshot_path = ""

        # Log incident
        alert_id = None
        try:
            _ensure_schema()
            conn = sqlite3.connect(DB_PATH)
            cur = conn.execute("""
                INSERT INTO incident_alerts (
                    suspect_id, suspect_name, confidence_score, mask_detected,
                    location, threat_level, snapshot_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                sid, suspect_info["name"], suspect_info["confidence_score"],
                1 if mask_info["is_masked"] else 0, "Webcam Surveillance Stream 1",
                suspect_info["threat_level"], snapshot_path,
            ))
            alert_id = cur.lastrowid
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Alert DB Log Error]: {e}")

        # Email the admin (with its own, longer cooldown to avoid spamming)
        if alert_id is not None:
            if now - cls._last_email_time.get(sid, 0) >= EMAIL_COOLDOWN_SEC:
                cls._last_email_time[sid] = now
                email_service.send_alert_async(alert_id, suspect_info, mask_info, snapshot_path)
            else:
                email_service._set_email_status(alert_id, "SKIPPED (already emailed recently)")
        return True


_ensure_schema()
