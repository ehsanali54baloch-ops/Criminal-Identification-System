#!/usr/bin/env python3
"""
==============================================================================
AI-Based Face Detection & Criminal Identification System
Database Initialization & Mock Data Seeder
Team: Ehsan Ali (Leader), Asif Ali, Sabir Ali
==============================================================================
This script initializes the local SQLite database ('criminals.db') and seeds
it with realistic mock criminal records, including face embeddings, threat
levels, crime details, and mugshot images for Codespaces testing.
"""

import os
import sqlite3
import json
import numpy as np
import cv2

# Configuration
DB_PATH = "criminals.db"
MUGSHOTS_DIR = "mock_mugshots"

def create_mock_face_image(filename: str, name: str, shirt_color: tuple, bg_color: tuple):
    """
    Generates a realistic synthetic avatar mugshot image for testing purposes
    if real sample photos are not yet supplied.
    """
    os.makedirs(MUGSHOTS_DIR, exist_ok=True)
    filepath = os.path.join(MUGSHOTS_DIR, filename)
    
    # 400x400 RGB canvas
    img = np.full((400, 400, 3), bg_color, dtype=np.uint8)
    
    # Background subtle grid pattern
    for y in range(0, 400, 20):
        cv2.line(img, (0, y), (400, y), (bg_color[0]-15, bg_color[1]-15, bg_color[2]-15), 1)
    
    # Height chart lines in background
    cv2.putText(img, "6'0\"", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    cv2.putText(img, "5'8\"", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    cv2.putText(img, "5'4\"", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    cv2.line(img, (40, 80), (390, 80), (100, 100, 100), 1)
    cv2.line(img, (40, 140), (390, 140), (100, 100, 100), 1)
    cv2.line(img, (40, 200), (390, 200), (100, 100, 100), 1)

    # Torso / Clothes
    cv2.ellipse(img, (200, 380), (130, 90), 0, 0, 180, shirt_color, -1)
    cv2.ellipse(img, (200, 380), (130, 90), 0, 0, 180, (40, 40, 40), 2)
    
    # Neck
    neck_color = (180, 205, 235)  # Light skin tone in BGR
    cv2.rectangle(img, (180, 240), (220, 290), neck_color, -1)
    
    # Head / Face oval
    cv2.ellipse(img, (200, 190), (70, 90), 0, 0, 360, neck_color, -1)
    cv2.ellipse(img, (200, 190), (70, 90), 0, 0, 360, (60, 60, 60), 2)
    
    # Hair
    hair_color = (30, 30, 40)
    cv2.ellipse(img, (200, 135), (75, 45), 0, 180, 360, hair_color, -1)
    
    # Eyes
    cv2.circle(img, (175, 180), 7, (255, 255, 255), -1)
    cv2.circle(img, (225, 180), 7, (255, 255, 255), -1)
    cv2.circle(img, (175, 180), 4, (60, 30, 10), -1)
    cv2.circle(img, (225, 180), 4, (60, 30, 10), -1)
    
    # Eyebrows
    cv2.line(img, (165, 168), (188, 168), hair_color, 3)
    cv2.line(img, (212, 168), (235, 168), hair_color, 3)
    
    # Nose
    cv2.line(img, (200, 182), (195, 202), (150, 175, 205), 2)
    cv2.line(img, (195, 202), (205, 202), (150, 175, 205), 2)
    
    # Mouth
    cv2.line(img, (185, 225), (215, 225), (100, 110, 170), 3)
    
    # Suspect ID Placard
    cv2.rectangle(img, (70, 330), (330, 385), (20, 20, 20), -1)
    cv2.rectangle(img, (70, 330), (330, 385), (200, 200, 200), 2)
    cv2.putText(img, f"POLICE ID: {name.upper()}", (85, 362), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite(filepath, img)
    return filepath

def generate_mock_embedding(seed: int, dims: int = 128) -> list:
    """
    Generates a deterministic 128-dimensional normalized unit vector
    representing facial feature embedding for offline/fallback matching.
    """
    rng = np.random.RandomState(seed)
    raw_vec = rng.randn(dims)
    norm_vec = raw_vec / np.linalg.norm(raw_vec)
    return norm_vec.tolist()

def init_database():
    """
    Creates SQLite tables and populates sample criminal records.
    """
    print("================================================================")
    print("AI Face Detection & Criminal Identification System")
    print("Initializing Database: 'criminals.db'...")
    print("================================================================")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table 1: Criminals / Suspects Registry
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS criminals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        suspect_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        alias TEXT,
        age INTEGER,
        gender TEXT,
        crime_category TEXT NOT NULL,
        threat_level TEXT NOT NULL, -- HIGH, MEDIUM, LOW
        bounty TEXT,
        status TEXT NOT NULL,       -- WANTED, IN_CUSTODY, SURVEILLANCE
        last_seen TEXT,
        image_path TEXT,
        face_embedding TEXT,       -- JSON list of 512-d FaceNet512 embeddings (one per photo)
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Table 2: Real-time Incident & Alert Logs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incident_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        suspect_id TEXT,
        suspect_name TEXT,
        confidence_score REAL NOT NULL,
        mask_detected INTEGER DEFAULT 0, -- 1=True, 0=False
        location TEXT DEFAULT 'Terminal Gate 4B Surveillance',
        threat_level TEXT NOT NULL,
        snapshot_path TEXT,
        email_status TEXT,         -- SENT / FAILED / NOT CONFIGURED ...
        alert_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Seed Mock Criminal Records
    mock_criminals = [
        {
            "suspect_id": "CRIM-8921",
            "name": "Marcus 'Ghost' Vance",
            "alias": "The Shadow Architect",
            "age": 38,
            "gender": "Male",
            "crime_category": "Grand Armed Robbery & Syndicate Leader",
            "threat_level": "HIGH",
            "bounty": "$150,000",
            "status": "WANTED",
            "last_seen": "Metropolitan Central Depot",
            "image_file": "marcus_vance.jpg",
            "shirt": (40, 60, 160),     # Red shirt
            "bg": (220, 220, 230),
            "seed": 101
        },
        {
            "suspect_id": "CRIM-4103",
            "name": "Elena Rostova",
            "alias": "Cipher Zero",
            "age": 29,
            "gender": "Female",
            "crime_category": "Critical Infrastructure Cyber Espionage",
            "threat_level": "HIGH",
            "bounty": "$200,000",
            "status": "WANTED",
            "last_seen": "Financial District Server Hub",
            "image_file": "elena_rostova.jpg",
            "shirt": (160, 40, 40),     # Navy blue shirt
            "bg": (210, 230, 220),
            "seed": 202
        },
        {
            "suspect_id": "CRIM-7734",
            "name": "Viktor Stone",
            "alias": "The Chameleon",
            "age": 45,
            "gender": "Male",
            "crime_category": "Multi-National Identity & Securities Fraud",
            "threat_level": "MEDIUM",
            "bounty": "$50,000",
            "status": "WANTED",
            "last_seen": "International Air Transit Hub",
            "image_file": "viktor_stone.jpg",
            "shirt": (50, 120, 50),     # Dark Green shirt
            "bg": (230, 220, 210),
            "seed": 303
        },
        {
            "suspect_id": "CRIM-5519",
            "name": "Damon 'Razor' Cross",
            "alias": "Razor",
            "age": 34,
            "gender": "Male",
            "crime_category": "High-Value Cargo Hijacking & Contraband",
            "threat_level": "HIGH",
            "bounty": "$85,000",
            "status": "SURVEILLANCE",
            "last_seen": "South Cargo Terminal Dock 12",
            "image_file": "damon_cross.jpg",
            "shirt": (20, 20, 20),      # Black shirt
            "bg": (220, 215, 235),
            "seed": 404
        }
    ]
    
    print("\n[+] Seeding criminal profiles & generating mock mugshots...")
    for crim in mock_criminals:
        img_path = create_mock_face_image(
            crim["image_file"],
            crim["name"],
            crim["shirt"],
            crim["bg"]
        )
        
        # The mugshots are cartoon drawings, not real faces, so they get NO
        # embedding. (The old version stored random numbers here, which can
        # never match a real person.) Enroll real photos from the dashboard.
        embedding_json = None
        
        cursor.execute("""
        INSERT OR REPLACE INTO criminals (
            suspect_id, name, alias, age, gender, crime_category,
            threat_level, bounty, status, last_seen, image_path, face_embedding
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            crim["suspect_id"],
            crim["name"],
            crim["alias"],
            crim["age"],
            crim["gender"],
            crim["crime_category"],
            crim["threat_level"],
            crim["bounty"],
            crim["status"],
            crim["last_seen"],
            img_path,
            embedding_json
        ))
        print(f"  -> Inserted Suspect: [{crim['suspect_id']}] {crim['name']} (Threat: {crim['threat_level']})")
    
    # Insert initial demo incident log
    cursor.execute("""
    INSERT INTO incident_alerts (
        suspect_id, suspect_name, confidence_score, mask_detected,
        location, threat_level, snapshot_path
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        "CRIM-8921",
        "Marcus 'Ghost' Vance",
        94.6,
        0,
        "North Sector Camera #04 - Gate B",
        "HIGH",
        os.path.join(MUGSHOTS_DIR, "marcus_vance.jpg")
    ))
    
    conn.commit()
    conn.close()
    
    print("\n[SUCCESS] SQLite Database initialized successfully: 'criminals.db'")
    print("[SUCCESS] All sample criminal mugshots generated in folder: 'mock_mugshots/'")
    print("[NOTE] Sample suspects have no face data. To test recognition, open the")
    print("       'Criminal Database Management' tab and enroll a REAL photo")
    print("       (e.g. your own face) - then stand in front of the webcam.")
    print("================================================================")

if __name__ == "__main__":
    init_database()
