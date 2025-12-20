"""
Project: Autonomous Warehouse Perception System
Author: Alfayez Ahmad
Copyright: (c) 2025 Alfayez Ahmad
License: MIT
Description: Hybrid pipeline integrating Meta SAM 2 (Vision) with OpenCV (Logic) 
             and SQLite (Memory) to detect object fragmentation events.
"""

import os
import sys
import cv2
import numpy as np
import sqlite3
import pandas as pd
from datetime import datetime
from PIL import Image
import torch

# Try to import SAM 2 
try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("[ERROR] SAM 2 not found. Please install via: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    sys.exit(1)

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "project_dir": "warehouse_data",
    "video_name": "synthetic_split.mp4",
    "mask_name": "mask.png",
    "db_name": "robot_memory.db",
    "model_weight": "sam2_hiera_large.pt",
    "model_cfg": "sam2_hiera_l.yaml",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# MODULE 1: DATA GENERATION
# ==========================================
def generate_synthetic_data(base_dir):
    """
    Generates a synthetic video of two objects splitting to benchmark the logic.
    Uses the 'Peanut Prompt' strategy to force SAM 2 to track both parts.
    """
    print("[INFO] Generating synthetic data...")
    os.makedirs(base_dir, exist_ok=True)
    
    width, height = 640, 480
    video_path = os.path.join(base_dir, CONFIG["video_name"])
    mask_path = os.path.join(base_dir, CONFIG["mask_name"])
    
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    
    # Animation Loop
    for i in range(100):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Logic: Objects start merged, then separate
        split_dist = 0
        if i > 30:
            split_dist = (i - 30) * 4
            
        # Draw two circles
        cv2.circle(frame, (300 - split_dist, 240), 40, (255, 255, 255), -1)
        cv2.circle(frame, (300 + split_dist, 240), 40, (255, 255, 255), -1)
        
        # Add timestamp simulation
        cv2.putText(frame, f"Frame: {i}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        out.write(frame)
        
    out.release()
    
    # Generate Prompt Mask (The "Peanut" Strategy)
    # We cover both object positions at Frame 0 to establish single-entity tracking
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (300, 240), 40, 255, -1)
    cv2.circle(mask, (300, 240), 45, 255, -1) # Slight overlap for safety
    Image.fromarray(mask).save(mask_path)
    
    # Extract Frames for SAM 2 ingestion
    frame_dir = os.path.join(base_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    cnt = 0
    while True:
        ret, f = cap.read()
        if not ret: break
        cv2.imwrite(f"{frame_dir}/{cnt:05d}.jpg", f)
        cnt += 1
    cap.release()
    
    print(f"[INFO] Data ready in {base_dir}")
    return frame_dir

# ==========================================
# MODULE 2: DATABASE BACKEND
# ==========================================
def init_database():
    """Initializes the SQLite schema for incident logging."""
    conn = sqlite3.connect(CONFIG["db_name"])
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS event_logs')
    cursor.execute('''
        CREATE TABLE event_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            frame_idx INTEGER,
            status TEXT,
            part_count INTEGER,
            risk_score FLOAT
        )
    ''')
    conn.commit()
    return conn

def log_event(conn, frame_idx, status, parts):
    """Inserts a single frame's telemetry into the DB."""
    risk = 0.1 if status == "Safe" else 0.95
    now = datetime.now().strftime("%H:%M:%S.%f")
    conn.execute('''
        INSERT INTO event_logs (timestamp, frame_idx, status, part_count, risk_score)
        VALUES (?, ?, ?, ?, ?)
    ''', (now, frame_idx, status, parts, risk))

# ==========================================
# MODULE 3: MAIN PIPELINE
# ==========================================
def run_pipeline():
    # 1. Setup
    frame_dir = generate_synthetic_data(CONFIG["project_dir"])
    conn = init_database()
    
    # 2. Check for Weights
    if not os.path.exists(CONFIG["model_weight"]):
        print(f"[INFO] Downloading weights to {CONFIG['model_weight']}...")
        os.system(f"wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/{CONFIG['model_weight']}")

    # 3. Load SAM 2
    print(f"[INFO] Loading Model on {CONFIG['device']}...")
    predictor = build_sam2_video_predictor(CONFIG["model_cfg"], CONFIG["model_weight"], device=CONFIG["device"])
    inference_state = predictor.init_state(video_path=frame_dir)

    # 4. Apply Prompt
    mask_path = os.path.join(CONFIG["project_dir"], CONFIG["mask_name"])
    mask = np.array(Image.open(mask_path).convert('L')) > 0
    predictor.add_new_mask(inference_state, 0, 1, mask.astype(np.float32))

    # 5. Processing Loop
    print("[INFO] Starting tracking pipeline...")
    output_path = "output_result.mp4"
    frames = sorted(os.listdir(frame_dir))
    h, w = cv2.imread(f"{frame_dir}/{frames[0]}").shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    for idx, ids, logits in predictor.propagate_in_video(inference_state):
        frame = cv2.imread(f"{frame_dir}/{frames[idx]}")
        
        status = "Safe"
        parts = 1
        
        if len(logits) > 0:
            # --- Vision Processing Layer ---
            mask_pred = (logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8)
            
            # --- Logic Layer (Topology Analysis) ---
            num_labels, labels = cv2.connectedComponents(mask_pred)
            # Filter noise < 50 pixels to avoid false positives
            parts = sum(1 for i in range(1, num_labels) if np.sum(labels == i) > 50)
            
            if parts > 1: status = "Critical"

            # --- Visualization Layer ---
            frame[mask_pred > 0] = [0, 0, 255] # Red Overlay
            color = (0, 255, 0) if status == "Safe" else (0, 0, 255)
            cv2.putText(frame, f"STATUS: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"PARTS: {parts}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # --- Telemetry Layer ---
            log_event(conn, idx, status, parts)

        out.write(frame)
        print(f"[INFO] Processed Frame {idx}/{len(frames)} | Status: {status}", end='\r')

    conn.commit()
    out.release()
    print(f"\n[INFO] Video saved: {output_path}")
    
    # 6. Generate Report
    print("\n[REPORT] SQL Database Summary:")
    df = pd.read_sql_query("SELECT status, COUNT(*) as frames, AVG(risk_score) as risk FROM event_logs GROUP BY status", conn)
    print(df)
    conn.close()

if __name__ == "__main__":
    run_pipeline()
