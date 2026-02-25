"""
Project: Autonomous Warehouse Perception System
Author: Alfayez Ahmad
Copyright: (c) 2026 Alfayez Ahmad
License: MIT
Description: Hybrid pipeline integrating Meta SAM 2 (Vision) with OpenCV (Logic) 
             and SQLite (Memory) to detect object fragmentation events.
Scenario: "The Big Bang" - Detecting a single object exploding into 4 independent shards.
"""

import os
import sys
import cv2
import numpy as np
import sqlite3
import pandas as pd
import torch
from datetime import datetime
from PIL import Image

# Try to import SAM 2 (Graceful error if not installed)
try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("[CRITICAL] SAM 2 not found. Please install via: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    sys.exit(1)

# ==========================================
# CONFIGURATION (Cloud-Native Environment Variables)
# ==========================================
CONFIG = {
    "project_dir": os.environ.get("DATA_DIR", "warehouse_data"),
    "video_name": "mitosis_simulation.mp4",
    "mask_name": "mask.png",
    "db_name": os.environ.get("DB_PATH", "warehouse_data/robot_memory.db"),
    "output_path": os.environ.get("OUTPUT_PATH", "warehouse_data/Final_System_Output.mp4"),
    "model_weight": os.environ.get("MODEL_WEIGHT_PATH", "sam2_hiera_large.pt"),
    "model_cfg": "sam2_hiera_l.yaml",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# MODULE 1: DATA GENERATION ("Big Bang" Strategy)
# ==========================================
def generate_synthetic_data(base_dir):
    """
    Generates a synthetic video of 1 object splitting into 4.
    Uses 'Fused Initialization' to force SAM 2 to track the group parent.
    """
    print(f"[INFO] Generating synthetic 'Big Bang' data (1 -> 4 shards)...")
    os.makedirs(base_dir, exist_ok=True)
    frame_dir = os.path.join(base_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    
    width, height = 640, 480
    cx, cy = 320, 240 # Center screen
    
    # 1. Generate Video
    video_path = os.path.join(base_dir, CONFIG["video_name"])
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    
    for i in range(150):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Physics: Objects start fused (dist=0) then explode outward
        dist = 0
        if i > 40: dist = (i - 40) * 3
        
        # Draw 4 circles originating from center
        offsets = [(-1, -1), (1, -1), (-1, 1), (1, 1)] # TopLeft, TopRight, BotLeft, BotRight
        for dx, dy in offsets:
            cv2.circle(frame, (cx + dist*dx, cy + dist*dy), 40, (255, 255, 255), -1)
            
        out.write(frame)
        cv2.imwrite(f"{frame_dir}/{i:05d}.jpg", frame)
    out.release()
    
    # 2. Generate Prompt Mask
    # Key Logic: Prompt with ONE central blob to teach SAM 2 "These 4 things are Object 1"
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), 45, 255, -1)
    Image.fromarray(mask).save(os.path.join(base_dir, CONFIG["mask_name"]))
    
    return frame_dir

# ==========================================
# MODULE 2: DATABASE BACKEND (Memory)
# ==========================================
def init_database():
    """Initializes the SQLite schema for incident logging."""
    # Ensure directory exists before connecting
    os.makedirs(os.path.dirname(CONFIG["db_name"]), exist_ok=True)
    conn = sqlite3.connect(CONFIG["db_name"])
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS event_logs')
    cursor.execute('''
        CREATE TABLE event_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            frame_idx INTEGER,
            status TEXT,
            shard_count INTEGER,
            risk_score FLOAT
        )
    ''')
    conn.commit()
    return conn

def log_event(conn, frame_idx, status, shards):
    """Inserts telemetry into DB."""
    risk = 0.1 if status == "STABLE" else 0.95
    now = datetime.now().strftime("%H:%M:%S.%f")
    conn.execute('''
        INSERT INTO event_logs (timestamp, frame_idx, status, shard_count, risk_score)
        VALUES (?, ?, ?, ?, ?)
    ''', (now, frame_idx, status, shards, risk))
    conn.commit()

# ==========================================
# MODULE 3: MAIN PIPELINE (Vision + Logic)
# ==========================================
def run_pipeline():
    # 1. Setup
    frame_dir = generate_synthetic_data(CONFIG["project_dir"])
    conn = init_database()
    
    # 2. Download Weights
    if not os.path.exists(CONFIG["model_weight"]):
        print(f"[INFO] Downloading SAM 2 weights to {CONFIG['model_weight']}...")
        os.system(f"wget -q -O {CONFIG['model_weight']} https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")

    # 3. Load SAM 2
    print(f"[INFO] Loading Model on {CONFIG['device']}...")
    predictor = build_sam2_video_predictor(CONFIG["model_cfg"], CONFIG["model_weight"], device=CONFIG["device"])
    inference_state = predictor.init_state(video_path=frame_dir)

    # 4. Apply Prompt
    mask_path = os.path.join(CONFIG["project_dir"], CONFIG["mask_name"])
    mask = np.array(Image.open(mask_path).convert('L')) > 0
    predictor.add_new_mask(inference_state, 0, 1, mask.astype(np.float32))

    # 5. Processing Loop
    print("[INFO] Starting Vision Pipeline...")
    frames = sorted(os.listdir(frame_dir))
    h, w = cv2.imread(f"{frame_dir}/{frames[0]}").shape[:2]
    out = cv2.VideoWriter(CONFIG["output_path"], cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    # Visualization colors for shards
    colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255), (255, 165, 0)]
    kernel = np.ones((5,5), np.uint8)

    for idx, ids, logits in predictor.propagate_in_video(inference_state):
        frame = cv2.imread(f"{frame_dir}/{frames[idx]}")
        
        status = "STABLE"
        shards = 1
        
        if len(logits) > 0:
            mask_pred = (logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8)
            
            # --- LOGIC LAYER ---
            # Erosion snaps thin connections to reveal independent components
            eroded = cv2.erode(mask_pred, kernel, iterations=3)
            num_labels, labels = cv2.connectedComponents(eroded)
            # Filter noise < 50 pixels
            shards = sum(1 for i in range(1, num_labels) if np.sum(labels == i) > 50)
            
            if shards > 1: status = "FRAGMENTED"

            # --- VISUALIZATION LAYER ---
            found_parts = 0
            for i in range(1, num_labels):
                if np.sum(labels == i) < 50: continue
                found_parts += 1
                c = colors[(found_parts - 1) % 5]
                overlay = np.zeros_like(frame)
                overlay[labels == i] = c
                frame = cv2.addWeighted(frame, 1.0, overlay, 0.7, 0)
                
                # ID Tag Logic
                M = cv2.moments((labels == i).astype(np.uint8))
                if M["m00"] != 0:
                    cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cv2.putText(frame, f"1.{found_parts}", (cX-15, cY+5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Dashboard Overlay
            cv2.rectangle(frame, (10, h-80), (280, h-10), (0,0,0), -1)
            color = (0, 255, 0) if shards <= 1 else (0, 0, 255)
            cv2.putText(frame, f"STATUS: {status}", (20, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"TRACKING: {shards} SHARDS", (20, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            log_event(conn, idx, status, shards)

        out.write(frame)
        print(f"[INFO] Processed Frame {idx}/{len(frames)}", end='\r')

    out.release()
    print(f"\n[SUCCESS] Video Saved: {CONFIG['output_path']}")

    # ==========================================
    # MODULE 4: THE MANAGER'S REPORT (DBMS)
    # ==========================================
    print("\n[REPORT] GENERATING INCIDENT AUDIT (FROM SQL):")
    print("-" * 50)

    # Query 1: Find the exact moment of fracture
    try:
        query = """
            SELECT MIN(frame_idx) as fracture_frame, timestamp, risk_score
            FROM event_logs 
            WHERE status = 'FRAGMENTED'
        """
        df_incident = pd.read_sql_query(query, conn)
        
        if df_incident['fracture_frame'][0] is not None:
            
            print(f" [ALERT] FRACTURE DETECTED AT FRAME: {df_incident['fracture_frame'][0]}")
            print(f"         Time of Incident: {df_incident['timestamp'][0]}")
            print(f"         Risk Score: {df_incident['risk_score'][0]} (CRITICAL)")
        else:
            print(" [OK] NO FRACTURE DETECTED. SYSTEM STABLE.")
            
        print("-" * 50)
        
        # Query 2: Summary Stats
        query_stats = """
            SELECT status, COUNT(*) as frames, MAX(shard_count) as max_shards
            FROM event_logs
            GROUP BY status
        """
        print(pd.read_sql_query(query_stats, conn))
        print("-" * 50)
        
    except Exception as e:
        print(f"[ERROR] SQL Reporting Error: {e}")

    conn.close()

if __name__ == "__main__":
    run_pipeline()
