"""
FINAL Face Mask Detection System
Uses properly trained model with GPU
"""

import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime
from collections import defaultdict

print("=" * 80)
print("            FACE MASK DETECTION - PRODUCTION SYSTEM")
print("                GPU-Trained Model - High Accuracy")
print("=" * 80)

# Try to load TensorFlow
try:
    import tensorflow as tf
    MODEL_PATH = "models/best_model.h5"
    
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[SUCCESS] Loaded trained model: {MODEL_PATH}")
        MODEL_SIZE = 128  # Model was trained on 128x128
    else:
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("[INFO] Please wait for training to complete")
        exit(1)
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)

# CSV log
LOG_FILE = "logs/detection_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'With_Mask', 'Without_Mask', 'Total_Faces', 'Compliance_%'])

# Load face detector
print("[INFO] Loading face detector...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Camera
print("[INFO] Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("[SUCCESS] Camera ready!")
print("\n" + "=" * 80)
print("CONTROLS: 'q' = Quit  |  's' = Snapshot  |  'r' = Reset Stats")
print("=" * 80 + "\n")


class FaceTracker:
    def __init__(self):
        self.faces = {}
        self.next_id = 0
        
    def update(self, detections):
        current_time = time.time()
        
        # Clean old
        for fid in list(self.faces.keys()):
            if current_time - self.faces[fid][-1] > 1.0:
                del self.faces[fid]
        
        matched = []
        for x, y, w, h in detections:
            best_id = None
            best_dist = float('inf')
            
            for fid, data in self.faces.items():
                if fid in matched:
                    continue
                fx, fy = data[0], data[1]
                dist = np.sqrt((x-fx)**2 + (y-fy)**2)
                if dist < 80 and dist < best_dist:
                    best_dist = dist
                    best_id = fid
            
            if best_id:
                self.faces[best_id] = [x, y, w, h, current_time]
                matched.append(best_id)
            else:
                best_id = self.next_id
                self.next_id += 1
                self.faces[best_id] = [x, y, w, h, current_time]
                matched.append(best_id)
        
        return matched


tracker = FaceTracker()
mask_history = defaultdict(list)

# Stats
total_with = 0
total_without = 0
fps_time = time.time()
fps_count = 0
fps = 0
last_log = time.time()

print("[INFO] Starting detection...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        enhanced,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(60, 60),
        maxSize=(500, 500)
    )
    
    # Fallback
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(60, 60)
        )
    
    # Filter
    valid = []
    for (x, y, w, h) in faces:
        if 0.6 <= w/h <= 1.5:
            valid.append((x, y, w, h))
    
    faces = valid
    face_ids = tracker.update(faces)
    
    current_with = 0
    current_without = 0
    
    for fid, (x, y, w, h) in zip(face_ids, faces):
        # Extract
        y1, y2 = max(0, y), min(frame.shape[0], y+h)
        x1, x2 = max(0, x), min(frame.shape[1], x+w)
        roi = gray[y1:y2, x1:x2]
        
        if roi.size == 0:
            continue
        
        # Prepare for model
        face_resized = cv2.resize(roi, (MODEL_SIZE, MODEL_SIZE))
        face_norm = face_resized / 255.0
        face_input = np.expand_dims(face_norm, axis=0)
        face_input = np.expand_dims(face_input, axis=-1)
        
        # Predict
        pred = model.predict(face_input, verbose=0)[0][0]
        
        # with_mask = class 0, without_mask = class 1
        # pred close to 0 = with_mask, close to 1 = without_mask
        has_mask = pred < 0.5
        conf = int((1-pred)*100) if has_mask else int(pred*100)
        
        # Smooth
        mask_history[fid].append(has_mask)
        if len(mask_history[fid]) > 5:
            mask_history[fid].pop(0)
        
        if len(mask_history[fid]) >= 3:
            has_mask = sum(mask_history[fid]) > len(mask_history[fid])/2
        
        # Count
        if has_mask:
            current_with += 1
            total_with += 1
            label = f"MASK: {conf}%"
            color = (0, 255, 0)
        else:
            current_without += 1
            total_without += 1
            label = f"NO MASK: {conf}%"
            color = (0, 0, 255)
        
        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x, y-45), (x+tw+10, y), color, -1)
        cv2.putText(frame, label, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    
    # FPS
    fps_count += 1
    if time.time() - fps_time > 1:
        fps = fps_count / (time.time() - fps_time)
        fps_count = 0
        fps_time = time.time()
    
    # Log every 30s
    if time.time() - last_log > 30 and len(faces) > 0:
        total = total_with + total_without
        compliance = (total_with/total*100) if total > 0 else 0
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                current_with, current_without, len(faces), f"{compliance:.1f}"
            ])
        last_log = time.time()
    
    # Panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 260), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Title
    cv2.putText(frame, "FACE MASK DETECTION", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.line(frame, (20, 55), (430, 55), (0, 255, 0), 2)
    
    # Info
    y = 90
    cv2.putText(frame, f"Model: GPU-Trained CNN", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    y += 35
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    y += 35
    cv2.putText(frame, f"Faces: {len(faces)}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y += 35
    cv2.putText(frame, f"With Mask: {current_with}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y += 35
    cv2.putText(frame, f"Without Mask: {current_without}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    y += 35
    
    total = total_with + total_without
    if total > 0:
        comp = total_with/total*100
        cv2.putText(frame, f"Compliance: {comp:.1f}%", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)
    
    cv2.imshow('Face Mask Detection System - Press Q to Quit', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        snap = f"snapshots/detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(snap, frame)
        print(f"[INFO] Saved: {snap}")
    elif key == ord('r'):
        total_with = 0
        total_without = 0
        print("[INFO] Statistics reset")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 80)
print("FINAL STATISTICS:")
print("=" * 80)
print(f"Total with mask:    {total_with}")
print(f"Total without mask: {total_without}")
total = total_with + total_without
if total > 0:
    print(f"Overall compliance: {total_with/total*100:.1f}%")
print(f"Log file: {LOG_FILE}")
print("=" * 80)
