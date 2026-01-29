"""
Face Mask Detection - Using Trained Model
Uses the actual trained CNN model for accurate detection
"""

import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime
from collections import defaultdict

print("=" * 70)
print("      FACE MASK DETECTION - USING TRAINED MODEL")
print("              Bharat AI-SoC Challenge 2026")
print("=" * 70)

# Check if TensorFlow is available
try:
    import tensorflow as tf
    print(f"[INFO] TensorFlow {tf.__version__} loaded successfully")
    MODEL_AVAILABLE = True
except ImportError:
    print("[WARNING] TensorFlow not available, will use fallback")
    MODEL_AVAILABLE = False

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)

# CSV log file
LOG_FILE = "logs/realtime_detection.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'With_Mask', 'Without_Mask', 'Total_Faces'])

# Load model
MODEL_PATH = "models/mask_detector_64x64.h5"
model = None

if MODEL_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[SUCCESS] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        MODEL_AVAILABLE = False
else:
    print(f"[WARNING] Model not found at {MODEL_PATH}")
    if not MODEL_AVAILABLE:
        print("[INFO] Will use simple heuristic detection")

# Load face detector
print("[INFO] Loading face detector...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print(f"[SUCCESS] Face detector loaded!")

# Open camera
print("[INFO] Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("[SUCCESS] Camera opened!")
print("\n" + "=" * 70)
print("CONTROLS:")
print("  'q' - Quit  |  's' - Save Snapshot  |  'r' - Reset Stats")
print("=" * 70 + "\n")


def detect_mask_with_model(face_gray):
    """Use trained model to detect mask"""
    # Resize to model input size (64x64)
    face_resized = cv2.resize(face_gray, (64, 64))
    
    # Normalize
    face_normalized = face_resized / 255.0
    
    # Add batch dimension
    face_input = np.expand_dims(face_normalized, axis=0)
    face_input = np.expand_dims(face_input, axis=-1)
    
    # Predict
    prediction = model.predict(face_input, verbose=0)[0][0]
    
    # prediction close to 1 = mask, close to 0 = no mask
    has_mask = prediction > 0.5
    confidence = int(prediction * 100) if has_mask else int((1 - prediction) * 100)
    
    return has_mask, confidence


def detect_mask_heuristic(face_gray):
    """
    Fallback heuristic detection
    INVERTED LOGIC: Lower face covered by mask = fewer details
    """
    h, w = face_gray.shape
    
    # Split into upper (eyes) and lower (nose/mouth)
    upper = face_gray[0:int(h*0.45), :]
    lower = face_gray[int(h*0.55):, :]
    
    # Count edges
    edges_upper = cv2.Canny(upper, 50, 150)
    edges_lower = cv2.Canny(lower, 50, 150)
    
    edge_count_upper = np.count_nonzero(edges_upper)
    edge_count_lower = np.count_nonzero(edges_lower)
    
    # If lower has significantly fewer edges, it's covered (mask)
    edge_ratio = edge_count_lower / (edge_count_upper + 1)
    
    # Calculate variance
    var_upper = np.var(upper)
    var_lower = np.var(lower)
    var_ratio = var_lower / (var_upper + 1)
    
    # Scoring
    score = 0
    
    # Fewer edges in lower face = mask
    if edge_ratio < 0.4:
        score += 40
    elif edge_ratio < 0.6:
        score += 25
    elif edge_ratio < 0.75:
        score += 10
    
    # Lower variance = mask
    if var_ratio < 0.5:
        score += 35
    elif var_ratio < 0.7:
        score += 20
    elif var_ratio < 0.85:
        score += 10
    
    # Brightness check
    mean_upper = np.mean(upper)
    mean_lower = np.mean(lower)
    if abs(mean_upper - mean_lower) > 15:
        score += 15
    
    has_mask = score >= 50
    confidence = min(95, max(60, score))
    
    return has_mask, confidence


# Face tracking
class FaceTracker:
    def __init__(self):
        self.faces = {}
        self.next_id = 0
        
    def update(self, detections):
        current_time = time.time()
        
        # Remove old faces
        for fid in list(self.faces.keys()):
            if current_time - self.faces[fid][5] > 1.0:
                del self.faces[fid]
        
        matched_ids = []
        
        for x, y, w, h in detections:
            best_id = None
            best_dist = float('inf')
            
            for fid, (fx, fy, fw, fh, votes, last_seen) in self.faces.items():
                if fid in matched_ids:
                    continue
                dist = np.sqrt((x - fx)**2 + (y - fy)**2)
                if dist < 100 and dist < best_dist:
                    best_dist = dist
                    best_id = fid
            
            if best_id is not None:
                self.faces[best_id] = [x, y, w, h, self.faces[best_id][4], current_time]
                matched_ids.append(best_id)
            else:
                best_id = self.next_id
                self.next_id += 1
                self.faces[best_id] = [x, y, w, h, [], current_time]
                matched_ids.append(best_id)
        
        return matched_ids


tracker = FaceTracker()
mask_history = defaultdict(list)

# Statistics
total_with_mask = 0
total_without_mask = 0
fps_start = time.time()
fps_counter = 0
current_fps = 0
last_log_time = time.time()

print("[INFO] Starting detection loop...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhance
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_enhanced,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(60, 60),
        maxSize=(600, 600)
    )
    
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(60, 60)
        )
    
    # Filter by aspect ratio
    valid_faces = []
    for (x, y, w, h) in faces:
        aspect_ratio = w / h
        if 0.6 <= aspect_ratio <= 1.5:
            valid_faces.append((x, y, w, h))
    
    faces = valid_faces
    face_ids = tracker.update(faces)
    
    mask_count = 0
    no_mask_count = 0
    
    for face_id, (x, y, w, h) in zip(face_ids, faces):
        y1 = max(0, y)
        y2 = min(frame.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(frame.shape[1], x + w)
        
        face_roi = gray[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            continue
        
        # Detect mask
        if model is not None:
            has_mask, confidence = detect_mask_with_model(face_roi)
        else:
            has_mask, confidence = detect_mask_heuristic(face_roi)
        
        # Temporal smoothing
        mask_history[face_id].append(has_mask)
        if len(mask_history[face_id]) > 7:
            mask_history[face_id].pop(0)
        
        if len(mask_history[face_id]) >= 4:
            mask_votes = sum(mask_history[face_id])
            has_mask = mask_votes > len(mask_history[face_id]) / 2
        
        # Update counts
        if has_mask:
            mask_count += 1
            total_with_mask += 1
            label = f"MASK: {confidence}%"
            color = (0, 255, 0)
        else:
            no_mask_count += 1
            total_without_mask += 1
            label = f"NO MASK: {confidence}%"
            color = (0, 0, 255)
        
        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.8, 2)
        cv2.rectangle(frame, (x, y-40), (x + tw + 10, y), color, -1)
        cv2.putText(frame, label, (x+5, y-12), font, 0.8, (255, 255, 255), 2)
    
    # FPS
    fps_counter += 1
    if time.time() - fps_start > 1.0:
        current_fps = fps_counter / (time.time() - fps_start)
        fps_counter = 0
        fps_start = time.time()
    
    # Log
    if time.time() - last_log_time > 30 and len(faces) > 0:
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                mask_count,
                no_mask_count,
                len(faces)
            ])
        last_log_time = time.time()
    
    # Info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 230), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, "FACE MASK DETECTION", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
    cv2.line(frame, (20, 50), (380, 50), (100, 255, 100), 2)
    
    y = 80
    mode_text = "Mode: ML Model" if model is not None else "Mode: Heuristic"
    cv2.putText(frame, mode_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y += 30
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y += 30
    cv2.putText(frame, f"Faces: {len(faces)}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 30
    cv2.putText(frame, f"With Mask: {mask_count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30
    cv2.putText(frame, f"Without Mask: {no_mask_count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y += 30
    
    total = total_with_mask + total_without_mask
    if total > 0:
        compliance = (total_with_mask / total) * 100
        cv2.putText(frame, f"Compliance: {compliance:.1f}%", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
    
    cv2.imshow('Face Mask Detection - Press Q to Quit', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"snapshots/snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Saved: {filename}")
    elif key == ord('r'):
        total_with_mask = 0
        total_without_mask = 0
        print("[INFO] Stats reset!")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("SESSION SUMMARY:")
print("=" * 70)
print(f"Detections with mask:    {total_with_mask}")
print(f"Detections without mask: {total_without_mask}")
total = total_with_mask + total_without_mask
if total > 0:
    print(f"Overall compliance:      {(total_with_mask/total)*100:.1f}%")
print(f"Log file: {LOG_FILE}")
print("=" * 70)
