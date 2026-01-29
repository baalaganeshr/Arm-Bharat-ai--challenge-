"""
Face Mask Detection - PRODUCTION VERSION
High accuracy, stable detection, minimal false positives
"""

import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime
from collections import defaultdict

print("=" * 70)
print("           FACE MASK DETECTION SYSTEM - PRODUCTION")
print("                Bharat AI-SoC Challenge 2026")
print("=" * 70)

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)

# CSV log file
LOG_FILE = "logs/realtime_detection.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'With_Mask', 'Without_Mask', 'Total_Faces'])

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
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

print("[SUCCESS] Camera opened!")
print("\n" + "=" * 70)
print("CONTROLS:")
print("  'q' - Quit  |  's' - Save Snapshot  |  'r' - Reset Stats")
print("=" * 70 + "\n")


class FaceTracker:
    """Track faces across frames for stability"""
    def __init__(self):
        self.faces = {}  # id: (x, y, w, h, mask_votes, last_seen)
        self.next_id = 0
        self.max_age = 30  # frames
        
    def update(self, detections):
        """Update tracked faces with new detections"""
        current_time = time.time()
        
        # Mark all as not seen
        for fid in list(self.faces.keys()):
            face_data = self.faces[fid]
            if current_time - face_data[5] > 1.0:  # 1 second timeout
                del self.faces[fid]
        
        matched_ids = []
        
        for det in detections:
            x, y, w, h = det
            
            # Find closest existing face
            best_id = None
            best_dist = float('inf')
            
            for fid, (fx, fy, fw, fh, votes, last_seen) in self.faces.items():
                if fid in matched_ids:
                    continue
                    
                # Calculate distance
                dist = np.sqrt((x - fx)**2 + (y - fy)**2)
                
                if dist < 100 and dist < best_dist:
                    best_dist = dist
                    best_id = fid
            
            if best_id is not None:
                # Update existing face
                matched_ids.append(best_id)
            else:
                # New face
                best_id = self.next_id
                self.next_id += 1
                self.faces[best_id] = [x, y, w, h, [], current_time]
                matched_ids.append(best_id)
        
        return matched_ids


def detect_mask_robust(face_gray):
    """
    ULTRA-SIMPLIFIED: Direct observation approach
    Masks are OBVIOUS - they cover lower face with uniform fabric
    Returns: (has_mask: bool, confidence: int)
    """
    h, w = face_gray.shape
    
    # Split face simply in HALF
    mid_point = h // 2
    upper_half = face_gray[0:mid_point, :]
    lower_half = face_gray[mid_point:, :]
    
    # Calculate SIMPLE metrics
    
    # 1. Edge counts (raw numbers)
    edges_upper = cv2.Canny(upper_half, 50, 150)
    edges_lower = cv2.Canny(lower_half, 50, 150)
    
    upper_edges = np.count_nonzero(edges_upper)
    lower_edges = np.count_nonzero(edges_lower)
    
    # 2. Standard deviation (texture measure)
    upper_std = np.std(upper_half)
    lower_std = np.std(lower_half)
    
    # 3. Variance
    upper_var = np.var(upper_half)
    lower_var = np.var(lower_half)
    
    # === SIMPLE LOGIC ===
    # If wearing mask: lower face has MUCH LESS detail
    
    # Calculate ratios
    edge_ratio = lower_edges / (upper_edges + 1)
    std_ratio = lower_std / (upper_std + 1)
    var_ratio = lower_var / (upper_var + 1)
    
    # Count how many indicators suggest a mask
    mask_indicators = 0
    
    # Indicator 1: Significantly fewer edges in lower face
    if edge_ratio < 0.7:
        mask_indicators += 1
    if edge_ratio < 0.5:
        mask_indicators += 1
    if edge_ratio < 0.3:
        mask_indicators += 1
    
    # Indicator 2: Lower std deviation
    if std_ratio < 0.85:
        mask_indicators += 1
    if std_ratio < 0.7:
        mask_indicators += 1
    
    # Indicator 3: Lower variance
    if var_ratio < 0.8:
        mask_indicators += 1
    if var_ratio < 0.6:
        mask_indicators += 1
    
    # === DECISION ===
    # If 3 or more indicators → MASK
    has_mask = mask_indicators >= 3
    
    # Confidence based on number of indicators
    if has_mask:
        confidence = 60 + (mask_indicators * 7)  # 67%, 74%, 81%, 88%, 95%
    else:
        confidence = 85 - (mask_indicators * 10)  # Strong NO MASK if few indicators
    
    confidence = min(95, max(60, confidence))
    
    return has_mask, confidence


# Initialize tracker
tracker = FaceTracker()
face_mask_history = defaultdict(list)  # face_id: [bool, bool, ...]

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
    
    # Mirror effect
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better detection
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    
    # Detect faces - MUCH MORE SENSITIVE to find all faces
    faces = face_cascade.detectMultiScale(
        gray_enhanced,
        scaleFactor=1.05,  # More sensitive (smaller steps)
        minNeighbors=3,    # Less strict (was 4)
        minSize=(60, 60),  # Smaller minimum (was 80)
        maxSize=(600, 600)
    )
    
    # Also try with regular gray if CLAHE didn't work
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(60, 60),
            maxSize=(600, 600)
        )
    
    # Filter by aspect ratio (faces should be roughly square) - MORE LENIENT
    valid_faces = []
    for (x, y, w, h) in faces:
        aspect_ratio = w / h
        if 0.6 <= aspect_ratio <= 1.5:  # More lenient (was 0.75-1.25)
            valid_faces.append((x, y, w, h))
    
    faces = valid_faces
    
    # Update tracker
    face_ids = tracker.update(faces)
    
    # Process each face
    mask_count = 0
    no_mask_count = 0
    
    for idx, (face_id, (x, y, w, h)) in enumerate(zip(face_ids, faces)):
        # Extract face region
        y1 = max(0, y)
        y2 = min(frame.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(frame.shape[1], x + w)
        
        face_roi = gray[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            continue
        
        # Resize to standard size
        face_resized = cv2.resize(face_roi, (128, 128))
        
        # Detect mask
        has_mask, confidence = detect_mask_robust(face_resized)
        
        # Temporal smoothing - use last 7 frames
        face_mask_history[face_id].append(has_mask)
        if len(face_mask_history[face_id]) > 7:
            face_mask_history[face_id].pop(0)
        
        # Majority vote
        if len(face_mask_history[face_id]) >= 4:
            mask_votes = sum(face_mask_history[face_id])
            has_mask = mask_votes > len(face_mask_history[face_id]) / 2
        
        # Update counts
        if has_mask:
            mask_count += 1
            total_with_mask += 1
            label = f"MASK: {confidence}%"
            color = (0, 255, 0)  # Green
        else:
            no_mask_count += 1
            total_without_mask += 1
            label = f"NO MASK: {confidence}%"
            color = (0, 0, 255)  # Red
        
        # Draw detection box
        thickness = 3
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
        # Draw label
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.8, 2)
        cv2.rectangle(frame, (x, y-40), (x + tw + 10, y), color, -1)
        cv2.putText(frame, label, (x+5, y-12), font, 0.8, (255, 255, 255), 2)
    
    # Calculate FPS
    fps_counter += 1
    if time.time() - fps_start > 1.0:
        current_fps = fps_counter / (time.time() - fps_start)
        fps_counter = 0
        fps_start = time.time()
    
    # Log every 30 seconds
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
    
    # Draw info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 230), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "FACE MASK DETECTION", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
    cv2.line(frame, (20, 50), (380, 50), (100, 255, 100), 2)
    
    # Stats
    y = 80
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y += 30
    cv2.putText(frame, f"Faces: {len(faces)}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 30
    cv2.putText(frame, f"With Mask: {mask_count}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30
    cv2.putText(frame, f"Without Mask: {no_mask_count}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y += 30
    
    total = total_with_mask + total_without_mask
    if total > 0:
        compliance = (total_with_mask / total) * 100
        cv2.putText(frame, f"Compliance: {compliance:.1f}%", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
    
    # Display
    cv2.imshow('Face Mask Detection - Press Q to Quit', frame)
    
    # Handle keys
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
        print("[INFO] Statistics reset!")

# Cleanup
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
