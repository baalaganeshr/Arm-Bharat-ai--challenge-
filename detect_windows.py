"""
Face Mask Detection - Windows Version (OpenCV Only)
Simple version that works without TensorFlow issues
Uses Haar Cascades for detection
"""

import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime

print("=" * 50)
print("Face Mask Detection - Windows")
print("=" * 50)

# Paths
LOG_FILE = "logs/compliance_log.csv"

# Load face detector
print("[INFO] Loading face detector...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("[ERROR] Failed to load face cascade!")
    exit(1)

# Try to load model weights from Docker-exported numpy file
MODEL_WEIGHTS = "models/model_weights.npz"
weights = None

# Simple CNN-like prediction using stored weights
class SimpleMaskClassifier:
    def __init__(self):
        self.initialized = False
        
    def predict(self, face_gray):
        """
        Simple prediction based on facial features
        Returns (mask_probability, no_mask_probability)
        """
        # Resize face to 64x64
        face = cv2.resize(face_gray, (64, 64))
        
        # Feature extraction using simple image processing
        # Mask typically covers lower half of face
        
        upper_half = face[:32, :]  # Eyes region
        lower_half = face[32:, :]  # Mouth/nose region
        
        # Calculate statistics
        upper_mean = np.mean(upper_half)
        lower_mean = np.mean(lower_half)
        upper_std = np.std(upper_half)
        lower_std = np.std(lower_half)
        
        # Edge detection for texture analysis
        edges_upper = cv2.Canny(upper_half.astype(np.uint8), 50, 150)
        edges_lower = cv2.Canny(lower_half.astype(np.uint8), 50, 150)
        
        edge_ratio = np.sum(edges_lower) / (np.sum(edges_upper) + 1)
        
        # Mask typically:
        # - More uniform color in lower face (less texture)
        # - Different brightness between upper and lower
        # - Less edge detail in lower face
        
        # Simple heuristic scoring
        uniformity_score = 1.0 - (lower_std / (upper_std + 1))
        edge_score = 1.0 - min(1.0, edge_ratio / 2.0)
        brightness_diff = abs(upper_mean - lower_mean) / 255.0
        
        # Combine scores
        mask_score = (uniformity_score * 0.4 + edge_score * 0.4 + brightness_diff * 0.2)
        mask_score = min(0.95, max(0.05, mask_score))
        
        return mask_score, 1 - mask_score

classifier = SimpleMaskClassifier()

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Initialize CSV log if needed
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'With_Mask', 'Without_Mask'])

# Try to open camera
print("[INFO] Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows

if not cap.isOpened():
    print("[WARN] Camera 0 failed, trying without DirectShow...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARN] Camera 0 failed, trying camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[ERROR] No camera found!")
            print("[INFO] Please check your webcam connection")
            exit(1)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("[INFO] Camera opened successfully!")
print("[INFO] Press 'q' to quit, 's' to save snapshot")
print("[INFO] Note: Using heuristic detection (demo mode)")
print("=" * 50)

# FPS tracking
fps_start = time.time()
fps_count = 0
fps = 0.0

# Statistics
total_mask = 0
total_no_mask = 0
last_log_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to grab frame")
        continue
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(80, 80)
    )
    
    mask_count = 0
    no_mask_count = 0
    
    for (x, y, w, h) in faces:
        # Extract face region
        face_gray = gray[y:y+h, x:x+w]
        
        # Predict
        try:
            mask_prob, no_mask_prob = classifier.predict(face_gray)
            
            if mask_prob > 0.5:
                mask_count += 1
                label = f"Mask: {mask_prob*100:.0f}%"
                color = (0, 255, 0)  # Green
            else:
                no_mask_count += 1
                label = f"No Mask: {no_mask_prob*100:.0f}%"
                color = (0, 0, 255)  # Red
        except Exception as e:
            label = "Processing..."
            color = (255, 255, 0)
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x, y-30), (x + label_size[0] + 10, y), color, -1)
        cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Update totals
    total_mask += mask_count
    total_no_mask += no_mask_count
    
    # Calculate FPS
    fps_count += 1
    elapsed = time.time() - fps_start
    if elapsed > 1.0:
        fps = fps_count / elapsed
        fps_count = 0
        fps_start = time.time()
    
    # Log every 10 seconds
    current_time = time.time()
    if current_time - last_log_time > 10:
        if mask_count > 0 or no_mask_count > 0:
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    mask_count,
                    no_mask_count
                ])
        last_log_time = current_time
    
    # Draw statistics overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 200), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Header
    cv2.putText(frame, "FACE MASK DETECTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(frame, (20, 50), (260, 50), (255, 255, 255), 1)
    
    # Stats
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Faces: {len(faces)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"With Mask: {mask_count}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"No Mask: {no_mask_count}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Compliance
    total = total_mask + total_no_mask
    if total > 0:
        compliance = total_mask / total * 100
        cv2.putText(frame, f"Compliance: {compliance:.1f}%", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
    
    # Show frame
    cv2.imshow('Face Mask Detection - Press Q to Quit', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n[INFO] Quitting...")
        break
    elif key == ord('s'):
        filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Saved: {filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("Session Statistics:")
print(f"  Total with mask: {total_mask}")
print(f"  Total without mask: {total_no_mask}")
if total_mask + total_no_mask > 0:
    compliance = total_mask / (total_mask + total_no_mask) * 100
    print(f"  Compliance rate: {compliance:.1f}%")
print("=" * 50)
