"""
Face Mask Detection - Windows (Simple Version)
Works without TensorFlow - uses computer vision heuristics
Perfect for testing and demonstration
"""

import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime

print("=" * 60)
print("    FACE MASK DETECTION - REAL-TIME")
print("    Bharat AI-SoC Challenge 2026")
print("=" * 60)

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

if face_cascade.empty():
    print("[ERROR] Failed to load face cascade!")
    exit(1)

print("[SUCCESS] Face detector loaded!")

# Open camera
print("[INFO] Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("[WARN] DirectShow failed, trying default...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        exit(1)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("[SUCCESS] Camera opened!")
print("\n" + "=" * 60)
print("CONTROLS:")
print("  'q' - Quit")
print("  's' - Save snapshot")
print("  'r' - Reset statistics")
print("=" * 60 + "\n")

# Detection algorithm - IMPROVED
def detect_mask(face_gray):
    """
    Enhanced mask detection using multiple features
    Returns: (has_mask: bool, confidence: float)
    """
    h, w = face_gray.shape
    
    # Divide face into 3 regions for better analysis
    upper_third = face_gray[0:int(h*0.33), :]           # Forehead/eyes
    middle_third = face_gray[int(h*0.33):int(h*0.67), :] # Nose area
    lower_third = face_gray[int(h*0.67):, :]            # Mouth/chin
    
    # Apply histogram equalization for better feature extraction
    upper_eq = cv2.equalizeHist(upper_third)
    middle_eq = cv2.equalizeHist(middle_third)
    lower_eq = cv2.equalizeHist(lower_third)
    
    # Feature 1: Texture variance (masks are more uniform)
    upper_variance = np.var(upper_eq)
    middle_variance = np.var(middle_eq)
    lower_variance = np.var(lower_eq)
    
    # Feature 2: Edge density (masks have fewer edges in lower face)
    edges_upper = cv2.Canny(upper_eq, 30, 100)
    edges_middle = cv2.Canny(middle_eq, 30, 100)
    edges_lower = cv2.Canny(lower_eq, 30, 100)
    
    edge_count_upper = np.sum(edges_upper > 0)
    edge_count_middle = np.sum(edges_middle > 0)
    edge_count_lower = np.sum(edges_lower > 0)
    
    # Feature 3: Gradient magnitude (skin has more variation)
    grad_x = cv2.Sobel(lower_eq, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(lower_eq, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    avg_gradient = np.mean(gradient_magnitude)
    
    # Scoring with improved weights
    score = 0.0
    
    # 1. Lower face has significantly less texture than upper (STRONG indicator)
    if lower_variance < upper_variance * 0.6:
        score += 0.35
    
    # 2. Lower face has fewer edges (mask covers details)
    if edge_count_lower < edge_count_middle * 0.5:
        score += 0.25
    
    # 3. Middle face has fewer edges than upper (nose covered)
    if edge_count_middle < edge_count_upper * 0.7:
        score += 0.20
    
    # 4. Low gradient in lower face (uniform mask surface)
    if avg_gradient < 15:
        score += 0.15
    
    # 5. Overall uniformity check
    variance_ratio = (upper_variance + 1) / (lower_variance + 1)
    if variance_ratio > 1.5:
        score += 0.05
    
    # Determine result
    has_mask = score > 0.45  # Lower threshold for better sensitivity
    confidence = min(0.95, max(0.60, score))
    
    return has_mask, confidence

# Statistics
total_with_mask = 0
total_without_mask = 0
fps_start_time = time.time()
fps_counter = 0
current_fps = 0
last_log_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame")
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces - IMPROVED PARAMETERS for better detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,      # More sensitive (was 1.1)
        minNeighbors=3,        # Less strict (was 5)
        minSize=(60, 60),      # Detect smaller faces (was 100x100)
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    mask_count = 0
    no_mask_count = 0
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract face with padding
        padding = int(h * 0.1)
        y1 = max(0, y - padding)
        y2 = min(gray.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(gray.shape[1], x + w + padding)
        
        face_gray = gray[y1:y2, x1:x2]
        
        # Resize for consistent analysis
        if face_gray.size > 0:
            face_gray = cv2.resize(face_gray, (128, 128))
        else:
            continue
        
        # Detect mask
        has_mask, confidence = detect_mask(face_gray)
        
        if has_mask:
            mask_count += 1
            label = f"MASK: {confidence*100:.0f}%"
            color = (0, 255, 0)  # Green
            total_with_mask += 1
        else:
            no_mask_count += 1
            label = f"NO MASK: {(1-confidence)*100:.0f}%"
            color = (0, 0, 255)  # Red
            total_without_mask += 1
        
        # Draw rectangle
        thickness = 3
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
        # Draw label with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        cv2.rectangle(frame, (x, y-35), (x + text_width + 10, y), color, -1)
        cv2.putText(frame, label, (x+5, y-10), font, font_scale, (255, 255, 255), font_thickness)
    
    # Calculate FPS
    fps_counter += 1
    elapsed = time.time() - fps_start_time
    if elapsed > 1.0:
        current_fps = fps_counter / elapsed
        fps_counter = 0
        fps_start_time = time.time()
    
    # Log to CSV every 30 seconds
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
    
    # Draw overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (420, 250), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Header
    cv2.putText(frame, "FACE MASK DETECTION SYSTEM", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(frame, (20, 50), (400, 50), (100, 255, 100), 2)
    
    # Statistics
    y_pos = 80
    line_height = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, y_pos), font, 0.7, (0, 255, 255), 2)
    y_pos += line_height
    
    cv2.putText(frame, f"Faces Detected: {len(faces)}", (20, y_pos), font, 0.7, (255, 255, 255), 2)
    y_pos += line_height
    
    # DEBUG: Show if faces are being detected
    if len(faces) == 0:
        cv2.putText(frame, "Looking for faces...", (20, y_pos), font, 0.6, (255, 255, 0), 1)
        y_pos += line_height
    
    cv2.putText(frame, f"With Mask: {mask_count}", (20, y_pos), font, 0.7, (0, 255, 0), 2)
    y_pos += line_height
    
    cv2.putText(frame, f"Without Mask: {no_mask_count}", (20, y_pos), font, 0.7, (0, 0, 255), 2)
    y_pos += line_height
    
    # Total statistics
    total = total_with_mask + total_without_mask
    if total > 0:
        compliance = (total_with_mask / total) * 100
        cv2.putText(frame, f"Compliance: {compliance:.1f}%", (20, y_pos), font, 0.7, (255, 200, 100), 2)
    
    # Display
    cv2.imshow('Face Mask Detection - Press Q to Quit', frame)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n[INFO] Quitting...")
        break
    elif key == ord('s'):
        filename = f"snapshots/snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Snapshot saved: {filename}")
    elif key == ord('r'):
        total_with_mask = 0
        total_without_mask = 0
        print("[INFO] Statistics reset!")

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Final statistics
print("\n" + "=" * 60)
print("SESSION STATISTICS:")
print("=" * 60)
print(f"Total detections with mask:    {total_with_mask}")
print(f"Total detections without mask: {total_without_mask}")
total = total_with_mask + total_without_mask
if total > 0:
    compliance = (total_with_mask / total) * 100
    print(f"Overall compliance rate:       {compliance:.1f}%")
print(f"Log file saved to: {LOG_FILE}")
print("=" * 60)
