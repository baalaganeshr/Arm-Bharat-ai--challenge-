"""
FPGA-Accelerated Face Mask Detection
Simplified version for real FPGA hardware integration
"""

import cv2
import numpy as np
from tensorflow import keras
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from our_improvements.fpga_interface import FPGAInterface

# Load model
print("[INFO] Loading model...")
model = keras.models.load_model("models/mask_detector_64x64.h5")

# Initialize FPGA
print("[INFO] Connecting to FPGA...")
# Change port to 'COM3' on Windows or '/dev/ttyUSB0' on Linux/RPi
FPGA_PORT = '/dev/ttyUSB0'  # Update this for your setup
fpga = FPGAInterface(port=FPGA_PORT, baudrate=115200)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open camera")
    sys.exit(1)

# FPS tracking
fps_start_time = time.time()
fps_counter = 0
fps = 0

# Mode toggle
use_fpga = True if fpga.is_connected() else False

print(f"[INFO] Mode: {'FPGA Accelerated' if use_fpga else 'CPU Only'}")
print("[INFO] Press 'q' to quit, 't' to toggle FPGA/CPU mode")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
    
    mask_count = 0
    no_mask_count = 0
    
    inference_start = time.time()
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face_normalized = face.reshape(1, 64, 64, 1) / 255.0
        
        if use_fpga and fpga.is_connected():
            # FPGA acceleration
            result = fpga.process_image(face / 255.0)
            if result is not None:
                class_id, confidence = result
                is_mask = (class_id == 0)
                pred_confidence = confidence
            else:
                # Fallback to CPU
                pred = model.predict(face_normalized, verbose=0)
                is_mask = pred[0][0] > pred[0][1]
                pred_confidence = max(pred[0])
        else:
            # CPU fallback
            pred = model.predict(face_normalized, verbose=0)
            is_mask = pred[0][0] > pred[0][1]
            pred_confidence = max(pred[0])
        
        if is_mask:
            mask_count += 1
            label = f"Mask: {pred_confidence*100:.0f}%"
            color = (0, 255, 0)
        else:
            no_mask_count += 1
            label = f"No Mask: {pred_confidence*100:.0f}%"
            color = (0, 0, 255)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y-25), (x + label_size[0], y), color, -1)
        cv2.putText(frame, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    inference_time = (time.time() - inference_start) * 1000
    
    # Calculate FPS
    fps_counter += 1
    if fps_counter % 30 == 0:
        fps_end_time = time.time()
        fps = 30 / (fps_end_time - fps_start_time)
        fps_start_time = time.time()
    
    # Draw statistics overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (280, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    mode_text = "FPGA Accelerated" if (use_fpga and fpga.is_connected()) else "CPU Only"
    mode_color = (0, 255, 0) if use_fpga else (0, 165, 255)
    
    cv2.putText(frame, f"Mode: {mode_text}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    cv2.putText(frame, f"With Mask: {mask_count}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"Without Mask: {no_mask_count}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(frame, f"Total: {mask_count + no_mask_count}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(frame, f"Latency: {inference_time:.1f}ms", (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
    
    cv2.imshow('Face Mask Detection - FPGA', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[INFO] Quitting...")
        break
    elif key == ord('t'):
        use_fpga = not use_fpga
        mode = 'FPGA' if use_fpga else 'CPU'
        print(f"[INFO] Switched to {mode} mode")

cap.release()
cv2.destroyAllWindows()
fpga.close()
print("[INFO] Detection stopped")
