"""
Real-Time Face Mask Detection with Dashboard Integration
Detects faces from webcam/video and logs results to dashboard
"""

import cv2
import numpy as np
from tensorflow import keras
import os
from datetime import datetime
import csv
import time

print("=" * 70)
print("  REAL-TIME FACE MASK DETECTION WITH DASHBOARD")
print("=" * 70)

# Configuration
MODEL_PATH = 'models/mask_detector_mobilenet.h5'
FACE_DETECTOR_PATH = 'haarcascade_frontalface_default.xml'
LOG_FILE = 'logs/compliance_log.csv'
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.5
LOG_INTERVAL = 2  # Log every 2 seconds

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Initialize CSV log file
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'With_Mask', 'Without_Mask', 'Compliance_Percent'])
    print(f"[INFO] Created log file: {LOG_FILE}")

# Load face detector
print(f"\n[INFO] Loading face detector...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load mask detection model
print(f"[INFO] Loading mask detection model...")
if not os.path.exists(MODEL_PATH):
    print(f"[WARNING] Model not found: {MODEL_PATH}")
    print(f"[INFO] Checking for alternative models...")
    
    # Check for other models
    alt_models = [
        'models/mask_detector_128x128_final.h5',
        'models/best_model.h5',
        'models/mask_detector_64x64_best.h5'
    ]
    
    for alt_model in alt_models:
        if os.path.exists(alt_model):
            MODEL_PATH = alt_model
            print(f"[SUCCESS] Using alternative model: {MODEL_PATH}")
            break
    else:
        print(f"[ERROR] No model found. Please train a model first.")
        exit(1)

model = keras.models.load_model(MODEL_PATH)
print(f"[SUCCESS] Model loaded: {MODEL_PATH}")

# Open webcam
print(f"\n[INFO] Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[WARNING] Webcam not available, using test video mode")
    USE_WEBCAM = False
else:
    print("[SUCCESS] Webcam opened")
    USE_WEBCAM = True

# Detection stats
last_log_time = time.time()
with_mask_count = 0
without_mask_count = 0
frame_count = 0

print("\n" + "=" * 70)
print("  DETECTION STARTED")
print("=" * 70)
print("  Press 'q' to quit")
print("  Press 's' to save screenshot")
print("  Logging to dashboard every", LOG_INTERVAL, "seconds")
print("=" * 70)

try:
    while True:
        if USE_WEBCAM:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Create a test frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No webcam - Test Mode", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frame_count += 1
        temp_with_mask = 0
        temp_without_mask = 0
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, 
                                              minNeighbors=5, minSize=(60, 60))
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess for model
            face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_array = np.array(face_resized, dtype="float32") / 255.0
            face_array = np.expand_dims(face_array, axis=0)
            
            # Predict
            predictions = model.predict(face_array, verbose=0)
            
            # Determine label
            if len(predictions[0]) == 2:  # MobileNet with softmax (2 classes)
                mask_prob = predictions[0][1]  # Class 1 = with_mask
                label = "Mask" if mask_prob > CONFIDENCE_THRESHOLD else "No Mask"
                confidence = mask_prob if mask_prob > CONFIDENCE_THRESHOLD else (1 - mask_prob)
                has_mask = mask_prob > CONFIDENCE_THRESHOLD
            else:  # Binary sigmoid output
                mask_prob = predictions[0][0]
                label = "Mask" if mask_prob > CONFIDENCE_THRESHOLD else "No Mask"
                confidence = mask_prob if mask_prob > CONFIDENCE_THRESHOLD else (1 - mask_prob)
                has_mask = mask_prob > CONFIDENCE_THRESHOLD
            
            # Update counters
            if has_mask:
                temp_with_mask += 1
                color = (0, 255, 0)  # Green
            else:
                temp_without_mask += 1
                color = (0, 0, 255)  # Red
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label_text = f"{label}: {confidence*100:.1f}%"
            cv2.putText(frame, label_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Update cumulative counts
        with_mask_count += temp_with_mask
        without_mask_count += temp_without_mask
        
        # Display statistics on frame
        total = with_mask_count + without_mask_count
        compliance = (with_mask_count / total * 100) if total > 0 else 0
        
        stats_text = [
            f"Faces Detected: {len(faces)}",
            f"With Mask: {with_mask_count} | Without: {without_mask_count}",
            f"Compliance: {compliance:.1f}%",
            f"Frame: {frame_count}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        # Log to CSV periodically
        current_time = time.time()
        if current_time - last_log_time >= LOG_INTERVAL:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            compliance_pct = round(compliance, 1)
            
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, with_mask_count, without_mask_count, compliance_pct])
            
            print(f"[{timestamp}] Logged: {with_mask_count} with mask, {without_mask_count} without | {compliance_pct}% compliance")
            last_log_time = current_time
        
        # Display frame
        cv2.imshow('Face Mask Detection - Dashboard Integration', frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"[INFO] Screenshot saved: {screenshot_name}")

except KeyboardInterrupt:
    print("\n\n[INFO] Detection interrupted by user")

finally:
    # Final log entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = with_mask_count + without_mask_count
    compliance = (with_mask_count / total * 100) if total > 0 else 0
    
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, with_mask_count, without_mask_count, round(compliance, 1)])
    
    # Cleanup
    if USE_WEBCAM:
        cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("  DETECTION SUMMARY")
    print("=" * 70)
    print(f"  Total Frames: {frame_count}")
    print(f"  Faces with Mask: {with_mask_count}")
    print(f"  Faces without Mask: {without_mask_count}")
    print(f"  Overall Compliance: {compliance:.1f}%")
    print(f"  Log file: {LOG_FILE}")
    print("=" * 70)
