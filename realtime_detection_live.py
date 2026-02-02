"""
Real-Time Face Mask Detection with Live Dashboard Integration
Detects faces from webcam and updates dashboard in real-time
"""

import cv2
import numpy as np
from tensorflow import keras
import os
from datetime import datetime
import csv
import time
import sys

print("=" * 70)
print("  REAL-TIME FACE MASK DETECTION - LIVE DASHBOARD")
print("=" * 70)

# Configuration
MODEL_PATH = '/app/models/mask_detector_mobilenet.h5'
LOG_FILE = '/app/logs/compliance_log.csv'
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.5
LOG_INTERVAL = 2  # Log every 2 seconds

# Create logs directory
os.makedirs('/app/logs', exist_ok=True)

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
    print(f"[ERROR] Model not found: {MODEL_PATH}")
    # Try alternative models
    alt_models = [
        '/app/models/mask_detector_128x128_final.h5',
        '/app/models/best_model.h5',
    ]
    
    for alt_model in alt_models:
        if os.path.exists(alt_model):
            MODEL_PATH = alt_model
            print(f"[SUCCESS] Using alternative model: {MODEL_PATH}")
            break
    else:
        print(f"[ERROR] No model found!")
        sys.exit(1)

model = keras.models.load_model(MODEL_PATH)
print(f"[SUCCESS] Model loaded: {MODEL_PATH}")

# Open webcam
print(f"\n[INFO] Opening video capture (camera 0)...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[WARNING] Cannot open camera 0, trying camera 1...")
    cap = cv2.VideoCapture(1)
    
if not cap.isOpened():
    print("[ERROR] No camera available!")
    print("[INFO] Running in TEST MODE with demo data...")
    USE_CAMERA = False
else:
    print("[SUCCESS] Camera opened successfully")
    USE_CAMERA = True
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

# Detection stats
last_log_time = time.time()
with_mask_count = 0
without_mask_count = 0
frame_count = 0

print("\n" + "=" * 70)
print("  DETECTION STARTED - LIVE DASHBOARD INTEGRATION")
print("=" * 70)
print("  Press 'q' to quit")
print("  Press 's' to save screenshot")
print("  Dashboard: http://localhost:9000")
print("  Logging interval:", LOG_INTERVAL, "seconds")
print("=" * 70 + "\n")

try:
    while True:
        frame_count += 1
        
        if USE_CAMERA:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break
        else:
            # Test mode - create synthetic frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Simulate detections for testing
            if frame_count % 30 < 20:
                with_mask_count += 1
            else:
                without_mask_count += 1
            time.sleep(0.033)  # ~30 FPS
        
        temp_with_mask = 0
        temp_without_mask = 0
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                # Preprocess for model
                face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                face_array = np.array(face_resized, dtype="float32") / 255.0
                face_array = np.expand_dims(face_array, axis=0)
                
                # Predict
                predictions = model.predict(face_array, verbose=0)
                
                # Determine label (MobileNetV2 has 2 classes: 0=without, 1=with)
                if len(predictions[0]) == 2:
                    mask_prob = predictions[0][1]  # Probability of with_mask
                    has_mask = mask_prob > CONFIDENCE_THRESHOLD
                    confidence = mask_prob if has_mask else (1 - mask_prob)
                    label = "Mask" if has_mask else "No Mask"
                else:
                    mask_prob = predictions[0][0]
                    has_mask = mask_prob > CONFIDENCE_THRESHOLD
                    confidence = mask_prob if has_mask else (1 - mask_prob)
                    label = "Mask" if has_mask else "No Mask"
                
                # Update counters
                if has_mask:
                    temp_with_mask += 1
                    color = (0, 255, 0)  # Green
                else:
                    temp_without_mask += 1
                    color = (0, 0, 255)  # Red
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                label_text = f"{label}: {confidence*100:.1f}%"
                
                # Background for text
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y-30), (x+text_width+10, y), color, -1)
                cv2.putText(frame, label_text, (x+5, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"[WARNING] Error processing face: {e}")
                continue
        
        # Update cumulative counts
        with_mask_count += temp_with_mask
        without_mask_count += temp_without_mask
        
        # Display statistics on frame
        total = with_mask_count + without_mask_count
        compliance = (with_mask_count / total * 100) if total > 0 else 0
        
        # Stats overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        stats_text = [
            f"SECURE GUARD PRO - Live Detection",
            f"Faces: {len(faces)} | Frame: {frame_count}",
            f"With Mask: {with_mask_count} | Without: {without_mask_count}",
            f"Compliance: {compliance:.1f}%",
        ]
        
        y_offset = 35
        for text in stats_text:
            cv2.putText(frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 25
        
        # Log to CSV periodically
        current_time = time.time()
        if current_time - last_log_time >= LOG_INTERVAL:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            compliance_pct = round(compliance, 1)
            
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, with_mask_count, without_mask_count, compliance_pct])
            
            print(f"[{timestamp}] ✓ Logged: {with_mask_count} masked, {without_mask_count} unmasked | {compliance_pct}% compliance | Faces: {len(faces)}")
            last_log_time = current_time
        
        # Display frame
        if USE_CAMERA:
            cv2.imshow('Face Mask Detection - Live Dashboard', frame)
        
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_name = f"/app/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(screenshot_name, frame)
                print(f"[INFO] Screenshot saved: {screenshot_name}")
        else:
            # In test mode, just continue without display
            time.sleep(0.1)
            if frame_count > 1000:  # Limit test mode
                break

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
    if USE_CAMERA:
        cap.release()
        cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("  DETECTION SUMMARY")
    print("=" * 70)
    print(f"  Total Frames Processed: {frame_count}")
    print(f"  Faces with Mask: {with_mask_count}")
    print(f"  Faces without Mask: {without_mask_count}")
    print(f"  Overall Compliance: {compliance:.1f}%")
    print(f"  Log file: {LOG_FILE}")
    print(f"  Dashboard: http://localhost:9000")
    print("=" * 70)
