"""
Real-Time Face Mask Detection for Windows
Live camera feed with confidence display and dashboard integration

Features:
- Real-time detection with OpenCV
- Confidence percentages for each face
- Green boxes for MASK, Red boxes for NO MASK
- Logs to CSV every 2 seconds for dashboard
- Keyboard controls: Q=quit, S=screenshot, R=reset

Usage: python realtime_detection_windows.py
"""

import cv2
import numpy as np
import os
from datetime import datetime
import csv
import time
import sys

def non_max_suppression(boxes, scores, threshold=0.3):
    """Remove overlapping face detections using Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    
    return keep

# Check TensorFlow
try:
    from tensorflow import keras
    import tensorflow as tf
    print("[OK] TensorFlow loaded successfully")
except ImportError:
    print("[X] TensorFlow not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    from tensorflow import keras
    import tensorflow as tf
    print("[OK] TensorFlow installed and loaded")

print("\n" + "=" * 70)
print("  SECURE GUARD PRO - REAL-TIME FACE MASK DETECTION")
print("=" * 70)
print("  Windows Live Detection System")
print("  Model: MobileNetV2 (99.01% accuracy)")
print("=" * 70 + "\n")

# ===== CONFIGURATION =====
MODEL_PATH = 'models/mask_detector_mobilenet.h5'
LOG_FILE = 'logs/compliance_log.csv'
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.70  # Increased to 70% for higher accuracy
MIN_FACE_SIZE = 120  # Minimum face size - ignores background/distant faces
NMS_THRESHOLD = 0.3  # Non-maximum suppression threshold
LOG_INTERVAL = 2  # Log every 2 seconds
CAMERA_ID = 0  # Change to 1 if camera 0 doesn't work
DETECT_LARGEST_ONLY = True  # Only detect closest/largest face

# Create logs directory
os.makedirs('logs', exist_ok=True)

# ===== INITIALIZE CSV LOG =====
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'With_Mask', 'Without_Mask', 'Compliance_Percent'])
    print(f"[INFO] Created log file: {LOG_FILE}")
else:
    print(f"[INFO] Using existing log file: {LOG_FILE}")

# ===== LOAD FACE DETECTOR =====
print(f"\n[INFO] Loading Haar Cascade face detector...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("[ERROR] Could not load face detector!")
    sys.exit(1)
print("[SUCCESS] Face detector loaded")

# ===== LOAD MASK DETECTION MODEL =====
print(f"[INFO] Loading mask detection model...")
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found: {MODEL_PATH}")
    print(f"\n[INFO] Trying alternative models...")
    
    # Try alternative models
    alt_models = [
        'models/best_model.h5',
        'models/mask_detector_128x128_final.h5',
        'models/mask_detector_64x64_best.h5'
    ]
    
    for alt_model in alt_models:
        if os.path.exists(alt_model):
            MODEL_PATH = alt_model
            IMG_SIZE = 64 if '64x64' in alt_model else 128
            print(f"[SUCCESS] Found alternative: {MODEL_PATH}")
            break
    else:
        print(f"\n[ERROR] No model found! Copy from Docker with:")
        print(f"  docker cp mask-detection-dev:/app/models/mask_detector_mobilenet.h5 models/")
        sys.exit(1)

try:
    # Load model without compilation to avoid version issues
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(f"[SUCCESS] Model loaded: {os.path.basename(MODEL_PATH)}")
    print(f"[INFO] Input size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"[INFO] Confidence threshold: {CONFIDENCE_THRESHOLD*100}%")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print(f"\n[INFO] Trying alternative loading method...")
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        print(f"[SUCCESS] ✓ Model loaded with safe_mode=False")
    except Exception as e2:
        print(f"[ERROR] Alternative method also failed: {e2}")
        sys.exit(1)

# ===== OPEN WEBCAM =====
print(f"\n[INFO] Opening camera {CAMERA_ID}...")
cap = cv2.VideoCapture(CAMERA_ID)

# Try different camera indices if first one fails
if not cap.isOpened():
    print(f"[WARNING] Camera {CAMERA_ID} failed, trying camera 1...")
    cap = cv2.VideoCapture(1)
    CAMERA_ID = 1

if not cap.isOpened():
    print("[ERROR] Could not open any camera!")
    print("[INFO] Available cameras:")
    for i in range(3):
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            print(f"  - Camera {i}: Available")
            test_cap.release()
        else:
            print(f"  - Camera {i}: Not available")
    sys.exit(1)

print(f"[SUCCESS] Camera {CAMERA_ID} opened")

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Get actual camera properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"[INFO] Camera resolution: {width}x{height} @ {fps}fps")

# ===== DETECTION STATS =====
last_log_time = time.time()
with_mask_count = 0
without_mask_count = 0
frame_count = 0
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

print("\n" + "=" * 70)
print("  DETECTION STARTED - LIVE MODE")
print("=" * 70)
print("  Controls:")
print("    'Q' - Quit detection")
print("    'S' - Save screenshot")
print("    'R' - Reset counters")
print()
print("  Dashboard: http://localhost:9000")
print("  Log file: " + LOG_FILE)
print("=" * 70 + "\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break
        
        frame_count += 1
        fps_counter += 1
        temp_with_mask = 0
        temp_without_mask = 0
        
        # Calculate FPS
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with improved parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Apply Non-Maximum Suppression and select largest face only
        if len(faces) > 0:
            # Calculate confidence scores based on face size
            scores = np.array([w * h for (x, y, w, h) in faces], dtype=np.float32)
            scores = scores / scores.max()  # Normalize to 0-1
            
            # Apply NMS
            keep_indices = non_max_suppression(faces, scores, NMS_THRESHOLD)
            faces = faces[keep_indices]
            
            # ONLY DETECT LARGEST FACE (closest person, ignore background)
            if DETECT_LARGEST_ONLY and len(faces) > 1:
                areas = [(x, y, w, h, w*h) for (x, y, w, h) in faces]
                largest = max(areas, key=lambda item: item[4])
                faces = np.array([[largest[0], largest[1], largest[2], largest[3]]])
        
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
                
                # Determine label - MODEL OUTPUT: [without_mask_prob, with_mask_prob]
                if len(predictions[0]) == 2:
                    without_mask_prob = predictions[0][0]
                    with_mask_prob = predictions[0][1]
                    
                    # Check if confidence is above threshold
                    max_confidence = max(with_mask_prob, without_mask_prob)
                    if max_confidence < CONFIDENCE_THRESHOLD:
                        continue  # Skip low-confidence predictions
                    
                    # FIXED: Use correct comparison
                    has_mask = (with_mask_prob > without_mask_prob)
                    confidence = with_mask_prob if has_mask else without_mask_prob
                    
                    # Debug output (remove later)
                    # print(f"Mask:{with_mask_prob:.2f} NoMask:{without_mask_prob:.2f} Result:{has_mask}")
                else:
                    mask_prob = predictions[0][0]
                    has_mask = mask_prob > CONFIDENCE_THRESHOLD
                    confidence = mask_prob if has_mask else (1 - mask_prob)
                
                # Update counters
                if has_mask:
                    temp_with_mask += 1
                    color = (0, 255, 0)  # Green
                    bg_color = (0, 200, 0)
                    label = "MASK"
                    emoji = "✓"
                else:
                    temp_without_mask += 1
                    color = (0, 0, 255)  # Red
                    bg_color = (0, 0, 200)
                    label = "NO MASK"
                    emoji = "✗"
                
                # Draw bounding box with shadow effect
                cv2.rectangle(frame, (x+2, y+2), (x+w+2, y+h+2), (0, 0, 0), 3)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                # Prepare label text
                label_text = f"{label}: {confidence*100:.1f}%"
                
                # Draw label background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(frame, (x, y-35), (x+text_width+10, y), bg_color, -1)
                cv2.rectangle(frame, (x, y-35), (x+text_width+10, y), color, 2)
                
                # Draw label text
                cv2.putText(frame, label_text, (x+5, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"[WARNING] Error processing face: {e}")
                continue
        
        # Update cumulative counts
        with_mask_count += temp_with_mask
        without_mask_count += temp_without_mask
        
        # Calculate statistics
        total = with_mask_count + without_mask_count
        compliance = (with_mask_count / total * 100) if total > 0 else 0
        
        # ===== DRAW UI OVERLAY =====
        
        # Draw statistics panel (top-left)
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (380, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (5, 5), (380, 150), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "SECURE GUARD PRO", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Stats - COMPACT
        stats_text = [
            f"FPS: {current_fps} | Frame: {frame_count} | Faces: {len(faces)}",
            f"Masked: {with_mask_count} | Unmasked: {without_mask_count}",
            f"Compliance: {compliance:.1f}% | Total: {total}",
            f"Confidence: >{CONFIDENCE_THRESHOLD*100:.0f}% | NMS: ON",
            f"Mode: CLOSEST PERSON ONLY"
        ]
        
        y_offset = 50
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 20
        
        # Status message
        if len(faces) == 0:
            status_msg = "[!] No faces - Move closer"
            status_col = (0, 165, 255)
        else:
            status_msg = "[OK] Detecting closest person"
            status_col = (0, 255, 0)
        cv2.putText(frame, status_msg, (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_col, 1, cv2.LINE_AA)
        
        # Draw compliance bar (bottom)
        bar_y = height - 50
        bar_width = width - 40
        bar_height = 30
        
        # Bar background
        cv2.rectangle(frame, (20, bar_y), (20 + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, bar_y), (20 + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Compliance fill
        if total > 0:
            fill_width = int((compliance / 100) * bar_width)
            if compliance >= 80:
                fill_color = (0, 255, 0)  # Green
            elif compliance >= 60:
                fill_color = (0, 165, 255)  # Orange
            else:
                fill_color = (0, 0, 255)  # Red
            cv2.rectangle(frame, (20, bar_y), (20 + fill_width, bar_y + bar_height), fill_color, -1)
        
        # Compliance text on bar
        cv2.putText(frame, f"Compliance: {compliance:.1f}%", (width//2 - 100, bar_y + 21),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw instructions (top-right)
        instructions = [
            "Q - Quit",
            "S - Screenshot",
            "R - Reset"
        ]
        inst_x = width - 150
        inst_y = 30
        for inst in instructions:
            cv2.putText(frame, inst, (inst_x, inst_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            inst_y += 20
        
        # ===== LOG TO CSV PERIODICALLY =====
        current_time = time.time()
        if current_time - last_log_time >= LOG_INTERVAL:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            compliance_pct = round(compliance, 1)
            
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, with_mask_count, without_mask_count, compliance_pct])
            
            print(f"[{timestamp}] LOGGED: {with_mask_count} masked, {without_mask_count} unmasked | {compliance_pct}% compliance | {len(faces)} faces")
            last_log_time = current_time
        
        # Display frame
        cv2.imshow('SECURE GUARD PRO - Real-Time Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n[INFO] Quitting...")
            break
        elif key == ord('s') or key == ord('S'):
            screenshot_name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"[INFO] Screenshot saved: {screenshot_name}")
        elif key == ord('r') or key == ord('R'):
            with_mask_count = 0
            without_mask_count = 0
            frame_count = 0
            print("[INFO] Counters reset")

except KeyboardInterrupt:
    print("\n\n[INFO] Detection interrupted by user (Ctrl+C)")

except Exception as e:
    print(f"\n[ERROR] Unexpected error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Final log entry
    try:
        total = with_mask_count + without_mask_count
        compliance = (with_mask_count / total * 100) if total > 0 else 0
    except:
        total = 0
        compliance = 0
    
    if total > 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, with_mask_count, without_mask_count, round(compliance, 1)])
        print(f"[INFO] Final entry logged")
    
    # Cleanup
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
    print("\n  [OK] System shutdown complete")
    print("  Thank you for using SECURE GUARD PRO")
