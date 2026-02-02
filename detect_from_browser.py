"""
Browser-based Face Mask Detection
Processes frames from browser webcam via API endpoint
"""

from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from datetime import datetime
import csv
import os

try:
    from tensorflow import keras
    HAS_TF = True
except:
    HAS_TF = False
    print("WARNING: TensorFlow not available, using dummy detection")

app = Flask(__name__)

# Configuration
MODEL_PATH = 'models/mask_detector_mobilenet.h5'
LOG_FILE = 'logs/compliance_log.csv'
IMG_SIZE = 128

# Load model
if HAS_TF and os.path.exists(MODEL_PATH):
    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"✓ Model loaded: {MODEL_PATH}")
    except:
        model = None
        print("✗ Model load failed")
else:
    model = None
    print("✗ No model available")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Counters
total_with_mask = 0
total_without_mask = 0

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """Process a frame from browser webcam"""
    global total_with_mask, total_without_mask
    
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        
        detections = []
        frame_with_mask = 0
        frame_without_mask = 0
        
        for (x, y, w, h) in faces:
            if model is not None:
                # Extract and preprocess face
                face_roi = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                face_array = np.array(face_resized, dtype="float32") / 255.0
                face_array = np.expand_dims(face_array, axis=0)
                
                # Predict
                predictions = model.predict(face_array, verbose=0)
                mask_prob = predictions[0][1]
                has_mask = mask_prob > 0.5
                confidence = mask_prob if has_mask else (1 - mask_prob)
            else:
                # Dummy detection
                has_mask = (x + y) % 2 == 0
                confidence = 0.85
            
            if has_mask:
                frame_with_mask += 1
                total_with_mask += 1
            else:
                frame_without_mask += 1
                total_without_mask += 1
            
            detections.append({
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h),
                'has_mask': bool(has_mask),
                'confidence': float(confidence * 100)
            })
        
        # Log to CSV
        total = total_with_mask + total_without_mask
        compliance = (total_with_mask / total * 100) if total > 0 else 0
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, total_with_mask, total_without_mask, round(compliance, 1)])
        
        return jsonify({
            'success': True,
            'detections': detections,
            'stats': {
                'total_with_mask': total_with_mask,
                'total_without_mask': total_without_mask,
                'compliance': round(compliance, 1)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    print("Starting detection API on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=False)
