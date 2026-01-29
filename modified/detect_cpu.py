"""
CPU-Only Face Mask Detection Script
Real-time detection using webcam with performance metrics

Features:
- Haar Cascade face detection
- CNN-based mask classification
- FPS tracking and display
- Detection statistics overlay
- Press 'q' to quit, 's' to save snapshot
"""

import cv2
import numpy as np
from tensorflow import keras
import time
import os
import argparse
from datetime import datetime

# ============== CONFIGURATION ==============
MODEL_PATH = "models/mask_detector_64x64.h5"
IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.5


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Face Mask Detection - CPU')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to model')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--save_dir', type=str, default='logs/snapshots', help='Snapshot save directory')
    return parser.parse_args()


class FPSTracker:
    """Track and smooth FPS calculations"""
    def __init__(self, smoothing=30):
        self.smoothing = smoothing
        self.times = []
        self.fps = 0
    
    def update(self):
        current_time = time.time()
        self.times.append(current_time)
        
        # Keep only recent times
        if len(self.times) > self.smoothing:
            self.times.pop(0)
        
        # Calculate FPS
        if len(self.times) > 1:
            self.fps = len(self.times) / (self.times[-1] - self.times[0])
        
        return self.fps


class MaskDetector:
    """Face Mask Detection using CNN"""
    
    def __init__(self, model_path, img_size=64):
        self.img_size = img_size
        self.model = None
        self.face_cascade = None
        
        # Load model
        print("[INFO] Loading mask detection model...")
        try:
            self.model = keras.models.load_model(model_path)
            print(f"[INFO] Model loaded from {model_path}")
        except Exception as e:
            print(f"[ERROR] Could not load model: {e}")
            print("[INFO] Please run 'python modified/train_simplified.py' first")
            raise
        
        # Load face detector
        print("[INFO] Loading face detector...")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise Exception("Could not load Haar cascade classifier")
        
        print("[INFO] Detector initialized successfully")
    
    def detect_faces(self, gray_frame):
        """Detect faces in grayscale frame"""
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def predict_mask(self, face_roi):
        """Predict mask/no-mask for a face ROI"""
        # Resize and normalize
        face = cv2.resize(face_roi, (self.img_size, self.img_size))
        face = face.reshape(1, self.img_size, self.img_size, 1) / 255.0
        
        # Predict
        prediction = self.model.predict(face, verbose=0)
        
        # Return (is_mask, confidence)
        mask_prob = prediction[0][0]
        no_mask_prob = prediction[0][1]
        
        if mask_prob > no_mask_prob:
            return True, mask_prob
        else:
            return False, no_mask_prob
    
    def process_frame(self, frame):
        """Process a single frame and return annotated frame with stats"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detect_faces(gray)
        
        # Statistics
        mask_count = 0
        no_mask_count = 0
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Predict mask
            is_mask, confidence = self.predict_mask(face_roi)
            
            # Update counts and set display properties
            if is_mask:
                mask_count += 1
                label = f"Mask: {confidence*100:.1f}%"
                color = (0, 255, 0)  # Green
            else:
                no_mask_count += 1
                label = f"No Mask: {confidence*100:.1f}%"
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y-25), (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, mask_count, no_mask_count


def draw_stats(frame, fps, mask_count, no_mask_count, inference_time):
    """Draw statistics overlay on frame"""
    total = mask_count + no_mask_count
    compliance = (mask_count / total * 100) if total > 0 else 0
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (220, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw stats
    stats = [
        f"FPS: {fps:.1f}",
        f"Inference: {inference_time*1000:.1f}ms",
        f"With Mask: {mask_count}",
        f"Without Mask: {no_mask_count}",
        f"Total: {total}",
        f"Compliance: {compliance:.1f}%"
    ]
    
    colors = [
        (255, 255, 0),   # FPS - Cyan
        (255, 255, 0),   # Inference - Cyan
        (0, 255, 0),     # Mask - Green
        (0, 0, 255),     # No Mask - Red
        (255, 255, 255), # Total - White
        (0, 255, 255) if compliance >= 80 else (0, 165, 255)  # Compliance
    ]
    
    for i, (stat, color) in enumerate(zip(stats, colors)):
        cv2.putText(frame, stat, (15, 30 + i*20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame


def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize detector
    try:
        detector = MaskDetector(args.model)
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    # Initialize video capture
    print(f"[INFO] Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        print("[INFO] Try specifying a different camera index with --camera")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Initialize FPS tracker
    fps_tracker = FPSTracker()
    
    print("[INFO] Starting detection... Press 'q' to quit, 's' to save snapshot")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame")
                break
            
            # Measure inference time
            start_time = time.time()
            
            # Process frame
            annotated_frame, mask_count, no_mask_count = detector.process_frame(frame)
            
            inference_time = time.time() - start_time
            
            # Update FPS
            fps = fps_tracker.update()
            
            # Draw statistics
            final_frame = draw_stats(annotated_frame, fps, mask_count, no_mask_count, inference_time)
            
            # Display
            cv2.imshow('Face Mask Detection - CPU', final_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("[INFO] Quitting...")
                break
            elif key == ord('s'):
                # Save snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_path = os.path.join(args.save_dir, f"snapshot_{timestamp}.jpg")
                cv2.imwrite(snapshot_path, final_frame)
                print(f"[INFO] Snapshot saved to {snapshot_path}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Detection stopped")


if __name__ == "__main__":
    main()
