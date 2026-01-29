"""
FPGA-Accelerated Face Mask Detection Script
Uses FPGA for CNN inference acceleration

Features:
- FPGA-accelerated CNN inference
- Fallback to CPU if FPGA not available
- Performance comparison metrics
- Real-time detection with statistics
"""

import cv2
import numpy as np
from tensorflow import keras
import time
import os
import argparse
from datetime import datetime
import sys

# Add our_improvements to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from our_improvements.fpga_interface import FPGAInterface, FPGASimulator
from our_improvements.dashboard import ComplianceDashboard, AlertSystem

# ============== CONFIGURATION ==============
MODEL_PATH = "models/mask_detector_64x64.h5"
IMG_SIZE = 64
FPGA_PORT = None  # Auto-detect, or set to 'COM3', '/dev/ttyUSB0', etc.


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Face Mask Detection - FPGA')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to model')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--port', type=str, default=FPGA_PORT, help='FPGA serial port')
    parser.add_argument('--simulate', action='store_true', help='Use FPGA simulator')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--log', action='store_true', help='Enable logging')
    return parser.parse_args()


class FPSTracker:
    """Track FPS with smoothing"""
    def __init__(self, smoothing=30):
        self.smoothing = smoothing
        self.times = []
        self.fps = 0
    
    def update(self):
        current_time = time.time()
        self.times.append(current_time)
        if len(self.times) > self.smoothing:
            self.times.pop(0)
        if len(self.times) > 1:
            self.fps = len(self.times) / (self.times[-1] - self.times[0])
        return self.fps


class HybridMaskDetector:
    """
    Hybrid Face Mask Detection using FPGA + CPU
    
    Uses FPGA for CNN inference when available, falls back to CPU otherwise.
    Haar cascade runs on CPU for face detection.
    """
    
    def __init__(self, model_path: str, fpga_interface=None, use_simulator: bool = False):
        self.img_size = IMG_SIZE
        self.model = None
        self.face_cascade = None
        self.fpga = fpga_interface
        self.use_fpga = False
        
        # Statistics
        self.cpu_inferences = 0
        self.fpga_inferences = 0
        self.cpu_total_time = 0.0
        self.fpga_total_time = 0.0
        
        # Load face detector (always CPU)
        print("[INFO] Loading face detector...")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load CNN model for CPU fallback
        print("[INFO] Loading mask detection model...")
        try:
            self.model = keras.models.load_model(model_path)
            print(f"[INFO] Model loaded from {model_path}")
        except Exception as e:
            print(f"[WARN] Could not load model: {e}")
            print("[INFO] Will use FPGA only if available")
        
        # Setup FPGA
        if fpga_interface:
            self.fpga = fpga_interface
            self.use_fpga = fpga_interface.is_connected()
        elif use_simulator:
            print("[INFO] Using FPGA simulator (5ms latency)")
            self.fpga = FPGASimulator(latency_ms=5.0)
            self.use_fpga = True
        
        mode = "FPGA" if self.use_fpga else "CPU"
        print(f"[INFO] Detector initialized - Mode: {mode}")
    
    def detect_faces(self, gray_frame):
        """Detect faces using Haar cascade (CPU)"""
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def predict_mask_cpu(self, face_roi) -> tuple:
        """Predict mask using CPU (TensorFlow)"""
        start = time.time()
        
        face = cv2.resize(face_roi, (self.img_size, self.img_size))
        face = face.reshape(1, self.img_size, self.img_size, 1) / 255.0
        
        prediction = self.model.predict(face, verbose=0)
        
        elapsed = time.time() - start
        self.cpu_inferences += 1
        self.cpu_total_time += elapsed
        
        mask_prob = prediction[0][0]
        no_mask_prob = prediction[0][1]
        
        if mask_prob > no_mask_prob:
            return True, mask_prob, elapsed
        else:
            return False, no_mask_prob, elapsed
    
    def predict_mask_fpga(self, face_roi) -> tuple:
        """Predict mask using FPGA acceleration"""
        start = time.time()
        
        face = cv2.resize(face_roi, (self.img_size, self.img_size))
        face_normalized = face.astype(np.float32) / 255.0
        
        result = self.fpga.process_image(face_normalized)
        
        elapsed = time.time() - start
        self.fpga_inferences += 1
        self.fpga_total_time += elapsed
        
        if result is None:
            # Fallback to CPU
            return self.predict_mask_cpu(face_roi)
        
        class_id, confidence = result
        is_mask = (class_id == 0)
        
        return is_mask, confidence, elapsed
    
    def predict_mask(self, face_roi) -> tuple:
        """Predict mask using best available method"""
        if self.use_fpga and self.fpga:
            return self.predict_mask_fpga(face_roi)
        elif self.model:
            return self.predict_mask_cpu(face_roi)
        else:
            return True, 0.5, 0.0  # Default if nothing available
    
    def process_frame(self, frame):
        """Process a frame and return annotated result"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)
        
        mask_count = 0
        no_mask_count = 0
        total_inference_time = 0.0
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            is_mask, confidence, inf_time = self.predict_mask(face_roi)
            total_inference_time += inf_time
            
            if is_mask:
                mask_count += 1
                label = f"Mask: {confidence*100:.1f}%"
                color = (0, 255, 0)
            else:
                no_mask_count += 1
                label = f"No Mask: {confidence*100:.1f}%"
                color = (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y-25), (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, mask_count, no_mask_count, total_inference_time
    
    def get_performance_stats(self) -> dict:
        """Get performance comparison statistics"""
        cpu_avg = (self.cpu_total_time / self.cpu_inferences * 1000) if self.cpu_inferences > 0 else 0
        fpga_avg = (self.fpga_total_time / self.fpga_inferences * 1000) if self.fpga_inferences > 0 else 0
        
        return {
            'cpu_inferences': self.cpu_inferences,
            'fpga_inferences': self.fpga_inferences,
            'cpu_avg_ms': cpu_avg,
            'fpga_avg_ms': fpga_avg,
            'speedup': cpu_avg / fpga_avg if fpga_avg > 0 else 0
        }


def draw_stats(frame, fps, mask_count, no_mask_count, inference_time, mode, perf_stats):
    """Draw statistics overlay"""
    total = mask_count + no_mask_count
    compliance = (mask_count / total * 100) if total > 0 else 0
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (280, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Mode indicator
    mode_color = (0, 255, 0) if mode == 'FPGA' else (255, 165, 0)
    cv2.putText(frame, f"Mode: {mode}", (15, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    
    # Stats
    stats = [
        (f"FPS: {fps:.1f}", (255, 255, 0)),
        (f"Inference: {inference_time*1000:.1f}ms", (255, 255, 0)),
        (f"With Mask: {mask_count}", (0, 255, 0)),
        (f"Without Mask: {no_mask_count}", (0, 0, 255)),
        (f"Total: {total}", (255, 255, 255)),
        (f"Compliance: {compliance:.1f}%", (0, 255, 255) if compliance >= 80 else (0, 165, 255))
    ]
    
    for i, (text, color) in enumerate(stats):
        cv2.putText(frame, text, (15, 50 + i*20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Performance comparison (if available)
    if perf_stats['fpga_inferences'] > 0 and perf_stats['cpu_inferences'] > 0:
        speedup_text = f"FPGA Speedup: {perf_stats['speedup']:.1f}x"
        cv2.putText(frame, speedup_text, (15, 170), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return frame


def main():
    args = parse_args()
    
    # Create directories
    os.makedirs('logs/snapshots', exist_ok=True)
    
    # Setup FPGA interface
    fpga = None
    if not args.simulate and args.port:
        try:
            fpga = FPGAInterface(port=args.port)
        except Exception as e:
            print(f"[WARN] FPGA not available: {e}")
    
    # Initialize detector
    detector = HybridMaskDetector(
        model_path=args.model,
        fpga_interface=fpga,
        use_simulator=args.simulate
    )
    
    # Initialize dashboard (if logging enabled)
    dashboard = None
    alert_system = None
    if args.log:
        dashboard = ComplianceDashboard()
        alert_system = AlertSystem(threshold=70.0)
    
    # Open camera
    print(f"[INFO] Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # FPS tracker
    fps_tracker = FPSTracker()
    
    mode = "FPGA" if detector.use_fpga else "CPU"
    print(f"[INFO] Starting detection (Mode: {mode})...")
    print("[INFO] Press 'q' to quit, 's' to save snapshot, 'f' to toggle FPGA/CPU")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated, mask_count, no_mask_count, inf_time = detector.process_frame(frame)
            
            # Update FPS
            fps = fps_tracker.update()
            
            # Get performance stats
            perf_stats = detector.get_performance_stats()
            
            # Draw overlay
            mode = "FPGA" if detector.use_fpga else "CPU"
            final_frame = draw_stats(annotated, fps, mask_count, no_mask_count, 
                                    inf_time, mode, perf_stats)
            
            # Log if enabled
            if dashboard:
                compliance = dashboard.log_detection(mask_count, no_mask_count, fps, mode)
                
                # Check alerts
                if alert_system:
                    alert = alert_system.check(compliance)
                    if alert:
                        print(alert)
            
            # Display
            cv2.imshow('Face Mask Detection - FPGA Accelerated', final_frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"logs/snapshots/snapshot_{timestamp}.jpg"
                cv2.imwrite(path, final_frame)
                print(f"[INFO] Snapshot saved to {path}")
            elif key == ord('f'):
                # Toggle FPGA/CPU mode
                detector.use_fpga = not detector.use_fpga
                mode = "FPGA" if detector.use_fpga else "CPU"
                print(f"[INFO] Switched to {mode} mode")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if fpga:
            fpga.close()
        
        # Print summary
        print("\n" + "=" * 50)
        print("PERFORMANCE SUMMARY")
        print("=" * 50)
        stats = detector.get_performance_stats()
        print(f"CPU Inferences: {stats['cpu_inferences']}")
        print(f"FPGA Inferences: {stats['fpga_inferences']}")
        print(f"CPU Avg Latency: {stats['cpu_avg_ms']:.2f} ms")
        print(f"FPGA Avg Latency: {stats['fpga_avg_ms']:.2f} ms")
        if stats['speedup'] > 0:
            print(f"FPGA Speedup: {stats['speedup']:.2f}x")
        print("=" * 50)
        
        if dashboard:
            dashboard.print_session_summary()


if __name__ == "__main__":
    main()
