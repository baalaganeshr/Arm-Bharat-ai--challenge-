"""
Face Mask Detection - DEBUG VERSION with Enhanced Detection
Shows preprocessing and uses multiple detection methods
"""

import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime

print("=" * 60)
print("    FACE MASK DETECTION - DEBUG MODE")
print("    Enhanced Multi-Cascade Detection")
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

# Load face detectors
print("[INFO] Loading face detectors...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

print(f"[INFO] Frontal detector loaded: {not face_cascade.empty()}")
print(f"[INFO] Alt detector loaded: {not face_cascade_alt.empty()}")

# Open camera
print("[INFO] Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        exit(1)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

print("[SUCCESS] Camera opened!")
print("\n" + "=" * 60)
print("CONTROLS:")
print("  'q' - Quit")
print("  's' - Save snapshot")
print("  'd' - Toggle debug view")
print("=" * 60 + "\n")

def detect_mask(face_gray, face_id=None):
    """Enhanced mask detection with better accuracy"""
    h, w = face_gray.shape
    
    # Divide face into regions - KEY: mask covers LOWER half
    upper_half = face_gray[0:int(h*0.45), :]       # Eyes/forehead (NOT covered)
    lower_half = face_gray[int(h*0.45):, :]        # Nose/mouth/chin (COVERED by mask)
    
    # Apply preprocessing
    upper_half = cv2.GaussianBlur(upper_half, (5, 5), 0)
    lower_half = cv2.GaussianBlur(lower_half, (5, 5), 0)
    
    # Histogram equalization
    upper_eq = cv2.equalizeHist(upper_half)
    lower_eq = cv2.equalizeHist(lower_half)
    
    # Calculate texture measures
    upper_std = np.std(upper_eq)
    lower_std = np.std(lower_eq)
    
    upper_var = np.var(upper_eq)
    lower_var = np.var(lower_eq)
    
    # Edge detection - masks have fewer natural face edges
    edges_upper = cv2.Canny(upper_eq, 50, 150)
    edges_lower = cv2.Canny(lower_eq, 50, 150)
    
    edge_density_upper = np.sum(edges_upper > 0) / edges_upper.size
    edge_density_lower = np.sum(edges_lower > 0) / edges_lower.size
    
    # Color uniformity - masks are typically solid colors
    # Calculate local standard deviation (texture)
    kernel = np.ones((5,5), np.float32) / 25
    lower_smoothed = cv2.filter2D(lower_eq, -1, kernel)
    uniformity = np.std(lower_eq - lower_smoothed)
    
    # Gradient magnitude - skin has more micro-variations
    grad_x_lower = cv2.Sobel(lower_eq, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_lower = cv2.Sobel(lower_eq, cv2.CV_64F, 0, 1, ksize=3)
    gradient_lower = np.mean(np.sqrt(grad_x_lower**2 + grad_y_lower**2))
    
    # SCORING SYSTEM - Multiple strong indicators
    score = 0.0
    reasons = []
    
    # 1. TEXTURE RATIO - Masks have uniform texture (STRONG indicator)
    texture_ratio = (upper_std + 1) / (lower_std + 1)
    if texture_ratio > 1.5:
        score += 0.30
        reasons.append("uniform_lower")
    elif texture_ratio > 1.2:
        score += 0.20
    
    # 2. EDGE DENSITY - Masks have fewer detailed edges
    edge_ratio = (edge_density_lower + 0.001) / (edge_density_upper + 0.001)
    if edge_ratio < 0.5:
        score += 0.25
        reasons.append("few_edges")
    elif edge_ratio < 0.7:
        score += 0.15
    
    # 3. VARIANCE - Lower variance indicates mask
    if lower_var < upper_var * 0.6:
        score += 0.20
        reasons.append("low_variance")
    elif lower_var < upper_var * 0.8:
        score += 0.10
    
    # 4. UNIFORMITY - Masks are uniform (low micro-texture)
    if uniformity < 10:
        score += 0.15
        reasons.append("very_uniform")
    elif uniformity < 15:
        score += 0.08
    
    # 5. GRADIENT - Low gradient = flat surface (mask)
    if gradient_lower < 12:
        score += 0.10
        reasons.append("low_gradient")
    
    # Determine result with LOWER threshold (more sensitive to masks)
    has_mask = score > 0.50  # If score > 50%, it's a mask
    
    # Convert score to percentage confidence
    if has_mask:
        confidence = min(95, max(60, int(score * 100)))
    else:
        confidence = min(95, max(60, int((1 - score) * 100)))
    
    # Temporal smoothing
    if face_id is not None:
        if face_id not in mask_history:
            mask_history[face_id] = []
        
        mask_history[face_id].append(has_mask)
        if len(mask_history[face_id]) > 5:
            mask_history[face_id].pop(0)
        
        # Majority vote over last 5 frames
        if len(mask_history[face_id]) >= 3:
            mask_count = sum(mask_history[face_id])
            has_mask = mask_count > len(mask_history[face_id]) / 2
    
    return has_mask, confidence
    
    # Temporal smoothing if face_id provided
    if face_id is not None:
        if face_id not in mask_history:
            mask_history[face_id] = []
        
        mask_history[face_id].append(has_mask)
        if len(mask_history[face_id]) > 5:
            mask_history[face_id].pop(0)
        
        # Use majority vote
        if len(mask_history[face_id]) >= 3:
            mask_count_hist = sum(mask_history[face_id])
            has_mask = mask_count_hist > len(mask_history[face_id]) / 2
    
    return has_mask, confidence

# Statistics
total_with_mask = 0
total_without_mask = 0
fps_start_time = time.time()
fps_counter = 0
current_fps = 0
last_log_time = time.time()
debug_mode = False

# Temporal smoothing for stable detection
previous_faces = []
face_buffer = []  # Buffer for smoothing
BUFFER_SIZE = 3
mask_history = {}  # Track mask status per face

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame")
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    
    # Use only the best detector to avoid duplicates
    faces = face_cascade.detectMultiScale(
        gray_enhanced,
        scaleFactor=1.08,
        minNeighbors=6,
        minSize=(100, 100),
        maxSize=(500, 500),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # If no faces found, try more sensitive settings
    if len(faces) == 0:
        faces = face_cascade_alt.detectMultiScale(
            gray_enhanced,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(80, 80),
            maxSize=(500, 500),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
    # Filter overlapping detections
    if len(faces) > 1:
        filtered_faces = []
        for i, (x1, y1, w1, h1) in enumerate(faces):
            is_duplicate = False
            for j, (x2, y2, w2, h2) in enumerate(faces):
                if i != j:
                    # Check overlap
                    overlap_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                    overlap_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
                    overlap_area = overlap_x * overlap_y
                    area1 = w1 * h1
                    
                    if overlap_area > area1 * 0.3:  # 30% overlap
                        if area1 < w2 * h2:  # Keep larger detection
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                # Quality check: reject faces that are too small or oddly shaped
                aspect_ratio = w1 / h1
                if 0.7 <= aspect_ratio <= 1.3 and w1 >= 100:  # Reasonable face proportions
                    filtered_faces.append([x1, y1, w1, h1])
        
        faces = np.array(filtered_faces) if len(filtered_faces) > 0 else faces
    else:
        # Single face quality check
        if len(faces) > 0:
            x1, y1, w1, h1 = faces[0]
            aspect_ratio = w1 / h1
            if not (0.7 <= aspect_ratio <= 1.3 and w1 >= 100):
                faces = []
    
    faces = np.array(faces) if len(faces) > 0 and not isinstance(faces, np.ndarray) else faces
    
    # Temporal smoothing: add to buffer
    face_buffer.append(faces)
    if len(face_buffer) > BUFFER_SIZE:
        face_buffer.pop(0)
    
    # Use smoothed detections if buffer is full
    if len(face_buffer) >= BUFFER_SIZE:
        # Average face positions across frames for stability
        all_buffered_faces = []
        for buffered_faces in face_buffer:
            if len(buffered_faces) > 0:
                all_buffered_faces.extend(buffered_faces)
        
        if len(all_buffered_faces) > 0:
            # Cluster nearby faces
            smoothed_faces = []
            used = [False] * len(all_buffered_faces)
            
            for i, (x1, y1, w1, h1) in enumerate(all_buffered_faces):
                if used[i]:
                    continue
                
                cluster = [(x1, y1, w1, h1)]
                used[i] = True
                
                for j, (x2, y2, w2, h2) in enumerate(all_buffered_faces):
                    if not used[j]:
                        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                        if dist < 50:  # Close enough to be same face
                            cluster.append((x2, y2, w2, h2))
                            used[j] = True
                
                # Average the cluster
                avg_x = int(np.mean([f[0] for f in cluster]))
                avg_y = int(np.mean([f[1] for f in cluster]))
                avg_w = int(np.mean([f[2] for f in cluster]))
                avg_h = int(np.mean([f[3] for f in cluster]))
                smoothed_faces.append([avg_x, avg_y, avg_w, avg_h])
            
            faces = np.array(smoothed_faces)
    
    mask_count = 0
    no_mask_count = 0
    
    # Process each face
    for idx, (x, y, w, h) in enumerate(faces):
        # Generate face ID based on position
        face_id = f"{x//50}_{y//50}"  # Grid-based ID for tracking
        
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
        
        # Detect mask with temporal smoothing
        has_mask, confidence = detect_mask(face_gray, face_id)
        
        if has_mask:
            mask_count += 1
            label = f"MASK: {confidence}%"
            color = (0, 255, 0)  # Green
            total_with_mask += 1
        else:
            no_mask_count += 1
            label = f"NO MASK: {confidence}%"
            color = (0, 0, 255)  # Red
            total_without_mask += 1
        
        # Draw rectangle
        thickness = 3
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
        # Draw label
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
    
    # Log to CSV
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
    
    # Draw overlay with stats
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 280), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Header
    cv2.putText(frame, "FACE MASK DETECTION - DEBUG MODE", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(frame, (20, 50), (430, 50), (100, 255, 100), 2)
    
    # Statistics
    y_pos = 80
    line_height = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, y_pos), font, 0.7, (0, 255, 255), 2)
    y_pos += line_height
    
    cv2.putText(frame, f"Faces Detected: {len(faces)}", (20, y_pos), font, 0.7, (255, 255, 255), 2)
    y_pos += line_height
    
    if len(faces) == 0:
        cv2.putText(frame, "No faces detected!", (20, y_pos), font, 0.6, (0, 255, 255), 2)
        y_pos += line_height
        cv2.putText(frame, "Tips:", (20, y_pos), font, 0.6, (255, 200, 100), 2)
        y_pos += line_height
        cv2.putText(frame, "- Face camera directly", (30, y_pos), font, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        cv2.putText(frame, "- Good lighting needed", (30, y_pos), font, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        cv2.putText(frame, "- Stay 2-3 feet away", (30, y_pos), font, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(frame, f"With Mask: {mask_count}", (20, y_pos), font, 0.7, (0, 255, 0), 2)
        y_pos += line_height
        
        cv2.putText(frame, f"Without Mask: {no_mask_count}", (20, y_pos), font, 0.7, (0, 0, 255), 2)
        y_pos += line_height
        
        # Total stats
        total = total_with_mask + total_without_mask
        if total > 0:
            compliance = (total_with_mask / total) * 100
            cv2.putText(frame, f"Compliance: {compliance:.1f}%", (20, y_pos), font, 0.7, (255, 200, 100), 2)
        
        # Info about percentage
        y_pos += line_height + 10
        cv2.putText(frame, "% = Confidence Level", (20, y_pos), font, 0.5, (150, 150, 150), 1)
    
    # Draw center guide only if no faces detected
    if len(faces) == 0:
        h, w = frame.shape[:2]
        cv2.circle(frame, (w//2, h//2), 150, (100, 100, 255), 2)
        cv2.putText(frame, "Position face here", (w//2 - 100, h//2 + 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
    
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
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"[INFO] Debug mode: {debug_mode}")

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Final stats
print("\n" + "=" * 60)
print("SESSION STATISTICS:")
print("=" * 60)
print(f"Total detections with mask:    {total_with_mask}")
print(f"Total detections without mask: {total_without_mask}")
total = total_with_mask + total_without_mask
if total > 0:
    compliance = (total_with_mask / total) * 100
    print(f"Overall compliance rate:       {compliance:.1f}%")
print(f"Log file: {LOG_FILE}")
print("=" * 60)
