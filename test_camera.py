"""
Test Camera Access on Windows
Simple script to verify webcam is working
"""

import cv2
import sys

print("=" * 50)
print("Camera Test - Windows")
print("=" * 50)

# Try to open camera
print("[INFO] Attempting to open camera 0...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow on Windows

if not cap.isOpened():
    print("[ERROR] Camera 0 failed!")
    print("[INFO] Trying camera 0 without DirectShow...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Camera 0 still failed!")
        print("[INFO] Trying camera 1...")
        cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            print("[ERROR] No camera found!")
            print("\n[TROUBLESHOOTING]")
            print("1. Check if webcam is connected")
            print("2. Check if another app is using the camera")
            print("3. Check Windows Camera Privacy settings")
            print("4. Restart your computer")
            sys.exit(1)

print("[SUCCESS] Camera opened!")

# Get camera properties
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"[INFO] Resolution: {int(width)}x{int(height)}")
print(f"[INFO] FPS: {fps}")

# Try to read a frame
print("[INFO] Attempting to capture frame...")
ret, frame = cap.read()

if ret:
    print("[SUCCESS] Frame captured successfully!")
    print(f"[INFO] Frame shape: {frame.shape}")
    
    # Show the frame
    cv2.imshow('Camera Test - Press Q to quit', frame)
    print("\n[INFO] Camera window opened.")
    print("[INFO] Press 'Q' to quit")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break
        
        frame_count += 1
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera Test - Press Q to quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"[INFO] Captured {frame_count} frames")
else:
    print("[ERROR] Failed to capture frame!")
    print("[INFO] Camera might be in use by another application")

cap.release()
cv2.destroyAllWindows()
print("[INFO] Camera test complete")
print("=" * 50)
