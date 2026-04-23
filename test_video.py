"""Quick script to diagnose video loading issues."""
import cv2
import sys

video_path = r"C:\Users\Shubham\Downloads\badminton1.mp4"

# Try all available backends
backends = {
    "AUTO": cv2.CAP_ANY,
    "FFMPEG": cv2.CAP_FFMPEG,
    "MSMF": cv2.CAP_MSMF,
    "DSHOW": cv2.CAP_DSHOW,
}

for name, backend in backends.items():
    cap = cv2.VideoCapture(video_path, backend)
    opened = cap.isOpened()
    if opened:
        ret, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        shape = frame.shape if ret else "read failed"
        print(f"{name}: OPENED | {w}x{h} @ {fps}fps | frames={fc} | first_read={shape}")
    else:
        print(f"{name}: FAILED to open")
    cap.release()
