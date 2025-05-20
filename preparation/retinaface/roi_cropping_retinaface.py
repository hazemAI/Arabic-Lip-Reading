# %%
# Import necessary libraries
import torch, os, cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from collections import deque
from skimage import transform as tf
from detector import LandmarksDetector
from video_process import VideoProcess


## Choose device: use CPU if CUDA is unavailable
device = "cuda:0" if torch.cuda.is_available() else "cpu"

## Initialize RetinaFace and face-alignment modules with ResNet50 and grayscale output
landmarks_detector = LandmarksDetector(device=device, model_name="resnet50")
video_processor   = VideoProcess(convert_gray=True)

def process_video(video_path):
    """Crop mouth ROI from the input video and save it to OUTPUT_DIR"""
    name = os.path.splitext(os.path.basename(video_path))[0]
    dst_path = os.path.join(OUTPUT_DIR, f"{name}.mp4")
    if os.path.exists(dst_path):
        print(f"Skipping {name}: already processed.")
        return
    print(f"Processing {name}...")

    # Read all frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print(f"No frames in {name}, skipping.")
        return

    video_array = np.stack(frames, axis=0)  # shape: (T, H, W, C)

    # Run detection and cropping
    landmarks = landmarks_detector(video_array)
    processed = video_processor(video_array, landmarks)
    if processed is None:
        print(f"Failed to crop {name}: no valid landmarks.")
        return

    # Save the cropped sequence
    H, W = processed.shape[1], processed.shape[2]
    is_color = processed.ndim == 4 and processed.shape[3] == 3
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(dst_path, fourcc, fps, (W, H), isColor=is_color)
    for frame in processed:
        if is_color:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            writer.write(frame)
    writer.release()
    print(f"Saved cropped video: {dst_path}")

if __name__ == '__main__':
    # Config: set your input and output directories here
    RAW_VID_DIR = r"D:/_hazem/Graduation Project/other/Arabic-Lib-Reading/Dataset/Video"
    OUTPUT_DIR = r"D:/_hazem/Graduation Project/Arabic-Lip-Reading-testing-public/Dataset/Preprocessed_Video"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all .mp4 files under RAW_VID_DIR
    video_files = []
    for root, _, files in os.walk(RAW_VID_DIR):
        for fname in files:
            if fname.lower().endswith('.mp4'):
                video_files.append(os.path.join(root, fname))
    print(f"Found {len(video_files)} videos to process.")
    # Process each video
    for idx, path in enumerate(video_files, start=1):
        print(f"[{idx}/{len(video_files)}] {path}")
        process_video(path)


