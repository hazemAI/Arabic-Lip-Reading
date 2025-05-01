import os
import cv2
import imageio
import numpy as np
from PIL import Image
from master_data_augmented import VideoAugmentation

# Paths (adjust if your working directory differs)
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
videos_dir = os.path.join(root, 'Dataset', 'Preprocessed_Video')
file_names = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
if not file_names:
    raise RuntimeError(f"No .mp4 files found in {videos_dir}")
video_path = os.path.join(videos_dir, file_names[0])

# Read raw grayscale frames
cap = cv2.VideoCapture(video_path)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(Image.fromarray(gray, 'L'))
cap.release()

# Create output folder
out_dir = os.path.join(os.path.dirname(__file__), 'aug_videos')
os.makedirs(out_dir, exist_ok=True)

# Save original video as MP4
auto_orig = [np.array(img) for img in frames]
imageio.mimwrite(os.path.join(out_dir, 'original.mp4'), auto_orig, fps=25)

# Apply train-time augmentation and save
aug = VideoAugmentation(is_train=True)
aug_tensor = aug(frames)  # shape: (C=1, T, H, W)
# Convert to numpy frames (T, H, W)
aug_np = (aug_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
imageio.mimwrite(os.path.join(out_dir, 'augmented.mp4'), aug_np, fps=25)

print(f"Saved original and augmented videos to {out_dir}") 