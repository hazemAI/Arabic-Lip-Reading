import os
import cv2
import imageio
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import kornia.augmentation as K
from master_data_augmented import VideoAugmentation, MEAN, STD

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

# Prepare video batch for individual augmentations
frame_tensors = [transforms.ToTensor()(img) for img in frames]
video = torch.stack(frame_tensors, dim=0)        # (T, C, H, W)
video = video.permute(1, 0, 2, 3)               # (C, T, H, W)
video_batch = video.unsqueeze(0)                # (1, C, T, H, W)

# Define and apply each Kornia augmentation separately
augmentations = [
    ('random_crop', K.RandomCrop((88, 88), p=1.0)),
    ('random_flip', K.RandomHorizontalFlip(p=1.0)),
    ('random_rotate', K.RandomRotation(5.0, p=1.0)),
]
for name, op in augmentations:
    pipe = K.VideoSequential(op, data_format='BCTHW', same_on_frame=True)
    aug_vid = pipe(video_batch)  # (1, C, T, H, W)
    aug_np = aug_vid.squeeze(0).cpu().numpy()  # (C, T, H, W)
    # Convert to a list of 2D frames (H, W)
    frames_aug = [(aug_np[0, i] * 255).astype(np.uint8) for i in range(aug_np.shape[1])]
    imageio.mimwrite(os.path.join(out_dir, f'{name}.mp4'), frames_aug, fps=25)

# Manual normalization-only pipeline and save
# Convert frames to tensor
_frame_tensors = [transforms.ToTensor()(img) for img in frames]  # list of (C, H, W)
norm_video = torch.stack(_frame_tensors, dim=0)  # (T, C, H, W)
# Apply normalization: (x - mean) / std
norm_video = (norm_video - MEAN) / STD
# Invert normalization for visualization: x = x*std + mean
vis_video = norm_video * STD + MEAN
# Scale to [0,255] and convert to uint8
vis_video = (vis_video * 255.0).clamp(0, 255).byte()  # (T, C, H, W)
norm_frames = [vis_video[i, 0].cpu().numpy() for i in range(vis_video.shape[0])]
imageio.mimwrite(os.path.join(out_dir, 'normalized.mp4'), norm_frames, fps=25)

# Visualize raw standardized intensities (clamp to ±2σ) and save
disp = (norm_video.clamp(-2.0, 2.0) + 2.0) / 4.0  # scale to [0,1]
disp = (disp * 255.0).clamp(0, 255).byte()        # to uint8
raw_norm_frames = [disp[i, 0].cpu().numpy() for i in range(disp.shape[0])]
imageio.mimwrite(os.path.join(out_dir, 'raw_normalized.mp4'), raw_norm_frames, fps=25)

print(f"Saved videos to {out_dir}: original.mp4, random_crop.mp4, random_flip.mp4, random_rotate.mp4, normalized.mp4, raw_normalized.mp4") 