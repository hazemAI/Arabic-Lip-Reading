# %%
import os
import torch
import torchvision
import av.video.frame
import re
import torchvision.io.video as _tvideo

# Explicitly set input/output directories and FPS
INPUT_DIR = "D:/_hazem/Graduation Project//test_input_mpc"
OUTPUT_DIR = "../../test_output_mpc"
FPS = 25

# %% [markdown]
# ## 1. Initialize AVSR Dataloader.
#     
# The AVSRDataLoader class is a pre-process pipeline from raw video to cropped mouth using the `retinaface` detector exclusively.

# %%
class AVSRDataLoader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Always use retinaface detector
        from retinaface.detector import LandmarksDetector
        from retinaface.video_process import VideoProcess
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.landmarks_detector = LandmarksDetector(device=dev)
        self.video_process = VideoProcess(convert_gray=True)

    def forward(self, data_filename):
        video = self.load_video(data_filename)
        landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        return video

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

# %% [markdown]
# ## 3. Save the cropped video using torchvision/PyAV writer

# %%
# Save the cropped video using torchvision.io.write_video
def save2vid(filename, vid, frames_per_second):
    """Save a numpy array or torch tensor video using torchvision.io.write_video."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Convert to torch Tensor if needed
    vid_t = vid if isinstance(vid, torch.Tensor) else torch.as_tensor(vid)
    # Ensure dtype is uint8
    if vid_t.dtype != torch.uint8:
        if vid_t.is_floating_point():
            vid_t = (vid_t * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            vid_t = vid_t.to(torch.uint8)
    # Handle shape [T, H, W] -> [T, H, W, 1]
    if vid_t.ndim == 3:
        vid_t = vid_t.unsqueeze(-1)
    # Expand channel dim to 3 if single-channel
    if vid_t.ndim == 4 and vid_t.shape[-1] == 1:
        vid_t = vid_t.expand(-1, -1, -1, 3)
    # Handle [T, C, H, W] -> [T, H, W, C]
    if vid_t.ndim == 4 and vid_t.shape[1] in (1, 3) and vid_t.shape[-1] not in (1, 3):
        vid_t = vid_t.permute(0, 2, 3, 1)
    # Write video (T, H, W, C)
    torchvision.io.write_video(filename, vid_t.cpu(), frames_per_second)
    print(f"[save2vid] saved video at resolution {vid_t.shape[2]}x{vid_t.shape[1]}, fps={frames_per_second}")

# Auto-process all .mp4 videos in INPUT_DIR and save to OUTPUT_DIR
if __name__ == '__main__':
    loader = AVSRDataLoader()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.lower().endswith('.mp4'):
            continue
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        print(f"Processing video: {in_path}")
        video = loader(in_path)
        print("Cropped video tensor shape:", video.shape)
        print(f"Saving cropped video to: {out_path}")
        save2vid(out_path, video, FPS)

# Runtime-patch for torchvision.io.write_video pict_type assignment bug with PyAV >=10
try:
    _video_file = _tvideo.__file__
    with open(_video_file, 'r', encoding='utf-8') as _f:
        _content = _f.read()
    # Replace frame.pict_type = "NONE" with integer enum
    if 'frame.pict_type = "NONE"' in _content:
        # ensure PictureType is imported
        if 'from av.video.frame import PictureType' not in _content:
            _content = _content.replace(
                'import av.video.frame',
                'import av.video.frame\nfrom av.video.frame import PictureType'
            )
        _content = _content.replace(
            'frame.pict_type = "NONE"',
            'frame.pict_type = PictureType.NONE'
        )
        with open(_video_file, 'w', encoding='utf-8') as _f:
            _f.write(_content)
except Exception:
    # ignore if patch fails (e.g. permissions)
    pass


