# %%
import os
import torch
import torchvision

# set input/output directories and FPS
INPUT_DIR = "D:/_hazem/Graduation Project/test_input_mpc"
OUTPUT_DIR = "D:/_hazem/Graduation Project/test_output_mpc"
FPS = 30

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Initialize AVSR Dataloader.
#     
# The AVSRDataLoader class is a pre-process pipeline from raw video to cropped mouth using the `retinaface` detector exclusively.

# %%
class AVSRDataLoader(torch.nn.Module):
    def __init__(self):
        super().__init__()
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

# %%
# Save the cropped video
def save2vid(filename, vid, frames_per_second):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Directly write preprocessed video array (shape [T, H, W, C], uint8)
    torchvision.io.write_video(filename, vid, frames_per_second)
    print(f"[save2vid] saved video to {filename} (fps={frames_per_second})")


if __name__ == '__main__':
    loader = AVSRDataLoader()
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

