import os
from pathlib import Path
import torch
import torchvision.io as io
import kornia.augmentation as KA


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def augment_and_save(src_dir, dst_dir, aug_type):
    """
    Read all mp4 in src_dir, apply transform (Kornia or callable), and save to dst_dir with suffix aug_name.
    """
    ensure_dir(dst_dir)
    for file_path in sorted(Path(src_dir).glob("*.mp4")):
        # skip if output already exists
        out_file = Path(dst_dir) / f"{file_path.stem}_{aug_type}.mp4"
        if out_file.exists():
            print(f"Skipping existing file {out_file}")
            continue
        # read only video frames (ignore audio)
        video, _, info = io.read_video(str(file_path))
        # normalize to [T, C, H, W]
        frames = video.permute(0, 3, 1, 2).float() / 255.0
        # apply deterministic augment per video
        if aug_type == "flip":
            flipper = KA.RandomHorizontalFlip(p=1.0, same_on_batch=True)
            aug = flipper(frames)
        elif aug_type == "rotation":
            rotator = KA.RandomRotation(degrees=10.0, p=1.0, same_on_batch=True)
            aug = rotator(frames)
        elif aug_type == "sigmoid":
            aug = torch.sigmoid(5.0 * (frames - 0.5))
        elif aug_type == "linear":
            aug = (frames * 1.2 - 0.1).clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")
        # convert back to uint8
        aug = (aug.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        aug = aug.permute(0, 2, 3, 1)  # [T, H, W, C]
        # prepare output path
        raw_fps = info.get("video_fps", 25.0)
        # ensure fps is a native int
        fps = int(round(float(raw_fps)))
        io.write_video(str(out_file), aug, fps=fps)
        print(f"Saved {out_file}")


if __name__ == "__main__":
    # Source directory containing original preprocessed videos
    src_dir = "D:\\_hazem\\Graduation Project\\Arabic-Lip-Reading-testing-public\\Dataset\\Preprocessed_Video"
    # Base destination directory root for augmented videos; suffix will add aug_type
    base_dst_dir = "D:\\_hazem\\Graduation Project\\Arabic-Lip-Reading-testing-public\\Dataset\\Preprocessed_Video"
    # Perform augmentations for each type, dynamically naming directories
    for aug_type in ["rotation"]:
        # Directory for this augmentation type
        dst_dir = f"{base_dst_dir}_{aug_type}"
        print(f"Running augmentation: {aug_type}")
        augment_and_save(src_dir, dst_dir, aug_type)
    print("augmentations completed.") 