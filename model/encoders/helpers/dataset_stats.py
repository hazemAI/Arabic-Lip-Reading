"""
dataset_stats.py

Compute minimum, maximum, and average label lengths (in tokens) and video frame lengths
across the Dataset folder containing 'Csv (with Diacritics)' and 'Preprocessed_Video' subfolders.
"""

import os
import cv2
import pandas as pd
import argparse

# Diacritics set used to merge with preceding characters
DIACRITICS = {
    '\u064B',  # Fathatan
    '\u064C',  # Dammatan
    '\u064D',  # Kasratan
    '\u064E',  # Fatha
    '\u064F',  # Damma
    '\u0650',  # Kasra
    '\u0651',  # Shadda
    '\u0652',  # Sukun
    '\u06E2',  # Small High meem
}

def extract_label(csv_path):
    """
    Read a CSV and return the list of diacritized-character tokens using the DIACRITICS merging.
    """
    labels = []
    df = pd.read_csv(csv_path)
    for word in df.get('word', []):
        for char in word:
            if char not in DIACRITICS:
                labels.append(char)
            else:
                # merge diacritic into previous character
                labels[-1] += char
    return labels


def compute_label_lengths(csv_dir):
    lengths = []
    for fname in sorted(os.listdir(csv_dir)):
        if not fname.lower().endswith('.csv'):
            continue
        path = os.path.join(csv_dir, fname)
        try:
            tokens = extract_label(path)
            lengths.append(len(tokens))
        except Exception as e:
            print(f"Warning: failed to parse {path}: {e}")
    return lengths


def compute_frame_lengths(video_dir):
    lengths = []
    for fname in sorted(os.listdir(video_dir)):
        if not fname.lower().endswith(('.mp4', '.avi', '.mov')):
            continue
        path = os.path.join(video_dir, fname)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Warning: could not open video {path}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        lengths.append(total)
        cap.release()
    return lengths


def summarize(lengths, name):
    if not lengths:
        print(f"No data for {name}")
        return
    print(f"{name} -> min: {min(lengths)}, max: {max(lengths)}, avg: {sum(lengths)/len(lengths):.2f}, count: {len(lengths)}")


def main(dataset_dir):
    csv_folder = os.path.join(dataset_dir, 'Csv (with Diacritics)')
    video_folder = os.path.join(dataset_dir, 'Preprocessed_Video')

    if not os.path.isdir(csv_folder):
        raise ValueError(f"CSV folder not found: {csv_folder}")
    if not os.path.isdir(video_folder):
        raise ValueError(f"Video folder not found: {video_folder}")

    print("Computing label lengths from:", csv_folder)
    label_lengths = compute_label_lengths(csv_folder)
    summarize(label_lengths, "Label lengths (tokens)")

    print("\nComputing frame lengths from:", video_folder)
    frame_lengths = compute_frame_lengths(video_folder)
    summarize(frame_lengths, "Frame lengths (frames)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset label and frame length stats')
    parser.add_argument('dataset_dir', nargs='?', default='Dataset', help='Path to the Dataset folder')
    args = parser.parse_args()
    main(args.dataset_dir) 