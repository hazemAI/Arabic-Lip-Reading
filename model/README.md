# Visual Speech Recognition (VSR) Model Directory

This directory contains the implementation of an Arabic Visual Speech Recognition (VSR) system using a hybrid CTC/Attention architecture. The system focuses on greedy decoding for efficient inference.

## Directory Structure

```text
model/
├── README.md               # This file, providing an overview of the model directory.
├── encoders/               # Contains modules for visual encoders and the pretrained visual frontend.
├── espnet/                 # Includes core ESPNet components for transformer and CTC functionalities.
├── Logs/                   # Directory for storing training, evaluation, and inference logs.
├── e2e_vsr_greedy.py       # End-to-end Visual Speech Recognition script (greedy decoding).
├── master_vsr_greedy.py    # Main training and evaluation script for VSR (greedy decoding).
├── inference_test.py       # Script for testing model inference.
└── utils.py                # Utility functions (e.g., data collation, metrics, learning rate schedulers).
```

## Key Files

- `e2e_vsr_greedy.py`: Core implementation of the end-to-end VSR model using greedy decoding.
- `master_vsr_greedy.py`: Main training and evaluation script for VSR, utilizing greedy decoding.
- `utils.py`: Helper functions for data loading, preprocessing, metrics, and learning rate schedulers.
- `inference_test.py`: Script dedicated to testing the inference capabilities of the VSR model with greedy decoding.
- `encoders/`: Directory containing visual encoder modules and the pretrained visual frontend.
- `espnet/`: Directory housing essential ESPNet components for transformer architecture and CTC loss.

## `utils.py` - Detailed Explanation

This file contains essential utility functions used in the VSR pipeline.

### File Structure

```python
import torch
import editdistance
import math
import logging
from model.espnet.nets_utils import make_pad_mask
```

- **torch**: Core PyTorch library for tensor computations and schedulers.
- **editdistance**: Library for computing Levenshtein (edit) distance between sequences.
- **math**: Provides mathematical functions like cosine for learning rate schedulers.
- **logging**: Enables debug and info logging within utility functions.
- **make_pad_mask**: ESPNet utility to generate masks for padded sequences.

### Function-by-Function Explanation

#### 1. `pad_packed_collate`

```python
def pad_packed_collate(batch):
    """Pads data and labels with different lengths in the same batch."""
    data_list, input_lengths, labels_list, label_lengths = zip(*batch)
    c, max_len, h, w = max(data_list, key=lambda x: x.shape[1]).shape

    data = torch.zeros((len(data_list), c, max_len, h, w))
    for idx in range(len(data_list)):
        data[idx, :, :input_lengths[idx], :, :] = data_list[idx][:, :input_lengths[idx], :, :]

    labels_flat = []
    for label_seq in labels_list:
        labels_flat.extend(label_seq)
    labels_flat = torch.LongTensor(labels_flat)

    input_lengths = torch.LongTensor(input_lengths)
    label_lengths = torch.LongTensor(label_lengths)
    return data, input_lengths, labels_flat, label_lengths
```

- Unpacks the batch into videos, input lengths, label sequences, and label lengths.
- Initializes a zero tensor for padded video data.
- Copies each sequence up to its actual length.
- Flattens label sequences for CTC loss.
- Returns padded data and length tensors.

#### 2. `indices_to_text`

```python
def indices_to_text(indices, idx2char):
    """Converts a list of indices to text using the reverse vocabulary mapping."""
    try:
        return ''.join([idx2char.get(i, '') for i in indices])
    except UnicodeEncodeError:
        safe_text = []
        for i in indices:
            char = idx2char.get(i, '')
            try:
                char.encode('cp1252') # Or appropriate encoding
                safe_text.append(char)
            except UnicodeEncodeError:
                safe_text.append(f"[{i}]") # Placeholder for unencodable chars
        return ''.join(safe_text)
```

- Maps token indices to characters and joins them.
- Includes basic error handling for character encoding issues.

#### 3. `compute_cer`

```python
def compute_cer(reference_indices, hypothesis_indices):
    """Computes Character Error Rate (CER) directly using token indices."""
    ref_tokens = reference_indices
    hyp_tokens = hypothesis_indices
    edit_distance_val = editdistance.eval(ref_tokens, hyp_tokens)
    cer = edit_distance_val / max(len(ref_tokens), 1)
    return cer, edit_distance_val
```

- Calculates Levenshtein distance and CER between reference and hypothesis token sequences.

#### 4. `WarmupCosineScheduler`

```python
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [
                self._step_count / self.warmup_steps * base_lr
                for base_lr in self.base_lrs
            ]
        decay_steps = self.total_steps - self.warmup_steps
        cos_val = math.cos(
            math.pi * (self._step_count - self.warmup_steps) / decay_steps
        )
        return [0.5 * base_lr * (1 + cos_val) for base_lr in self.base_lrs]
```

- Implements a learning rate scheduler with a linear warmup phase followed by cosine annealing.

## `master_vsr_greedy.py` - Main VSR Script

This Python script implements the primary visual speech recognition (VSR) training and evaluation pipeline. It covers:
- **Data Preparation**: Handles frame extraction from videos, mapping of text labels to indices, and setting up DataLoaders using `pad_packed_collate` from `utils.py`.
- **Model Configuration**: Configures the VSR model, choosing between temporal encoders like `conformer`, `densetcn`, or `mstcn`. It builds the `VisualTemporalEncoder` and loads a pretrained visual frontend.
- **Training Loop**: Manages the training process using a hybrid CTC + attention loss. It includes optimizer setup (e.g., AdamW) and learning rate scheduling (e.g., `WarmupCosineScheduler`), along with checkpoint saving.
- **Evaluation**: Performs evaluation on a validation set using **greedy decoding** and computes the Character Error Rate (CER) metric.

Configurations for this script (e.g., model hyperparameters, paths, training settings) are typically managed within the script itself or through a dedicated configuration mechanism loaded by the script.

## `e2e_vsr_greedy.py` - End-to-End VSR Script

This Python script provides an interface for end-to-end visual speech recognition. It utilizes the `E2EVSR` class, which builds upon ESPNet's E2E ASR framework, adapted for VSR.

The script is designed to handle both model training and inference, employing **greedy decoding** for generating text from visual inputs. Configurations regarding encoder types, dataset paths, CTC weight, label smoothing, and other relevant hyperparameters are managed internally within the script or through its specific setup routines.

## Training Flow

1.  **Data Processing**:
    *   Load video frames and corresponding Arabic text labels.
    *   Map Arabic characters to unique integer indices using a predefined vocabulary.
    *   Create training, validation, and test dataset splits.
    *   Apply necessary data augmentations (if any) and transformations (e.g., normalization).

2.  **Model Initialization**:
    *   Load a pretrained visual frontend (e.g., a ResNet-based model) to extract initial frame-level features.
    *   Initialize the chosen temporal encoder (e.g., Conformer, DenseTCN, MSTCN) to model temporal dynamics.
    *   Set up the Arabic-specific transformer decoder for sequence generation.
    *   Define loss functions (CTC loss for encoder outputs, attention-based cross-entropy loss for decoder outputs) and the optimizer (e.g., AdamW with a learning rate scheduler).

3.  **Training Loop**:
    *   Iterate over batches of video sequences from the training DataLoader.
    *   For each batch, extract visual features using the visual frontend and temporal encoder.
    *   Compute the CTC loss based on the encoder outputs and target labels.
    *   Feed encoder outputs to the transformer decoder (using teacher forcing with ground truth labels during training) to generate output sequences.
    *   Compute the attention-based cross-entropy loss based on the decoder outputs.
    *   Combine the CTC and attention losses (weighted sum) to get the final loss.
    *   Perform backpropagation and update model parameters using the optimizer.

4.  **Evaluation**:
    *   Periodically, or after training completes, evaluate the model on the validation set.
    *   Perform **greedy decoding** using the trained model to generate predicted text sequences from validation videos.
    *   Calculate Character Error Rate (CER) by comparing predicted sequences with ground truth labels.
    *   Save model checkpoints, especially the one yielding the best validation performance.

## Inference

1.  **Input Processing**:
    *   Load and preprocess the input video: extract frames, normalize them, and arrange into the format expected by the model.
2.  **Feature Extraction**:
    *   Pass the processed video frames through the pretrained visual frontend and the trained temporal encoder to obtain a sequence of visual features.
3.  **Decoding**:
    *   Employ **greedy decoding** with the trained transformer decoder. The decoder autoregressively predicts the most likely token at each step based on the visual features and previously generated tokens, until an end-of-sequence token is produced or a maximum length is reached.
4.  **Output Generation**:
    *   Convert the sequence of predicted token indices back to readable Arabic text using the character-to-index mapping.

## Integration Points

- **Visual Frontend → Temporal Encoder → Transformer Decoder**: This defines the core pipeline for processing visual information and generating text.
- `utils.py`: Provides critical support functions for data handling (`pad_packed_collate`), metric calculation (`compute_cer`), and learning rate scheduling (`WarmupCosineScheduler`), used across various parts of the main scripts.
- **ESPNet Components**: The underlying transformer architecture and CTC implementation are leveraged from ESPNet, providing robust and well-tested building blocks.
- **Hybrid CTC/Attention Loss**: The training process relies on a hybrid loss that combines the strengths of CTC (for alignment) and attention-based sequence-to-sequence modeling (for context).
