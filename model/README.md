# Arabic Lip Reading Model Directory Structure

This directory contains the implementation of an Arabic lip reading system using a hybrid CTC/Attention architecture. The system consists of several key components that work together to perform lip reading on Arabic videos.

## Directory Structure

```text
model/
├── encoders/           # Encoder modules and pretrained frontend
├── espnet/             # ESPNet transformer and CTC components
├── Logs/               # Training and inference logs
├── master_vsr.ipynb    # Main VSR training and evaluation notebook
├── e2e_avsr.py         # End-to-end audio-visual speech recognition script
└── utils.py            # Utility functions (collation, metrics, scheduler)
```

## utils.py - Detailed Explanation

This file contains essential utility functions used in the lip reading and audio-visual speech recognition pipelines.

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
    # Only copy up to the actual sequence length
    for idx in range(len(data_list)):
        data[idx, :, :input_lengths[idx], :, :] = data_list[idx][:, :input_lengths[idx], :, :]

    # Flatten labels for CTC loss
    labels_flat = []
    for label_seq in labels_list:
        labels_flat.extend(label_seq)
    labels_flat = torch.LongTensor(labels_flat)

    # Convert lengths to tensor
    input_lengths = torch.LongTensor(input_lengths)
    label_lengths = torch.LongTensor(label_lengths)
    return data, input_lengths, labels_flat, label_lengths
```

- Unpacks the batch into videos, input lengths, label sequences, and label lengths.
- Initializes a zero tensor of shape `(batch_size, channels, max_time, height, width)`.
- Copies each sequence up to its actual length to avoid padding noise.
- Flattens all label sequences into a single 1D tensor for CTC loss processing.
- Returns: `data` (padded video tensor), `input_lengths`, `labels_flat`, and `label_lengths`.

#### 2. `indices_to_text`

```python
def indices_to_text(indices, idx2char):
    """
    Converts a list of indices to text using the reverse vocabulary mapping.
    """
    try:
        return ''.join([idx2char.get(i, '') for i in indices])
    except UnicodeEncodeError:
        safe_text = []
        for i in indices:
            char = idx2char.get(i, '')
            try:
                char.encode('cp1252')
                safe_text.append(char)
            except UnicodeEncodeError:
                safe_text.append(f"[{i}]")
        return ''.join(safe_text)
```

- Maps each token index to its character via `idx2char` and joins into a string.
- Handles Windows encoding errors by replacing unencodable characters with their index in brackets.

#### 3. `compute_cer`

```python
def compute_cer(reference_indices, hypothesis_indices):
    """
    Computes Character Error Rate (CER) directly using token indices. Returns `(cer, edit_distance)`.
    """
    ref_tokens = reference_indices
    hyp_tokens = hypothesis_indices

    try:
        logging.info(f"Debug - Reference tokens ({len(ref_tokens)}): {ref_tokens}")
        logging.info(f"Debug - Hypothesis tokens ({len(hyp_tokens)}): {hyp_tokens}")
    except UnicodeEncodeError:
        logging.info("Debug - Token indices omitted due to encoding issues")

    edit_distance = editdistance.eval(ref_tokens, hyp_tokens)
    cer = edit_distance / max(len(ref_tokens), 1)
    return cer, edit_distance
```

- Uses the `editdistance` library to compute Levenshtein distance between reference and hypothesis index lists.
- Logs token sequences for debugging purposes.
- Calculates CER as `edit_distance / max(reference_length, 1)` to avoid division by zero.

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
            # Linear warmup
            return [
                self._step_count / self.warmup_steps * base_lr
                for base_lr in self.base_lrs
            ]
        # Cosine decay after warmup
        decay_steps = self.total_steps - self.warmup_steps
        cos_val = math.cos(
            math.pi * (self._step_count - self.warmup_steps) / decay_steps
        )
        return [0.5 * base_lr * (1 + cos_val) for base_lr in self.base_lrs]
```

- Performs a linear warmup phase for the first `warmup_epochs`.
- After warmup, applies cosine annealing over the remaining steps to adjust learning rate.
- Inherits from PyTorch's `_LRScheduler` to integrate with optimizers.

### Usage Patterns

#### 1. DataLoader Setup

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True,
    collate_fn=pad_packed_collate
)
```

#### 2. Mask Creation

```python
# Create memory mask based on actual encoder output lengths
memory_mask = torch.zeros((batch_size, encoder_features.size(1)), device=device).bool()
for b in range(batch_size):
    memory_mask[b, :output_lengths[b]] = True
```

#### 3. Text Conversion

```python
# Convert predictions to text
pred_text = indices_to_text(pred_indices, idx2char)
target_text = indices_to_text(target_idx, idx2char)
```

#### 4. Metric Calculation

```python
# Calculate and log CER
cer, edit_distance = compute_cer(target_idx, pred_indices)
print(f"CER: {cer:.4f}, Edit Distance: {edit_distance}")
```

## master_vsr.ipynb - Main VSR Notebook

This Jupyter notebook implements the video-only speech recognition (VSR) pipeline. It covers:
- **Data Preparation**: Frame extraction, label mapping, DataLoader setup with `pad_packed_collate`.
- **Model Configuration**: Choosing `conformer`, `densetcn`, or `mstcn` temporal encoder, building `Lipreading` model, loading pretrained frontend.
- **Training Loop**: Hybrid CTC + attention training, optimizer and scheduler setup, checkpoint saving.
- **Evaluation**: Beam search decoding on validation set, CER computation.

### Usage
1. Launch the notebook in JupyterLab or Jupyter Notebook:
   ```bash
   jupyter notebook master_vsr.ipynb
   ```
2. Follow cells in order to configure paths, hyperparameters, and run each stage.

## e2e_avsr.py - End-to-End AVSR Script

This Python script provides a command-line interface for audio-visual speech recognition using the `E2EAVSR` class (inheriting from ESPNet's E2E ASR):

```bash
python e2e_avsr.py --help
```

- **Training**: Use flags like `--train` to start model training with specified encoder, dataset paths, and hyperparameters.
- **Inference**: Use flags like `--decode` along with `--checkpoint` to perform beam search decoding on new videos.
- **Key Options**: `--encoder_type`, `--ctc_weight`, `--beam_size`, `--length_bonus_weight`, `--label_smoothing`.

Refer to the script docstring and `--help` output for full option list.

## Training Flow

1. **Data Processing**:

   - Load video frames and Arabic text labels
   - Map Arabic characters to indices
   - Create train/validation/test splits
   - Apply data transformations

2. **Model Initialization**:

   - Load pretrained visual frontend
   - Initialize temporal encoder (Conformer/DenseTCN/MSTCN)
   - Setup Arabic-specific transformer decoder
   - Define loss functions and optimizer

3. **Training Loop**:

   - Process batches of video sequences
   - Extract visual features using encoder
   - Compute CTC loss on encoder outputs
   - Generate sequences with transformer decoder
   - Compute cross-entropy loss on decoder outputs
   - Update model parameters

4. **Evaluation**:

   - Perform beam search decoding
   - Calculate Character Error Rate (CER)
   - Monitor validation performance
   - Save checkpoints and best model

5. **Inference**:
   - Process input video
   - Extract visual features
   - Perform beam search with hybrid scoring
   - Convert indices to Arabic text

## Integration Points

- **Visual Frontend** → **Temporal Encoder** → **Transformer Decoder**
- **utils.py** provides supporting functions throughout the pipeline
- **ESPNet Components** provide transformer building blocks
- **Hybrid Loss** combines CTC and attention-based approaches
