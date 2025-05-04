import torch
import editdistance
import math
import logging


def pad_packed_collate(batch):
    """Pads data and labels with different lengths in the same batch"""
    data_list, input_lengths, labels_list, label_lengths = zip(*batch)
    c, max_len, h, w = max(data_list, key=lambda x: x.shape[1]).shape

    data = torch.zeros((len(data_list), c, max_len, h, w))
    # Copy up to actual lengths
    for idx in range(len(data_list)):
        data[idx, :, :input_lengths[idx], :, :] = data_list[idx][:, :input_lengths[idx], :, :]

    # Flatten labels for CTC loss
    labels_flat = []
    for seq in labels_list:
        labels_flat.extend(seq)
    labels_flat = torch.LongTensor(labels_flat)

    input_lengths = torch.LongTensor(input_lengths)
    label_lengths = torch.LongTensor(label_lengths)
    return data, input_lengths, labels_flat, label_lengths


def indices_to_text(indices, idx2token):
    """
    Converts a list of word indices to a space-separated string using the reverse vocabulary mapping.
    """
    return ' '.join([idx2token.get(i, '') for i in indices])


def compute_wer(reference_indices, hypothesis_indices):
    """
    Computes Word Error Rate (WER) directly using token indices.
    Returns a tuple of (WER, edit_distance).
    """
    # Calculate edit distance between reference and hypothesis tokens
    edit_distance = editdistance.eval(reference_indices, hypothesis_indices)
    wer = edit_distance / max(len(reference_indices), 1)
    return wer, edit_distance


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [self._step_count / self.warmup_steps * base_lr for base_lr in self.base_lrs]
        decay_steps = self.total_steps - self.warmup_steps
        cos_val = math.cos(math.pi * (self._step_count - self.warmup_steps) / decay_steps)
        return [0.5 * base_lr * (1 + cos_val) for base_lr in self.base_lrs] 