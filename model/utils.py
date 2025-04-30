import torch
import editdistance
import math
import logging

def pad_packed_collate(batch):
    """Pads data and labels with different lengths in the same batch
    """
    data_list, input_lengths, labels_list, label_lengths = zip(*batch)
    c, max_len, h, w = max(data_list, key=lambda x: x.shape[1]).shape

    data = torch.zeros((len(data_list), c, max_len, h, w))
    
    # Only copy up to the actual sequence length
    for idx in range(len(data)):
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



def indices_to_text(indices, idx2char):
    """
    Converts a list of indices to text using the reverse vocabulary mapping.
    """
    try:
        return ''.join([idx2char.get(i, '') for i in indices])
    except UnicodeEncodeError:
        # Handle encoding issues in Windows console
        # Return a safe representation that won't cause encoding errors
        safe_text = []
        for i in indices:
            char = idx2char.get(i, '')
            try:
                # Test if character can be encoded
                char.encode('cp1252')
                safe_text.append(char)
            except UnicodeEncodeError:
                # Replace with a placeholder for characters that can't be displayed
                safe_text.append(f"[{i}]")
        return ''.join(safe_text)

def compute_cer(reference_indices, hypothesis_indices):
    """
    Computes Character Error Rate (CER) directly using token indices.
    Takes raw token indices from our vocabulary (class_mapping.txt).
    
    Returns a tuple of (CER, edit_distance)
    """
    # Use the indices directly - each index is one token in our vocabulary
    ref_tokens = reference_indices
    hyp_tokens = hypothesis_indices
    
    try:
        logging.info(f"Debug - Reference tokens ({len(ref_tokens)} tokens): {ref_tokens}")
        logging.info(f"Debug - Hypothesis tokens ({len(hyp_tokens)} tokens): {hyp_tokens}")
    except UnicodeEncodeError:
        # Handle encoding issues in Windows console
        logging.info(f"Debug - Reference tokens ({len(ref_tokens)} tokens): [Token indices omitted due to encoding issues]")
        logging.info(f"Debug - Hypothesis tokens ({len(hyp_tokens)} tokens): [Token indices omitted due to encoding issues]")
    
    # Calculate edit distance using the editdistance library
    edit_distance = editdistance.eval(ref_tokens, hyp_tokens)
    
    # Calculate CER
    cer = edit_distance / max(len(ref_tokens), 1)  # Avoid division by zero
    
    return cer, edit_distance


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
