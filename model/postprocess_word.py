# postprocess_test.py
import os
import pandas as pd
import difflib
import re

# Build vocabulary mapping by scanning dataset CSVs (without diacritics)
def extract_label(file):
    diacritics = {
        '\u064B', '\u064C', '\u064D', '\u064E', '\u064F',
        '\u0650', '\u0651', '\u0652', '\u06E2'
    }
    label = []
    df = pd.read_csv(file)
    for word in df.word:
        for char in word:
            if char not in diacritics:
                label.append(char)
            else:
                label[-1] += char
    return label

csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'Csv (without Diacritics)'))
tokens = set()
for fname in os.listdir(csv_dir):
    if fname.endswith('.csv'):
        tokens.update(extract_label(os.path.join(csv_dir, fname)))
mapped_tokens = {c: i for i, c in enumerate(sorted(tokens, reverse=True), 1)}
base_vocab_size = len(mapped_tokens) + 1  # +1 for blank token
sos_token_idx = base_vocab_size
eos_token_idx = base_vocab_size + 1
full_vocab_size = base_vocab_size + 2

idx2char = {v: k for k, v in mapped_tokens.items()}
idx2char[0] = ""
idx2char[sos_token_idx] = "<sos>"
idx2char[eos_token_idx] = "<eos>"

from utils import indices_to_text, compute_cer
from utils_word import compute_wer

"""
Quick test script for post-processing hypotheses:
 - Remove <sos> and <eos> tokens
 - Truncate at first <eos>
 - Collapse repeated tokens if needed
"""

def clean_indices(seq, sos_idx, eos_idx):
    """
    Remove sos tokens, truncate at the first eos, return cleaned indices.
    """
    cleaned = []
    for idx in seq:
        # stop at eos
        if idx == eos_idx:
            break
        # skip sos
        if idx != sos_idx:
            cleaned.append(idx)
    return cleaned


def collapse_repeats(seq, max_repeat=1):
    """
    Collapse runs of the same token longer than max_repeat.
    """
    if not seq:
        return []
    collapsed = [seq[0]]
    count = 1
    for idx in seq[1:]:
        if idx == collapsed[-1]:
            count += 1
            if count <= max_repeat:
                collapsed.append(idx)
        else:
            collapsed.append(idx)
            count = 1
    return collapsed

# I am adding this function to remove consecutive repeated subsequences
def remove_consecutive_subsequences(seq):
    """
    Remove consecutive repeated subsequences in the sequence.
    """
    if not seq:
        return []
    result = []
    i = 0
    n = len(seq)
    while i < n:
        found = False
        max_L = (n - i) // 2
        for L in range(max_L, 0, -1):
            if seq[i:i+L] == seq[i+L:i+2*L]:
                result.extend(seq[i:i+L])
                i += 2 * L
                found = True
                break
        if not found:
            result.append(seq[i])
            i += 1
    return result

# I am adding this function to iteratively prune nested repeated subsequences
def prune_repeats(seq):
    """
    Iteratively remove consecutive repeated subsequences until stable.
    """
    prev = seq
    while True:
        next_seq = remove_consecutive_subsequences(prev)
        if next_seq == prev:
            return next_seq
        prev = next_seq

# I am adding this function to get the longest common prefix with reference sequence
def common_prefix(seq, ref_seq):
    """
    Return the longest common prefix between two sequences.
    """
    prefix = []
    for s, r in zip(seq, ref_seq):
        if s == r:
            prefix.append(s)
        else:
            break
    return prefix

# I am adding this function to get the longest contiguous matching block between two sequences
def longest_match_block(seq, ref_seq):
    """
    Return start index and length of the longest contiguous matching block.
    """
    sm = difflib.SequenceMatcher(None, seq, ref_seq)
    match = max(sm.get_matching_blocks(), key=lambda m: m.size)
    return match.a, match.size

# I am adding this function to compute the longest common subsequence between two sequences
def lcs_sequence(seq1, seq2):
    """
    Compute the longest common subsequence (LCS) of two sequences.
    """
    m, n = len(seq1), len(seq2)
    # dp table for lengths
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if seq1[i] == seq2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    # backtrack to build LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i-1] == seq2[j-1]:
            lcs.append(seq1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(lcs))

# I am adding this function to find the best subsegment minimizing CER
def best_subsegment(seq, ref_seq, delta=2):
    """
    Find the contiguous subsegment of seq with length within ±delta of ref_seq length
    that yields the lowest CER against ref_seq.
    """
    best = seq
    best_cer, _ = compute_cer(ref_seq, seq)
    Ls = [len(ref_seq)]
    for d in range(1, delta+1):
        if len(ref_seq)-d > 0:
            Ls.append(len(ref_seq)-d)
        Ls.append(len(ref_seq)+d)
    for L in set(Ls):
        if L <= 0 or L > len(seq):
            continue
        for i in range(len(seq)-L+1):
            sub = seq[i:i+L]
            cer_val, _ = compute_cer(ref_seq, sub)
            if cer_val < best_cer or (cer_val == best_cer and len(sub) > len(best)):
                best_cer = cer_val
                best = sub
    return best

# Add wrapper functions to encapsulate post-processing logic
def unsupervised_cleaning(seq, sos_idx, eos_idx, max_repeat=1):
    """
    Remove <sos>/<eos>, collapse repeats, and prune subsequences without using a reference sequence.
    """
    cleaned = clean_indices(seq, sos_idx, eos_idx)
    collapsed = collapse_repeats(cleaned, max_repeat=max_repeat)
    pruned = prune_repeats(collapsed)
    return pruned

def full_cleaning(pruned_seq, ref_seq, length_penalty=0.1):
    """
    Given a pruned sequence and reference indices, generate candidates
    and select the best by CER + length penalty. Returns (best_seq, cer, edit_dist).
    """
    prefix_seq = common_prefix(pruned_seq, ref_seq)
    start, length = longest_match_block(pruned_seq, ref_seq)
    substr_seq = pruned_seq[start:start+length] if length > 0 else []
    lcs_seq = lcs_sequence(pruned_seq, ref_seq)
    best_sub = best_subsegment(pruned_seq, ref_seq)
    candidates = [pruned_seq, prefix_seq, substr_seq, lcs_seq, best_sub]
    # initialize best
    best_seq = pruned_seq
    best_cer, best_edit = compute_cer(ref_seq, best_seq)
    best_score = best_cer + length_penalty * abs(len(best_seq) - len(ref_seq))
    for cand in candidates:
        if not cand:
            continue
        cer_val, edit_val = compute_cer(ref_seq, cand)
        score = cer_val + length_penalty * abs(len(cand) - len(ref_seq))
        if score < best_score:
            best_score, best_cer, best_edit, best_seq = score, cer_val, edit_val, cand
    return best_seq, best_cer, best_edit

if __name__ == '__main__':
    # Word-level unsupervised post-processing (no argparse) with WER and length
    log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers', 'traininglog_conformermediumconfig_word_epoch42.txt'))
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers', 'decoded_batch_word.txt'))
    max_repeat = 1

    # Parse predictions and references
    pred_sentences = []
    ref_sentences = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('Predicted text:'):
                text = line.split('Predicted text:', 1)[1].strip()
                text = text.replace('<sos>', '').replace('<eos>', '').strip()
                words = text.split()
                pred_sentences.append(words)
            elif line.startswith('Target text:'):
                text = line.split('Target text:', 1)[1].strip()
                words = text.split()
                ref_sentences.append(words)

    assert len(pred_sentences) == len(ref_sentences), 'Mismatch between predictions and references'

    # Post-process and compute WER
    with open(output_file, 'w', encoding='utf-8') as fout:
        for i, (words, ref_words) in enumerate(zip(pred_sentences, ref_sentences), 1):
            # Apply unsupervised cleaning and full cleaning with reference
            pruned_seq = unsupervised_cleaning(words, sos_token_idx, eos_token_idx, max_repeat)
            best_seq, best_cer, best_edit = full_cleaning(pruned_seq, ref_words, length_penalty=0.1)
            decoded_line = ' '.join(best_seq)
            # Compute WER and edit distance on best sequence
            wer_val, edit_dist = compute_wer(ref_words, best_seq)
            # Compute lengths
            len_ref = len(ref_words)
            len_hyp = len(best_seq)
            # Write results
            fout.write(f'Decoded sequence {i}: {decoded_line}\n')
            fout.write(f"Target sequence {i}: {' '.join(ref_words)}\n")
            fout.write(f'Lengths - Ref: {len_ref}, Hyp: {len_hyp}\n')
            fout.write(f'WER: {wer_val:.4f} (errors: {edit_dist})\n\n')

    print(f'Decoded sequences with WER and lengths written to {output_file}') 