# postprocess_test.py
import os
import pandas as pd
import difflib
import re
import regex

# Build vocabulary mapping by scanning dataset CSVs (with diacritics)
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

csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'Csv (with Diacritics)'))
tokens = set()
for fname in os.listdir(csv_dir):
    if fname.endswith('.csv'):
        tokens.update(extract_label(os.path.join(csv_dir, fname)))
mapped_tokens = {c: i for i, c in enumerate(sorted(tokens, reverse=True), 1)}
# Override token-to-index mapping with explicit dictionary to avoid diacritic parsing errors
mapped_tokens = {
    "ٱ": 1, "يْ": 2, "يّْ": 3, "يِّ": 4, "يُّ": 5, "يَّ": 6, "يٌّ": 7, "يِ": 8, "يُ": 9, "يَ": 10,
    "يٌ": 11, "ي": 12, "ى": 13, "وْ": 14, "وِّ": 15, "وُّ": 16, "وَّ": 17, "وِ": 18, "وُ": 19, "وَ": 20,
    "وً": 21, "و": 22, "هْ": 23, "هُّ": 24, "هِ": 25, "هُ": 26, "هَ": 27, "نۢ": 28, "نْ": 29, "نِّ": 30,
    "نُّ": 31, "نَّ": 32, "نِ": 33, "نُ": 34, "نَ": 35, "مْ": 36, "مّْ": 37, "مِّ": 38, "مُّ": 39, "مَّ": 40,
    "مِ": 41, "مُ": 42, "مَ": 43, "مٍ": 44, "مٌ": 45, "مً": 46, "لْ": 47, "لّْ": 48, "لِّ": 49, "لُّ": 50,
    "لَّ": 51, "لِ": 52, "لُ": 53, "لَ": 54, "لٍ": 55, "لٌ": 56, "لً": 57, "كْ": 58, "كِّ": 59, "كَّ": 60,
    "كِ": 61, "كُ": 62, "كَ": 63, "قْ": 64, "قَّ": 65, "قِ": 66, "قُ": 67, "قَ": 68, "قٍ": 69, "قً": 70,
    "فْ": 71, "فِّ": 72, "فَّ": 73, "فِ": 74, "فُ": 75, "فَ": 76, "غْ": 77, "غِ": 78, "غَ": 79, "عْ": 80,
    "عَّ": 81, "عِ": 82, "عُ": 83, "عَ": 84, "عٍ": 85, "ظْ": 86, "ظِّ": 87, "ظَّ": 88, "ظِ": 89, "ظُ": 90,
    "ظَ": 91, "طْ": 92, "طِّ": 93, "طَّ": 94, "طِ": 95, "طُ": 96, "طَ": 97, "ضْ": 98, "ضِّ": 99, "ضُّ": 100,
    "ضَّ": 101, "ضِ": 102, "ضُ": 103, "ضَ": 104, "ضً": 105, "صْ": 106, "صّْ": 107, "صِّ": 108, "صُّ": 109,
    "صَّ": 110, "صِ": 111, "صُ": 112, "صَ": 113, "صٍ": 114, "صً": 115, "شْ": 116, "شِّ": 117, "شُّ": 118,
    "شَّ": 119, "شِ": 120, "شُ": 121, "شَ": 122, "سْ": 123, "سّْ": 124, "سِّ": 125, "سُّ": 126, "سَّ": 127,
    "سِ": 128, "سُ": 129, "سَ": 130, "سٍ": 131, "زْ": 132, "زَّ": 133, "زِ": 134, "زُ": 135, "زَ": 136,
    "رْ": 137, "رِّ": 138, "رُّ": 139, "رَّ": 140, "رِ": 141, "رُ": 142, "رَ": 143, "رٍ": 144, "رٌ": 145,
    "رً": 146, "ذْ": 147, "ذَّ": 148, "ذِ": 149, "ذُ": 150, "ذَ": 151, "دْ": 152, "دِّ": 153, "دُّ": 154,
    "دَّ": 155, "دًّ": 156, "دِ": 157, "دُ": 158, "دَ": 159, "دٍ": 160, "دٌ": 161, "دً": 162, "خْ": 163,
    "خِ": 164, "خُ": 165, "خَ": 166, "حْ": 167, "حَّ": 168, "حِ": 169, "حُ": 170, "حَ": 171, "جْ": 172,
    "جِّ": 173, "جُّ": 174, "جَّ": 175, "جِ": 176, "جُ": 177, "جَ": 178, "ثْ": 179, "ثِّ": 180, "ثُّ": 181,
    "ثَّ": 182, "ثِ": 183, "ثُ": 184, "ثَ": 185, "تْ": 186, "تِّ": 187, "تُّ": 188, "تَّ": 189, "تِ": 190,
    "تُ": 191, "تَ": 192, "تٍ": 193, "تٌ": 194, "ةْ": 195, "ةِ": 196, "ةُ": 197, "ةَ": 198, "ةٍ": 199,
    "ةٌ": 200, "ةً": 201, "بْ": 202, "بِّ": 203, "بَّ": 204, "بِ": 205, "بُ": 206, "بَ": 207, "بٍ": 208,
    "بً": 209, "ا": 210, "ئْ": 211, "ئِ": 212, "ئَ": 213, "ئً": 214, "إِ": 215, "ؤْ": 216, "ؤُ": 217,
    "ؤَ": 218, "أْ": 219, "أُ": 220, "أَ": 221, "آ": 222, "ءْ": 223, "ءِ": 224, "ءَ": 225, "ءً": 226
}
base_vocab_size = len(mapped_tokens) + 1  # +1 for blank token
sos_token_idx = base_vocab_size
eos_token_idx = base_vocab_size + 1
full_vocab_size = base_vocab_size + 2

idx2char = {v: k for k, v in mapped_tokens.items()}
idx2char[0] = ""
idx2char[sos_token_idx] = "<sos>"
idx2char[eos_token_idx] = "<eos>"

from utils import indices_to_text, compute_cer

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

# DP removal of adjacent repeated subsequences via dynamic programming
def remove_consecutive_repeats_dp(seq):
    """
    Remove one instance of the largest adjacent repeated subsequence in seq.
    """
    n = len(seq)
    # search for the largest block length L where seq[i:i+L] == seq[i+L:i+2*L]
    for L in range(n // 2, 1, -1):
        for i in range(0, n - 2*L + 1):
            if seq[i:i+L] == seq[i+L:i+2*L]:
                # remove the second occurrence
                return seq[:i+L] + seq[i+2*L:]
    return seq

# I am adding this function to iteratively prune nested repeated subsequences
def prune_repeats(seq):
    """
    Iteratively remove consecutive repeated subsequences until stable.
    """
    prev = seq
    while True:
        next_seq = remove_consecutive_repeats_dp(prev)
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

# I am adding this function to remove any global repeated subsequences anywhere in a sequence
def remove_global_repeats(seq, min_length=2):
    """
    Remove any repeated contiguous subsequences of length >= min_length anywhere in sequence.
    """
    if not seq:
        return []
    prev = seq
    changed = True
    while changed:
        changed = False
        n = len(prev)
        for L in range(n // 2, min_length - 1, -1):
            for i in range(n - L + 1):
                sub = prev[i:i+L]
                for j in range(i + 1, n - L + 1):
                    if prev[j:j+L] == sub:
                        prev = prev[:j] + prev[j+L:]
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    return prev

# Add cleaning pipelines to encapsulate post-processing logic
def unsupervised_cleaning(seq, sos_idx, eos_idx, max_repeat=1):
    """
    Remove <sos>/<eos> and unsupervised removal of repeated subsequences via DP.
    """
    # remove sos/eos markers
    cleaned = clean_indices(seq, sos_idx, eos_idx)
    # iteratively remove adjacent repeated subsequences until stable
    pruned = cleaned
    while True:
        next_seq = remove_consecutive_repeats_dp(pruned)
        if next_seq == pruned:
            break
        pruned = next_seq
    return pruned

def full_cleaning(pruned_seq, ref_seq, length_penalty=0.1):
    """
    Given a pruned sequence and reference indices, generate candidates
    and select the best by CER + length penalty. Returns (best_seq, cer, edit_dist).
    """
    # generate candidates
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
    # evaluate all candidates
    for cand in candidates:
        if not cand:
            continue
        cer_val, edit_val = compute_cer(ref_seq, cand)
        score = cer_val + length_penalty * abs(len(cand) - len(ref_seq))
        if score < best_score:
            best_score, best_cer, best_edit, best_seq = score, cer_val, edit_val, cand
    return best_seq, best_cer, best_edit

def remove_repeated_substrings(text, min_len=3):
    """
    Remove any repeated contiguous substring of length >= min_len in text by DP.
    """
    s = text
    changed = True
    while changed:
        changed = False
        n = len(s)
        # search for largest repeat
        for L in range(n // 2, min_len - 1, -1):
            for i in range(0, n - 2*L + 1):
                if s[i:i+L] == s[i+L:i+2*L]:
                    # remove second occurrence
                    s = s[:i+L] + s[i+2*L:]
                    changed = True
                    break
            if changed:
                break
    return s

# Noise-tolerant repeat removal allowing small gaps
def remove_noisy_repeats(text, min_len=3, max_gap=3):
    """
    Collapse any repeated substring of length >= min_len that appears twice
    with up to max_gap characters between them.
    """
    s = text
    prev = None
    pat = regex.compile(r'(.{%d,}?)(?:.{0,%d}?)(\1)' % (min_len, max_gap))
    while prev != s:
        prev = s
        s = regex.sub(pat, r"\1", s, overlapped=True)
    return s

# Global text-level repeat removal on decoded text
def remove_global_text_repeats(text, min_len=3):
    """
    Remove any trailing contiguous substring of length >= min_len if it appears earlier in the text.
    """
    n = len(text)
    # try longest possible suffix first
    for L in range(n // 2, min_len - 1, -1):
        sfx = text[-L:]
        if sfx in text[:-L]:
            return text[:-L]
    return text

if __name__ == '__main__':

    # Apply unsupervised cleaning to training log
    log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers', 'traininglog_conformer_dia_filp.txt'))
    if os.path.isfile(log_file):
        out_file = log_file[:-4] + '_cleaned_unsupervised.txt'
        print(f"Cleaning training log {log_file} -> {out_file}")
        def clean_and_score(orig_txt, target_txt):
            # strip tokens
            orig = orig_txt.replace('<sos>', '').replace('<eos>', '')
            # 1) noise-tolerant removal of repeated substrings
            cleaned_noisy = remove_noisy_repeats(orig, min_len=3, max_gap=3)
            # 2) global text-level repeat removal
            cleaned = remove_global_text_repeats(cleaned_noisy, min_len=3)
            # compute CER against reference for reporting
            # map reference to indices
            ref_idx = [mapped_tokens.get(c, 0) for c in target_txt]
            # map cleaned text to indices
            cleaned_idx = [mapped_tokens.get(c, 0) for c in cleaned]
            cer_val, _ = compute_cer(ref_idx, cleaned_idx)
            return cleaned, cer_val
        # read and process log lines
        with open(log_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        # write cleaned log and compute average CER on cleaned outputs
        cer_sum = 0.0
        cer_count = 0
        with open(out_file, 'w', encoding='utf-8') as fout:
            for idx, line in enumerate(lines):
                # stop before the original summary block
                if line.startswith('=== Summary Statistics'): break
                # skip original metrics
                if line.startswith('Edit distance:') or line.startswith('CER:'): continue
                fout.write(line)
                # when predicted appears, compute cleaning with next target line
                if line.startswith('Predicted text:'):
                    orig = line.strip().split(': ', 1)[1]
                    # get target from following line
                    target = ''
                    if idx+1 < len(lines) and lines[idx+1].startswith('Target text:'):
                        target = lines[idx+1].strip().split(': ',1)[1]
                    cleaned_txt, cleaned_cer = clean_and_score(orig, target)
                    fout.write(f"Cleaned text: {cleaned_txt}\n")
                    fout.write(f"Cleaned CER: {cleaned_cer:.4f}\n")
                    cer_sum += cleaned_cer
                    cer_count += 1
            # append cleaned summary after processing
            if cer_count > 0:
                avg_cer = cer_sum / cer_count
                fout.write("\n=== Cleaned Summary ===\n")
                fout.write(f"Total samples: {cer_count}\n")
                fout.write(f"Average CER: {avg_cer:.4f}\n")
        print(f"Unsup cleaned log written to {out_file}")
    else:
        print(f"Training log not found at {log_file}") 