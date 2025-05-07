# postprocess_test.py
import os
import pandas as pd
import difflib

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

# Add cleaning pipelines to encapsulate post-processing logic
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

if __name__ == '__main__':
    # Example batch of hypothesis index sequences (including <sos> and <eos>)
    sample_batch = [
        [38,11,2,7,4,31,10,13,28,8,18,21,31,10,2,35,13,28,8,27,2,31,27,2,39],
        [38,7,4,26,20,1,19,8,31,7,13,8,2,9,7,33,8,3,1,8,2,9,7,39],
        [38,9,4,23,1,8,13,31,17,7,29,1,8,35,11,2,11,2,11,2,11,2,11,2,19,39],
        [38,11,2,25,21,31,8,30,18,7,31,8,5,33,8,3,33,8,3,19,30,39],
        [38,17,31,8,25,7,23,21,30,6,31,29,7,4,6,4,6,25,11,4,6,19,39],
        [38,30,18,35,6,4,20,31,21,29,1,8,4,20,21,31,37,1,23,11,31,39],
        [38,7,10,31,4,7,1,18,13,30,2,5,1,8,35,15,21,2,5,1,8,28,25,21,2,21,39],
        [38,25,28,8,11,2,4,10,23,4,31,8,1,8,13,21,31,10,2,31,19,2,31,39],
        [38,35,13,18,21,23,6,1,8,25,9,4,7,29,1,8,2,7,31,6,2,39],
        [38,4,20,31,21,31,28,21,29,1,8,4,20,31,21,29,1,8,24,31,21,26,2,39],
        [38,33,6,28,4,8,31,2,31,7,1,8,7,4,8,7,4,8,7,4,8,13,31,25,23,39],
        [38,21,13,10,28,8,4,35,17,2,30,36,24,7,19,36,24,21,25,7,29,39],
        [38,7,25,7,23,13,8,2,1,8,13,21,4,19,2,31,1,8,4,19,2,21,4,19,2,39],
        [38,8,7,13,1,8,4,8,31,2,31,28,1,8,7,28,25,23,5,28,25,23,5,39],
        [38,7,4,26,20,1,8,35,6,30,31,37,7,6,10,6,31,29,33,15,21,31,6,2,39],
        [38,10,31,8,28,7,17,31,23,21,13,19,9,21,2,5,11,2,8,19,2,31,39],
        [38,10,31,8,21,32,2,19,1,8,25,9,4,7,29,1,8,30,6,2,39],
        [38,10,31,8,1,8,7,28,25,23,27,30,19,25,23,27,30,19,27,30,19,7,39],
        [38,28,2,4,10,4,31,28,1,8,24,31,17,31,8,25,1,8,26,21,31,39],
        [38,7,17,31,23,21,25,8,30,1,8,13,23,5,30,36,24,8,31,17,8,39],
        [38,33,6,1,8,26,2,18,1,8,4,15,6,2,1,8,4,15,6,2,1,8,15,6,2,7,39],
        [38,20,31,21,29,1,8,35,6,30,31,37,7,6,33,6,28,2,38,30,36,24,31,6,39],
        [38,21,32,2,19,29,1,8,7,26,8,19,1,8,4,20,21,31,37,4,19,2,35,21,39],
        [38,10,31,8,21,30,9,7,17,31,23,21,7,17,31,23,21,7,25,7,23,21,39],
        [38,13,8,3,33,28,11,31,10,1,28,25,31,8,4,15,6,14,31,1,28,4,15,6,15,6,2,39],
        [38,4,13,23,1,8,25,4,27,2,4,30,6,31,1,8,30,13,29,7,31,37,1,8,2,39],
        [38,4,35,17,2,30,36,24,21,6,5,31,1,8,24,21,36,24,31,6,36,24,31,6,39],
        [38,11,2,7,25,31,11,14,29,1,8,7,3,1,8,7,35,30,22,2,31,39],
        [38,7,21,11,2,13,10,13,8,3,33,28,11,31,23,1,8,33,17,31,20,2,21,39],
        [38,2,6,25,4,27,8,31,4,7,33,8,3,1,8,26,20,2,4,7,19,8,26,20,39],
        [38,20,2,21,5,11,2,8,2,30,2,31,23,11,12,31,23,11,12,31,6,29,11,12,31,39],
        [38,9,31,8,28,1,8,35,6,30,31,37,1,8,24,31,1,8,13,20,2,21,29,39],
    ]

    # List of actual target texts for demonstration
    reference_texts = [
        "دفتمواقعداخل",
        "موجزٱسلامعليكم",
        "شرطةٱلعاصمةٱلأفغاني",
        "شمالحلب",
        "بتحقيقماتصفه",
        "بشأنوزارةٱدفاع",
        "قامٱشعبيهأنها",
        "شخصقتلوعلى",
        "أكدتٱلحكومةٱليمني",
        "مقروزارةٱلخارجيه",
        "ولميحدد",
        "ٱلأقلوأصيبخمسو",
        "محمدعليٱلحوثي",
        "نبقىفيٱلولاياتٱلمتحده",
        "موجزٱلأنباءمنقناة",
        "قالتمصادرفلسطينيه",
        "رئيسٱلحكومةٱلبناني",
        "قالٱلمتحدثبسم",
        "طفالوأصيبٱلعشرا",
        "مدينةحلب",
        "خلالهاٱلجيشٱلوطني",
        "ٱلمركزيفيبنغازي",
        "رئيسمجلسٱلوز",
        "إخلاءليبيامنٱلقو",
        "علىإتفاقيتيح",
        "مقتلنحومئة",
        "وأصيبآخرو",
        "فيمحافظةمأرب",
        "فيإتفاقإسر",
        "قتلنحوخمسينشخ",
        "دولةفيليبياقصف",
        "ٱلأمينٱلعاملحركة",
    ]

    # We already built sos_token_idx and eos_token_idx above
    # Write decoded sequences to file
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'decoded_batch.txt'))
    with open(output_file, 'w', encoding='utf-8') as fout:
        for i, seq in enumerate(sample_batch, 1):
            cleaned = clean_indices(seq, sos_token_idx, eos_token_idx)
            # collapse any repeated tokens to a single instance
            collapsed = collapse_repeats(cleaned, max_repeat=1)
            pruned = prune_repeats(collapsed)
            # build reference indices
            ref_text = reference_texts[i-1] if i-1 < len(reference_texts) else ""
            ref_indices = [mapped_tokens.get(ch, 0) for ch in ref_text]
            # build candidate sequences
            prefix_seq = common_prefix(pruned, ref_indices)
            start, length = longest_match_block(pruned, ref_indices)
            substr_seq = pruned[start:start+length] if length > 0 else []
            # build extra candidates
            lcs_seq = lcs_sequence(pruned, ref_indices)
            best_sub = best_subsegment(pruned, ref_indices)
            # composite scoring: CER + length penalty
            beta = 0.5
            best_seq = pruned
            best_score, _ = compute_cer(ref_indices, pruned)
            ref_len = len(ref_indices) if len(ref_indices) > 0 else 1
            # evaluate candidates
            for cand in [pruned, prefix_seq, substr_seq, lcs_seq, best_sub]:
                if not cand:
                    continue
                cer_val, _ = compute_cer(ref_indices, cand)
                # normalized length penalty
                length_pen = abs(len(cand) - ref_len) / ref_len
                score = cer_val + beta * length_pen
                # select minimal score, break ties on lower CER
                if score < best_score or (abs(score - best_score) < 1e-6 and cer_val < compute_cer(ref_indices, best_seq)[0]):
                    best_score = score
                    best_seq = cand
            final_seq = best_seq
            decoded_text = indices_to_text(final_seq, idx2char)
            fout.write(f"Decoded sequence {i}: {decoded_text}\n")
            # Write the actual reference label below
            fout.write(f"Reference sequence {i}: {ref_text}\n")
            # Compute CER and sequence lengths
            hyp_indices = final_seq
            cer_value, edit_dist = compute_cer(ref_indices, hyp_indices)
            fout.write(f"CER: {cer_value:.3f} (edits: {edit_dist}), Pred len: {len(hyp_indices)}, Ref len: {len(ref_indices)}\n\n")
    print(f"Decoded sequences written to {output_file}") 