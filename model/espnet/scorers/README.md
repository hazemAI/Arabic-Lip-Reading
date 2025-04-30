# Scorers Module

This directory contains scoring modules used during beam search and decoding in the ESPNet-based lip reading system.

## Directory Structure

```text
scorers/
├── ctc.py           # CTC prefix scoring and interface wrapper
├── length_bonus.py  # Length bonus scorer for beam search
└── __init__.py      # Package initialization
```

## ctc.py - CTC Prefix Scorer

Provides a decoder interface for CTC-based scoring:

- **CTCPrefixScorer** (BatchPartialScorerInterface):
  - Wraps `espnet.ctc_prefix_score.CTCPrefixScore` and `CTCPrefixScoreTH`.
  - Implements methods for prefix scoring (`score_partial`), state initialization (`init_state`/`batch_init_state`), state selection, and streaming (`extend_prob`/`extend_state`).
  - Used to compute hybrid CTC/attention scores during beam search.

## length_bonus.py - Length Bonus Scorer

Implements a simple length bonus for beam search:

- **LengthBonus** (BatchScorerInterface):
  - Returns a constant positive score for each token to favor longer hypotheses.
  - Used in beam search to control output length bias.

## Usage

These scorers are registered in the beam search pipeline to provide additional scoring functions. For example:

```python
from espnet.scorers.ctc import CTCPrefixScorer
from espnet.scorers.length_bonus import LengthBonus

ctc_scorer = CTCPrefixScorer(ctc_module, eos_id)
length_scorer = LengthBonus(vocab_size)
```

Integrate with the beam search API to adjust hypothesis scores during decoding. 