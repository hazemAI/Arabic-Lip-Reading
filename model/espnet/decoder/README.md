# Decoder Components

This directory contains the transformer-based decoder implementations used for autoregressive sequence generation in lip reading and speech recognition.

## File Structure

```text
decoder/
└── transformer_decoder.py  # TransformerDecoder and DecoderLayer implementation
```

## transformer_decoder.py - Detailed Explanation

- **DecoderLayer**: Single decoder layer combining self-attention, source-attention, and position-wise feed-forward sublayers with optional pre-norm or post-norm and concatenation.

- **TransformerDecoder**: Stacked decoder layers that produce output logits over the vocabulary. Inherits from `BatchScorerInterface` and `torch.nn.Module`.

  - Constructor parameters:
    - `odim`: Output vocabulary size.
    - `attention_dim`: Dimension for attention (model size).
    - `attention_heads`: Number of attention heads.
    - `linear_units`: Hidden units in feed-forward layers.
    - `num_blocks`: Number of decoder layers.
    - `dropout_rate`, `self_attention_dropout_rate`, etc.

  - Key methods:
    - `forward(tgt, tgt_mask, memory, memory_mask)`: Compute decoder outputs during training.
    - `forward_one_step(tgt, tgt_mask, memory, memory_mask, cache)`: Incremental decoding step with caching support.
    - `score(ys, state, x)`: Compute scores for a single token in beam search.
    - `batch_score(ys, states, xs)`: Compute batch scores for beam search.

## Usage Example

```python
from espnet.decoder.transformer_decoder import TransformerDecoder

decoder = TransformerDecoder(
    odim=full_vocab_size,
    attention_dim=512,
    attention_heads=8,
    linear_units=2048,
    num_blocks=6,
    dropout_rate=0.1
)
``` 