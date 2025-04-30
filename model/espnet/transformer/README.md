# Transformer Components

This directory contains the core transformer components used in the lip reading system. Each file implements a specific part of the transformer architecture.

## File Structure

```text
transformer/
├── add_sos_eos.py                 # Add SOS and EOS tokens to sequences
├── label_smoothing_loss.py        # Label smoothing loss implementation
├── attention.py                   # Attention mechanisms (multi-head, relative positions)
├── embedding.py                   # Input embeddings and positional encoding
├── layer_norm.py                  # Layer normalization module
├── mask.py                        # Sequence masking utilities
├── positionwise_feed_forward.py   # Position-wise FeedForward networks
├── repeat.py                      # Module repetition utility
```

## Detailed Explanations

### 1. attention.py

This file implements attention mechanisms for transformer models, including the standard multi-head attention and relative positional multi-head attention.

#### Key Components

##### MultiHeadedAttention

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate):
        # Implementation...
```

- **Purpose**: Implements the multi-head attention mechanism as described in "Attention Is All You Need"
- **Key Parameters**:
  - `n_head`: Number of attention heads
  - `n_feat`: Feature dimension
  - `dropout_rate`: Dropout rate for regularization
- **Key Methods**:
  - `forward_qkv`: Transforms query, key, and value tensors into multi-head format
  - `forward_attention`: Computes attention weights and applies them to values
  - `forward`: Main method that computes scaled dot-product attention
- **Implementation Details**:
  - Splits input into multiple heads with dimension `d_k = n_feat // n_head`
  - Performs scaled dot-product attention separately for each head
  - Concatenates the outputs from each head and applies a linear transformation
  - Maintains attention weights for visualization or analysis

##### RelPositionMultiHeadedAttention

```python
class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        # Implementation...
```

- **Purpose**: Extends multi-head attention with relative positional encoding for better sequence modeling
- **Additional Parameters**:
  - `zero_triu`: Option to zero out the upper triangular part of the attention matrix
- **Additional Components**:
  - `linear_pos`: Linear transformation for positional embeddings
  - `pos_bias_u` and `pos_bias_v`: Learnable biases for relative position calculations
  - `rel_shift`: Method to compute relative positional encoding
- **Unique Features**:
  - Implements the attention mechanism from "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
  - Computes four attention terms (matrices a, b, c, d) for better relative position modeling
  - Supports causal attention through the zero_triu option

#### Implementation Details

The attention implementation follows these steps:

1. **Query, Key, Value Transformation**:

   ```python
   q, k, v = self.forward_qkv(query, key, value)
   ```

   - Applies linear projections to query, key, and value
   - Reshapes tensors for multi-head processing

2. **Attention Score Calculation**:

   ```python
   scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
   ```

   - Computes dot product between query and key
   - Scales by √d_k to stabilize gradients

3. **Masking and Softmax**:

   ```python
   if mask is not None:
       mask = mask.unsqueeze(1).eq(0)
       scores = scores.masked_fill(mask, min_value)
   self.attn = torch.softmax(scores, dim=-1)
   ```

   - Applies mask to prevent attention to pad tokens or future tokens
   - Computes softmax to get attention weights

4. **Output Computation**:
   ```python
   x = torch.matmul(p_attn, value)
   x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
   return self.linear_out(x)
   ```
   - Applies attention weights to values
   - Concatenates outputs from all heads
   - Applies final linear transformation

#### Usage in the System

- Used in both encoder and decoder self-attention mechanisms
- The RelPositionMultiHeadedAttention is used in the Conformer encoder for better modeling of sequential data
- Enables the model to capture long-range dependencies in lip movements

### 2. embedding.py

This file implements positional encoding modules for transformer models, providing information about the position of tokens in a sequence.

#### Key Components

##### PositionalEncoding

```python
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        # Implementation...
```

- **Purpose**: Adds positional information to input embeddings using sine and cosine functions
- **Key Parameters**:
  - `d_model`: Embedding dimension
  - `dropout_rate`: Dropout rate for regularization
  - `max_len`: Maximum sequence length
  - `reverse`: Whether to reverse position indices (legacy parameter)
- **Key Methods**:
  - `extend_pe`: Creates or extends the positional encoding matrix
  - `forward`: Adds positional encoding to input embeddings
- **Implementation Details**:
  - Uses sine for even indices and cosine for odd indices
  - Scales input embeddings by √d_model before adding positional encoding
  - Supports extension to longer sequences dynamically

##### ScaledPositionalEncoding

```python
class ScaledPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model, dropout_rate, max_len=5000):
        # Implementation...
```

- **Purpose**: Adds a learnable scalar parameter to control the influence of positional encoding
- **Additional Components**:
  - `alpha`: Learnable parameter to scale the positional encoding
- **Implementation Details**:
  - Instead of adding PE directly, adds α·PE to the input
  - Allows the model to learn the optimal balance between content and position

##### RelPositionalEncoding

```python
class RelPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=5000):
        # Implementation...
```

- **Purpose**: Creates relative positional encodings for more effective sequence modeling
- **Key Methods**:
  - `extend_pe`: Creates or extends the relative positional encoding matrix
  - `forward`: Returns both scaled input and positional embedding separately
- **Implementation Details**:
  - Creates both positive and negative positional encodings
  - Manages a sliding window of positions for efficient relative position calculation
  - Returns both the scaled input and position embeddings separately for use in RelPositionMultiHeadedAttention

#### Key Concepts

1. **Absolute vs. Relative Positional Encoding**:

   - Absolute: Each position has a fixed encoding
   - Relative: Encodes relationships between positions

2. **Sinusoidal Position Encoding**:
   ```python
   position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
   div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
   pe[:, 0::2] = torch.sin(position * div_term)
   pe[:, 1::2] = torch.cos(position * div_term)
   ```
   - Uses sine and cosine functions of different frequencies
   - Provides unique encoding for each position
   - Allows model to generalize to sequence lengths not seen during training

#### Usage in the System

- PositionalEncoding is used in standard Transformer implementations
- RelPositionalEncoding is used in the Conformer encoder for lip reading
- Provides critical sequential information that helps the model understand the order of lip movements

### 3. layer_norm.py

This file implements layer normalization for transformer models, which normalizes the inputs across features.

#### Key Components

##### LayerNorm

```python
class LayerNorm(torch.nn.LayerNorm):
    def __init__(self, nout, dim=-1):
        # Implementation...
```

- **Purpose**: Extends PyTorch's LayerNorm to support normalization along different dimensions
- **Key Parameters**:
  - `nout`: Output dimension size (normalized feature dimension)
  - `dim`: Dimension to normalize along (default: -1)
- **Implementation Details**:
  - Wraps PyTorch's LayerNorm with a smaller epsilon (1e-12) for numerical stability
  - Adds support for normalizing along different dimensions through transpose operations
  - Preserves the original tensor shape after normalization

#### Implementation Details

The implementation handles normalization along different dimensions:

```python
def forward(self, x):
    if self.dim == -1:
        return super(LayerNorm, self).forward(x)
    return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)
```

- If `dim=-1`, applies standard layer normalization along the last dimension
- Otherwise, transposes the tensor to normalize along the specified dimension, then transposes back

#### Usage in the System

- Used throughout the transformer architecture, particularly:
  - Before and/or after attention mechanisms
  - Before and/or after feed-forward networks
  - As the final normalization in encoder and decoder blocks
- Critical for training stability by keeping activations in a reasonable range
- Helps the model converge faster and achieve better performance

### 4. mask.py

This file implements masking utilities for transformer models, particularly for self-attention in the decoder.

#### Key Components

##### subsequent_mask

```python
def subsequent_mask(size, device="cpu", dtype=torch.bool):
    """Create mask for subsequent steps (size, size)."""
```

- **Purpose**: Creates a lower triangular mask to prevent attending to future positions
- **Parameters**:
  - `size`: Size of the mask (sequence length)
  - `device`: Device for the tensor
  - `dtype`: Data type for the mask
- **Returns**: A lower triangular mask where 1s indicate positions that can be attended to
- **Implementation Details**:
  - Creates a matrix of ones
  - Applies `torch.tril` to zero out the upper triangular part
  - Used for causal (autoregressive) attention in decoders

##### target_mask

```python
def target_mask(ys_in_pad, ignore_id):
    """Create mask for decoder self-attention."""
```

- **Purpose**: Combines padding mask and subsequent mask for decoder self-attention
- **Parameters**:
  - `ys_in_pad`: Batch of padded target sequences
  - `ignore_id`: Index for padding
- **Returns**: Combined mask where False indicates positions to mask
- **Implementation Details**:
  - Creates a padding mask where padding positions are False
  - Creates a subsequent mask to prevent attending to future positions
  - Combines the two masks with logical AND to mask both padding and future positions

#### Usage in the System

- The subsequent_mask is used in the decoder to implement causal attention
- The target_mask is used in the decoder self-attention to handle both padding and causality
- Essential for autoregressive decoding in the lip reading system
- Ensures the decoder can only attend to previous positions during generation

### 5. positionwise_feed_forward.py

This file implements the position-wise feed-forward networks used in transformer models.

#### Key Components

##### PositionwiseFeedForward

```python
class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, idim, hidden_units, dropout_rate):
        # Implementation...
```

- **Purpose**: Implements the feed-forward network used in transformer layers
- **Key Parameters**:
  - `idim`: Input dimension
  - `hidden_units`: Number of hidden units
  - `dropout_rate`: Dropout rate for regularization
- **Implementation Details**:
  - Two linear transformations with a ReLU activation in between
  - Dropout applied after the first linear transformation
  - Follows the formula: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
  - Processes each position independently (position-wise)

#### Usage in the System

- Used in both encoder and decoder layers after the attention mechanism
- Provides additional non-linearity and capacity to the model
- In the Conformer encoder, used in macaron-style configuration (before and after attention)
- Helps transform the attention outputs into more useful representations

### 6. repeat.py

This file implements utilities for repeating layers in transformer models, with support for layer dropout.

#### Key Components

##### MultiSequential

```python
class MultiSequential(torch.nn.Sequential):
    def __init__(self, *args, layer_drop_rate=0.0):
        # Implementation...
```

- **Purpose**: Extends PyTorch's Sequential to handle multiple inputs/outputs and layer dropout
- **Key Parameters**:
  - `*args`: Sequence of modules to be added
  - `layer_drop_rate`: Probability of dropping out each layer during training
- **Implementation Details**:
  - Passes multiple arguments through each layer in sequence
  - Supports stochastic depth (layer dropout) during training
  - Each layer has a probability `layer_drop_rate` of being skipped

##### repeat

```python
def repeat(N, fn, layer_drop_rate=0.0):
    """Repeat module N times."""
```

- **Purpose**: Creates a stack of N identical layers with optional layer dropout
- **Parameters**:
  - `N`: Number of times to repeat the module
  - `fn`: Function to generate the module
  - `layer_drop_rate`: Probability of dropping each layer during training
- **Returns**: A MultiSequential container with N instances of the module
- **Implementation Details**:
  - Creates N instances of the module using the provided function
  - Wraps them in a MultiSequential container with the specified layer drop rate
  - Used to create stacks of encoder or decoder layers

#### Usage in the System

- Used to create multiple encoder layers in the Conformer encoder
- Supports stochastic depth, which can improve training of very deep networks
- Handles the passing of multiple arguments between layers
- Simplifies the creation of repeated layer stacks

## Integration in the Lip Reading System

These transformer components work together to form the building blocks of the lip reading system:

1. **Encoder-Decoder Architecture**:

   - The encoder processes visual features from lip movements
   - The decoder generates text predictions autoregressively

2. **Self-Attention Mechanism**:

   - Captures global dependencies in the visual sequence
   - Uses relative positional encoding for better sequence modeling

3. **Cross-Attention Mechanism**:

   - Allows the decoder to attend to encoder outputs
   - Creates alignment between lip movements and text

4. **Positional Information**:

   - Positional encodings provide sequence order information
   - Relative positions help model relationships between frames

5. **Layer Normalization and Feed-Forward Networks**:

   - Stabilize training and increase model capacity
   - Process information at each position independently

6. **Masking**:
   - Enables handling of variable-length sequences
   - Ensures autoregressive generation in the decoder

These components form a powerful sequence-to-sequence architecture that can effectively map lip movements to text in the Arabic lip reading system.
