# Conformer Encoder Implementation

This directory contains the implementation of the Conformer encoder, which combines self-attention and convolution modules for robust speech/lip reading feature extraction.

## File Structure

```text
encoder/
├── conformer_encoder.py   # Conformer encoder implementation (self-attention + convolution)
└── README.md              # This file
```

## conformer_encoder.py - Detailed Explanation

The `conformer_encoder.py` file implements the Conformer architecture, a state-of-the-art encoder that combines Transformer and CNN approaches for sequence modeling. The Conformer was originally designed for speech recognition but has been adapted for lip reading due to its ability to model both global and local dependencies effectively.

### Key Components

#### 1. ConvolutionModule

```python
class ConvolutionModule(torch.nn.Module):
    def __init__(self, channels, kernel_size, bias=True):
        # Implementation...
```

- **Purpose**: Implements the convolution module that captures local features using depthwise separable convolutions
- **Architecture**:
  - Pointwise convolution (1x1) that expands channels by a factor of 2
  - Gated Linear Unit (GLU) for non-linearity
  - Depthwise convolution with specified kernel size for capturing local context
  - Batch normalization for stable training
  - SiLU (Swish) activation function
  - Final pointwise convolution to restore the original channel dimension
- **Key Parameters**:
  - `channels`: Number of input/output channels
  - `kernel_size`: Size of the depthwise convolution kernel (typically large, e.g., 31)
- **Implementation Details**:
  - Uses transposed convolutions to handle sequence data (batch, time, channels)
  - Applies GLU activation after the first pointwise convolution
  - Separates convolution into depthwise and pointwise for efficiency

#### 2. EncoderLayer

```python
class EncoderLayer(torch.nn.Module):
    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        macaron_style=False,
    ):
        # Implementation...
```

- **Purpose**: Implements a single Conformer encoder layer that combines self-attention, convolution, and feed-forward modules
- **Architecture Components**:
  - Self-attention module (with optional relative positional encoding)
  - Feed-forward module (standard or macaron-style)
  - Convolution module (optional)
  - Layer normalization for each sub-module
  - Residual connections and dropout
- **Key Parameters**:
  - `size`: Dimension of the model
  - `self_attn`: Self-attention module for capturing global context
  - `feed_forward`: Position-wise feed-forward network
  - `conv_module`: Convolution module for capturing local features
  - `dropout_rate`: Dropout rate for regularization
  - `normalize_before`: Whether to apply layer normalization before each block
  - `concat_after`: Whether to concatenate attention input and output
  - `macaron_style`: Whether to use macaron-style architecture
- **Implementation Notes**:
  - Supports "macaron style" with feed-forward networks before and after self-attention
  - Uses half-scale (0.5) for feed-forward networks in macaron style
  - Supports caching for efficient inference

#### 3. ConformerEncoder

```python
class ConformerEncoder(torch.nn.Module):
    def __init__(
        self,
        attention_dim=768,
        attention_heads=12,
        linear_units=3072,
        num_blocks=12,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        normalize_before=True,
        concat_after=False,
        macaron_style=True,
        use_cnn_module=True,
        zero_triu=False,
        cnn_module_kernel=31,
    ):
        # Implementation...
```

- **Purpose**: Main Conformer encoder that stacks multiple encoder layers for sequence modeling
- **Architecture**:
  - Relative positional encoding instead of absolute positional encoding
  - Multiple encoder layers with self-attention, feed-forward, and convolution modules
  - Optional final layer normalization
- **Key Parameters**:
  - `attention_dim`: Dimension of attention and model (default: 768)
  - `attention_heads`: Number of attention heads (default: 12)
  - `linear_units`: Dimension of feed-forward network (default: 3072)
  - `num_blocks`: Number of encoder layers (default: 12)
  - `dropout_rate`: Dropout rate (default: 0.1)
  - `positional_dropout_rate`: Dropout rate for positional encoding (default: 0.1)
  - `attention_dropout_rate`: Dropout rate for attention (default: 0.0)
  - `normalize_before`: Whether to use layer normalization before each block (default: True)
  - `concat_after`: Whether to concatenate attention input and output (default: False)
  - `macaron_style`: Whether to use macaron-style architecture (default: True)
  - `use_cnn_module`: Whether to use convolution module (default: True)
  - `cnn_module_kernel`: Kernel size of convolution module (default: 31)
- **Implementation Notes**:
  - Uses repeated encoder layers with same architecture
  - Employs relative positional encoding for better generalization
  - Registers a pre-hook for loading pretrained weights

### Conformer Architecture Details

The Conformer architecture combines the strengths of both Transformers (for global context modeling) and CNNs (for local feature extraction) with several innovative components:

1. **Macaron-style Feed-Forward Modules**:

   - Sandwiches the self-attention and convolution modules between two feed-forward networks
   - Each feed-forward module contributes half of the residual connection (0.5 scale)

   ```
   FFN_1/2 → Self-Attention → Convolution → FFN_1/2
   ```

2. **Multi-Head Self-Attention with Relative Positional Encoding**:

   - Uses relative positional encoding instead of absolute positional encoding
   - Better captures the relative relationships between sequence elements
   - Improves generalization to sequences of different lengths

3. **Convolution Module**:

   - Captures local context using depthwise separable convolutions
   - Uses pointwise convolutions (1x1) before and after depthwise convolution
   - Applies batch normalization and SiLU activation
   - Significantly enhances local feature extraction

4. **Layer Normalization and Residual Connections**:
   - Applies layer normalization for training stability
   - Uses residual connections around each module to facilitate gradient flow
   - Supports both pre-norm and post-norm variations (default: pre-norm)

### Key Methods

#### ConformerEncoder.forward

```python
def forward(self, xs, masks):
    """Encode input sequence."""
```

- **Purpose**: Processes the input sequence through the entire encoder
- **Parameters**:
  - `xs`: Input tensor of shape (batch, time, features)
  - `masks`: Mask tensor for valid sequence positions
- **Returns**: Encoded sequence and updated masks
- **Process**:
  1. Applies positional encoding to the input
  2. Processes through multiple encoder layers
  3. Applies final normalization if required
  4. Returns the encoded sequence and masks
- **Usage**: Main method for encoding sequence data during training and inference

#### ConformerEncoder.forward_one_step

```python
def forward_one_step(self, xs, masks, cache=None):
    """Encode input frame."""
```

- **Purpose**: Processes a single step of the input sequence with caching for efficient inference
- **Parameters**:
  - `xs`: Input tensor for the current step
  - `masks`: Mask tensor for valid positions
  - `cache`: Cache from previous steps for efficient processing
- **Returns**: Encoded frame, updated masks, and new cache
- **Use Case**: Used for incremental decoding during inference, particularly for streaming applications

### Integration with Lip Reading System

In the Arabic lip reading system, the Conformer encoder serves as a powerful temporal modeling component that processes visual features:

1. **Feature Processing**:

   - Takes visual features extracted from the visual frontend (ResNet/ShuffleNetV2)
   - Applies relative positional encoding to capture temporal relationships
   - Processes sequences of arbitrary length with appropriate masking

2. **Advantages for Lip Reading**:

   - The self-attention mechanism captures global dependencies in the lip sequence
   - The convolution module captures local motion patterns critical for lip reading
   - The combination results in more accurate lip movement interpretation

3. **Integration with Decoder**:
   - The encoded representations serve as "memory" for the transformer decoder
   - The attention mechanism in the decoder attends to these encoded features
   - The masked self-attention in the decoder handles autoregressive generation

### Usage Example

```python
import torch
from model.espnet.encoder.conformer_encoder import ConformerEncoder
from model.espnet.nets_utils import make_non_pad_mask

# Create a Conformer encoder
encoder = ConformerEncoder(
    attention_dim=256,            # Feature dimension
    attention_heads=4,            # Number of attention heads
    linear_units=1024,            # Feed-forward dimension
    num_blocks=6,                 # Number of encoder blocks
    dropout_rate=0.1,             # Dropout rate
    cnn_module_kernel=15          # Kernel size for convolution module
)

# Prepare input sequence
batch_size = 2
seq_length = 50
feature_dim = 256
sequence = torch.randn(batch_size, seq_length, feature_dim)
lengths = torch.tensor([50, 30])  # Actual sequence lengths

# Create padding mask
masks = make_non_pad_mask(lengths).unsqueeze(-2)

# Encode sequence
encoded_sequence, encoded_masks = encoder(sequence, masks)
# encoded_sequence shape: [2, 50, 256]
```

The Conformer encoder represents a powerful advancement in sequence modeling by effectively combining the global modeling capabilities of Transformers with the local feature extraction of CNNs. This makes it particularly well-suited for lip reading, where both global context and local motion patterns are crucial for accurate interpretation.
