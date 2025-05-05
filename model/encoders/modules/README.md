# Lip Reading Model Architectures

This directory contains various neural network architectures used in the lip reading system. Each file implements a specific model architecture optimized for visual feature extraction and temporal modeling.

## File Structure

```
models/
├── densetcn.py          # Dense Temporal Convolutional Network
├── resnet.py            # ResNet backbone implementation
├── tcn.py               # Temporal Convolutional Network
└── se_module.py         # Squeeze-and-Excitation module
```

## Detailed File Explanations

### 1. densetcn.py

This file implements the Dense Temporal Convolutional Network (DenseTCN), a variant of TCN inspired by DenseNet architecture principles for temporal modeling. DenseTCN is designed for efficient processing of sequential data with dense connections between layers.

#### Key Components

##### Chomp1d

```python
class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        # Implementation...
```

- Handles padding removal after convolution to maintain causal property
- Supports symmetric chomping (removing equal amounts from beginning and end)
- Ensures output sequences align properly with inputs

##### TemporalConvLayer

```python
class TemporalConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type):
        # Implementation...
```

- Basic building block for temporal convolutions
- Combines Conv1d, BatchNorm1d, Chomp1d, and activation
- Supports various activation functions (ReLU, PReLU, SiLU/Swish)
- Maintains temporal causality through padding and chomping

##### _ConvBatchChompRelu

```python
class _ConvBatchChompRelu(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size_set, stride, dilation, dropout, relu_type, se_module=False):
        # Implementation...
```

- Multi-branch convolutional block that processes inputs with different kernel sizes
- Optionally integrates Squeeze-Excitation (SE) attention
- Uses parallel convolution paths for multi-scale feature extraction
- Combines features from multiple kernel sizes
- Includes residual connection for gradient flow

##### _DenseBlock

```python
class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate,
                 kernel_size_set, dilation_size_set,
                 dropout, relu_type, squeeze_excitation):
        # Implementation...
```

- Core component implementing dense connectivity pattern
- Concatenates features from all previous layers in the block
- Applies increasing dilation rates for expanding receptive field
- Optionally includes Squeeze-Excitation for attention
- Forward method concatenates all features for efficient feature reuse

##### _Transition

```python
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, relu_type):
        # Implementation...
```

- Reduces feature dimensions between dense blocks
- Uses 1x1 convolutions for channel reduction
- Includes batch normalization and activation
- Controls model capacity and computational efficiency

##### DenseTemporalConvNet

```python
class DenseTemporalConvNet(nn.Module):
    def __init__(self, block_config, growth_rate_set, input_size, reduced_size,
                 kernel_size_set, dilation_size_set,
                 dropout=0.2, relu_type='prelu',
                 squeeze_excitation=False):
        # Implementation...
```

- Main model class that integrates dense blocks and transitions
- Highly configurable architecture with parameters for:
  - Dense block depth (`block_config`)
  - Growth rates for each block (`growth_rate_set`)
  - Kernel sizes for multi-scale processing (`kernel_size_set`)
  - Dilation rates for expanding receptive field (`dilation_size_set`)
  - Squeeze-Excitation attention (`squeeze_excitation`)
- Creates a sequential network of alternating dense blocks and transitions
- Final batch normalization for feature stabilization

#### Implementation Details

- **Dense Connectivity**: Each layer receives feature maps from all preceding layers, encouraging feature reuse and improving gradient flow
- **Multi-Scale Processing**: Parallel convolutions with different kernel sizes (typically 3, 5, 7) to capture patterns at different temporal scales
- **Dilated Convolutions**: Exponentially increasing dilation rates to expand the receptive field without increasing parameter count
- **Attention Mechanism**: Optional Squeeze-Excitation blocks for channel-wise attention
- **Feature Reduction**: Transition blocks to control model capacity and reduce feature map dimensions

#### Usage Example

```python
# Create a DenseTemporalConvNet
model = DenseTemporalConvNet(
    block_config=[3, 3, 3],              # 3 dense blocks with 3 layers each
    growth_rate_set=[64, 64, 64],        # Growth rate for each block
    input_size=512,                      # Input feature dimension
    reduced_size=256,                    # Reduced dimension between blocks
    kernel_size_set=[3, 5, 7],           # Multi-scale kernel sizes
    dilation_size_set=[1, 2, 4, 8],      # Dilation rates
    dropout=0.2,                         # Dropout rate
    relu_type='prelu',                   # Activation type
    squeeze_excitation=True              # Use SE attention
)

# Forward pass
input_tensor = torch.randn(32, 512, 100)  # (batch_size, channels, time_steps)
output = model(input_tensor)              # Shape: (32, final_channels, time_steps)
```

### 2. resnet.py

This file implements ResNet architecture variants for the visual backbone of the lip reading system. It provides the spatial feature extraction component, processing the output from the 3D convolutional frontend.

#### Key Components

##### conv3x3

```python
def conv3x3(in_planes, out_planes, stride=1):
    # Implementation...
```

- Helper function for creating 3x3 convolution with padding
- Standard building block used throughout ResNet

##### downsample_basic_block and downsample_basic_block_v2

```python
def downsample_basic_block(inplanes, outplanes, stride):
    # Implementation...

def downsample_basic_block_v2(inplanes, outplanes, stride):
    # Implementation...
```

- Two options for downsampling in residual connections
- v1: Simple 1x1 convolution with stride
- v2: AvgPool followed by 1x1 convolution (better preserves information)

##### BasicBlock

```python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type='prelu'):
        # Implementation...
```

- Core building block of ResNet architecture
- Contains two 3x3 convolutional layers with batch normalization
- Supports different activation functions (ReLU, PReLU, SiLU/Swish)
- Includes skip connection (residual) for improved gradient flow
- Downsample option for handling dimension changes

##### ResNet

```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_tokens=1000, relu_type='relu',
                 gamma_zero=False, avg_pool_downsample=False):
        # Implementation...
```

- Main ResNet model implementation
- Configurable number of layers for different ResNet variants
- Takes BasicBlock as input parameter
- Creates four layer groups with increasing channel dimensions
- Supports different initialization strategies (gamma_zero)
- Option for average pooling in downsampling paths

#### Implementation Details

- **Residual Learning**: Skip connections that enable direct gradient flow and mitigate the vanishing gradient problem
- **Layer Structure**: Four layer groups with increasing channel dimensions (64, 128, 256, 512)
- **Downsampling Options**: Two strategies for residual connections when dimensions change
- **Initialization**: Xavier/Kaiming initialization for weights, with optional zero-gamma for last BN in each block
- **Adaptive Pooling**: Final adaptive average pooling to create fixed-size feature vectors regardless of input dimensions

#### Key Methods

- **\_make_layer**: Builds a layer group with specified number of blocks

  ```python
  def _make_layer(self, block, planes, blocks, stride=1):
      # Creates a group of residual blocks
      # Handles downsample for first block if needed
      # Returns a sequential container of blocks
  ```

- **forward**: Processes input through the ResNet
  ```python
  def forward(self, x):
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)
      x = self.avgpool(x)
      x = x.view(x.size(0), -1)
      return x
  ```

#### Usage Example

```python
# Create ResNet-18
model = ResNet(
    block=BasicBlock,
    layers=[2, 2, 2, 2],      # ResNet-18 configuration
    relu_type='prelu',        # Use PReLU activation
    gamma_zero=True,          # Zero-initialize last BN in each block
    avg_pool_downsample=True  # Use average pooling for downsampling
)

# Forward pass
input_tensor = torch.randn(32, 64, 112, 112)  # (batch_size, channels, height, width)
output = model(input_tensor)                 # Shape: (32, 512)
```

### 3. se_module.py

This file implements the Squeeze-and-Excitation (SE) module, a channel attention mechanism that adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels.

#### Key Components

##### \_average_batch

```python
def _average_batch(x, lengths):
    # Implementation...
```

- Utility function for sequence averaging
- Takes variable-length sequences into account
- Used for creating sequence-level representations

##### SELayer

```python
class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        # Implementation...
```

- Main Squeeze-and-Excitation module implementation
- Performs channel-wise attention through:
  1. **Squeeze**: Global average pooling to capture channel-wise statistics
  2. **Excitation**: Two fully-connected layers with bottleneck structure
  3. **Scale**: Channel-wise multiplication with original features

#### Implementation Details

- **Channel Reduction**: Configurable reduction ratio to control model capacity
- **Activation Functions**: SiLU (Swish) for non-linearity in the bottleneck
- **Sigmoid Gating**: Final sigmoid activation to produce attention weights between 0 and 1
- **Feature Recalibration**: Element-wise multiplication of original features with channel attention weights

#### Key Methods

- **forward**: Processes input through the SE module
  ```python
  def forward(self, x):
      b, c, T = x.size()
      y = self.avg_pool(x).view(b, c)  # Global average pooling
      y = self.fc(y).view(b, c, 1)    # Channel-wise attention weights
      return x * y.expand_as(x)       # Scale input features
  ```

#### Usage Example

```python
# Create SE module
se_module = SELayer(
    channel=256,       # Number of input channels
    reduction=16       # Reduction ratio for bottleneck
)

# Forward pass
input_tensor = torch.randn(32, 256, 100)  # (batch_size, channels, time_steps)
output = se_module(input_tensor)         # Same shape: (32, 256, 100)
```

### 4. tcn.py

This file implements the Temporal Convolutional Network (TCN) architecture, a specialized convolutional network for sequence modeling that maintains causal relationships through dilated convolutions.

#### Key Components

##### Chomp1d

```python
class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        # Implementation...
```

- Removes padding from the temporal dimension to maintain causality
- Handles both asymmetric (causal) and symmetric chomping
- Essential for ensuring output sequences align properly with inputs

##### ConvBatchChompRelu

```python
class ConvBatchChompRelu(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, relu_type, dwpw=False):
        # Implementation...
```

- Basic building block for temporal convolutions
- Combines Conv1d, BatchNorm1d, Chomp1d, and activation
- Optional depthwise-pointwise (dwpw) convolution for efficiency
- Supports various activation functions (ReLU, PReLU, SiLU/Swish)

##### MultibranchTemporalBlock

```python
class MultibranchTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation,
                 padding, dropout=0.2, relu_type='relu', dwpw=False):
        # Implementation...
```

- Multi-scale block that processes inputs with parallel convolutions of different kernel sizes
- Creates multiple branches with different receptive fields
- Includes residual connection for improved gradient flow
- Two layers of convolutions with dropout between them

##### MultibranchTemporalConvNet

```python
class MultibranchTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2,
                 relu_type='relu', dwpw=False):
        # Implementation...
```

- Main multi-branch TCN implementation
- Creates a stack of temporal blocks with exponentially increasing dilation
- Configurable channel count for each layer
- Configurable kernel sizes for multi-scale processing

##### TemporalBlock

```python
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, dropout=0.2, symm_chomp=False, no_padding=False,
                 relu_type='relu', dwpw=False):
        # Implementation...
```

- Standard TCN block for single-branch processing
- Two convolution layers with batch normalization, chomping, and activation
- Residual connection with optional downsampling
- Support for depthwise-pointwise convolutions

##### TemporalConvNet

```python
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2,
                 relu_type='relu', dwpw=False):
        # Implementation...
```

- Standard single-branch TCN implementation
- Creates a stack of temporal blocks with exponentially increasing dilation
- Configurable channel count for each layer
- Fixed kernel size for all layers

#### Implementation Details

- **Dilated Convolutions**: Exponentially increasing dilation rates (1, 2, 4, 8...) to expand receptive field
- **Multi-Branch Architecture**: Parallel convolutions with different kernel sizes for capturing patterns at different scales
- **Residual Connections**: Skip connections to improve gradient flow and training stability
- **Causal Convolutions**: Padding and chomping strategy ensures no information leakage from future to past
- **Efficiency Options**: Depthwise-pointwise convolutions for reduced computation

#### Usage Examples

##### Standard TCN

```python
# Create a standard TCN
tcn_model = TemporalConvNet(
    num_inputs=512,                     # Input feature dimension
    num_channels=[64, 128, 256, 512],   # Channel dimensions for each layer
    tcn_options={'kernel_size': 3},     # Kernel size for all layers
    dropout=0.2,                        # Dropout rate
    relu_type='relu'                    # Activation type
)

# Forward pass
input_tensor = torch.randn(32, 512, 100)  # (batch_size, channels, time_steps)
output = tcn_model(input_tensor)          # Shape: (32, 512, 100)
```

##### Multi-Branch TCN

```python
# Create a multi-branch TCN
mb_tcn_model = MultibranchTemporalConvNet(
    num_inputs=512,                     # Input feature dimension
    num_channels=[64, 128, 256, 512],   # Channel dimensions for each layer
    tcn_options={
        'kernel_size': [3, 5, 7]        # Multiple kernel sizes for parallel branches
    },
    dropout=0.2,                        # Dropout rate
    relu_type='prelu'                   # Activation type
)

# Forward pass
input_tensor = torch.randn(32, 512, 100)  # (batch_size, channels, time_steps)
output = mb_tcn_model(input_tensor)       # Shape: (32, 512, 100)
```

## Integration with the Lip Reading System

These model architectures are used in different parts of the lip reading pipeline:

1. **Visual Frontend**: Processes raw video frames using 3D convolutions
2. **Visual Backbone**: ResNet or ShuffleNetV2 extract spatial features from each frame
3. **Temporal Encoder**: TCN or DenseTCN model the temporal relationships in the sequence
4. **Attention Mechanism**: SE modules enhance feature representation through channel attention

The choice of components creates a flexible architecture that can be adapted to different hardware constraints and performance requirements.
