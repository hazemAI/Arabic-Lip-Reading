## Files in this Directory

- `densetcn.py`: Implements Dense Temporal Convolutional Networks.
- `resnet.py`: Provides ResNet backbone implementations.
- `tcn.py`: Contains implementations for standard and Multibranch Temporal Convolutional Networks.
- `se_module.py`: Implements the Squeeze-and-Excitation (SE) layer.

## Detailed File Explanations

### 1. `densetcn.py`

This file implements the Dense Temporal Convolutional Network (DenseTCN), a variant of TCN inspired by DenseNet architecture principles for temporal modeling. DenseTCN is designed for efficient processing of sequential data with dense connections between layers.

#### Key Components

##### `Chomp1d`

```python
class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        # Implementation...
```
- Handles padding removal after 1D convolution to maintain causality.
- Supports symmetric chomping.

##### `TemporalConvLayer`

```python
class TemporalConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type):
        # Implementation...
```
- Basic building block: Conv1d, BatchNorm1d, `Chomp1d`, and an activation function (ReLU, PReLU, SiLU/Swish).

##### `_ConvBatchChompRelu`

```python
class _ConvBatchChompRelu(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size_set, stride, dilation, dropout, relu_type, se_module=False):
        # Implementation...
```
- A multi-branch convolutional block that processes inputs with different kernel sizes.
- Optionally integrates Squeeze-Excitation (SE) attention.
- Uses parallel convolution paths for multi-scale feature extraction.
- Combines features from multiple kernel sizes.
- Includes residual connection for gradient flow.

##### `_DenseBlock`

```python
class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate,
                 kernel_size_set, dilation_size_set,
                 dropout, relu_type, squeeze_excitation):
        # Implementation...
```
- Core component implementing dense connectivity pattern: concatenates features from all previous layers in the block.
- Applies increasing dilation rates for expanding receptive field.
- Optionally includes Squeeze-Excitation for attention.
- Forward method concatenates all features for efficient feature reuse.

##### `_Transition`

```python
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, relu_type):
        # Implementation...
```
- Reduces feature dimensions between dense blocks using a 1x1 convolution.

##### `DenseTemporalConvNet`

```python
class DenseTemporalConvNet(nn.Module):
    def __init__(self, block_config, growth_rate_set, input_size, reduced_size,
                 kernel_size_set, dilation_size_set,
                 dropout=0.2, relu_type='prelu',
                 squeeze_excitation=False):
        # Implementation...
```
- The main DenseTCN model class, integrating `_DenseBlock`s and `_Transition` layers.
- Highly configurable for block depth, growth rates, kernel sizes, dilation rates, and optional SE attention.

#### Implementation Details

- **Dense Connectivity**: Encourages feature reuse and improves gradient flow.
- **Multi-Scale Processing**: Parallel convolutions with different kernel sizes.
- **Dilated Convolutions**: Expands receptive field efficiently.
- **Attention Mechanism**: Optional Squeeze-Excitation blocks.

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

### 2. `resnet.py`

This file implements ResNet architecture variants, primarily used as the spatial feature extraction backbone within the `VisualFrontend` of the VSR system.

#### Key Components

##### `conv3x3`

```python
def conv3x3(in_planes, out_planes, stride=1):
    # Implementation...
```
- Helper function for a 3x3 convolution with padding.

##### `downsample_basic_block` / `downsample_basic_block_v2`

```python
def downsample_basic_block(inplanes, outplanes, stride):
    # Implementation...

def downsample_basic_block_v2(inplanes, outplanes, stride):
    # Implementation...
```
- Options for downsampling in residual connections when feature map sizes change (v2 uses AvgPool).

##### `BasicBlock`

```python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type='prelu'):
        # Implementation...
```
- The core building block of ResNet, consisting of two 3x3 convolutional layers, batch normalization, and an activation function.
- Includes a skip connection (residual).

##### `ResNet`

```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_tokens=1000, relu_type='relu',
                 gamma_zero=False, avg_pool_downsample=False):
        # Implementation...
```
- Main ResNet model implementation (e.g., ResNet-18, ResNet-34).
- Composed of an initial convolutional layer, followed by multiple stages of `BasicBlock`s, and an adaptive average pooling layer at the end.

#### Implementation Details

- **Residual Learning**: Enables training of very deep networks.
- **Configurable Depth**: Number of layers in each stage can be configured (e.g., `layers=[2, 2, 2, 2]` for ResNet-18).

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

### 3. `tcn.py`

This file provides implementations for Temporal Convolutional Networks (TCNs), including a standard version and a multibranch version, designed for sequence modeling.

#### Key Components

##### `Chomp1d`

```python
class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        # Implementation...
```
- Utility for causal 1D convolutions, removing padding from the end (or symmetrically) to prevent leakage from future time steps.

##### `ConvBatchChompRelu`

```python
class ConvBatchChompRelu(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, relu_type, dwpw=False):
        # Implementation...
```
- A single convolutional layer unit comprising: Conv1d, BatchNorm1d, `Chomp1d` for causal output, and an activation function (ReLU, PReLU, or SiLU/Swish).
- Supports optional depthwise-separable convolutions (`dwpw=True`).

##### `TemporalBlock` (Standard TCN)

```python
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, dropout=0.2, symm_chomp=False, no_padding=False,
                 relu_type='relu', dwpw=False):
        # Implementation...
```
- A residual block for standard TCNs. It consists of two `ConvBatchChompRelu` layers with the same dilation factor, dropout, and a residual connection that adds the input to the output of the second layer.
- If input and output channels differ, a 1x1 convolution is used for the residual connection.

##### `TemporalConvNet` (Standard TCN)

```python
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2,
                 relu_type='relu', dwpw=False):
        # Implementation...
```
- The main network for a standard TCN. It stacks multiple `TemporalBlock` layers, where each subsequent block typically has an exponentially increasing dilation factor (e.g., 1, 2, 4, 8, ...).
- This allows the receptive field to grow exponentially with depth, enabling capture of long-range temporal dependencies.

##### `MultibranchTemporalBlock`

```python
class MultibranchTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation,
                 padding, dropout=0.2, relu_type='relu', dwpw=False):
        # Implementation...
```
- A residual block designed for multibranch TCNs. It contains two sets of multi-branch convolutions.
- In each set, the input is passed through multiple parallel `ConvBatchChompRelu` layers, each with a different kernel size.
- The outputs of these parallel branches are concatenated.
- A residual connection is included.

##### `MultibranchTemporalConvNet`

```python
class MultibranchTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2,
                 relu_type='relu', dwpw=False):
        # Implementation...
```
- The main network for a multibranch TCN. It stacks `MultibranchTemporalBlock` layers, typically with increasing dilation factors for each level.
- This architecture allows the network to simultaneously process temporal information at different scales corresponding to the different kernel sizes in its branches.

#### Implementation Details

- **Causality**: Ensures that predictions for a given time step depend only on past and current inputs, not future ones, crucial for auto-regressive tasks.
- **Dilated Convolutions**: Key for achieving large receptive fields with fewer layers and parameters compared to standard CNNs.
- **Residual Connections**: Stabilize training and enable deeper networks.
- **Multibranch Architecture**: Allows capturing temporal patterns at multiple resolutions simultaneously.

#### Usage Example

```python
# Standard TCN
# num_channels = [256] * 3 # Example: 3 layers, 256 channels each
# model_std_tcn = TemporalConvNet(num_inputs=512, num_channels=num_channels, tcn_options={'kernel_size': 3})

# Multibranch TCN
# num_channels_mb = [64, 64, 64] # Output channels per level, will be divided among branches
# model_mb_tcn = MultibranchTemporalConvNet(num_inputs=512, num_channels=num_channels_mb, tcn_options={'kernel_size': [3, 5, 7]})

# input_tensor = torch.randn(32, 512, 100) # (batch, channels, time)
# output_std = model_std_tcn(input_tensor)
# output_mb = model_mb_tcn(input_tensor)
```

### 4. `se_module.py`

This file implements the Squeeze-and-Excitation (SE) layer, a channel-wise attention mechanism that can be integrated into various convolutional neural network architectures to improve performance.

#### Key Components

##### `SELayer`

```python
class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        # Implementation...
```
- **Squeeze Operation**: Global information is aggregated from each channel by using adaptive average pooling to reduce spatial dimensions to 1x1, resulting in a channel descriptor vector.
- **Excitation Operation**: This channel descriptor is fed through two fully connected (FC) layers. The first FC layer reduces dimensionality (controlled by the `reduction` parameter), and the second FC layer brings it back to the original number of channels. A sigmoid activation then generates channel-wise weights (attention scores) between 0 and 1.
- **Rescale Operation**: The original input feature map is rescaled by multiplying each channel with its corresponding attention score from the excitation step.

#### Implementation Details

- The `SELayer` takes the number of input channels and a reduction ratio as arguments.
- It is designed to be easily inserted into existing CNN architectures, typically after convolutional layers.
- By adaptively recalibrating channel-wise feature responses, SE blocks can emphasize informative features and suppress less useful ones.

#### Usage Example

```python
# model_se = SELayer(channel=256, reduction=4)
# input_tensor = torch.randn(32, 256, 100) # (batch, channels, time/sequence_length for 1D)
# output_se = model_se(input_tensor)
```
