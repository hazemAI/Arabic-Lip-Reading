## Directory Structure

```text
encoders/
├── README.md                        # This file, providing an overview of the encoders directory.
├── encoder_models.py                # Defines high-level encoder architectures.
├── modules/                         # Contains core neural network building blocks.
└── pretrained_visual_frontend.pth   # Pretrained weights for the VisualFrontend component.
```

## `encoder_models.py`

This script defines several key PyTorch `nn.Module` classes that constitute the visual encoding part of the VSR model:

- **`VisualFrontend`**: Implements the initial visual feature extraction. It typically consists of a 3D convolution layer followed by a 2D ResNet backbone (e.g., ResNet-18). This module processes the input video frames (or frame patches) to produce a sequence of spatial features for each time step.

- **`DenseTCN`**: A temporal encoder based on Dense Temporal Convolutional Networks. It uses densely connected convolutional blocks to capture temporal patterns in the sequence of visual features. It can be configured with various block structures, growth rates, kernel sizes, and dilation factors.

- **`MultiscaleTCN`**: Another temporal encoder variant, this one employing Multibranch Temporal Convolutional Networks. It processes the input sequence through multiple parallel TCN branches with different kernel sizes, allowing it to capture temporal dependencies at various scales. The outputs of these branches are then typically combined.

- **`VisualTemporalEncoder`**: This is a higher-level module that combines the `VisualFrontend` with a chosen temporal encoding backbone. The temporal backbone can be one of the TCN variants defined in this file (`DenseTCN`, `MultiscaleTCN`) or a `ConformerEncoder` (imported from ESPNet). This class orchestrates the flow from raw visual input to a sequence of encoded features ready for the CTC loss calculation and the subsequent decoder stage of the VSR model. It also includes an adapter layer if the frontend output dimension doesn't match the temporal encoder input dimension and can project features for CTC and decoder stages.

These modules are designed to be configurable and are used within the main VSR model pipeline (`e2e_vsr_greedy.py` and `master_vsr_greedy.py`).

## `modules/` Directory

This directory contains the fundamental neural network building blocks and layers that are utilized by the models defined in `encoder_models.py`. The key components are:

- `resnet.py`: Implements the ResNet architecture (e.g., ResNet-18, ResNet-34), including `BasicBlock` definitions, used as the backbone in the `VisualFrontend`.
- `tcn.py`: Provides the implementation for Multibranch Temporal Convolutional Networks (`MultibranchTemporalConvNet`), used by the `MultiscaleTCN` encoder.
- `densetcn.py`: Contains the implementation for Dense Temporal Convolutional Networks (`DenseTemporalConvNet`), serving as the core for the `DenseTCN` encoder.
- `se_module.py`: Implements the Squeeze-and-Excitation (SE) block, which can be optionally integrated into models like `DenseTCN` for channel-wise feature recalibration.

These modules provide the foundational layers for constructing the more complex encoder architectures.

## `pretrained_visual_frontend.pth`

This binary file contains pretrained weights specifically for the `VisualFrontend` component (which typically combines a 3D convolution layer with a ResNet). These weights are often learned from a large dataset and can significantly improve performance and reduce training time when used as a starting point for the VSR model.

To load these weights into your `VisualFrontend` instance:
```python
import torch

# Assuming `visual_frontend` is an instance of the VisualFrontend class
# and `device` is your target device (e.g., torch.device('cuda') or torch.device('cpu'))

state = torch.load('Working/model/encoders/pretrained_visual_frontend.pth', map_location=device)

# It's common for pretrained weights to be saved under a 'state_dict' key.
# Adjust the key if necessary based on how the weights were saved.
if 'state_dict' in state:
    visual_frontend.load_state_dict(state['state_dict'], strict=False)
elif 'model_state_dict' in state: # Another common key
    visual_frontend.load_state_dict(state['model_state_dict'], strict=False)
else:
    visual_frontend.load_state_dict(state, strict=False)

print("Pretrained visual frontend weights loaded.")
```
Using `strict=False` is often helpful to ignore mismatches if the model architecture has slight differences from when the weights were saved, or if only partial loading is intended (e.g., when fine-tuning).

These weights can be kept frozen during initial VSR model training or fine-tuned along with the rest of the model.