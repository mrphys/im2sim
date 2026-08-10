# Configurable Half U-Net

This module provides a configurable PyTorch implementation of a **Half U-Net** architecture.

The purpose of this implementation is to allow rapid experimentation with different CNN architectures without changing the model code. The network structure is controlled through configuration objects and reusable presets.

The model is built around:

- `HalfUNet` - the PyTorch model implementation
- `HalfUNetConfig` - defines the architecture configuration


The architecture supports:

- 1D, 2D, and 3D inputs
- configurable encoder depth
- configurable feature channels
- custom convolution blocks
- custom pooling and upsampling layers
- configurable skip connection fusion
- residual blocks
- dilated bottlenecks
- depthwise separable convolutions
- Ghost convolution variants

---

# Basic Usage

A default Half U-Net can be created using a configuration object:

```python
import torch

from half_unet import HalfUNet, HalfUNetConfig


cfg = HalfUNetConfig(num_downsamples=2, hidden_channels=32)

model = HalfUNet.build(rank=3, in_channels=1, out_channels=1, cfg=cfg)


x = torch.randn(1, 1, 64, 64, 64)

y = model(x)
```

---

# Applying Presets

Architectures can be modified using presets. Presets are composable, meaning multiple architectural changes can be applied together.

Example:

```python
cfg = HalfUNetConfig(num_downsamples=2)

cfg = cfg.apply_presets(["ghost_depthwise", "residual"])

model = HalfUNet.build(rank=3, in_channels=1, out_channels=1, cfg=cfg)
```

---

# Available Presets

## Residual Encoder Blocks

Adds residual connections inside convolution blocks.

Useful when deeper networks require improved gradient flow.

```python
cfg = cfg.apply_presets(["residual"])
```

---

## Dilated Bottleneck

Replaces the deepest encoder block with dilated convolutions to increase the receptive field.

Useful for:

- large structures
- segmentation
- low-resolution feature processing

```python
cfg = cfg.apply_presets(["dilated_bottleneck"])
```

---

## Reconstruction

Applies convolution block settings intended for image reconstruction tasks.

Example applications:

- MRI reconstruction
- denoising
- image restoration

```python
cfg = cfg.apply_presets(["reconstruction"])
```

---

## Segmentation

Applies convolution block settings intended for segmentation tasks.

Example applications:

- medical image segmentation
- semantic segmentation

```python
cfg = cfg.apply_presets(["segmentation"])
```

---

## Depthwise Separable Convolutions

Replaces standard convolutions with depthwise separable convolutions.

This reduces:

- parameter count
- memory usage
- computational cost

```python
cfg = cfg.apply_presets(["depthwise_separable"])
```

---

## Ghost Convolution Variants

Uses Ghost convolution based blocks to reduce computational cost.

### Ghost depthwise

```python
cfg = cfg.apply_presets(["ghost_depthwise"])
```

### Ghost depthwise separable

```python
cfg = cfg.apply_presets(["ghost_depthwise_separable"])
```

---

# Combining Presets

Presets can be stacked to create task-specific architectures.

Example: a segmentation model using residual blocks and a dilated bottleneck:

```python
cfg = HalfUNetConfig(num_downsamples=3)

cfg = cfg.apply_presets(["segmentation", "residual", "dilated_bottleneck"])

model = HalfUNet.build(rank=3, in_channels=1, out_channels=3, cfg=cfg)
```

Example: a lightweight reconstruction model:

```python
cfg = HalfUNetConfig(num_downsamples=4)

cfg = cfg.apply_presets(["reconstruction", "ghost_depthwise"])

model = HalfUNet.build(rank=3, in_channels=1, out_channels=1, cfg=cfg)
```

---

# Custom Encoder Configurations

Each encoder level can optionally use a different convolution block configuration.

Example:

```python
cfg = HalfUNetConfig(num_downsamples=3, encoder_block_cfg=[block_cfg_1, block_cfg_2, block_cfg_3])
```

This allows heterogeneous architectures such as:

```
Encoder level 1:
    Standard convolution blocks

Encoder level 2:
    Depthwise separable convolution blocks

Encoder level 3:
    Dilated convolution blocks
```

---

# Pooling and Upsampling

Pooling and upsampling operations are configurable.

Example:

```python
cfg = HalfUNetConfig(pool_spec=LayerConfig(name="AveragePool", kwargs={"kernel_size": 2}))
```

Different operations can also be provided at each level:

```python
cfg.pool_spec = [
    LayerConfig(name="MaxPool", kwargs={"kernel_size": 2}),
    LayerConfig(name="AveragePool", kwargs={"kernel_size": 2}),
]
```

Supported pooling layers:

- `MaxPool`
- `AveragePool`

Supported upsampling layers:

- `Upsample`
- `PixelShuffle`

---

# Skip Connection Fusion

Encoder features are fused with decoder features using the configured fusion type.

Default:

```python
fusion_type = ResidualConnectionType.ADD
```

which performs:

```
decoder + encoder
```

Alternatively:

```python
fusion_type = ResidualConnectionType.CONCAT
```

which performs:

```
[decoder, encoder]
```

When using concatenation, the output block automatically adjusts the number of input channels.

