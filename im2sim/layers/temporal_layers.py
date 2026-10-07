# ==============================================================================
# Copyright 2026 University College London.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Temporal (rank + 1)D versions of the image layers used by the im2sim models.

All layers accept inputs of shape `(N, C, T, *spatial)`, where `spatial` is `(H, W)` for rank `2`
and `(D, H, W)` for rank `3`. Each layer is registered under the same name as its spatial
counterpart (e.g. `Conv`, `MaxPool`), so `get_image_layer(name, rank, temporal=True)` returns it.

Parameters such as `kernel_size` and `stride` can be given as a `(T, *spatial)` tuple to set the
time component explicitly. Otherwise:

- `ConvTime`: a scalar `kernel_size` or `padding` also applies to time, while scalar `stride` and
  `dilation` only apply spatially.
- Resampling layers (`ConvTransposeTime`, `MaxPoolTime`, `AvgPoolTime`, `UpsampleTime`): scalars
  and spatial-only tuples only apply spatially, so the time axis is not resampled.

Temporal ops that would be the identity (e.g. a time kernel of `1`) are skipped entirely.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from im2sim.utils.layer_util import register_temporal_layer, register_with_ranks

TEMPORAL_RANKS = (2, 3)

SUPPORTED_TEMPORAL_LAYERS = (
    "Conv",
    "ConvTranspose",
    "MaxPool",
    "AvgPool",
    "BatchNorm",
    "InstanceNorm",
    "Upsample",
    "Dropout",
)

_LINEAR_MODES = {2: "bilinear", 3: "trilinear"}


class TimeDistributed(nn.Module):
    """
    Apply a module independently to each time step.

    Expected input: (N, C, T, *spatial) where spatial is HW for rank=2 or DHW for rank=3
    Wrapped module should accept: (N, C, *spatial)
    Output: (N, C, T, *spatial_out)
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (5, 6):
            raise ValueError(
                f"TimeDistributed expects 5D or 6D input [N, C, T, ...], got shape {tuple(x.shape)}"
            )

        n, c, t = x.shape[:3]
        spatial = x.shape[3:]

        # (N, C, T, *S) -> (N, T, C, *S)
        permute_order = [0, 2, 1] + list(range(3, x.ndim))
        x = x.permute(*permute_order).contiguous()

        # (N, T, C, *S) -> (N*T, C, *S)
        x = x.reshape(n * t, c, *spatial)

        y = self.module(x)  # (N*T, C_out, *S_out)

        c_out = y.shape[1]
        spatial_out = y.shape[2:]

        # (N*T, C_out, *S_out) -> (N, T, C_out, *S_out)
        y = y.reshape(n, t, c_out, *spatial_out)

        # (N, T, C_out, *S_out) -> (N, C_out, T, *S_out)
        permute_back = [0, 2, 1] + list(range(3, y.ndim))
        y = y.permute(*permute_back).contiguous()

        return y


class SpaceDistributed(nn.Module):
    """
    Apply a module independently at each spatial location over time.

    Expected input: (N, C, T, *spatial)
    Wrapped module should accept: (N * prod(spatial), C, T)
    Output: (N, C_out, T_out, *spatial)
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (5, 6):
            raise ValueError(
                f"SpaceDistributed expects 5D or 6D input [N, C, T, ...], got shape {tuple(x.shape)}"
            )

        n, c, t = x.shape[:3]
        spatial = x.shape[3:]
        spatial_rank = len(spatial)
        spatial_prod = math.prod(spatial)

        # (N, C, T, *S) -> (N, *S, C, T)
        permute_order = [0] + list(range(3, x.ndim)) + [1, 2]
        x = x.permute(*permute_order).contiguous()

        # (N, *S, C, T) -> (N*prod(S), C, T)
        x = x.reshape(n * spatial_prod, c, t)

        y = self.module(x)  # (N*prod(S), C_out, T_out)

        c_out = y.shape[1]
        t_out = y.shape[2]

        # (N*prod(S), C_out, T_out) -> (N, *S, C_out, T_out)
        y = y.reshape(n, *spatial, c_out, t_out)

        # (N, *S, C_out, T_out) -> (N, C_out, T_out, *S)
        permute_back = [0, spatial_rank + 1, spatial_rank + 2] + list(range(1, spatial_rank + 1))
        y = y.permute(*permute_back).contiguous()

        return y


def _check_rank(rank: int, layer_name: str):
    if rank not in TEMPORAL_RANKS:
        raise ValueError(f"{layer_name} only supports rank 2 or 3, got {rank}")


def _check_input(x: torch.Tensor, rank: int, layer_name: str):
    if x.ndim != rank + 3:
        raise ValueError(
            f"{layer_name} with rank {rank} expects {rank + 3}D input (N, C, T, *spatial), "
            f"got shape {tuple(x.shape)}"
        )


def _split_time_param(param, rank: int, name: str, time_default, scalar_applies_to_time=False):
    """
    Split a parameter into its time and spatial components.

    Args:
        param: An int/float, a string (e.g. `padding="same"`), a spatial-only tuple of length `rank`
            or a `(T, *spatial)` tuple of length `rank + 1`.
        rank: Spatial rank.
        name: Parameter name for error messages.
        time_default: Time component used for spatial-only tuples (and for scalars, unless
            `scalar_applies_to_time` is True).
        scalar_applies_to_time: If True, a scalar is used for the time component as well.

    Returns:
        (time_component, spatial_component): strings are returned unchanged for both components.
    """
    if isinstance(param, str):
        return param, param
    if isinstance(param, (int, float)) and not isinstance(param, bool):
        return (param if scalar_applies_to_time else time_default), (param,) * rank
    if isinstance(param, (tuple, list)):
        if len(param) == rank + 1:
            return param[0], tuple(param[1:])
        if len(param) == rank:
            return time_default, tuple(param)
    raise ValueError(
        f"{name} must be a scalar, a spatial tuple of length {rank} or a (T, *spatial) tuple of "
        f"length {rank + 1}, got {param!r}"
    )


@register_with_ranks("Conv", ranks=TEMPORAL_RANKS, register=register_temporal_layer)
class ConvTime(nn.Module):
    """
    Factorized (rank + 1)D convolution: a spatial `ConvNd` applied to each time step, followed by a
    temporal `Conv1d` applied at each spatial location.

    A scalar `kernel_size` or `padding` applies to time as well, so `kernel_size=3, padding="same"`
    gives a 3-tap temporal convolution that preserves `T`. Scalar `stride` and `dilation` only apply
    spatially. Use `(T, *spatial)` tuples to set the time components explicitly. The temporal
    convolution is skipped if its kernel and stride are both `1`.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int | tuple): Kernel size.
        rank (int): Spatial rank (`2` or `3`).
        stride (int | tuple): Stride. Default is `1`.
        padding (int | str | tuple): Padding, including `"same"` and `"valid"`. Default is `0`.
        dilation (int | tuple): Dilation. Default is `1`.
        groups (int): Groups of the spatial convolution. Default is `1`.
        bias (bool): If True, both convolutions have a bias. Default is `True`.
        padding_mode (str): Padding mode of both convolutions. Default is `"zeros"`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        rank,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        _check_rank(rank, "ConvTime")
        self.rank = rank

        time_kernel, space_kernel = _split_time_param(
            kernel_size, rank, "kernel_size", 1, scalar_applies_to_time=True
        )
        time_stride, space_stride = _split_time_param(stride, rank, "stride", 1)
        time_padding, space_padding = _split_time_param(
            padding, rank, "padding", 0, scalar_applies_to_time=True
        )
        time_dilation, space_dilation = _split_time_param(dilation, rank, "dilation", 1)

        spatial_conv = nn.Conv2d if rank == 2 else nn.Conv3d
        self.spatial_op = TimeDistributed(
            spatial_conv(
                in_channels,
                out_channels,
                kernel_size=space_kernel,
                stride=space_stride,
                padding=space_padding,
                dilation=space_dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            )
        )

        is_identity = time_kernel == 1 and time_stride == 1 and time_padding in (0, "same", "valid")
        self.temporal_op = (
            None
            if is_identity
            else SpaceDistributed(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=time_kernel,
                    stride=time_stride,
                    padding=time_padding,
                    dilation=time_dilation,
                    bias=bias,
                    padding_mode=padding_mode,
                )
            )
        )

    def forward(self, x):
        _check_input(x, self.rank, "ConvTime")
        x = self.spatial_op(x)
        if self.temporal_op is not None:
            x = self.temporal_op(x)
        return x


@register_with_ranks("ConvTranspose", ranks=TEMPORAL_RANKS, register=register_temporal_layer)
class ConvTransposeTime(nn.Module):
    """
    Factorized (rank + 1)D transposed convolution: a spatial `ConvTransposeNd` applied to each time
    step, followed by a temporal `ConvTranspose1d` applied at each spatial location.

    Scalar parameters only apply spatially, so the time axis is not upsampled by default. The
    temporal transposed convolution is only created if a `(T, *spatial)` tuple sets a non-identity
    time component.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int | tuple): Kernel size.
        rank (int): Spatial rank (`2` or `3`).
        stride (int | tuple): Stride. Default is `1`.
        padding (int | tuple): Padding. Default is `0`.
        output_padding (int | tuple): Output padding. Default is `0`.
        groups (int): Groups of the spatial transposed convolution. Default is `1`.
        bias (bool): If True, both transposed convolutions have a bias. Default is `True`.
        dilation (int | tuple): Dilation. Default is `1`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        rank,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
    ):
        super().__init__()
        _check_rank(rank, "ConvTransposeTime")
        self.rank = rank

        time_kernel, space_kernel = _split_time_param(kernel_size, rank, "kernel_size", 1)
        time_stride, space_stride = _split_time_param(stride, rank, "stride", 1)
        time_padding, space_padding = _split_time_param(padding, rank, "padding", 0)
        time_output_padding, space_output_padding = _split_time_param(
            output_padding, rank, "output_padding", 0
        )
        time_dilation, space_dilation = _split_time_param(dilation, rank, "dilation", 1)

        spatial_conv_transpose = nn.ConvTranspose2d if rank == 2 else nn.ConvTranspose3d
        self.spatial_op = TimeDistributed(
            spatial_conv_transpose(
                in_channels,
                out_channels,
                kernel_size=space_kernel,
                stride=space_stride,
                padding=space_padding,
                output_padding=space_output_padding,
                groups=groups,
                bias=bias,
                dilation=space_dilation,
            )
        )

        is_identity = (
            time_kernel == 1 and time_stride == 1 and time_padding == 0 and time_output_padding == 0
        )
        self.temporal_op = (
            None
            if is_identity
            else SpaceDistributed(
                nn.ConvTranspose1d(
                    out_channels,
                    out_channels,
                    kernel_size=time_kernel,
                    stride=time_stride,
                    padding=time_padding,
                    output_padding=time_output_padding,
                    bias=bias,
                    dilation=time_dilation,
                )
            )
        )

    def forward(self, x):
        _check_input(x, self.rank, "ConvTransposeTime")
        x = self.spatial_op(x)
        if self.temporal_op is not None:
            x = self.temporal_op(x)
        return x


@register_with_ranks("MaxPool", ranks=TEMPORAL_RANKS, register=register_temporal_layer)
class MaxPoolTime(nn.Module):
    """
    Factorized (rank + 1)D max pooling: spatial `MaxPoolNd` applied to each time step, followed by
    a temporal `MaxPool1d` applied at each spatial location.

    Scalar parameters only apply spatially, so time is only pooled if a `(T, *spatial)` tuple is
    given, e.g. `kernel_size=(2, 2, 2, 2)` for rank `3`.

    Args:
        kernel_size (int | tuple): Pooling window size.
        rank (int): Spatial rank (`2` or `3`).
        stride (int | tuple | None): Stride. Default is `kernel_size`.
        padding (int | tuple): Padding. Default is `0`.
        dilation (int | tuple): Dilation. Default is `1`.
        ceil_mode (bool): If True, use ceil instead of floor to compute the output size.
    """

    def __init__(self, kernel_size, rank, stride=None, padding=0, dilation=1, ceil_mode=False):
        super().__init__()
        _check_rank(rank, "MaxPoolTime")
        self.rank = rank

        stride = kernel_size if stride is None else stride
        time_kernel, space_kernel = _split_time_param(kernel_size, rank, "kernel_size", 1)
        time_stride, space_stride = _split_time_param(stride, rank, "stride", 1)
        time_padding, space_padding = _split_time_param(padding, rank, "padding", 0)
        time_dilation, space_dilation = _split_time_param(dilation, rank, "dilation", 1)

        spatial_pool = nn.MaxPool2d if rank == 2 else nn.MaxPool3d
        self.spatial_op = TimeDistributed(
            spatial_pool(
                kernel_size=space_kernel,
                stride=space_stride,
                padding=space_padding,
                dilation=space_dilation,
                ceil_mode=ceil_mode,
            )
        )

        is_identity = time_kernel == 1 and time_stride == 1 and time_padding == 0
        self.temporal_op = (
            None
            if is_identity
            else SpaceDistributed(
                nn.MaxPool1d(
                    kernel_size=time_kernel,
                    stride=time_stride,
                    padding=time_padding,
                    dilation=time_dilation,
                    ceil_mode=ceil_mode,
                )
            )
        )

    def forward(self, x):
        _check_input(x, self.rank, "MaxPoolTime")
        x = self.spatial_op(x)
        if self.temporal_op is not None:
            x = self.temporal_op(x)
        return x


@register_with_ranks("AvgPool", ranks=TEMPORAL_RANKS, register=register_temporal_layer)
class AvgPoolTime(nn.Module):
    """
    Factorized (rank + 1)D average pooling: spatial `AvgPoolNd` applied to each time step, followed
    by a temporal `AvgPool1d` applied at each spatial location.

    Scalar parameters only apply spatially, so time is only pooled if a `(T, *spatial)` tuple is
    given.

    Args:
        kernel_size (int | tuple): Pooling window size.
        rank (int): Spatial rank (`2` or `3`).
        stride (int | tuple | None): Stride. Default is `kernel_size`.
        padding (int | tuple): Padding. Default is `0`.
        ceil_mode (bool): If True, use ceil instead of floor to compute the output size.
        count_include_pad (bool): If True, include the zero-padding in the averaging calculation.
    """

    def __init__(
        self, kernel_size, rank, stride=None, padding=0, ceil_mode=False, count_include_pad=True
    ):
        super().__init__()
        _check_rank(rank, "AvgPoolTime")
        self.rank = rank

        stride = kernel_size if stride is None else stride
        time_kernel, space_kernel = _split_time_param(kernel_size, rank, "kernel_size", 1)
        time_stride, space_stride = _split_time_param(stride, rank, "stride", 1)
        time_padding, space_padding = _split_time_param(padding, rank, "padding", 0)

        spatial_pool = nn.AvgPool2d if rank == 2 else nn.AvgPool3d
        self.spatial_op = TimeDistributed(
            spatial_pool(
                kernel_size=space_kernel,
                stride=space_stride,
                padding=space_padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
            )
        )

        is_identity = time_kernel == 1 and time_stride == 1 and time_padding == 0
        self.temporal_op = (
            None
            if is_identity
            else SpaceDistributed(
                nn.AvgPool1d(
                    kernel_size=time_kernel,
                    stride=time_stride,
                    padding=time_padding,
                    ceil_mode=ceil_mode,
                    count_include_pad=count_include_pad,
                )
            )
        )

    def forward(self, x):
        _check_input(x, self.rank, "AvgPoolTime")
        x = self.spatial_op(x)
        if self.temporal_op is not None:
            x = self.temporal_op(x)
        return x


class _Interpolate(nn.Module):
    """Module wrapper of `torch.nn.functional.interpolate`."""

    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

    def extra_repr(self):
        resize = f"size={self.size}" if self.size is not None else f"scale={self.scale_factor}"
        return f"{resize}, mode={self.mode}"


@register_with_ranks("Upsample", ranks=TEMPORAL_RANKS, register=register_temporal_layer)
class UpsampleTime(nn.Module):
    """
    Factorized (rank + 1)D upsampling: spatial interpolation applied to each time step, followed by
    temporal interpolation applied at each spatial location.

    Scalar or spatial-only `scale_factor`/`size` only resize the spatial dims. Use a `(T, *spatial)`
    tuple to also resize time. As in `im2sim.layers.custom_image_layers.Upsample`, modes other than
    `"nearest"` are replaced by the linear mode matching the rank. Time is interpolated linearly
    for linear modes and with nearest neighbours otherwise.

    Args:
        rank (int): Spatial rank (`2` or `3`).
        scale_factor (float | tuple): Multiplier for the input size. Default is `2`.
        mode (str): Interpolation mode, `"nearest"` or linear. Default is `"nearest"`.
        align_corners (bool | None): Passed to `torch.nn.functional.interpolate` for linear modes.
        size (int | tuple | None): Output size. Overrides `scale_factor` if given.
    """

    def __init__(self, rank, scale_factor=2, mode="nearest", align_corners=None, size=None):
        super().__init__()
        _check_rank(rank, "UpsampleTime")
        self.rank = rank

        is_linear = mode != "nearest"
        space_mode = _LINEAR_MODES[rank] if is_linear else "nearest"
        time_mode = "linear" if is_linear else "nearest"
        align_corners = align_corners if is_linear else None

        if size is not None:
            time_size, space_size = _split_time_param(size, rank, "size", None)
            self.spatial_op = TimeDistributed(
                _Interpolate(size=space_size, mode=space_mode, align_corners=align_corners)
            )
            temporal_kwargs = None if time_size is None else {"size": time_size}
        else:
            time_scale, space_scale = _split_time_param(scale_factor, rank, "scale_factor", 1)
            self.spatial_op = TimeDistributed(
                _Interpolate(scale_factor=space_scale, mode=space_mode, align_corners=align_corners)
            )
            temporal_kwargs = None if time_scale == 1 else {"scale_factor": time_scale}

        self.temporal_op = (
            None
            if temporal_kwargs is None
            else SpaceDistributed(
                _Interpolate(**temporal_kwargs, mode=time_mode, align_corners=align_corners)
            )
        )

    def forward(self, x):
        _check_input(x, self.rank, "UpsampleTime")
        x = self.spatial_op(x)
        if self.temporal_op is not None:
            x = self.temporal_op(x)
        return x


def _joint_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a 3D norm jointly over all (T, *spatial) dims of a (N, C, T, *spatial) tensor."""
    if x.ndim == 5:
        return norm(x)
    n, c = x.shape[:2]
    # (N, C, T, D, H, W) -> (N, C, T*D, H, W): the norm statistics only depend on (N, C)
    return norm(x.reshape(n, c, -1, *x.shape[4:])).reshape(x.shape)


@register_with_ranks("BatchNorm", ranks=TEMPORAL_RANKS, register=register_temporal_layer)
class BatchNormTime(nn.Module):
    """
    Batch normalization over the batch and all (T, *spatial) dims of a `(N, C, T, *spatial)` input.

    Unlike the factorized layers, time is not normalized separately: per-channel statistics are
    computed jointly over the batch, time and space.

    Args:
        num_features (int): Number of channels.
        rank (int): Spatial rank (`2` or `3`).
        eps, momentum, affine, track_running_stats: As in `torch.nn.BatchNorm3d`.
    """

    def __init__(
        self, num_features, rank, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
    ):
        super().__init__()
        _check_rank(rank, "BatchNormTime")
        self.rank = rank
        self.norm = nn.BatchNorm3d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        _check_input(x, self.rank, "BatchNormTime")
        return _joint_norm(self.norm, x)


@register_with_ranks("InstanceNorm", ranks=TEMPORAL_RANKS, register=register_temporal_layer)
class InstanceNormTime(nn.Module):
    """
    Instance normalization over all (T, *spatial) dims of each sample of a `(N, C, T, *spatial)`
    input.

    Unlike the factorized layers, time is not normalized separately, so the temporal evolution of
    each voxel is preserved.

    Args:
        num_features (int): Number of channels.
        rank (int): Spatial rank (`2` or `3`).
        eps, momentum, affine, track_running_stats: As in `torch.nn.InstanceNorm3d`.
    """

    def __init__(
        self, num_features, rank, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False
    ):
        super().__init__()
        _check_rank(rank, "InstanceNormTime")
        self.rank = rank
        self.norm = nn.InstanceNorm3d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        _check_input(x, self.rank, "InstanceNormTime")
        return _joint_norm(self.norm, x)


@register_with_ranks("Dropout", ranks=TEMPORAL_RANKS, register=register_temporal_layer)
class DropoutTime(nn.Module):
    """
    Channel dropout (`torch.nn.Dropout2d`/`Dropout3d`) applied independently to each time step.

    Args:
        rank (int): Spatial rank (`2` or `3`).
        p (float): Probability of zeroing a channel. Default is `0.5`.
        inplace (bool): If True, apply the dropout in-place. Default is `False`.
    """

    def __init__(self, rank, p=0.5, inplace=False):
        super().__init__()
        _check_rank(rank, "DropoutTime")
        self.rank = rank
        dropout = nn.Dropout2d if rank == 2 else nn.Dropout3d
        self.spatial_op = TimeDistributed(dropout(p=p, inplace=inplace))

    def forward(self, x):
        _check_input(x, self.rank, "DropoutTime")
        return self.spatial_op(x)
