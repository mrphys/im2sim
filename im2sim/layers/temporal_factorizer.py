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

import math

import torch
import torch.nn as nn


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


def _get_spatial_op(rank: int, op2d: type, op3d: type, name: str = "spatial op") -> type:
    """Select 2D or 3D operation based on rank."""
    if rank == 2:
        return op2d
    if rank == 3:
        return op3d
    raise ValueError(f"{name} only supports rank=2 or rank=3, got {rank}")


def _default_spatial_mode(rank: int) -> str:
    """Return default interpolation mode for spatial operations."""
    return "bilinear" if rank == 2 else "trilinear"


class TemporalFactorizer(nn.Module):
    """
    Wraps any torch.nn.Module to apply temporal factorization.

    Splits parameters into time and spatial components, applies spatial operations
    per timestep using TimeDistributed, and temporal operations per spatial location
    using SpaceDistributed.

    Supported modules are 2D/3D convolutions, transposed convolutions, max/average pooling,
    batch/instance normalization, channel dropout, and upsampling (``torch.nn.Upsample`` and
    ``im2sim.layers.custom_image_layers.Upsample``). Parameters like kernel_size, stride,
    padding and dilation are split into (T, *spatial) components. Other modules raise a ValueError.

    Args:
        module: The torch.nn.Module to factorize
        rank: Spatial rank (2 or 3)
        params_to_split: List of parameter names that may be given as (T, *spatial) to set the time
            component explicitly. Parameters not in the list apply to the spatial dims only, and the
            time component uses its default (1 for kernel_size/stride/dilation/scale_factor/size,
            0 for padding/output_padding). If None, auto-detected from the module type.
    """

    # Parameters that should be split into time/spatial components
    DEFAULT_SPLIT_PARAMS = {"kernel_size", "stride", "padding", "dilation", "output_padding"}

    # Module types that need temporal factorization
    CONV_TYPES = (nn.Conv2d, nn.Conv3d)
    POOL_TYPES = (nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool2d, nn.AvgPool3d)
    NORM_TYPES = (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    TRANSPOSE_TYPES = (nn.ConvTranspose2d, nn.ConvTranspose3d)
    UPSAMPLE_TYPES = (nn.Upsample,)
    DROPOUT_TYPES = (nn.Dropout2d, nn.Dropout3d)

    @classmethod
    def _init_custom_types(cls):
        """Initialize custom layer types from im2sim if available."""
        try:
            from im2sim.layers.custom_image_layers import Upsample

            # Add custom Upsample to the tuple (create a new tuple with both types)
            if Upsample not in cls.UPSAMPLE_TYPES:
                cls.UPSAMPLE_TYPES = cls.UPSAMPLE_TYPES + (Upsample,)
        except (ImportError, AttributeError):
            pass

    def __init__(
        self,
        module: nn.Module,
        rank: int,
        params_to_split: list | None = None,
    ):
        super().__init__()
        self.module = module
        self.rank = rank

        # Initialize custom layer types from im2sim
        self._init_custom_types()

        if rank not in (2, 3):
            raise ValueError(f"TemporalFactorizer only supports rank=2 or rank=3, got {rank}")

        # Auto-detect which parameters to split
        if params_to_split is None:
            params_to_split = self._detect_split_params(module)

        self.params_to_split = params_to_split

        # Create the factorized architecture based on module type
        self._factorize_module()

    def _detect_split_params(self, module: nn.Module) -> list:
        """Auto-detect which parameters need splitting based on module type."""
        # For most modules, we split standard spatial parameters
        if isinstance(module, self.TRANSPOSE_TYPES):
            return ["kernel_size", "stride", "padding", "dilation", "output_padding"]
        elif isinstance(module, (self.CONV_TYPES, self.POOL_TYPES)):
            return ["kernel_size", "stride", "padding", "dilation"]
        elif isinstance(module, self.NORM_TYPES):
            return []  # BatchNorm parameters don't split; just wrap
        elif isinstance(module, self.DROPOUT_TYPES):
            return []  # Dropout doesn't have spatial parameters
        elif isinstance(module, self.UPSAMPLE_TYPES):
            return ["scale_factor", "size"]
        else:
            # Fallback: try common parameter names
            return ["kernel_size", "stride", "padding", "dilation"]

    def _split_param(self, param, name: str = "parameter", time_default: int = 0):
        """Split parameter into time and spatial components.

        Scalar parameters apply to spatial dimensions only.
        Tuple parameters can be (T, *spatial) if `name` is in `params_to_split`, otherwise just (*spatial).
        String parameters like 'same' are returned as-is for spatial dims, but time gets time_default.

        Args:
            param: An int, float, string, or tuple/list of length (rank+1) or rank
            name: Name of parameter for error messages
            time_default: Default value for time component (0 for padding/dilation, 1 for kernel_size/stride)

        Returns:
            (time_component, spatial_component_tuple or string)
        """
        if isinstance(param, str):
            # String parameters like padding="same" - apply as-is to spatial, time uses default
            # Keep string as-is (not as tuple) because Conv2d/Conv3d expect a string, not tuple
            return time_default, param
        elif isinstance(param, (int, float)):
            # Scalar: apply to spatial only, time defaults to time_default
            # Convert to int if it's a float
            param = int(param) if isinstance(param, float) else param
            return time_default, (param,) * self.rank
        elif isinstance(param, (tuple, list)):
            if len(param) == self.rank + 1 and name in self.params_to_split:
                # Format: (T, *spatial)
                # Convert floats to ints
                time_val = int(param[0]) if isinstance(param[0], float) else param[0]
                space_val = tuple(int(v) if isinstance(v, float) else v for v in param[1:])
                return time_val, space_val
            elif len(param) == self.rank:
                # Format: just spatial dims, use time_default for time
                space_val = tuple(int(v) if isinstance(v, float) else v for v in param)
                return time_default, space_val
            elif name in self.params_to_split:
                raise ValueError(
                    f"{name} must have length {self.rank} or {self.rank + 1}, got {len(param)}"
                )
            else:
                raise ValueError(
                    f"{name} is not in params_to_split, so it must have length {self.rank}, "
                    f"got {len(param)}"
                )
        else:
            raise TypeError(f"{name} must be int, float, string, or tuple/list, got {type(param)}")

    def _factorize_module(self):
        """Create temporal factorization based on module type."""
        module = self.module

        # Handle convolution layers
        if isinstance(module, self.CONV_TYPES):
            self._factorize_conv(module)
        # Handle pooling layers
        elif isinstance(module, self.POOL_TYPES):
            self._factorize_pool(module)
        # Handle batch normalization
        elif isinstance(module, self.NORM_TYPES):
            self._factorize_norm(module)
        # Handle transposed convolution
        elif isinstance(module, self.TRANSPOSE_TYPES):
            self._factorize_conv_transpose(module)
        # Handle upsampling
        elif isinstance(module, self.UPSAMPLE_TYPES):
            self._factorize_upsample(module)
        # Handle dropout
        elif isinstance(module, self.DROPOUT_TYPES):
            self._factorize_dropout(module)
        else:
            # Unknown modules cannot be factorized
            raise ValueError(
                f"Unsupported module type for temporal factorization: {type(module)} \n Supported modules: {self.CONV_TYPES + self.POOL_TYPES + self.NORM_TYPES + self.TRANSPOSE_TYPES + self.UPSAMPLE_TYPES + self.DROPOUT_TYPES}"
            )

    def _factorize_conv(self, module: nn.Module):
        """Factorize convolutional layer."""
        # Extract parameters
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size
        stride = getattr(module, "stride", 1)
        padding = getattr(module, "padding", 0)
        dilation = getattr(module, "dilation", 1)
        groups = getattr(module, "groups", 1)
        bias = module.bias is not None
        padding_mode = getattr(module, "padding_mode", "zeros")

        # Split parameters (kernel_size/stride/dilation default to 1, padding defaults to 0)
        time_kernel, space_kernel = self._split_param(kernel_size, "kernel_size", time_default=1)
        time_stride, space_stride = self._split_param(stride, "stride", time_default=1)
        time_padding, space_padding = self._split_param(padding, "padding", time_default=0)
        time_dilation, space_dilation = self._split_param(dilation, "dilation", time_default=1)

        # Create spatial convolution (applied per timestep)
        spatial_conv = _get_spatial_op(self.rank, nn.Conv2d, nn.Conv3d, "Conv spatial")
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

        # Create temporal convolution (applied per spatial location)
        self.temporal_op = SpaceDistributed(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=time_kernel,
                stride=time_stride,
                padding=time_padding,
                dilation=time_dilation,
                groups=1,
                bias=bias,
            )
        )

    def _factorize_pool(self, module: nn.Module):
        """Factorize pooling layer."""
        kernel_size = module.kernel_size
        stride = getattr(module, "stride", None)
        padding = getattr(module, "padding", 0)
        dilation = getattr(module, "dilation", 1)
        ceil_mode = getattr(module, "ceil_mode", False)

        # Split parameters (kernel_size/stride/dilation default to 1, padding defaults to 0)
        time_kernel, space_kernel = self._split_param(kernel_size, "kernel_size", time_default=1)
        time_stride, space_stride = self._split_param(
            stride or kernel_size, "stride", time_default=1
        )
        time_padding, space_padding = self._split_param(padding, "padding", time_default=0)
        time_dilation, space_dilation = self._split_param(dilation, "dilation", time_default=1)

        # Determine pool type
        is_max_pool = isinstance(module, (nn.MaxPool2d, nn.MaxPool3d))

        # Create spatial pooling (AvgPool doesn't support dilation)
        spatial_pool_class = _get_spatial_op(
            self.rank,
            nn.MaxPool2d if is_max_pool else nn.AvgPool2d,
            nn.MaxPool3d if is_max_pool else nn.AvgPool3d,
            "Pool spatial",
        )
        if is_max_pool:
            self.spatial_op = TimeDistributed(
                spatial_pool_class(
                    kernel_size=space_kernel,
                    stride=space_stride,
                    padding=space_padding,
                    dilation=space_dilation,
                    ceil_mode=ceil_mode,
                )
            )
        else:
            self.spatial_op = TimeDistributed(
                spatial_pool_class(
                    kernel_size=space_kernel,
                    stride=space_stride,
                    padding=space_padding,
                    ceil_mode=ceil_mode,
                )
            )

        # Create temporal pooling (AvgPool doesn't support dilation)
        temporal_pool_class = nn.MaxPool1d if is_max_pool else nn.AvgPool1d
        if is_max_pool:
            self.temporal_op = SpaceDistributed(
                temporal_pool_class(
                    kernel_size=time_kernel,
                    stride=time_stride,
                    padding=time_padding,
                    dilation=time_dilation,
                    ceil_mode=ceil_mode,
                )
            )
        else:
            self.temporal_op = SpaceDistributed(
                temporal_pool_class(
                    kernel_size=time_kernel,
                    stride=time_stride,
                    padding=time_padding,
                    ceil_mode=ceil_mode,
                )
            )

    def _factorize_norm(self, module: nn.Module):
        """Factorize batch/instance normalization."""
        num_features = module.num_features
        eps = getattr(module, "eps", 1e-5)
        momentum = getattr(module, "momentum", 0.1)
        affine = getattr(module, "affine", True)
        track_running_stats = getattr(module, "track_running_stats", True)

        is_batch_norm = isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d))

        # Create spatial normalization
        spatial_norm_class = _get_spatial_op(
            self.rank,
            nn.BatchNorm2d if is_batch_norm else nn.InstanceNorm2d,
            nn.BatchNorm3d if is_batch_norm else nn.InstanceNorm3d,
            "Norm spatial",
        )
        self.spatial_op = TimeDistributed(
            spatial_norm_class(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
            )
        )

        # Create temporal normalization
        temporal_norm_class = nn.BatchNorm1d if is_batch_norm else nn.InstanceNorm1d
        self.temporal_op = SpaceDistributed(
            temporal_norm_class(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
            )
        )

    def _factorize_conv_transpose(self, module: nn.Module):
        """Factorize transposed convolution."""
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size
        stride = getattr(module, "stride", 1)
        padding = getattr(module, "padding", 0)
        output_padding = getattr(module, "output_padding", 0)
        dilation = getattr(module, "dilation", 1)
        bias = module.bias is not None

        # Split parameters (kernel_size/stride/dilation default to 1, padding/output_padding default to 0)
        time_kernel, space_kernel = self._split_param(kernel_size, "kernel_size", time_default=1)
        time_stride, space_stride = self._split_param(stride, "stride", time_default=1)
        time_padding, space_padding = self._split_param(padding, "padding", time_default=0)
        time_output_pad, space_output_pad = self._split_param(
            output_padding, "output_padding", time_default=0
        )
        time_dilation, space_dilation = self._split_param(dilation, "dilation", time_default=1)

        # Create spatial transposed convolution
        spatial_conv_trans = _get_spatial_op(
            self.rank, nn.ConvTranspose2d, nn.ConvTranspose3d, "ConvTranspose spatial"
        )
        self.spatial_op = TimeDistributed(
            spatial_conv_trans(
                in_channels,
                out_channels,
                kernel_size=space_kernel,
                stride=space_stride,
                padding=space_padding,
                output_padding=space_output_pad,
                dilation=space_dilation,
                bias=bias,
            )
        )

        # Create temporal transposed convolution
        self.temporal_op = SpaceDistributed(
            nn.ConvTranspose1d(
                out_channels,
                out_channels,
                kernel_size=time_kernel,
                stride=time_stride,
                padding=time_padding,
                output_padding=time_output_pad,
                dilation=time_dilation,
                bias=bias,
            )
        )

    def _factorize_upsample(self, module: nn.Module):
        """Factorize upsampling layer."""
        scale_factor = getattr(module, "scale_factor", None)
        size = getattr(module, "size", None)
        mode = getattr(module, "mode", _default_spatial_mode(self.rank))
        align_corners = getattr(module, "align_corners", None)

        # Adjust mode for rank=2 if trilinear is specified
        if self.rank == 2 and mode == "trilinear":
            mode = "bilinear"

        # Split scale_factor or size
        # When size is provided as spatial-only, don't apply temporal upsampling
        if scale_factor is not None:
            time_value, space_value = self._split_param(
                scale_factor, "scale_factor", time_default=1
            )
            use_scale = True
            upscale_time = True  # Always upscale time with scale_factor
        elif size is not None:
            # Check if size includes temporal dimension
            if isinstance(size, (tuple, list)) and len(size) == self.rank + 1:
                time_value, space_value = self._split_param(size, "size", time_default=1)
                upscale_time = True
            else:
                time_value = 1
                space_value = self._split_param(size, "size", time_default=1)[1]
                upscale_time = False  # Don't upscale time if only spatial size specified
            use_scale = False
        else:
            raise ValueError("Either scale_factor or size must be provided")

        # Create spatial upsampling
        if use_scale:
            self.spatial_op = TimeDistributed(
                nn.Upsample(
                    scale_factor=space_value,
                    mode=mode,
                    align_corners=align_corners if "linear" in mode else None,
                )
            )
        else:
            self.spatial_op = TimeDistributed(
                nn.Upsample(
                    size=space_value,
                    mode=mode,
                    align_corners=align_corners if "linear" in mode else None,
                )
            )

        # Create temporal upsampling only if needed
        if upscale_time:
            if use_scale:
                self.temporal_op = SpaceDistributed(
                    nn.Upsample(
                        scale_factor=time_value,
                        mode="linear",
                        align_corners=align_corners,
                    )
                )
            else:
                self.temporal_op = SpaceDistributed(
                    nn.Upsample(
                        size=time_value,
                        mode="linear",
                        align_corners=align_corners,
                    )
                )
        else:
            self.temporal_op = None

    def _factorize_dropout(self, module: nn.Module):
        """Factorize dropout layer."""
        p = getattr(module, "p", 0.5)
        inplace = getattr(module, "inplace", False)

        # For dropout, just apply the same dropout spatially and temporally
        # Since dropout is applied per element, we don't need true factorization
        # Just wrap the original module with TimeDistributed
        self.spatial_op = TimeDistributed(module.__class__(p=p, inplace=inplace))
        self.temporal_op = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal factorization.

        Args:
            x: Tensor of shape (N, C, T, *spatial)

        Returns:
            Tensor of shape (N, C, T_out, *spatial_out)
        """
        # Apply spatial operations per timestep
        x = self.spatial_op(x)

        # Apply temporal operations per spatial location (if present)
        # Normalizing over a single timestep is degenerate (InstanceNorm1d raises on T=1)
        skip_temporal_norm = isinstance(self.module, self.NORM_TYPES) and x.shape[2] == 1
        if self.temporal_op is not None and not skip_temporal_norm:
            x = self.temporal_op(x)

        return x
