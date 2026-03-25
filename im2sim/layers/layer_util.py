from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import math


def get_image_layer(name, rank):
  """Get an N-D layer object.

  Args:
    name: A `str`. The name of the requested layer.
    rank: An `int`. The rank of the requested layer.

  Returns:
    A `torch.nn.Module` object.

  Raises:
    ValueError: If the requested layer is unknown.
  """
  try:
    return _IMAGE_LAYERS[(name, rank)]
  except KeyError as err:
    raise ValueError(
        f"Could not find a layer with name '{name}' and rank {rank}.") from err
  
  
def get_graph_layer(name):
  """Get an graph layer object.

  Args:
    name: A `str`. The name of the requested layer.

  Returns:
    A `torch.nn.Module` object.

  Raises:
    ValueError: If the requested layer is unknown.
  """
  try:
    return _GRAPH_LAYERS[name] 
  except KeyError as err:
    raise ValueError(
        f"Could not find an activation with name '{name}'") from err
  
def get_default_kwargs(name):
  """Get an graph layer object.

  Args:
    name: A `str`. The name of the requested layer.

  Returns:
    A `torch.nn.Module` object.

  Raises:
    ValueError: If the requested layer is unknown.
  """
  try:
    return _LAYER_KWARGS[name] 
  except KeyError as err:
    raise ValueError(
        f"Could not find an activation with name '{name}'") from err
  
  
def get_activation(name):
  """Get an activation object.

  Args:
    name: A `str`. The name of the requested layer.

  Returns:
    A `torch.nn.Module` object.

  Raises:
    ValueError: If the requested activation is unknown.
  """
  try:
    return _ACTIVATIONS[name]
  except KeyError as err:
    raise ValueError(
        f"Could not find an activation with name '{name}'") from err


_IMAGE_LAYERS = {
    ('AveragePooling', 1): nn.AvgPool1d,
    ('AveragePooling', 2): nn.AvgPool2d,
    ('AveragePooling', 3): nn.AvgPool3d,
    ('Conv', 1): nn.Conv1d,
    ('Conv', 2): nn.Conv2d,
    ('Conv', 3): nn.Conv3d,
    ('ConvTranspose', 1): nn.ConvTranspose1d,
    ('ConvTranspose', 2): nn.ConvTranspose2d,
    ('ConvTranspose', 3): nn.ConvTranspose3d,
    ('MaxPool', 1): nn.MaxPool1d,
    ('MaxPool', 2): nn.MaxPool2d,
    ('MaxPool', 3): nn.MaxPool3d,
    ('Dropout', 1): nn.Dropout1d,
    ('Dropout', 2): nn.Dropout2d,
    ('Dropout', 3): nn.Dropout3d,
    ('ZeroPadding', 1): nn.ZeroPad1d,
    ('ZeroPadding', 2): nn.ZeroPad2d,
    ('ZeroPadding', 3): nn.ZeroPad3d,
    ('BatchNorm', 1): nn.BatchNorm1d,
    ('BatchNorm', 2): nn.BatchNorm2d,
    ('BatchNorm', 3): nn.BatchNorm3d,
    ('InstanceNorm', 1): nn.InstanceNorm1d,
    ('InstanceNorm', 2): nn.InstanceNorm2d,
    ('InstanceNorm', 3): nn.InstanceNorm3d
}

_GRAPH_LAYERS = {
  'ChebConv': gnn.ChebConv,
  'GraphConv': gnn.GraphConv,
  'GCNConv': gnn.GCNConv,
  'GATConv': gnn.GATConv,
  'InstanceNorm': gnn.InstanceNorm,
  'BatchNorm': gnn.BatchNorm,
  'GraphNorm': gnn.GraphNorm,
}

_LAYER_KWARGS = {
  'ChebConv': {'K':3},
  'GraphConv': {},
  'GCNConv': {},
  'GATConv': {}
}
_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
    "softmax": nn.Softmax
}


def init_weights(m):
    if isinstance(m, gnn.ChebConv):
        for lin in m.lins:
            nn.init.kaiming_normal_(lin.weight, nonlinearity='leaky_relu')
            lin.weight.data *= 0.1
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

def standardize_spatial_factors(factors, rank):
    """
    Convert a sequence of spatial factors into a standardized list of tuples.
    """
    standardized = []

    for f in factors:
        if isinstance(f, int):
            standardized.append(tuple([f] * rank))
        elif isinstance(f, (tuple, list)):
            standardized.append(tuple(f))
        else:
            raise TypeError(
                f"Each factor must be an int, tuple, or list, got {type(f).__name__}"
            )
        
    return standardized

def _expand_to_4d(param, name="parameter"):
    """
    Convert an int or 4-tuple/list into a 4-tuple: (T, D, H, W)
    """
    if isinstance(param, int):
        return (param, param, param, param)
    if isinstance(param, (tuple, list)):
        if len(param) != 4:
            raise ValueError(f"{name} must have 4 elements (T, D, H, W), got {param}")
        return tuple(param)
    raise TypeError(f"{name} must be an int or a tuple/list of length 4")


def _same_padding_4d(kernel_size):
    """
    Compute 'same-like' padding for odd kernel sizes.
    """
    k = _expand_to_4d(kernel_size, "kernel_size")
    return tuple(kk // 2 for kk in k)


def _get_spatial_op(rank, op2d, op3d, name="spatial op"):
    if rank == 2:
        return op2d
    if rank == 3:
        return op3d
    raise ValueError(f"{name} only supports rank=2 or rank=3, got {rank}")


def _default_spatial_mode(rank):
    return "bilinear" if rank == 2 else "trilinear"

def _same_padding_time(kernel_size, rank):
    """
    Return same-style padding for (T + spatial) kernel sizes.

    rank=2 -> expects int or (T, H, W)
    rank=3 -> expects int or (T, D, H, W)
    """
    if isinstance(kernel_size, int):
        k = (kernel_size,) * (rank + 1)
    elif isinstance(kernel_size, (tuple, list)):
        if len(kernel_size) != rank + 1:
            raise ValueError(
                f"kernel_size must have length {rank + 1} for rank={rank}, got {kernel_size}"
            )
        k = tuple(kernel_size)
    else:
        raise TypeError("kernel_size must be an int or tuple/list")

    return tuple(kk // 2 for kk in k)


class TimeDistributed(nn.Module):
    """
    Apply a module independently to each time step.

    Expected input:
        2D+1: (N, C, T, H, W)
        3D+1: (N, C, T, D, H, W)

    Wrapped module should accept:
        2D+1: (N, C, H, W)
        3D+1: (N, C, D, H, W)
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
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

    Expected input:
        2D+1: (N, C, T, H, W)
        3D+1: (N, C, T, D, H, W)

    Wrapped module should accept:
        (N_flat, C, T)
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
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


class ConvTime(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        rank=3,
    ):
        super().__init__()
        self.rank = rank

        self.time_kernel, self.space_kernel = self._split_param(kernel_size, "kernel_size")
        self.time_stride, self.space_stride = self._split_param(stride, "stride")
        self.time_padding, self.space_padding = self._split_param(padding, "padding")
        self.time_dilation, self.space_dilation = self._split_param(dilation, "dilation")

        spatial_conv = _get_spatial_op(rank, nn.Conv2d, nn.Conv3d, "ConvTime spatial conv")

        self.conv_spatial = TimeDistributed(
            spatial_conv(
                in_channels,
                out_channels,
                kernel_size=self.space_kernel,
                stride=self.space_stride,
                padding=self.space_padding,
                dilation=self.space_dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
                device=device,
                dtype=dtype,
            )
        )

        self.conv_time = SpaceDistributed(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=self.time_kernel,
                stride=self.time_stride,
                padding=self.time_padding,
                dilation=self.time_dilation,
                groups=1,
                bias=bias,
                device=device,
                dtype=dtype,
            )
        )

    def _split_param(self, param, name="parameter"):
        if isinstance(param, (tuple, list)):
            if len(param) != self.rank + 1:
                raise ValueError(f"{name} must have {self.rank + 1} elements (T + spatial dims)")
            return param[0], tuple(param[1:])
        return param, (param,) * self.rank

    def forward(self, x):
        x = self.conv_spatial(x)
        x = self.conv_time(x)
        return x


class UpsampleTime(nn.Module):
    """
    Separable (rank)+1D upsampling:
      - spatial upsampling per time step
      - temporal upsampling per spatial location

    rank=2:
        input  (B, C, T, H, W)
        output (B, C, T_out, H_out, W_out)

    rank=3:
        input  (B, C, T, D, H, W)
        output (B, C, T_out, D_out, H_out, W_out)
    """
    def __init__(self, scale_factor=None, size=None, mode=None, align_corners=False, rank=3):
        super().__init__()
        self.rank = rank

        if (scale_factor is None) == (size is None):
            raise ValueError("Provide exactly one of scale_factor or size")

        if mode is None:
            mode = _default_spatial_mode(rank)
        elif rank == 2 and mode == "trilinear":
            mode = "bilinear"

        self.use_scale_factor = scale_factor is not None

        if self.use_scale_factor:
            self.time_value, self.space_value = self._split_param(scale_factor, "scale_factor")
        else:
            self.time_value, self.space_value = self._split_param(size, "size")

        if self.use_scale_factor:
            self.up_spatial = TimeDistributed(
                nn.Upsample(
                    scale_factor=self.space_value,
                    mode=mode,
                    align_corners=align_corners if "linear" in mode else None,
                )
            )
            self.up_time = SpaceDistributed(
                nn.Upsample(
                    scale_factor=self.time_value,
                    mode="linear",
                    align_corners=align_corners,
                )
            )
        else:
            self.up_spatial = TimeDistributed(
                nn.Upsample(
                    size=self.space_value,
                    mode=mode,
                    align_corners=align_corners if "linear" in mode else None,
                )
            )
            self.up_time = SpaceDistributed(
                nn.Upsample(
                    size=self.time_value,
                    mode="linear",
                    align_corners=align_corners,
                )
            )

    def _split_param(self, param, name):
        if isinstance(param, (tuple, list)):
            if len(param) != self.rank + 1:
                raise ValueError(f"{name} must have {self.rank + 1} elements (T + spatial dims)")
            return param[0], tuple(param[1:])
        return param, (param,) * self.rank

    def forward(self, x):
        x = self.up_spatial(x)
        x = self.up_time(x)
        return x


class ConvTransTime(nn.Module):
    """
    Separable (rank)+1D transposed convolution:
      1) spatial ConvTransposeNd applied independently per time step
      2) temporal ConvTranspose1d applied independently per spatial location
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        bias=True,
        dilation=1,
        rank=3,
    ):
        super().__init__()
        self.rank = rank

        self.time_kernel, self.space_kernel = self._split_param(kernel_size, "kernel_size")
        self.time_stride, self.space_stride = self._split_param(stride, "stride")
        self.time_padding, self.space_padding = self._split_param(padding, "padding")
        self.time_outpad, self.space_outpad = self._split_param(output_padding, "output_padding")
        self.time_dilation, self.space_dilation = self._split_param(dilation, "dilation")

        spatial_conv_trans = _get_spatial_op(
            rank, nn.ConvTranspose2d, nn.ConvTranspose3d, "ConvTransTime spatial conv"
        )

        self.up_spatial = TimeDistributed(
            spatial_conv_trans(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.space_kernel,
                stride=self.space_stride,
                padding=self.space_padding,
                output_padding=self.space_outpad,
                dilation=self.space_dilation,
                bias=bias,
            )
        )

        self.up_time = SpaceDistributed(
            nn.ConvTranspose1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=self.time_kernel,
                stride=self.time_stride,
                padding=self.time_padding,
                output_padding=self.time_outpad,
                dilation=self.time_dilation,
                bias=bias,
            )
        )

    def _split_param(self, param, name="parameter"):
        if isinstance(param, (tuple, list)):
            if len(param) != self.rank + 1:
                raise ValueError(f"{name} must have {self.rank + 1} elements (T + spatial dims)")
            return param[0], tuple(param[1:])
        return param, (param,) * self.rank

    def forward(self, x):
        x = self.up_spatial(x)
        x = self.up_time(x)
        return x


class MaxPoolTime(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, rank=3):
        super().__init__()
        self.rank = rank

        stride = kernel_size if stride is None else stride

        self.time_kernel, self.space_kernel = self._split_param(kernel_size, "kernel_size")
        self.time_stride, self.space_stride = self._split_param(stride, "stride")
        self.time_padding, self.space_padding = self._split_param(padding, "padding")
        self.time_dilation, self.space_dilation = self._split_param(dilation, "dilation")

        spatial_pool = _get_spatial_op(rank, nn.MaxPool2d, nn.MaxPool3d, "MaxPoolTime spatial pool")

        self.pool_spatial = TimeDistributed(
            spatial_pool(
                kernel_size=self.space_kernel,
                stride=self.space_stride,
                padding=self.space_padding,
                dilation=self.space_dilation,
                ceil_mode=ceil_mode,
            )
        )

        self.pool_time = SpaceDistributed(
            nn.MaxPool1d(
                kernel_size=self.time_kernel,
                stride=self.time_stride,
                padding=self.time_padding,
                dilation=self.time_dilation,
                ceil_mode=ceil_mode,
            )
        )

    def _split_param(self, param, name="parameter"):
        if isinstance(param, (tuple, list)):
            if len(param) != self.rank + 1:
                raise ValueError(f"{name} must have {self.rank + 1} elements (T + spatial dims)")
            return param[0], tuple(param[1:])
        return param, (param,) * self.rank

    def forward(self, x):
        x = self.pool_spatial(x)
        x = self.pool_time(x)
        return x


class BatchNormTime(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, rank=3):
        super().__init__()

        spatial_bn = _get_spatial_op(rank, nn.BatchNorm2d, nn.BatchNorm3d, "BatchNormTime spatial norm")

        self.bn_spatial = TimeDistributed(
            spatial_bn(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
            )
        )

        self.bn_time = SpaceDistributed(
            nn.BatchNorm1d(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
            )
        )

    def forward(self, x):
        x = self.bn_spatial(x)
        x = self.bn_time(x)
        return x