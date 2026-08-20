import math

import torch

from im2sim.utils.layer_util import get_image_layer, register_with_ranks


@register_with_ranks("DepthwiseConv", ranks=(1, 2, 3))
class DepthwiseConv(torch.nn.Module):
    """
    Depthwise convolution layer that applies a separate convolutional filter to each input channel.

    This operation is useful for reducing the number of parameters and computational cost in convolutional neural networks, especially in mobile and embedded applications[1].

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (should be equal to in_channels for depthwise convolution).
        rank (int): The rank of the convolution (`1` for 1D, `2` for 2D, `3` for 3D).
        kernel_size (int | tuple): Size of the convolving kernel. Default is `3`.
        stride (int | tuple): Stride of the convolution. Default is `1`.
        padding (str or int | tuple): Padding added to all four sides of the input. Default is `"same"`.
        dilation (int | tuple): Spacing between kernel elements. Default is `1`.
        bias (bool): If `True`, adds a learnable bias to the output. Default is `True`.

    References:
        .. [1]	A. G. Howard et al., MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,
            Apr. 17, 2017, arXiv: arXiv:1704.04861. doi: 10.48550/arXiv.1704.04861.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        rank,
        kernel_size=3,
        stride=1,
        padding="same",
        dilation=1,
        bias=True,
    ):
        super().__init__()
        assert out_channels % in_channels == 0, (
            "For depthwise convolution, out_channels should be a multiple of in_channels."
        )
        self.conv = get_image_layer("Conv", rank)(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x):
        """
        Forward pass of the depthwise convolution layer.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, in_channels, *spatial_dims)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, out_channels, *spatial_dims)`.
        """
        return self.conv(x)


@register_with_ranks("DepthwiseSeparableConv", ranks=(1, 2, 3))
class DepthwiseSeparableConv(torch.nn.Module):
    """
    Depthwise separable convolution layer that consists of a depthwise convolution followed by a pointwise convolution.

    This operation is useful for reducing the number of parameters and computational cost in convolutional neural networks,
    while retaining more representational power compared to standard depthwise convolution[1].

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): The rank of the convolution (`1` for 1D, `2`for 2D, `3` for 3D).
        kernel_size (int | tuple): Size of the convolving kernel. Default is `3`.
        stride (int | tuple): Stride of the convolution. Default is `1`.
        padding (str or int | tuple): Padding added to all four sides of the input. Default is `"same"`.
        dilation (int | tuple): Spacing between kernel elements. Default is `1`.
        bias (bool): If `True`, adds a learnable bias to the output. Default is `True`.

    References:
        .. [1] F. Chollet, Xception: Deep Learning with Depthwise Separable Convolutions,
            Apr. 04, 2017, arXiv: arXiv:1610.02357. doi: 10.48550/arXiv.1610.02357.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        rank,
        kernel_size=3,
        stride=1,
        padding="same",
        dilation=1,
        bias=True,
    ):
        super().__init__()

        self.depthwise = get_image_layer("Conv", rank)(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = get_image_layer("Conv", rank)(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        """
        Forward pass of the depthwise separable convolution layer.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, in_channels, *spatial_dims)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, out_channels, *spatial_dims)`.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


@register_with_ranks("GhostConv", ranks=(1, 2, 3))
class GhostConv(torch.nn.Module):
    """
    Ghost convolution layer that generates more feature maps from cheap operations.

    This operation is useful for reducing the number of parameters and computational cost in convolutional neural networks.
    The cheap operation can either be a depthwise convolution as per the original GhostNet paper[1] or a depthwise separable convolution as in the HalfUNet paper[2

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): The rank of the convolution (`1` for 1D, `2` for 2D, `3` for 3D).
        kernel_size (int | tuple): Size of the convolving kernel for the primary convolution. Default is `3`.
        ratio (int): Ratio of the number of output channels to the number of primary convolution channels. Default is `2`.
        dw_kernel_size (int | tuple): Size of the convolving kernel for the cheap operation. Default is `3`.
        stride (int | tuple): Stride of the primary convolution. Default is `1`.
        padding (str or int | tuple): Padding added to all four sides of the input for the primary convolution. Default is `"same"`.
        separable (bool): If `True`, uses depthwise separable convolution for the cheap operation. Default is `False`.
        bias (bool): If `True`, adds a learnable bias to the output. Default is `True`.

    References:
        .. [1] K. Han, Y. Wang, Q. Tian, J. Guo, C. Xu, and C. Xu, GhostNet: More Features from Cheap Operations,
            Mar. 13, 2020, arXiv: arXiv:1911.11907. doi: 10.48550/arXiv.1911.11907.
        .. [2] H. Lu, Y. She, J. Tie, and S. Xu, Half-UNet: A Simplified U-Net Architecture for Medical Image Segmentation,
            Front. Neuroinformatics, vol. 16, Jun. 2022, doi: 10.3389/fninf.2022.911679.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        rank,
        kernel_size=3,
        ratio=2,
        dw_kernel_size=3,
        stride=1,
        padding="same",
        separable=False,
        bias=True,
    ):
        super().__init__()
        self.rank = rank
        self.out_channels = out_channels
        self.init_channels = int(out_channels / ratio)
        self.new_channels = out_channels - self.init_channels

        self.primary_conv = get_image_layer("Conv", rank)(
            in_channels=in_channels,
            out_channels=self.init_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        cheap_conv_type = "DepthwiseSeparableConv" if separable else "DepthwiseConv"
        self.cheap_operation = get_image_layer(cheap_conv_type, rank)(
            in_channels=self.init_channels,
            out_channels=self.new_channels,
            kernel_size=dw_kernel_size,
            stride=1,
            padding="same",
            bias=bias,
        )

    def forward(self, x):
        """
        Forward pass of the Ghost convolution layer.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, in_channels, *spatial_dims)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, out_channels, *spatial_dims)`.
        """
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


@register_with_ranks("EfficientChannelAttn", ranks=(1, 2, 3))
class EfficientChannelAttn(torch.nn.Module):
    """
    Efficient Channel Attention (ECA) layer that adaptively selects important channels based on global context.

    This operation is useful for improving the representational power of convolutional neural networks by focusing on the most informative channels[1].

    Args:
        channels (int): Number of input channels.
        rank (int): The rank of the input tensor (`1` for 1D, `2` for 2D, `3` for 3D).

    References:
        .. [1] Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, and Q. Hu, ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks,
            Mar. 04, 2020, arXiv: arXiv:1910.03151. doi: 10.48550/arXiv.1910.03151.
    """

    def __init__(self, channels: int, rank: int):
        super().__init__()
        self.rank = rank
        k_raw = math.log2(channels) / 2 + 0.5
        k = max(3, int(k_raw) if int(k_raw) % 2 == 1 else int(k_raw) + 1)
        self.avg_pool = get_image_layer("AdaptiveAvgPool", rank)(1)
        self.conv1d = torch.nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Efficient Channel Attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, channels, *spatial_dims).`

        Returns:
           torch.Tensor: Output tensor of shape `(batch_size, channels, *spatial_dims)` with channel-wise attention applied.
        """
        B, C = x.shape[:2]
        w = self.avg_pool(x).view(B, 1, C)  # [B, 1, C]
        w = self.conv1d(w)  # [B, 1, C]
        out_shape = [B, C] + [1] * self.rank
        w = self.sigmoid(w).view(*out_shape)
        return x * w


@register_with_ranks("SqueezeExcite", ranks=(1, 2, 3))
class SqueezeExcite(torch.nn.Module):
    """
    Squeeze-and-Excitation (SE) layer that adaptively recalibrates channel-wise feature responses.

    This operation is useful for improving the representational power of convolutional neural networks by explicitly modeling interdependencies between channels[1].

    Args:
        channels (int): Number of input channels.
        rank (int): The rank of the input tensor (`1` for 1D, `2` for 2D, `3` for 3D).
        reduction (int): Reduction ratio for the hidden layer in the SE block. Default is `8`.

    References:
        .. [1] J. Hu, L. Shen, and G. Sun, Squeeze-and-Excitation Networks,
            Mar. 27, 2018, arXiv: arXiv:1709.01507. doi: 10.48550/arXiv.1709.01507.
    """

    def __init__(self, channels: int, rank: int, reduction: int = 8):
        super().__init__()
        self.rank = rank
        hidden = max(8, channels // reduction)
        self.avg_pool = get_image_layer("AdaptiveAvgPool", rank)(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, channels),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Efficient Channel Attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, channels, *spatial_dims)`.

        Returns:
           torch.Tensor: Output tensor of shape `(batch_size, channels, *spatial_dims)` with channel-wise attention applied.
        """
        B, C = x.shape[:2]
        z = self.avg_pool(x).view(B, C)
        out_shape = [B, C] + [1] * self.rank
        s = self.sigmoid(self.fc(z)).view(*out_shape)
        return x * s


@register_with_ranks("ConditionedSqueezeExcite", ranks=(1, 2, 3))
class ConditionedSqueezeExcite(torch.nn.Module):
    """
    Conditioned Squeeze-and-Excitation [1] (SE) layer that adaptively recalibrates channel-wise feature responses based on an additional conditioning input.

    This operation is useful for improving the representational power of convolutional neural networks by explicitly modeling interdependencies between channels,
    while also allowing for external conditioning information to influence the recalibration process.

    Args:
        channels (int): Number of input channels.
        rank (int): The rank of the input tensor (`1` for 1D, `2` for 2D, `3` for 3D).
        n_cond (int): Number of conditioning channels. Default is `6`.
        reduction (int): Reduction ratio for the hidden layer in the SE block. Default is `8`.
        mode (str): Mode of combining the feature and conditioning information. Can be `"concat"` or `"add"`. Default is `"add"`.

    References:
        .. [1] J. Hu, L. Shen, and G. Sun, Squeeze-and-Excitation Networks,
            Mar. 27, 2018, arXiv: arXiv:1709.01507. doi: 10.48550/arXiv.1709.01507.

    """

    def __init__(
        self, channels: int, rank: int, n_cond: int = 6, reduction: int = 8, mode: str = "add"
    ):
        super().__init__()
        self.rank = rank
        if mode not in ("concat", "add"):
            raise ValueError(f"unknown se_cond_mode {mode!r}")
        self.mode = mode
        hidden = max(8, channels // reduction)
        self.avg_pool = get_image_layer("AdaptiveAvgPool", self.rank)(1)
        if mode == "concat":
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(channels + n_cond, hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden, channels),
            )
        else:  # add
            self.feat_fc = torch.nn.Linear(channels, hidden)
            self.cond_fc = torch.nn.Linear(n_cond, hidden)
            self.act = torch.nn.ReLU(inplace=True)
            self.out_fc = torch.nn.Linear(hidden, channels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Conditioned Squeeze-and-Excitation layer.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, channels, *spatial_dims)`.
            cond (torch.Tensor): Conditioning tensor of shape `(batch_size, n_cond)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, channels, *spatial_dims)` with channel-wise attention applied based on the conditioning input.
        """
        B, C = x.shape[:2]
        z = self.avg_pool(x).view(B, C)
        if self.mode == "concat":
            s = self.fc(torch.cat([z, cond], dim=1))
        else:
            s = self.out_fc(self.act(self.feat_fc(z) + self.cond_fc(cond)))

        out_shape = [B, C] + [1] * self.rank
        s = self.sigmoid(s).view(*out_shape)
        return x * s


@register_with_ranks("Upsample", ranks=(1, 2, 3))
class Upsample(torch.nn.Module):
    """
    Upsample layer that increases the spatial resolution of the input tensor using interpolation.

    Wrapper of `torch.nn.functional.interpolate` to provide a consistent interface for upsampling across different spatial ranks (1D, 2D, 3D).

    This operation is useful for tasks such as image super-resolution, semantic segmentation, and generative modeling.

    Args:
        rank (int): The rank of the input tensor (`1` for 1D, `2` for 2D, `3` for 3D).
        scale_factor (int | tuple): The multiplier for the spatial size. Default is `2`.
        mode (str): The algorithm used for upsampling. Options are `"nearest"`, `"linear"`, `"bilinear"`, `"bicubic"`, `"trilinear"`. Default for 1D is `"linear"`, for 2D is `"bilinear"`, and for 3D is `"trilinear"`.
        align_corners (bool | None): If `True`, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels. Default is `None`.
    """

    def __init__(self, rank, scale_factor=2, mode="nearest", align_corners=None):
        super().__init__()
        self.rank = rank
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """
        Forward pass of the Upsample layer.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, channels, *spatial_dims)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, channels, *upsampled_spatial_dims)`.
        """

        accepted_modes = {
            1: ["linear", "nearest"],
            2: ["bilinear", "nearest"],
            3: ["trilinear", "nearest"],
        }

        if self.mode not in accepted_modes[self.rank]:
            self.mode = accepted_modes[self.rank][0]

        return torch.nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
