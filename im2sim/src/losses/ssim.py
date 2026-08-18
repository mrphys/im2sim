from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from im2sim.src.utils import api_util


@api_util.export("losses.SSIMLoss")
class SSIMLoss(nn.Module):
    r"""Structural Similarity Index Measure (SSIM) loss.

    The SSIM loss is defined as `1 - SSIM`, where SSIM is the Structural Similarity Index Measure[1].

    .. math::
        \operatorname{SSIM}(x, y) =
        \frac{
            (2\mu_x\mu_y + C_1)
            (2\sigma_{xy} + C_2)
        }{
            (\mu_x^2 + \mu_y^2 + C_1)
            (\sigma_x^2 + \sigma_y^2 + C_2)
        }


        C_1 = (k_1 \, \mathrm{max\_val})^2,
        \qquad
        C_2 = (k_2 \, \mathrm{max\_val})^2.

    `rank` specifies the number of image dimensions over which SSIM is
    calculated. The dimensions before the channel dimension are treated as
    batch dimensions.

    This allows, for example, 2D SSIM to be calculated independently over
    every slice of a 3D image.

    Args:
        max_val:
            Dynamic range of the images. If `None`, defaults to `1`
            for floating-point inputs and the largest representable positive
            value for integer inputs.

        filter_size:
            Size of the Gaussian filter. Must be a positive odd
            integer. Defaults to `11`.

        filter_sigma:
            Standard deviation of the Gaussian filter.
            Defaults to `1.5`.

        k1:
            Factor used to calculate the luminance regularisation constant.
            Defaults to `0.01`.

        k2:
            Factor used to calculate the contrast regularisation constant.
            Defaults to `0.03`.

        batch_dims:
            Number of dimensions before the channel dimension that
            should be treated as batch dimensions.
            If `None`, it is inferred from `image_dims` and the input
            tensor rank.

        image_dims:
            Number of spatial dimensions used for calculating SSIM.
            If `None`, it is inferred from `batch_dims` and the input
            tensor rank.

        rank:
            Number of spatial dimensions used by the Gaussian filter.
            Must be `2` or `3`.
            If `None`, it is inferred from `image_dims`. If both are
            `None`, it defaults to `2`.

    Examples:
        Standard 2D SSIM:

        ..  code-block:: python

            loss = SSIMLoss()
            y_true.shape == (B, C, H, W)
            y_pred.shape == (B, C, H, W)

        3D SSIM:

        ..  code-block:: python

            loss = SSIMLoss(rank=3)
            y_true.shape == (B, C, D, H, W)
            y_pred.shape == (B, C, D, H, W)

        2D SSIM applied independently to every slice of a 3D image:

        ..  code-block:: python

            loss = SSIMLoss(
                rank=2,
                batch_dims=2,
            )

            y_true.shape == (B, C, D, H, W)

            # B and D are treated as batch dimensions.
            # H and W are used as the SSIM image dimensions.

    References:
        .. [1] Zhao, H., Gallo, O., Frosio, I., & Kautz, J. (2016). Loss functions
            for image restoration with neural networks. IEEE Transactions on
            computational imaging, 3(1), 47-57.
    """

    def __init__(
        self,
        max_val: float | None = None,
        filter_size: int = 11,
        filter_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        batch_dims: int | None = None,
        image_dims: int | None = None,
        rank: int | None = None,
    ) -> None:
        """"""
        super().__init__()

        if filter_size <= 0 or filter_size % 2 == 0:
            raise ValueError("filter_size must be a positive odd integer.")

        if filter_sigma <= 0:
            raise ValueError("filter_sigma must be greater than zero.")

        if k1 < 0 or k2 < 0:
            raise ValueError("k1 and k2 must be non-negative.")

        if batch_dims is not None and batch_dims < 0:
            raise ValueError("batch_dims must be non-negative.")

        if image_dims is not None and image_dims <= 0:
            raise ValueError("image_dims must be positive.")

        if rank is not None and rank not in (2, 3):
            raise ValueError("rank must be either 2 or 3.")

        if rank is not None and image_dims is not None and rank != image_dims:
            raise ValueError("rank and image_dims must have the same value.")

        self.max_val = max_val
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.k1 = k1
        self.k2 = k2

        self.batch_dims = batch_dims
        self.image_dims = image_dims
        self.rank = rank

        self.register_buffer(
            "_kernel",
            torch.empty(0),
            persistent=False,
        )

        self._kernel_channels: int | None = None
        self._kernel_rank: int | None = None

    @staticmethod
    def _gaussian_1d(
        size: int,
        sigma: float,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Create a normalised 1D Gaussian kernel."""
        x = torch.arange(
            size,
            dtype=dtype,
            device=device,
        )

        x = x - (size - 1) / 2

        kernel = torch.exp(-(x**2) / (2 * sigma**2))

        return kernel / kernel.sum()

    def _create_kernel(
        self,
        rank: int,
        channels: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Create a depthwise Gaussian convolution kernel."""
        kernel_1d = self._gaussian_1d(
            self.filter_size,
            self.filter_sigma,
            dtype=dtype,
            device=device,
        )

        if rank == 2:
            kernel = kernel_1d[:, None] * kernel_1d[None, :]

            kernel = kernel[None, None, :, :]
            kernel = kernel.expand(
                channels,
                1,
                -1,
                -1,
            )

        elif rank == 3:
            kernel = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]

            kernel = kernel[None, None, :, :, :]
            kernel = kernel.expand(
                channels,
                1,
                -1,
                -1,
                -1,
            )

        else:
            raise ValueError("rank must be either 2 or 3.")

        return kernel.contiguous()

    def _get_kernel(
        self,
        x: Tensor,
        rank: int,
    ) -> Tensor:
        """Get a Gaussian kernel matching the input."""
        channels = x.shape[1]

        if (
            self._kernel.numel() == 0
            or self._kernel_channels != channels
            or self._kernel_rank != rank
            or self._kernel.dtype != x.dtype
            or self._kernel.device != x.device
        ):
            self._kernel = self._create_kernel(
                rank,
                channels,
                dtype=x.dtype,
                device=x.device,
            )

            self._kernel_channels = channels
            self._kernel_rank = rank

        return self._kernel

    def _resolve_dims(
        self,
        x: Tensor,
    ) -> tuple[int, int]:
        """Resolve batch and image dimensions."""
        ndim = x.ndim

        if self.image_dims is not None:
            image_dims = self.image_dims

        elif self.rank is not None:
            image_dims = self.rank

        elif self.batch_dims is not None:
            image_dims = ndim - self.batch_dims - 1

        else:
            # Default: [B, C, H, W]
            image_dims = 2

        batch_dims = self.batch_dims if self.batch_dims is not None else ndim - image_dims - 1

        if image_dims not in (2, 3):
            raise ValueError(f"SSIM requires 2 or 3 image dimensions, got {image_dims}.")

        if batch_dims <= 0:
            raise ValueError("Invalid combination of batch_dims and image_dims.")

        expected_ndim = batch_dims + 1 + image_dims

        if ndim != expected_ndim:
            raise ValueError(
                f"Expected {expected_ndim} dimensions, but input has {ndim} dimensions."
            )

        return batch_dims, image_dims

    def _resolve_max_val(self, x: Tensor) -> float:
        """Resolve the dynamic range."""
        if self.max_val is not None:
            return float(self.max_val)

        if x.is_floating_point():
            return 1.0

        if x.dtype == torch.bool:
            return 1.0

        return float(torch.iinfo(x.dtype).max)

    def _filter(
        self,
        x: Tensor,
        kernel: Tensor,
        rank: int,
    ) -> Tensor:
        """Apply Gaussian filtering independently per channel."""
        padding = self.filter_size // 2
        channels = x.shape[1]

        if rank == 2:
            return F.conv2d(
                x,
                kernel,
                padding=padding,
                groups=channels,
            )

        return F.conv3d(
            x,
            kernel,
            padding=padding,
            groups=channels,
        )

    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
    ) -> Tensor:
        """Calculate `1 - SSIM`."""
        if y_pred.shape != y_true.shape:
            raise ValueError(
                "y_pred and y_true must have identical shapes. "
                f"Got {tuple(y_pred.shape)} and "
                f"{tuple(y_true.shape)}."
            )

        batch_dims, image_dims = self._resolve_dims(y_true)

        max_val = self._resolve_max_val(y_true)

        if max_val <= 0:
            raise ValueError("max_val must be greater than zero.")

        # Convert to a floating-point type.
        if y_pred.is_floating_point():
            dtype = y_pred.dtype
        elif y_true.is_floating_point():
            dtype = y_true.dtype
        else:
            dtype = torch.float32

        y_pred = y_pred.to(dtype)
        y_true = y_true.to(dtype)

        # Layout:
        #
        #   [batch..., channels, image...]
        #
        # Flatten all batch dimensions so that the convolution sees
        # a standard [N, C, ...] tensor.
        batch_shape = y_true.shape[:batch_dims]

        channels = y_true.shape[batch_dims]

        spatial_shape = y_true.shape[batch_dims + 1 :]

        n_batch = 1

        for size in batch_shape:
            n_batch *= size

        y_true = y_true.reshape(
            n_batch,
            channels,
            *spatial_shape,
        )

        y_pred = y_pred.reshape(
            n_batch,
            channels,
            *spatial_shape,
        )

        kernel = self._get_kernel(
            y_true,
            image_dims,
        )

        # Local means.
        mu_true = self._filter(
            y_true,
            kernel,
            image_dims,
        )

        mu_pred = self._filter(
            y_pred,
            kernel,
            image_dims,
        )

        # Local variances.
        sigma_true = (
            self._filter(
                y_true * y_true,
                kernel,
                image_dims,
            )
            - mu_true.square()
        )

        sigma_pred = (
            self._filter(
                y_pred * y_pred,
                kernel,
                image_dims,
            )
            - mu_pred.square()
        )

        # Local covariance.
        sigma_cross = (
            self._filter(
                y_true * y_pred,
                kernel,
                image_dims,
            )
            - mu_true * mu_pred
        )

        # Avoid tiny negative variances caused by floating-point
        # round-off.
        sigma_true = sigma_true.clamp_min(0.0)
        sigma_pred = sigma_pred.clamp_min(0.0)

        c1 = (self.k1 * max_val) ** 2
        c2 = (self.k2 * max_val) ** 2

        luminance = (2 * mu_true * mu_pred + c1) / (mu_true.square() + mu_pred.square() + c1)

        contrast_structure = (2 * sigma_cross + c2) / (sigma_true + sigma_pred + c2)

        ssim = luminance * contrast_structure

        return 1.0 - ssim.mean()
