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

import pytest
import torch
import torch.nn as nn

from im2sim.layers.temporal_factorizer import (
    TemporalFactorizer,
    TimeDistributed,
    SpaceDistributed,
)


class TestTimeDistributed:
    """Test TimeDistributed wrapper."""

    def test_forward_5d_input(self):
        """Test TimeDistributed with 5D input (N, C, T, H, W)."""
        module = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        wrapper = TimeDistributed(module)

        # Input: (N=2, C=3, T=4, H=8, W=8)
        x = torch.randn(2, 3, 4, 8, 8)
        y = wrapper(x)

        # Output should be (2, 16, 4, 8, 8)
        assert y.shape == (2, 16, 4, 8, 8)

    def test_forward_6d_input(self):
        """Test TimeDistributed with 6D input (N, C, T, D, H, W)."""
        module = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        wrapper = TimeDistributed(module)

        # Input: (N=2, C=3, T=4, D=4, H=8, W=8)
        x = torch.randn(2, 3, 4, 4, 8, 8)
        y = wrapper(x)

        # Output should be (2, 16, 4, 4, 8, 8)
        assert y.shape == (2, 16, 4, 4, 8, 8)

    def test_invalid_input_shape(self):
        """Test TimeDistributed raises error for invalid input shape."""
        module = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        wrapper = TimeDistributed(module)

        x = torch.randn(2, 3, 8, 8)  # 4D input (invalid)
        with pytest.raises(ValueError):
            wrapper(x)


class TestSpaceDistributed:
    """Test SpaceDistributed wrapper."""

    def test_forward_5d_input(self):
        """Test SpaceDistributed with 5D input (N, C, T, H, W)."""
        module = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        wrapper = SpaceDistributed(module)

        # Input: (N=2, C=16, T=4, H=8, W=8)
        x = torch.randn(2, 16, 4, 8, 8)
        y = wrapper(x)

        # Output should have same or similar shape
        assert y.ndim == 5
        assert y.shape[0] == 2
        assert y.shape[1] == 16

    def test_forward_6d_input(self):
        """Test SpaceDistributed with 6D input (N, C, T, D, H, W)."""
        module = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        wrapper = SpaceDistributed(module)

        # Input: (N=2, C=16, T=4, D=4, H=8, W=8)
        x = torch.randn(2, 16, 4, 4, 8, 8)
        y = wrapper(x)

        # Output should have same or similar shape
        assert y.ndim == 6
        assert y.shape[0] == 2
        assert y.shape[1] == 16


class TestTemporalFactorizerParamSplitting:
    """Test parameter splitting logic in TemporalFactorizer."""

    def test_split_param_scalar(self):
        """Test splitting scalar parameter."""
        factorizer = TemporalFactorizer(nn.Conv2d(3, 16, 3), rank=2)
        # With time_default=1 (kernel_size)
        time_val, space_val = factorizer._split_param(3, "kernel_size", time_default=1)
        assert time_val == 1
        assert space_val == (3, 3)

    def test_split_param_tuple_rank_plus_one(self):
        """Test splitting tuple of length rank+1."""
        factorizer = TemporalFactorizer(nn.Conv2d(3, 16, 3), rank=2)
        time_val, space_val = factorizer._split_param((1, 3, 3), "kernel_size", time_default=1)
        assert time_val == 1
        assert space_val == (3, 3)

    def test_split_param_tuple_rank_3d(self):
        """Test splitting tuple for rank=3."""
        factorizer = TemporalFactorizer(nn.Conv3d(3, 16, 3), rank=3)
        time_val, space_val = factorizer._split_param((1, 3, 3, 3), "kernel_size", time_default=1)
        assert time_val == 1
        assert space_val == (3, 3, 3)

    def test_split_param_valid_rank_len(self):
        """Test splitting tuple with length=rank (just spatial dims)."""
        factorizer = TemporalFactorizer(nn.Conv2d(3, 16, 3), rank=2)
        # Length 2 is valid for rank=2 (just spatial)
        time_val, space_val = factorizer._split_param((3, 3), "kernel_size", time_default=1)
        assert time_val == 1
        assert space_val == (3, 3)


class TestTemporalFactorizerConv2d:
    """Test TemporalFactorizer with Conv2d."""

    def test_conv2d_factorization(self):
        """Test Conv2d factorization output shape."""
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        factorizer = TemporalFactorizer(conv, rank=2)

        # Input: (N=2, C=3, T=4, H=8, W=8)
        x = torch.randn(2, 3, 4, 8, 8)
        y = factorizer(x)

        # Output should be (2, 16, 4, 8, 8)
        assert y.shape == (2, 16, 4, 8, 8)

    def test_conv2d_with_stride(self):
        """Test Conv2d factorization with stride."""
        conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        factorizer = TemporalFactorizer(conv, rank=2)

        x = torch.randn(2, 3, 4, 8, 8)
        y = factorizer(x)

        # Spatial dimensions should be halved due to stride=2
        # Time dimension unchanged with default time_kernel=1
        assert y.shape[0] == 2
        assert y.shape[1] == 16
        assert y.shape[2] == 4  # Time unchanged
        assert y.shape[3] == 4  # H halved
        assert y.shape[4] == 4  # W halved

    def test_conv2d_factorization_gradient_flow(self):
        """Test that gradients flow through factorized conv."""
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        factorizer = TemporalFactorizer(conv, rank=2)

        x = torch.randn(2, 3, 4, 8, 8, requires_grad=True)
        y = factorizer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestTemporalFactorizerConv3d:
    """Test TemporalFactorizer with Conv3d."""

    def test_conv3d_factorization(self):
        """Test Conv3d factorization output shape."""
        conv = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        factorizer = TemporalFactorizer(conv, rank=3)

        # Input: (N=2, C=3, T=4, D=4, H=8, W=8)
        x = torch.randn(2, 3, 4, 4, 8, 8)
        y = factorizer(x)

        # Output should be (2, 16, 4, 4, 8, 8)
        assert y.shape == (2, 16, 4, 4, 8, 8)

    def test_conv3d_with_tuple_kernel(self):
        """Test Conv3d with tuple kernel size."""
        conv = nn.Conv3d(3, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        factorizer = TemporalFactorizer(conv, rank=3)

        x = torch.randn(2, 3, 4, 4, 8, 8)
        y = factorizer(x)

        assert y.shape == (2, 16, 4, 4, 8, 8)


class TestTemporalFactorizerPooling:
    """Test TemporalFactorizer with pooling layers."""

    def test_maxpool2d_factorization(self):
        """Test MaxPool2d factorization."""
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        factorizer = TemporalFactorizer(pool, rank=2)

        x = torch.randn(2, 16, 4, 8, 8)
        y = factorizer(x)

        # Spatial dimensions should be halved, time unchanged (time_kernel=1 by default)
        assert y.shape == (2, 16, 4, 4, 4)

    def test_avgpool3d_factorization(self):
        """Test AvgPool3d factorization."""
        pool = nn.AvgPool3d(kernel_size=2, stride=2)
        factorizer = TemporalFactorizer(pool, rank=3)

        x = torch.randn(2, 16, 4, 4, 8, 8)
        y = factorizer(x)

        # Spatial dimensions halved, time unchanged (time_kernel=1 by default)
        assert y.shape == (2, 16, 4, 2, 4, 4)

    def test_maxpool_with_asymmetric_kernel(self):
        """Test MaxPool with asymmetric kernel (time=1, spatial=2)."""
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        factorizer = TemporalFactorizer(pool, rank=2)

        x = torch.randn(2, 16, 4, 8, 8)
        y = factorizer(x)

        # Time should be halved too if not handled specially
        # Actual behavior depends on SpaceDistributed handling
        assert y.shape[0] == 2
        assert y.shape[1] == 16


class TestTemporalFactorizerBatchNorm:
    """Test TemporalFactorizer with BatchNorm layers."""

    def test_batchnorm2d_factorization(self):
        """Test BatchNorm2d factorization."""
        norm = nn.BatchNorm2d(16)
        factorizer = TemporalFactorizer(norm, rank=2)

        x = torch.randn(2, 16, 4, 8, 8)
        y = factorizer(x)

        # Shape should be preserved
        assert y.shape == x.shape

    def test_batchnorm3d_factorization(self):
        """Test BatchNorm3d factorization."""
        norm = nn.BatchNorm3d(16)
        factorizer = TemporalFactorizer(norm, rank=3)

        x = torch.randn(2, 16, 4, 4, 8, 8)
        y = factorizer(x)

        # Shape should be preserved
        assert y.shape == x.shape


class TestTemporalFactorizerConvTranspose:
    """Test TemporalFactorizer with transposed convolution."""

    def test_convtranspose2d_factorization(self):
        """Test ConvTranspose2d factorization."""
        conv_t = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        factorizer = TemporalFactorizer(conv_t, rank=2)

        x = torch.randn(2, 16, 4, 4, 4)
        y = factorizer(x)

        # Spatial dimensions should be doubled, time unchanged
        assert y.shape[0] == 2
        assert y.shape[1] == 8
        assert y.shape[2] == 4  # Time unchanged
        assert y.shape[3] == 8  # H doubled
        assert y.shape[4] == 8  # W doubled


class TestTemporalFactorizerUpsample:
    """Test TemporalFactorizer with upsampling."""

    def test_upsample_scale_factor(self):
        """Test Upsample with scale_factor."""
        upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        factorizer = TemporalFactorizer(upsample, rank=2)

        x = torch.randn(2, 16, 4, 4, 4)
        y = factorizer(x)

        # Spatial dimensions doubled, time unchanged (scale_factor=1 for time by default)
        assert y.shape == (2, 16, 4, 8, 8)

    def test_upsample_size(self):
        """Test Upsample with size."""
        upsample = nn.Upsample(size=(8, 8), mode="bilinear", align_corners=False)
        factorizer = TemporalFactorizer(upsample, rank=2)

        x = torch.randn(2, 16, 4, 4, 4)
        y = factorizer(x)

        # Spatial dimensions should match size, time unchanged
        assert y.shape == (2, 16, 4, 8, 8)


class TestTemporalFactorizerDropout:
    """Test TemporalFactorizer with dropout layers."""

    def test_dropout2d_factorization(self):
        """Test Dropout2d factorization."""
        dropout = nn.Dropout2d(p=0.5)
        factorizer = TemporalFactorizer(dropout, rank=2)

        x = torch.randn(2, 16, 4, 8, 8)
        y = factorizer(x)

        # Shape should be preserved
        assert y.shape == x.shape


class TestTemporalFactorizerRankValidation:
    """Test TemporalFactorizer rank validation."""

    def test_invalid_rank(self):
        """Test that invalid rank raises error."""
        with pytest.raises(ValueError):
            TemporalFactorizer(nn.Conv2d(3, 16, 3), rank=4)

    def test_valid_ranks(self):
        """Test that valid ranks don't raise error."""
        # rank=2
        factorizer_2d = TemporalFactorizer(nn.Conv2d(3, 16, 3), rank=2)
        assert factorizer_2d.rank == 2

        # rank=3
        factorizer_3d = TemporalFactorizer(nn.Conv3d(3, 16, 3), rank=3)
        assert factorizer_3d.rank == 3


class TestTemporalFactorizerIntegration:
    """Integration tests for TemporalFactorizer."""

    def test_sequential_factorized_layers(self):
        """Test sequential application of factorized layers."""
        layers = nn.Sequential(
            TemporalFactorizer(nn.Conv2d(3, 16, 3, padding=1), rank=2),
            nn.ReLU(),
            TemporalFactorizer(nn.MaxPool2d(2, 2), rank=2),
            TemporalFactorizer(nn.Conv2d(16, 32, 3, padding=1), rank=2),
        )

        x = torch.randn(2, 3, 4, 8, 8)
        y = layers(x)

        # Convs preserve spatial dims (padding=1), MaxPool halves them, time unchanged
        assert y.shape == (2, 32, 4, 4, 4)

    def test_backward_pass_sequential(self):
        """Test backward pass through sequential layers."""
        layers = nn.Sequential(
            TemporalFactorizer(nn.Conv2d(3, 16, 3, padding=1), rank=2),
            TemporalFactorizer(nn.BatchNorm2d(16), rank=2),
            TemporalFactorizer(nn.Conv2d(16, 32, 3, padding=1), rank=2),
        )

        x = torch.randn(2, 3, 4, 8, 8, requires_grad=True)
        y = layers(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
