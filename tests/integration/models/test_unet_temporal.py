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

from im2sim.configs.image_blocks import ImageConvBlockConfig
from im2sim.configs.unet import UNetConfig
from im2sim.configs.core import LayerConfig
from im2sim.models.unet import UNet


class TestUNetTemporal2D:
    """Test UNet with temporal support for 2D data (2D+time)."""

    def test_unet_temporal_2d_forward(self):
        """Test forward pass through temporal UNet 2D."""
        cfg = UNetConfig(
            filters=[32, 64, 128],
            enable_temporal=True,
        )
        model = UNet(
            in_channels=3,
            out_channels=1,
            rank=2,
            cfg=cfg,
        )

        # Input: (N=2, C=3, T=4, H=64, W=64)
        x = torch.randn(2, 3, 4, 64, 64)
        y = model(x)

        # Output should have same spatial size, channels=1, time=4
        assert y.shape == (2, 1, 4, 64, 64)

    def test_unet_temporal_2d_non_temporal_comparison(self):
        """Test that temporal flag changes behavior."""
        cfg_non_temporal = UNetConfig(filters=[32, 64, 128], enable_temporal=False)
        cfg_temporal = UNetConfig(filters=[32, 64, 128], enable_temporal=True)

        model_non_temporal = UNet(in_channels=3, out_channels=1, rank=2, cfg=cfg_non_temporal)
        model_temporal = UNet(in_channels=3, out_channels=1, rank=2, cfg=cfg_temporal)

        # Non-temporal: input 2D (N, C, H, W)
        x_2d = torch.randn(2, 3, 64, 64)
        y_2d = model_non_temporal(x_2d)
        assert y_2d.shape == (2, 1, 64, 64)

        # Temporal: input 3D (N, C, T, H, W)
        x_3d = torch.randn(2, 3, 4, 64, 64)
        y_3d = model_temporal(x_3d)
        assert y_3d.shape == (2, 1, 4, 64, 64)

    def test_unet_temporal_2d_different_time_steps(self):
        """Test temporal UNet with different time dimensions."""
        cfg = UNetConfig(filters=[32, 64], enable_temporal=True)
        model = UNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)

        # Test with T=2
        x_t2 = torch.randn(2, 1, 2, 32, 32)
        y_t2 = model(x_t2)
        assert y_t2.shape == (2, 1, 2, 32, 32)

        # Test with T=8
        x_t8 = torch.randn(2, 1, 8, 32, 32)
        y_t8 = model(x_t8)
        assert y_t8.shape == (2, 1, 8, 32, 32)

    def test_unet_temporal_2d_small_spatial(self):
        """Test temporal UNet with small spatial dimensions."""
        cfg = UNetConfig(filters=[16, 32], enable_temporal=True)
        model = UNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)

        # Small input (N, C, T, H, W)
        x = torch.randn(1, 1, 4, 16, 16)
        y = model(x)

        assert y.shape == (1, 1, 4, 16, 16)

    def test_unet_temporal_2d_gradient_flow(self):
        """Test gradient flow through temporal UNet 2D."""
        cfg = UNetConfig(filters=[32, 64], enable_temporal=True)
        model = UNet(in_channels=3, out_channels=1, rank=2, cfg=cfg)

        x = torch.randn(2, 3, 4, 32, 32, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_unet_temporal_2d_different_configs(self):
        """Test temporal UNet with different block configurations."""
        cfg = UNetConfig(
            filters=[32, 64, 128],
            encoder_block_cfg=ImageConvBlockConfig(depth=2, activation="ReLU"),
            decoder_block_cfg=ImageConvBlockConfig(depth=2, activation="ReLU"),
            enable_temporal=True,
        )
        model = UNet(in_channels=3, out_channels=1, rank=2, cfg=cfg)

        x = torch.randn(2, 3, 4, 64, 64)
        y = model(x)

        assert y.shape == (2, 1, 4, 64, 64)


class TestUNetTemporal3D:
    """Test UNet with temporal support for 3D data (3D+time)."""

    def test_unet_temporal_3d_forward(self):
        """Test forward pass through temporal UNet 3D."""
        cfg = UNetConfig(
            filters=[32, 64],
            enable_temporal=True,
        )
        model = UNet(
            in_channels=1,
            out_channels=1,
            rank=3,
            cfg=cfg,
        )

        # Input: (N=1, C=1, T=2, D=16, H=16, W=16)
        x = torch.randn(1, 1, 2, 16, 16, 16)
        y = model(x)

        # Output should have same spatial and temporal size, channels=1
        assert y.shape == (1, 1, 2, 16, 16, 16)

    def test_unet_temporal_3d_larger_volume(self):
        """Test temporal UNet 3D with larger volume."""
        cfg = UNetConfig(filters=[16, 32], enable_temporal=True)
        model = UNet(in_channels=1, out_channels=1, rank=3, cfg=cfg)

        # Input: (N=1, C=1, T=4, D=32, H=32, W=32)
        x = torch.randn(1, 1, 4, 32, 32, 32)
        y = model(x)

        assert y.shape == (1, 1, 4, 32, 32, 32)

    def test_unet_temporal_3d_gradient_flow(self):
        """Test gradient flow through temporal UNet 3D."""
        cfg = UNetConfig(filters=[16, 32], enable_temporal=True)
        model = UNet(in_channels=1, out_channels=1, rank=3, cfg=cfg)

        x = torch.randn(1, 1, 2, 16, 16, 16, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None


class TestUNetTemporalAdvanced:
    """Advanced tests for temporal UNet features."""

    def test_temporal_with_asymmetric_pool_sizes(self):
        """Test temporal UNet with asymmetric pool sizes (time=1, spatial=2)."""
        pool_cfg = [
            LayerConfig(name="MaxPool", kwargs={"kernel_size": (1, 2, 2)}),
            LayerConfig(name="MaxPool", kwargs={"kernel_size": (1, 2, 2)}),
        ]
        cfg = UNetConfig(
            filters=[32, 64, 128],
            pool_cfg=pool_cfg,
            enable_temporal=True,
        )
        model = UNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)

        x = torch.randn(1, 1, 4, 64, 64)
        y = model(x)

        # Time dimension should be preserved with (1, 2, 2) pooling (1x pooling in time)
        assert y.shape[2] == 4  # Time unchanged

    def test_temporal_with_batch_norm(self):
        """Test temporal UNet with batch normalization."""
        cfg = UNetConfig(
            filters=[32, 64],
            encoder_block_cfg=ImageConvBlockConfig(
                depth=1,
                activation="ReLU",
                norm_cfg=LayerConfig(name="BatchNorm", kwargs={}),
            ),
            enable_temporal=True,
        )
        model = UNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)

        x = torch.randn(2, 1, 4, 32, 32)
        y = model(x)

        assert y.shape == (2, 1, 4, 32, 32)

    def test_temporal_with_deep_supervision(self):
        """Test temporal UNet with deep supervision."""
        cfg = UNetConfig(
            filters=[32, 64, 128],
            enable_temporal=True,
        )
        model = UNet(
            in_channels=1,
            out_channels=1,
            rank=2,
            cfg=cfg,
            supervision_levels=[0, 1],
        )

        x = torch.randn(1, 1, 4, 64, 64)
        y = model(x)

        # With deep supervision, should return list of outputs
        assert isinstance(y, list)
        assert len(y) == 2
        # All outputs should have temporal dimension
        for out in y:
            assert out.shape[2] == 4  # Time preserved

    def test_temporal_model_eval_mode(self):
        """Test temporal UNet in eval mode (for batch norm)."""
        cfg = UNetConfig(
            filters=[32, 64],
            encoder_block_cfg=ImageConvBlockConfig(
                depth=1,
                norm_cfg=LayerConfig(name="BatchNorm", kwargs={}),
            ),
            enable_temporal=True,
        )
        model = UNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)
        model.eval()

        x = torch.randn(1, 1, 4, 32, 32)
        with torch.no_grad():
            y = model(x)

        assert y.shape == (1, 1, 4, 32, 32)

    def test_temporal_different_num_channels(self):
        """Test temporal UNet with various channel configurations."""
        configs = [
            (1, 1, 2),
            (3, 2, 4),
            (4, 3, 4),
        ]

        for in_ch, out_ch, t in configs:
            cfg = UNetConfig(filters=[32, 64], enable_temporal=True)
            model = UNet(in_channels=in_ch, out_channels=out_ch, rank=2, cfg=cfg)

            x = torch.randn(1, in_ch, t, 32, 32)
            y = model(x)

            assert y.shape == (1, out_ch, t, 32, 32)

    def test_temporal_state_dict_save_load(self):
        """Test saving and loading temporal UNet state."""
        cfg = UNetConfig(filters=[32, 64], enable_temporal=True)
        model = UNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)

        # Get model state
        state = model.state_dict()

        # Create new model and load state
        model2 = UNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)
        model2.load_state_dict(state)

        # Compare outputs
        x = torch.randn(1, 1, 4, 32, 32)
        with torch.no_grad():
            y1 = model(x)
            y2 = model2(x)

        assert torch.allclose(y1, y2)

    def test_temporal_inference_speed(self):
        """Test that temporal UNet can handle inference efficiently."""
        cfg = UNetConfig(filters=[16, 32], enable_temporal=True)
        model = UNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)
        model.eval()

        x = torch.randn(1, 1, 8, 64, 64)

        with torch.no_grad():
            y = model(x)

        assert y.shape == (1, 1, 8, 64, 64)


class TestUNetTemporalEdgeCases:
    """Test edge cases and error conditions."""

    def test_temporal_minimum_spatial_size(self):
        """Test temporal UNet with minimum spatial size."""
        cfg = UNetConfig(filters=[16, 32], enable_temporal=True)
        model = UNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)

        x = torch.randn(1, 1, 2, 8, 8)
        y = model(x)

        assert y.shape == (1, 1, 2, 8, 8)

    def test_temporal_single_timestep(self):
        """Test temporal UNet with single timestep."""
        cfg = UNetConfig(filters=[32, 64], enable_temporal=True)
        model = UNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)

        x = torch.randn(1, 1, 1, 32, 32)
        y = model(x)

        assert y.shape == (1, 1, 1, 32, 32)

    def test_temporal_many_timesteps(self):
        """Test temporal UNet with many timesteps."""
        cfg = UNetConfig(filters=[16, 32], enable_temporal=True)
        model = UNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)

        x = torch.randn(1, 1, 32, 32, 32)
        y = model(x)

        assert y.shape == (1, 1, 32, 32, 32)

    def test_non_temporal_still_works(self):
        """Test that non-temporal UNet still works as expected."""
        cfg = UNetConfig(filters=[32, 64], enable_temporal=False)
        model = UNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)

        # Non-temporal input
        x = torch.randn(1, 1, 32, 32)
        y = model(x)

        assert y.shape == (1, 1, 32, 32)
