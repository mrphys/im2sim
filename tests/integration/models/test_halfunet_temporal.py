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

from im2sim.configs.core import LayerConfig
from im2sim.configs.halfunet import HalfUNetConfig
from im2sim.models.halfunet import HalfUNet


class TestHalfUNetTemporal2D:
    """Test HalfUNet with temporal support for 2D data (2D+time)."""

    @pytest.mark.parametrize("fusion_type", ["add", "concat"])
    def test_halfunet_temporal_2d_forward(self, fusion_type):
        """Test forward pass through temporal HalfUNet 2D."""
        cfg = HalfUNetConfig(hidden_channels=8, fusion_type=fusion_type, enable_temporal=True)
        model = HalfUNet(in_channels=3, out_channels=2, rank=2, cfg=cfg)

        # Input: (N=2, C=3, T=4, H=32, W=32)
        x = torch.randn(2, 3, 4, 32, 32)
        y = model(x)

        assert y.shape == (2, 2, 4, 32, 32)

    def test_halfunet_temporal_2d_gradient_flow(self):
        """Test gradient flow through temporal HalfUNet 2D."""
        cfg = HalfUNetConfig(hidden_channels=8, enable_temporal=True)
        model = HalfUNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)

        x = torch.randn(2, 1, 4, 32, 32, requires_grad=True)
        model(x).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_halfunet_temporal_2d_odd_time_with_temporal_pooling(self):
        """Test skip connections are aligned when pooling over an odd number of time steps."""
        cfg = HalfUNetConfig(
            hidden_channels=8,
            n_levels=2,
            pool_cfg=LayerConfig(name="MaxPool", kwargs={"kernel_size": (2, 2, 2)}),
            upsample_cfg=LayerConfig(
                name="Upsample", kwargs={"scale_factor": (2, 2, 2), "mode": "trilinear"}
            ),
            enable_temporal=True,
        )
        model = HalfUNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)

        # T=5 is pooled to 2 and upsampled to 4, so the T=5 encoder output must be resized
        x = torch.randn(1, 1, 5, 16, 16)
        y = model(x)

        assert y.shape == (1, 1, 4, 16, 16)


class TestHalfUNetTemporal3D:
    """Test HalfUNet with temporal support for 3D data (3D+time)."""

    @pytest.mark.parametrize("fusion_type", ["add", "concat"])
    def test_halfunet_temporal_3d_forward(self, fusion_type):
        """Test forward pass through temporal HalfUNet 3D."""
        cfg = HalfUNetConfig(hidden_channels=8, fusion_type=fusion_type, enable_temporal=True)
        model = HalfUNet(in_channels=1, out_channels=1, rank=3, cfg=cfg)

        # Input: (N=1, C=1, T=2, D=16, H=16, W=16)
        x = torch.randn(1, 1, 2, 16, 16, 16)
        y = model(x)

        assert y.shape == (1, 1, 2, 16, 16, 16)

    def test_halfunet_temporal_3d_gradient_flow(self):
        """Test gradient flow through temporal HalfUNet 3D."""
        cfg = HalfUNetConfig(hidden_channels=8, enable_temporal=True)
        model = HalfUNet(in_channels=1, out_channels=1, rank=3, cfg=cfg)

        x = torch.randn(1, 1, 2, 16, 16, 16, requires_grad=True)
        model(x).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    @pytest.mark.parametrize("spatial", [(15, 17, 13), (9, 16, 11)])
    def test_halfunet_temporal_3d_odd_spatial_sizes(self, spatial):
        """Test encoder outputs are aligned when pooling doesn't divide the volume evenly."""
        cfg = HalfUNetConfig(hidden_channels=8, enable_temporal=True)
        model = HalfUNet(in_channels=1, out_channels=2, rank=3, cfg=cfg)

        x = torch.randn(1, 1, 3, *spatial)
        y = model(x)

        # Spatial size should match the equivalent non-temporal 3D HalfUNet, time is unchanged
        model_3d = HalfUNet(
            in_channels=1, out_channels=2, rank=3, cfg=cfg.mod(enable_temporal=False)
        )
        expected_spatial = model_3d(torch.randn(1, 1, *spatial)).shape[2:]
        assert y.shape == (1, 2, 3, *expected_spatial)


class TestHalfUNetTemporalConfig:
    """Test temporal flag handling in HalfUNet configs."""

    def test_non_temporal_still_works(self):
        """Test that non-temporal HalfUNet still works as expected."""
        cfg = HalfUNetConfig(hidden_channels=8, enable_temporal=False)
        model = HalfUNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)

        x = torch.randn(1, 1, 32, 32)
        y = model(x)

        assert y.shape == (1, 1, 32, 32)

    def test_enable_temporal_save_load(self, tmp_path):
        """Test enable_temporal round-trips through config save/load."""
        cfg = HalfUNetConfig(hidden_channels=8, enable_temporal=True)
        path = tmp_path / "cfg.yaml"
        cfg.save(path)

        assert HalfUNetConfig.load(path).enable_temporal is True
