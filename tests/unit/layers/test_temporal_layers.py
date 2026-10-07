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

from im2sim.configs.core import LayerConfig
from im2sim.configs.image_blocks import ImageConvBlockConfig
from im2sim.configs.unet import UNetConfig
from im2sim.layers.temporal_layers import (
    SUPPORTED_TEMPORAL_LAYERS,
    AvgPoolTime,
    BatchNormTime,
    ConvTime,
    ConvTransposeTime,
    DropoutTime,
    InstanceNormTime,
    MaxPoolTime,
    SpaceDistributed,
    TimeDistributed,
    UpsampleTime,
    _split_time_param,
)
from im2sim.models.unet import UNet
from im2sim.utils.layer_util import (
    UnsupportedTemporalLayerError,
    call_with_supported_kwargs,
    get_image_layer,
)


def _input(rank, n=2, c=3, t=4, s=8):
    return torch.randn(n, c, t, *([s] * rank))


def _assert_all_params_used(module, x):
    module(x).sum().backward()
    for name, param in module.named_parameters():
        assert param.grad is not None, f"{name} is unused in the forward pass"


class TestDistributed:
    def test_time_distributed_shape(self):
        y = TimeDistributed(nn.Conv2d(3, 16, 3, padding=1))(_input(2))
        assert y.shape == (2, 16, 4, 8, 8)

    def test_space_distributed_shape(self):
        y = SpaceDistributed(nn.Conv1d(3, 16, 3, padding=1))(_input(3))
        assert y.shape == (2, 16, 4, 8, 8, 8)

    def test_invalid_input_shape(self):
        with pytest.raises(ValueError):
            TimeDistributed(nn.Identity())(torch.randn(2, 3, 8, 8))


class TestSplitTimeParam:
    def test_scalar_spatial_only(self):
        assert _split_time_param(3, 2, "kernel_size", 1) == (1, (3, 3))

    def test_scalar_applies_to_time(self):
        assert _split_time_param(3, 3, "kernel_size", 1, scalar_applies_to_time=True) == (
            3,
            (3, 3, 3),
        )

    def test_spatial_tuple(self):
        assert _split_time_param((3, 5), 2, "kernel_size", 1) == (1, (3, 5))

    def test_time_tuple(self):
        assert _split_time_param((1, 3, 5, 7), 3, "kernel_size", 1) == (1, (3, 5, 7))

    def test_string(self):
        assert _split_time_param("same", 3, "padding", 0) == ("same", "same")

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="kernel_size"):
            _split_time_param((3, 3, 3, 3, 3), 3, "kernel_size", 1)


@pytest.mark.parametrize("rank", [2, 3])
class TestConvTime:
    def test_same_padding_preserves_shape(self, rank):
        conv = ConvTime(3, 16, kernel_size=3, rank=rank, padding="same")
        assert conv(_input(rank)).shape == (2, 16, 4, *([8] * rank))

    def test_scalar_kernel_is_temporal(self, rank):
        conv = ConvTime(3, 16, kernel_size=3, rank=rank, padding=1)
        assert conv.temporal_op.module.kernel_size == (3,)
        assert conv(_input(rank)).shape == (2, 16, 4, *([8] * rank))

    def test_temporal_receptive_field(self, rank):
        conv = ConvTime(1, 4, kernel_size=3, rank=rank, padding="same").eval()
        x = _input(rank, c=1, t=6)
        x_perturbed = x.clone()
        x_perturbed[:, :, 0] += 1.0
        with torch.no_grad():
            diff = (conv(x_perturbed) - conv(x)).abs().amax(dim=(0, 1, *range(3, rank + 3)))
        # t=0 reaches t=1 through the 3-tap time kernel, but not t=5 (only float noise)
        assert diff[1] > 1e-3
        assert diff[5] < 1e-5

    def test_kernel_one_skips_temporal_op(self, rank):
        conv = ConvTime(3, 16, kernel_size=1, rank=rank, padding=0)
        assert conv.temporal_op is None
        assert conv(_input(rank)).shape == (2, 16, 4, *([8] * rank))

    def test_explicit_spatial_only_kernel(self, rank):
        conv = ConvTime(3, 16, kernel_size=(1, *([3] * rank)), rank=rank, padding="same")
        assert conv.temporal_op is None

    def test_scalar_stride_is_spatial_only(self, rank):
        conv = ConvTime(3, 16, kernel_size=3, rank=rank, stride=2, padding=1)
        assert conv(_input(rank)).shape == (2, 16, 4, *([4] * rank))

    def test_all_params_used(self, rank):
        _assert_all_params_used(ConvTime(3, 16, kernel_size=3, rank=rank, padding=1), _input(rank))


@pytest.mark.parametrize("rank", [2, 3])
class TestConvTransposeTime:
    def test_scalar_upsamples_space_only(self, rank):
        conv_t = ConvTransposeTime(8, 4, kernel_size=2, rank=rank, stride=2)
        assert conv_t.temporal_op is None
        assert conv_t(_input(rank, c=8)).shape == (2, 4, 4, *([16] * rank))

    def test_time_tuple_upsamples_time(self, rank):
        factor = (2,) * (rank + 1)
        conv_t = ConvTransposeTime(8, 4, kernel_size=factor, rank=rank, stride=factor)
        assert conv_t(_input(rank, c=8)).shape == (2, 4, 8, *([16] * rank))
        _assert_all_params_used(conv_t, _input(rank, c=8))


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("pool_cls", [MaxPoolTime, AvgPoolTime])
class TestPoolTime:
    def test_scalar_pools_space_only(self, rank, pool_cls):
        pool = pool_cls(kernel_size=2, rank=rank)
        assert pool.temporal_op is None
        assert pool(_input(rank)).shape == (2, 3, 4, *([4] * rank))

    def test_time_tuple_pools_time(self, rank, pool_cls):
        pool = pool_cls(kernel_size=(2,) * (rank + 1), rank=rank)
        assert pool(_input(rank)).shape == (2, 3, 2, *([4] * rank))


@pytest.mark.parametrize("rank", [2, 3])
class TestUpsampleTime:
    def test_scalar_scale_upsamples_space_only(self, rank):
        up = UpsampleTime(rank=rank, scale_factor=2, mode="trilinear")
        assert up.temporal_op is None
        assert up(_input(rank)).shape == (2, 3, 4, *([16] * rank))

    def test_time_tuple_scale(self, rank):
        up = UpsampleTime(rank=rank, scale_factor=(2,) * (rank + 1), mode="trilinear")
        assert up.temporal_op.module.mode == "linear"
        assert up(_input(rank)).shape == (2, 3, 8, *([16] * rank))

    def test_nearest_mode_is_nearest_in_time(self, rank):
        up = UpsampleTime(rank=rank, scale_factor=(2,) * (rank + 1), mode="nearest")
        assert up.temporal_op.module.mode == "nearest"

    def test_size(self, rank):
        up = UpsampleTime(rank=rank, size=(6, *([10] * rank)), mode="trilinear")
        assert up(_input(rank)).shape == (2, 3, 6, *([10] * rank))

    def test_spatial_size_keeps_time(self, rank):
        up = UpsampleTime(rank=rank, size=(10,) * rank)
        assert up.temporal_op is None
        assert up(_input(rank)).shape == (2, 3, 4, *([10] * rank))


@pytest.mark.parametrize("rank", [2, 3])
class TestNormTime:
    def test_instance_norm_is_joint_over_time_and_space(self, rank):
        x = _input(rank) * 3 + 1
        y = InstanceNormTime(3, rank=rank)(x)
        dims = tuple(range(2, rank + 3))
        assert torch.allclose(y.mean(dim=dims), torch.zeros(2, 3), atol=1e-5)
        assert torch.allclose(y.var(dim=dims, unbiased=False), torch.ones(2, 3), atol=1e-3)

    def test_instance_norm_keeps_temporal_means(self, rank):
        # A signal that only changes over time must survive normalization
        x = torch.arange(4.0).view(1, 1, 4, *([1] * rank)).expand(1, 1, 4, *([8] * rank))
        y = InstanceNormTime(1, rank=rank)(x.contiguous())
        voxel_means_over_space = y.mean(dim=tuple(range(3, rank + 3)))[0, 0]
        assert torch.all(voxel_means_over_space.diff() > 0)

    def test_batch_norm(self, rank):
        norm = BatchNormTime(3, rank=rank)
        y = norm(_input(rank) * 3 + 1)
        dims = (0, *range(2, rank + 3))
        assert torch.allclose(y.mean(dim=dims), torch.zeros(3), atol=1e-5)
        norm.eval()
        assert norm(_input(rank)).shape == (2, 3, 4, *([8] * rank))

    def test_single_timestep(self, rank):
        y = InstanceNormTime(3, rank=rank, affine=True)(_input(rank, t=1))
        assert y.shape == (2, 3, 1, *([8] * rank))

    def test_wrong_input_rank_raises(self, rank):
        with pytest.raises(ValueError, match="expects"):
            BatchNormTime(3, rank=rank)(_input(rank)[:, :, 0])


@pytest.mark.parametrize("rank", [2, 3])
class TestDropoutTime:
    def test_eval_is_identity(self, rank):
        dropout = DropoutTime(rank=rank, p=0.5).eval()
        x = _input(rank)
        assert torch.equal(dropout(x), x)

    def test_train_drops_channels_per_timestep(self, rank):
        dropout = DropoutTime(rank=rank, p=0.5).train()
        y = dropout(torch.ones(4, 8, 4, *([4] * rank)))
        flat = y.flatten(start_dim=3)
        # Each (sample, channel, time) slice is either fully dropped or fully kept
        assert torch.all((flat == 0).all(-1) | (flat == 2).all(-1))


class TestTemporalLayerRegistry:
    @pytest.mark.parametrize("rank", [2, 3])
    @pytest.mark.parametrize("name", SUPPORTED_TEMPORAL_LAYERS)
    def test_supported_layers_are_registered(self, name, rank):
        layer_cls = get_image_layer(name, rank, temporal=True)
        assert layer_cls.__name__.endswith(f"Time{rank}d")

    def test_registered_layer_signature_has_no_rank(self):
        conv = call_with_supported_kwargs(
            get_image_layer("Conv", 3, temporal=True),
            {"in_channels": 2, "out_channels": 4, "kernel_size": 3, "padding": "same", "foo": 1},
        )
        assert isinstance(conv, ConvTime)
        assert conv.rank == 3

    @pytest.mark.parametrize(
        "name", ["GhostConv", "DepthwiseSeparableConv", "SqueezeExcite", "EfficientChannelAttn"]
    )
    def test_unsupported_layer_raises(self, name):
        with pytest.raises(UnsupportedTemporalLayerError, match=name):
            get_image_layer(name, 3, temporal=True)

    def test_unsupported_rank_raises(self):
        with pytest.raises(UnsupportedTemporalLayerError, match="rank 1"):
            get_image_layer("Conv", 1, temporal=True)

    def test_none_is_identity(self):
        assert get_image_layer(None, 3, temporal=True) is nn.Identity

    def test_unsupported_layer_in_model_raises(self):
        cfg = UNetConfig(
            filters=[8, 16],
            block_cfg=ImageConvBlockConfig(attn_cfg=LayerConfig(name="SqueezeExcite", kwargs={})),
            enable_temporal=True,
        )
        with pytest.raises(UnsupportedTemporalLayerError):
            UNet(in_channels=1, out_channels=1, rank=2, cfg=cfg)

    def test_temporal_unet_uses_all_params(self):
        model = UNet(1, 1, 3, UNetConfig(filters=[4, 8], enable_temporal=True))
        _assert_all_params_used(model, torch.randn(1, 1, 3, 8, 8, 8))
