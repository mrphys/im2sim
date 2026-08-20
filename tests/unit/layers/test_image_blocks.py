import pytest
import torch

from im2sim.layers.image_conv_blocks import (
    ImageConvBlock,
    ImageConvBlockConfig,
)
from im2sim.layers.module_config import LayerConfig


# ---------------------------------------------------------------------------
# ImageConvBlockConfig
# ---------------------------------------------------------------------------


class TestImageConvBlockConfig:
    def test_default_config(self):
        cfg = ImageConvBlockConfig()

        assert cfg.depth == 1
        assert cfg.activation == "ReLU"
        assert cfg.out_activation is None

        assert cfg.conv_cfg.name == "Conv"
        assert cfg.conv_cfg.kwargs == {
            "kernel_size": 3,
            "padding": "same",
        }

        assert cfg.norm_cfg.name == "InstanceNorm"
        assert cfg.norm_cfg.kwargs == {"affine": True}

        assert cfg.dropout_cfg.name is None
        assert cfg.attn_cfg.name is None

        assert cfg.dropout_position == 1
        assert cfg.residual_connections is None
        assert cfg.residual_type == "add"

    def test_layer_config_defaults_are_independent(self):
        cfg1 = ImageConvBlockConfig()
        cfg2 = ImageConvBlockConfig()

        cfg1.conv_cfg.kwargs["kernel_size"] = 5

        assert cfg2.conv_cfg.kwargs["kernel_size"] == 3

    def test_to_single_conv(self):
        cfg = ImageConvBlockConfig(
            depth=3,
            activation="ReLU",
            out_activation="Sigmoid",
        )

        result = cfg.to_single_conv()

        assert result is cfg
        assert cfg.depth == 1
        assert cfg.activation is None
        assert cfg.norm_cfg.name is None
        assert cfg.dropout_cfg.name is None
        assert cfg.attn_cfg.name is None

    def test_to_single_block(self):
        cfg = ImageConvBlockConfig(
            depth=3,
            activation="ReLU",
            residual_connections={2: [0]},
        )

        result = cfg.to_single_block()

        assert result is cfg
        assert cfg.depth == 1
        assert cfg.activation == "ReLU"
        assert cfg.norm_cfg.name == "InstanceNorm"
        assert cfg.dropout_cfg.name is None
        assert cfg.residual_connections is None

    def test_add_input_residual(self):
        cfg = ImageConvBlockConfig(depth=3)

        result = cfg.add_input_residual()

        assert result is cfg
        assert cfg.residual_connections == {2: [0]}
        assert cfg.residual_type == "add"

    def test_add_conv1_residual(self):
        cfg = ImageConvBlockConfig(depth=3)

        result = cfg.add_conv1_residual()

        assert result is cfg
        assert cfg.residual_connections == {2: [1]}
        assert cfg.residual_type == "add"

    def test_add_conv1_residual_requires_depth_greater_than_one(self):
        cfg = ImageConvBlockConfig(depth=1)

        with pytest.raises(AssertionError, match="Depth must be greater than 1"):
            cfg.add_conv1_residual()

    def test_add_input_concat_residual(self):
        cfg = ImageConvBlockConfig(depth=3)

        result = cfg.add_input_concat_residual()

        assert result is cfg
        assert cfg.residual_connections == {2: [0]}
        assert cfg.residual_type == "concat"

    def test_reconstruction_mode(self):
        cfg = ImageConvBlockConfig()

        result = cfg.reconstruction_mode()

        assert result is cfg
        assert cfg.norm_cfg.name is None
        assert cfg.dropout_cfg.name is None

    def test_segmentation_mode(self):
        cfg = ImageConvBlockConfig()

        result = cfg.segmentation_mode()

        assert result is cfg
        assert cfg.norm_cfg.name == "InstanceNorm"
        assert cfg.norm_cfg.kwargs == {"affine": True}

    def test_dilate_convs(self):
        cfg = ImageConvBlockConfig()

        result = cfg.dilate_convs()

        assert result is cfg
        assert cfg.conv_cfg.kwargs["dilation"] == 2

    def test_dilate_convs_custom_dilation(self):
        cfg = ImageConvBlockConfig()

        cfg.dilate_convs(dilation=4)

        assert cfg.conv_cfg.kwargs["dilation"] == 4

    def test_add_eca(self):
        cfg = ImageConvBlockConfig()

        result = cfg.add_eca()

        assert result is cfg
        assert cfg.attn_cfg.name == "EfficientChannelAttn"
        assert cfg.attn_cfg.kwargs == {}

    def test_add_se(self):
        cfg = ImageConvBlockConfig()

        result = cfg.add_se()

        assert result is cfg
        assert cfg.attn_cfg.name == "SqueezeExcite"
        assert cfg.attn_cfg.kwargs == {}

    def test_nullify(self):
        cfg = ImageConvBlockConfig()

        result = cfg.nullify()

        assert result is cfg
        assert cfg.conv_cfg.name is None
        assert cfg.norm_cfg.name is None
        assert cfg.dropout_cfg.name is None
        assert cfg.attn_cfg.name is None


# ---------------------------------------------------------------------------
# ImageConvBlock construction
# ---------------------------------------------------------------------------


class TestImageConvBlock:
    @pytest.fixture
    def input_tensor(self):
        return torch.randn(2, 8, 32, 32)

    def test_default_block(self):
        cfg = ImageConvBlockConfig()

        model = ImageConvBlock(
            in_channels=8,
            out_channels=16,
            rank=2,
            cfg=cfg,
        )

        assert model.in_channels == 8
        assert model.out_channels == 16
        assert model.rank == 2
        assert model.depth == 1
        assert len(model.layers) == 1

    def test_forward_shape(self, input_tensor):
        cfg = ImageConvBlockConfig(depth=3)

        model = ImageConvBlock(
            in_channels=8,
            out_channels=16,
            rank=2,
            cfg=cfg,
        )

        output = model(input_tensor)

        assert output.shape == (2, 16, 32, 32)

    def test_single_conv_shape(self, input_tensor):
        cfg = ImageConvBlockConfig().to_single_conv()

        model = ImageConvBlock(
            in_channels=8,
            out_channels=16,
            rank=2,
            cfg=cfg,
        )

        output = model(input_tensor)

        assert output.shape == (2, 16, 32, 32)

    def test_different_spatial_dimensions(self):
        cfg = ImageConvBlockConfig(depth=2)

        model = ImageConvBlock(
            in_channels=4,
            out_channels=8,
            rank=2,
            cfg=cfg,
        )

        x = torch.randn(3, 4, 64, 48)
        y = model(x)

        assert y.shape == (3, 8, 64, 48)

    def test_depth_creates_correct_number_of_layers(self):
        for depth in [1, 2, 3, 5]:
            cfg = ImageConvBlockConfig(depth=depth)

            model = ImageConvBlock(
                in_channels=8,
                out_channels=8,
                rank=2,
                cfg=cfg,
            )

            assert len(model.layers) == depth

    def test_activation_is_applied_between_layers(self):
        cfg = ImageConvBlockConfig(
            depth=3,
            activation="ReLU",
        )

        model = ImageConvBlock(
            in_channels=8,
            out_channels=8,
            rank=2,
            cfg=cfg,
        )

        # The activation should be present on all but the final layer.
        assert isinstance(model.layers[0][-1], torch.nn.ReLU)
        assert isinstance(model.layers[1][-1], torch.nn.ReLU)
        assert isinstance(model.layers[2][-1], torch.nn.Identity)

    def test_no_activation(self, input_tensor):
        cfg = ImageConvBlockConfig(
            activation=None,
            out_activation=None,
        )

        model = ImageConvBlock(
            in_channels=8,
            out_channels=8,
            rank=2,
            cfg=cfg,
        )

        assert isinstance(model.activation, torch.nn.Identity)
        assert isinstance(model.out_activation, torch.nn.Identity)

        output = model(input_tensor)

        assert output.shape == input_tensor.shape

    def test_output_activation(self, input_tensor):
        cfg = ImageConvBlockConfig().nullify()
        cfg.out_activation = "Sigmoid"
        model = ImageConvBlock(
            in_channels=8,
            out_channels=8,
            rank=2,
            cfg=cfg,
        )

        output = model(input_tensor)

        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    # -----------------------------------------------------------------------
    # Normalisation
    # -----------------------------------------------------------------------

    def test_instance_norm(self):
        cfg = ImageConvBlockConfig(
            norm_cfg=LayerConfig(
                name="InstanceNorm",
                kwargs={"affine": True},
            )
        )

        model = ImageConvBlock(
            in_channels=8,
            out_channels=16,
            rank=2,
            cfg=cfg,
        )

        assert isinstance(model.layers[0][1], torch.nn.InstanceNorm2d)

    def test_batch_norm(self):
        cfg = ImageConvBlockConfig(
            norm_cfg=LayerConfig(
                name="BatchNorm",
                kwargs={},
            )
        )

        model = ImageConvBlock(
            in_channels=8,
            out_channels=16,
            rank=2,
            cfg=cfg,
        )

        assert isinstance(model.layers[0][1], torch.nn.BatchNorm2d)

    def test_no_normalisation(self):
        cfg = ImageConvBlockConfig().nullify()

        model = ImageConvBlock(
            in_channels=8,
            out_channels=16,
            rank=2,
            cfg=cfg,
        )

        assert isinstance(model.layers[0][1], torch.nn.Identity)

    # -----------------------------------------------------------------------
    # Dropout
    # -----------------------------------------------------------------------

    def test_dropout_position(self):
        cfg = ImageConvBlockConfig(
            depth=3,
            dropout_cfg=LayerConfig(
                name="Dropout",
                kwargs={"p": 0.5},
            ),
            dropout_position=1,
        )

        model = ImageConvBlock(
            in_channels=8,
            out_channels=8,
            rank=2,
            cfg=cfg,
        )

        assert isinstance(model.layers[0][2], torch.nn.Dropout2d)
        assert isinstance(model.layers[1][2], torch.nn.Identity)
        assert isinstance(model.layers[2][2], torch.nn.Identity)

    def test_multiple_dropout_positions(self):
        cfg = ImageConvBlockConfig(
            depth=4,
            dropout_cfg=LayerConfig(
                name="Dropout",
                kwargs={"p": 0.5},
            ),
            dropout_position=[1, 3],
        )

        model = ImageConvBlock(
            in_channels=8,
            out_channels=8,
            rank=2,
            cfg=cfg,
        )

        assert isinstance(model.layers[0][2], torch.nn.Dropout2d)
        assert isinstance(model.layers[1][2], torch.nn.Identity)
        assert isinstance(model.layers[2][2], torch.nn.Dropout2d)
        assert isinstance(model.layers[3][2], torch.nn.Identity)

    # -----------------------------------------------------------------------
    # Residual connections
    # -----------------------------------------------------------------------

    def test_input_residual(self, input_tensor):
        cfg = ImageConvBlockConfig(
            depth=3,
            norm_cfg=LayerConfig(name=None, kwargs={}),
        ).add_input_residual()

        model = ImageConvBlock(
            in_channels=8,
            out_channels=8,
            rank=2,
            cfg=cfg,
        )

        output = model(input_tensor)

        assert output.shape == input_tensor.shape

    def test_conv1_residual(self, input_tensor):
        cfg = ImageConvBlockConfig(
            depth=3,
            norm_cfg=LayerConfig(name=None, kwargs={}),
        ).add_conv1_residual()

        model = ImageConvBlock(
            in_channels=8,
            out_channels=8,
            rank=2,
            cfg=cfg,
        )

        output = model(input_tensor)

        assert output.shape == input_tensor.shape

    def test_concat_residual(self, input_tensor):
        cfg = ImageConvBlockConfig(
            depth=3,
            norm_cfg=LayerConfig(name=None, kwargs={}),
        ).add_input_concat_residual()

        model = ImageConvBlock(
            in_channels=8,
            out_channels=8,
            rank=2,
            cfg=cfg,
        )

        output = model(input_tensor)

        # Concatenatiion should not modify the final channel dim 
        assert output.shape == (2, 8, 32, 32)

    # -----------------------------------------------------------------------
    # Attention
    # -----------------------------------------------------------------------

    def test_eca_attention(self):
        cfg = ImageConvBlockConfig(depth=2).add_eca()

        model = ImageConvBlock(
            in_channels=8,
            out_channels=16,
            rank=2,
            cfg=cfg,
        )

        # With no residual connections, attention is placed on the final
        # layer.
        assert isinstance(model.layers[0][3], torch.nn.Identity)
        assert not isinstance(model.layers[1][3], torch.nn.Identity)

    def test_se_attention(self):
        cfg = ImageConvBlockConfig(depth=2).add_se()

        model = ImageConvBlock(
            in_channels=8,
            out_channels=16,
            rank=2,
            cfg=cfg,
        )

        assert isinstance(model.layers[0][3], torch.nn.Identity)
        assert not isinstance(model.layers[1][3], torch.nn.Identity)

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def test_invalid_norm_cfg(self):
        cfg = ImageConvBlockConfig(
            norm_cfg=LayerConfig(
                name="InvalidNorm",
                kwargs={},
            )
        )

        with pytest.raises(AssertionError, match="Unsupported norm type"):
            ImageConvBlock(
                in_channels=8,
                out_channels=8,
                rank=2,
                cfg=cfg,
            )

    def test_invalid_attention_config(self):
        cfg = ImageConvBlockConfig(
            attn_cfg=LayerConfig(
                name="InvalidAttention",
                kwargs={},
            )
        )

        with pytest.raises(AssertionError, match="Unsupported attention type"):
            ImageConvBlock(
                in_channels=8,
                out_channels=8,
                rank=2,
                cfg=cfg,
            )

    def test_dropout_position_must_be_less_than_depth(self):
        cfg = ImageConvBlockConfig(
            depth=2,
            dropout_cfg=LayerConfig(
                name="Dropout",
                kwargs={"p": 0.5},
            ),
            dropout_position=2,
        )

        with pytest.raises(
            AssertionError,
            match="Dropout position must be less than depth",
        ):
            ImageConvBlock(
                in_channels=8,
                out_channels=8,
                rank=2,
                cfg=cfg,
            )

    # -----------------------------------------------------------------------
    # Default config handling
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "attribute",
        [
            "conv_cfg",
            "norm_cfg",
            "dropout_cfg",
            "attn_cfg",
        ],
    )
    def test_none_config_gets_default(self, attribute):
        cfg = ImageConvBlockConfig()
        setattr(cfg, attribute, None)

        model = ImageConvBlock(
            in_channels=8,
            out_channels=8,
            rank=2,
            cfg=cfg,
        )

        assert getattr(model, attribute) is not None

    # -----------------------------------------------------------------------
    # Gradients
    # -----------------------------------------------------------------------

    def test_backward(self, input_tensor):
        cfg = ImageConvBlockConfig(depth=2)

        model = ImageConvBlock(
            in_channels=8,
            out_channels=16,
            rank=2,
            cfg=cfg,
        )

        output = model(input_tensor)
        loss = output.mean()
        loss.backward()

        for parameter in model.parameters():
            assert parameter.grad is not None

    # -----------------------------------------------------------------------
    # Different ranks
    # -----------------------------------------------------------------------

    def test_1d(self):
        cfg = ImageConvBlockConfig(depth=2)

        model = ImageConvBlock(
            in_channels=4,
            out_channels=8,
            rank=1,
            cfg=cfg,
        )

        x = torch.randn(2, 4, 32)
        y = model(x)

        assert y.shape == (2, 8, 32)

    def test_2d(self):
        cfg = ImageConvBlockConfig(depth=2)

        model = ImageConvBlock(
            in_channels=4,
            out_channels=8,
            rank=2,
            cfg=cfg,
        )

        x = torch.randn(2, 4, 32, 32)
        y = model(x)

        assert y.shape == (2, 8, 32, 32)

    def test_3d(self):
        cfg = ImageConvBlockConfig(depth=2)

        model = ImageConvBlock(
            in_channels=4,
            out_channels=8,
            rank=3,
            cfg=cfg,
        )

        x = torch.randn(2, 4, 8, 16, 16)
        y = model(x)

        assert y.shape == (2, 8, 8, 16, 16)