import pytest
import torch

from im2sim.configs.core import LayerConfig
from im2sim.models.halfunet import HalfUNet, HalfUNetConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_cfg():
    return HalfUNetConfig(
        # Explicitly use the 2D interpolation mode.
        upsample_cfg=LayerConfig(
            name="Upsample",
            kwargs={"scale_factor": 2, "mode": "bilinear"},
        )
    )


@pytest.fixture
def model(default_cfg):
    return HalfUNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=default_cfg,
    )


# ---------------------------------------------------------------------------
# HalfUNetConfig - defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = HalfUNetConfig()

    assert cfg.hidden_channels == 64
    assert cfg.n_levels == 3
    assert cfg.encoder_blocks_per_level == 1
    assert cfg.fusion_type == "add"
    assert cfg.out_activation is None


def test_config_expands_stem_config_from_block_config():
    block_cfg = HalfUNetConfig().block_cfg

    cfg = HalfUNetConfig(block_cfg=block_cfg)

    assert cfg.stem_block_cfg is not None
    assert cfg.stem_block_cfg is not block_cfg


def test_config_creates_encoder_config_for_each_level():
    cfg = HalfUNetConfig(n_levels=4)

    assert isinstance(cfg.encoder_block_cfg, list)
    assert len(cfg.encoder_block_cfg) == 4


def test_encoder_configs_are_independent():
    cfg = HalfUNetConfig(n_levels=3)

    cfg.encoder_block_cfg[0].depth = 5

    assert cfg.encoder_block_cfg[1].depth != 5
    assert cfg.encoder_block_cfg[2].depth != 5


def test_config_repeats_single_encoder_config():
    encoder_cfg = HalfUNetConfig().block_cfg

    cfg = HalfUNetConfig(
        n_levels=3,
        encoder_block_cfg=encoder_cfg,
    )

    assert len(cfg.encoder_block_cfg) == 3
    assert all(
        encoder_cfg is item
        for item in cfg.encoder_block_cfg
    )


def test_config_creates_single_output_conv():
    cfg = HalfUNetConfig()

    assert cfg.out_block_cfg is not None
    assert cfg.out_block_cfg.conv_cfg.kwargs["kernel_size"] == 1
    assert cfg.out_block_cfg.conv_cfg.kwargs["stride"] == 1
    assert cfg.out_block_cfg.conv_cfg.kwargs["padding"] == 0


# ---------------------------------------------------------------------------
# HalfUNetConfig - validation
# ---------------------------------------------------------------------------


def test_encoder_config_length_must_match_number_of_levels():
    with pytest.raises(
        AssertionError,
        match="Length of encoder_block_cfg",
    ):
        HalfUNetConfig(
            n_levels=3,
            encoder_block_cfg=[
                HalfUNetConfig().block_cfg,
                HalfUNetConfig().block_cfg,
            ],
        )


def test_pool_config_length_must_equal_levels_minus_one():
    with pytest.raises(
        AssertionError,
        match="Length of pool_cfg",
    ):
        HalfUNetConfig(
            n_levels=3,
            pool_cfg=[
                LayerConfig(name="MaxPool", kwargs={"kernel_size": 2}),
                LayerConfig(name="MaxPool", kwargs={"kernel_size": 2}),
                LayerConfig(name="MaxPool", kwargs={"kernel_size": 2}),
            ],
        )


def test_upsample_config_length_must_equal_levels_minus_one():
    with pytest.raises(
        AssertionError,
        match="Length of upsample_cfg",
    ):
        HalfUNetConfig(
            n_levels=3,
            upsample_cfg=[
                LayerConfig(
                    name="Upsample",
                    kwargs={"scale_factor": 2, "mode": "bilinear"},
                ),
                LayerConfig(
                    name="Upsample",
                    kwargs={"scale_factor": 2, "mode": "bilinear"},
                ),
                LayerConfig(
                    name="Upsample",
                    kwargs={"scale_factor": 2, "mode": "bilinear"},
                ),
            ],
        )


# ---------------------------------------------------------------------------
# HalfUNetConfig - configuration presets
# ---------------------------------------------------------------------------


def test_add_residual():
    cfg = HalfUNetConfig(n_levels=3)

    result = cfg.add_residual()

    assert result is cfg

    for encoder_cfg in cfg.encoder_block_cfg[1:]:
        assert encoder_cfg.residual_connections is not None


def test_add_residual_does_not_modify_first_encoder():
    cfg = HalfUNetConfig(n_levels=3)

    original = cfg.encoder_block_cfg[0].residual_connections

    cfg.add_residual()

    assert cfg.encoder_block_cfg[0].residual_connections == original


def test_dilate_bottleneck():
    cfg = HalfUNetConfig(n_levels=3)

    original_depth = cfg.encoder_block_cfg[-1].depth

    result = cfg.dilate_bottleneck(3)

    assert result is cfg
    assert cfg.encoder_block_cfg[-1].depth == original_depth

    # Verify that the convolution configuration was changed.
    conv_kwargs = cfg.encoder_block_cfg[-1].conv_cfg.kwargs
    assert conv_kwargs["dilation"] == 3


def test_double_bottleneck():
    cfg = HalfUNetConfig(n_levels=3)

    original_depth = cfg.encoder_block_cfg[-1].depth

    result = cfg.double_bottleneck()

    assert result is cfg
    assert cfg.encoder_block_cfg[-1].depth == 2 * original_depth


def test_reconstruction_mode():
    cfg = HalfUNetConfig(
        out_activation="relu",
    )

    result = cfg.reconstruction_mode()

    assert result is cfg
    assert cfg.out_activation is None
    assert cfg.out_block_cfg.out_activation is None


def test_segmentation_mode():
    cfg = HalfUNetConfig()

    result = cfg.segmentation_mode()

    assert result is cfg
    assert cfg.out_block_cfg.out_activation is None


def test_single_class_segmentation_mode():
    cfg = HalfUNetConfig()

    result = cfg.single_class_segmentation_mode()

    assert result is cfg
    assert cfg.out_activation == "sigmoid"
    assert cfg.out_block_cfg.out_activation == "sigmoid"


def test_multiclass_segmentation_mode():
    cfg = HalfUNetConfig()

    result = cfg.multiclass_segmentation_mode()

    assert result is cfg
    assert cfg.out_activation == "softmax"
    assert cfg.out_block_cfg.out_activation == "softmax"


def test_to_depthwise_separable():
    cfg = HalfUNetConfig(n_levels=3)

    result = cfg.to_depthwise_separable()

    assert result is cfg
    assert cfg.stem_block_cfg.conv_cfg.name == "DepthwiseSeparableConv"

    for encoder_cfg in cfg.encoder_block_cfg:
        assert encoder_cfg.conv_cfg.name == "DepthwiseSeparableConv"


def test_to_ghost_depthwise():
    cfg = HalfUNetConfig(n_levels=3)

    result = cfg.to_ghost_depthwise()

    assert result is cfg
    assert cfg.stem_block_cfg.conv_cfg.name == "GhostConv"

    for encoder_cfg in cfg.encoder_block_cfg:
        assert encoder_cfg.conv_cfg.name == "GhostConv"


def test_to_ghost_depthwise_separable():
    cfg = HalfUNetConfig(n_levels=3)

    result = cfg.to_ghost_depthwise_separable()

    assert result is cfg
    assert cfg.stem_block_cfg.conv_cfg.name == "GhostConv"

    for encoder_cfg in cfg.encoder_block_cfg:
        assert encoder_cfg.conv_cfg.name == "GhostConv"
        assert encoder_cfg.conv_cfg.kwargs["separable"] is True


def test_add_eca():
    cfg = HalfUNetConfig(n_levels=3)

    result = cfg.add_eca()

    assert result is cfg

    # Check that the preset propagated to every encoder.
    for encoder_cfg in cfg.encoder_block_cfg:
        assert "EfficientChannelAttn" in encoder_cfg.attn_cfg.name 


def test_add_se():
    cfg = HalfUNetConfig(n_levels=3)

    result = cfg.add_se()

    assert result is cfg

    for encoder_cfg in cfg.encoder_block_cfg:
        assert "SqueezeExcite" in encoder_cfg.attn_cfg.name 


# ---------------------------------------------------------------------------
# HalfUNet - construction
# ---------------------------------------------------------------------------


def test_model_has_expected_number_of_levels(model, default_cfg):
    assert model.n_levels == default_cfg.n_levels
    assert len(model.encoders) == default_cfg.n_levels
    assert len(model.pools) == default_cfg.n_levels - 1
    assert len(model.ups) == default_cfg.n_levels - 1


def test_model_stem(model):
    assert model.stem is not None


def test_model_has_output_block(model):
    assert model.out_block is not None


@pytest.mark.parametrize("n_levels", [1, 2, 3, 4])
def test_model_constructs_for_different_numbers_of_levels(n_levels):
    cfg = HalfUNetConfig(
        n_levels=n_levels,
        hidden_channels=8,
        upsample_cfg=LayerConfig(
            name="Upsample",
            kwargs={"scale_factor": 2, "mode": "bilinear"},
        ),
    )

    model = HalfUNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
    )

    assert len(model.encoders) == n_levels
    assert len(model.pools) == n_levels - 1
    assert len(model.ups) == n_levels - 1


# ---------------------------------------------------------------------------
# HalfUNet - forward pass
# ---------------------------------------------------------------------------


def test_forward_shape(model):
    x = torch.randn(2, 3, 64, 64)

    y = model(x)

    assert y.shape == (2, 2, 64, 64)


def test_forward_preserves_spatial_dimensions(model):
    x = torch.randn(1, 3, 128, 128)

    y = model(x)

    assert y.shape[-2:] == x.shape[-2:]


@pytest.mark.parametrize(
    "spatial_size",
    [32, 64, 128],
)
def test_forward_different_spatial_sizes(spatial_size):
    cfg = HalfUNetConfig(
        hidden_channels=8,
        upsample_cfg=LayerConfig(
            name="Upsample",
            kwargs={"scale_factor": 2, "mode": "bilinear"},
        ),
    )

    model = HalfUNet(
        in_channels=3,
        out_channels=1,
        rank=2,
        cfg=cfg,
    )

    x = torch.randn(1, 3, spatial_size, spatial_size)
    y = model(x)

    assert y.shape == (1, 1, spatial_size, spatial_size)


# ---------------------------------------------------------------------------
# HalfUNet - fusion types
# ---------------------------------------------------------------------------


def test_additive_fusion():
    cfg = HalfUNetConfig(
        hidden_channels=8,
        fusion_type="add",
        upsample_cfg=LayerConfig(
            name="Upsample",
            kwargs={"scale_factor": 2, "mode": "bilinear"},
        ),
    )

    model = HalfUNet(
        in_channels=3,
        out_channels=1,
        rank=2,
        cfg=cfg,
    )

    x = torch.randn(1, 3, 64, 64)
    y = model(x)

    assert y.shape == (1, 1, 64, 64)


def test_concatenation_fusion():
    cfg = HalfUNetConfig(
        hidden_channels=8,
        fusion_type="concat",
        upsample_cfg=LayerConfig(
            name="Upsample",
            kwargs={"scale_factor": 2, "mode": "bilinear"},
        ),
    )

    model = HalfUNet(
        in_channels=3,
        out_channels=1,
        rank=2,
        cfg=cfg,
    )

    x = torch.randn(1, 3, 64, 64)
    y = model(x)

    assert y.shape == (1, 1, 64, 64)


# ---------------------------------------------------------------------------
# HalfUNet - activations / segmentation
# ---------------------------------------------------------------------------


def test_single_class_segmentation_forward():
    cfg = HalfUNetConfig(
        hidden_channels=8,
        upsample_cfg=LayerConfig(
            name="Upsample",
            kwargs={"scale_factor": 2, "mode": "bilinear"},
        ),
    ).single_class_segmentation_mode()

    model = HalfUNet(
        in_channels=3,
        out_channels=1,
        rank=2,
        cfg=cfg,
    )

    x = torch.randn(1, 3, 64, 64)
    y = model(x)

    assert y.shape == (1, 1, 64, 64)
    assert torch.all(y >= 0)
    assert torch.all(y <= 1)


# ---------------------------------------------------------------------------
# HalfUNet - backward / gradients
# ---------------------------------------------------------------------------


def test_backward(model):
    x = torch.randn(
        1,
        3,
        64,
        64,
        requires_grad=True,
    )

    y = model(x)
    loss = y.sum()

    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_parameter_gradients(model):
    x = torch.randn(1, 3, 64, 64)

    y = model(x)
    y.sum().backward()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, (
                f"Missing gradient for parameter: {name}"
            )
            assert torch.isfinite(parameter.grad).all(), (
                f"Non-finite gradient for parameter: {name}"
            )


def test_forward_is_deterministic_in_eval_mode(model):
    model.eval()

    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)

    assert torch.equal(y1, y2)


# ---------------------------------------------------------------------------
# 3D support
# ---------------------------------------------------------------------------


def test_3d_forward():
    cfg = HalfUNetConfig(
        hidden_channels=8,
        upsample_cfg=LayerConfig(
            name="Upsample",
            kwargs={
                "scale_factor": 2,
                "mode": "trilinear",
            },
        ),
    )

    model = HalfUNet(
        in_channels=2,
        out_channels=1,
        rank=3,
        cfg=cfg,
    )

    x = torch.randn(1, 2, 16, 32, 32)
    y = model(x)

    assert y.shape == (1, 1, 16, 32, 32)