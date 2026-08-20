import pytest
import torch

from im2sim.configs.core import LayerConfig
from im2sim.models.reverse_halfunet import (
    ReverseHalfUNet,
    ReverseHalfUNetConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_upsample_cfg():
    return LayerConfig(
        name="Upsample",
        kwargs={
            "scale_factor": 2,
            "mode": "bilinear",
        },
    )


@pytest.fixture
def default_cfg(default_upsample_cfg):
    return ReverseHalfUNetConfig(
        hidden_channels=8,
        n_levels=3,
        upsample_cfg=default_upsample_cfg,
    )


@pytest.fixture
def model(default_cfg):
    return ReverseHalfUNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=default_cfg,
    )


# ===========================================================================
# ReverseHalfUNetConfig
# ===========================================================================


def test_config_defaults():
    cfg = ReverseHalfUNetConfig()

    assert cfg.hidden_channels == 64
    assert cfg.n_levels == 3
    assert cfg.decoder_blocks_per_level == 1
    assert cfg.fusion_type == "add"
    assert cfg.out_activation is None


def test_config_creates_stem_from_block_config():
    cfg = ReverseHalfUNetConfig()

    assert cfg.stem_block_cfg is not None
    assert cfg.stem_block_cfg is not cfg.block_cfg


def test_config_creates_decoder_config_for_each_level():
    cfg = ReverseHalfUNetConfig(n_levels=4)

    assert isinstance(cfg.decoder_block_cfg, list)
    assert len(cfg.decoder_block_cfg) == 4


def test_decoder_configs_are_independent():
    cfg = ReverseHalfUNetConfig(n_levels=3)

    cfg.decoder_block_cfg[0].depth = 5

    assert cfg.decoder_block_cfg[1].depth != 5
    assert cfg.decoder_block_cfg[2].depth != 5


def test_single_decoder_config_is_repeated():
    block_cfg = ReverseHalfUNetConfig().block_cfg

    cfg = ReverseHalfUNetConfig(
        n_levels=3,
        decoder_block_cfg=block_cfg,
    )

    assert len(cfg.decoder_block_cfg) == 3

    # The implementation intentionally repeats the same object here.
    assert all(
        decoder_cfg is block_cfg
        for decoder_cfg in cfg.decoder_block_cfg
    )


def test_output_config_is_single_conv():
    cfg = ReverseHalfUNetConfig()

    assert cfg.out_block_cfg is not None
    assert cfg.out_block_cfg.conv_cfg.kwargs["kernel_size"] == 1
    assert cfg.out_block_cfg.conv_cfg.kwargs["stride"] == 1
    assert cfg.out_block_cfg.conv_cfg.kwargs["padding"] == 0


# ===========================================================================
# Configuration validation
# ===========================================================================


def test_decoder_config_length_must_match_levels():
    block_cfg = ReverseHalfUNetConfig().block_cfg

    with pytest.raises(
        AssertionError,
        match="Length of decoder_block_cfg",
    ):
        ReverseHalfUNetConfig(
            n_levels=3,
            decoder_block_cfg=[
                block_cfg,
                block_cfg,
            ],
        )


def test_pool_config_length_must_equal_levels_minus_one():
    pool_cfg = LayerConfig(
        name="MaxPool",
        kwargs={"kernel_size": 2},
    )

    with pytest.raises(
        AssertionError,
        match="Length of pool_cfg",
    ):
        ReverseHalfUNetConfig(
            n_levels=3,
            pool_cfg=[
                pool_cfg,
                pool_cfg,
                pool_cfg,
            ],
        )


def test_upsample_config_length_must_equal_levels_minus_one():
    upsample_cfg = LayerConfig(
        name="Upsample",
        kwargs={
            "scale_factor": 2,
            "mode": "bilinear",
        },
    )

    with pytest.raises(
        AssertionError,
        match="Length of upsample_cfg",
    ):
        ReverseHalfUNetConfig(
            n_levels=3,
            upsample_cfg=[
                upsample_cfg,
                upsample_cfg,
                upsample_cfg,
            ],
        )


# ===========================================================================
# Configuration presets
# ===========================================================================


def test_add_residual():
    cfg = ReverseHalfUNetConfig(n_levels=3)

    result = cfg.add_residual()

    assert result is cfg

    for decoder_cfg in cfg.decoder_block_cfg[1:]:
        assert decoder_cfg.residual_connections is not None


def test_add_residual_does_not_modify_first_decoder():
    cfg = ReverseHalfUNetConfig(n_levels=3)

    original = cfg.decoder_block_cfg[0].residual_connections

    cfg.add_residual()

    assert cfg.decoder_block_cfg[0].residual_connections == original


def test_dilate_bottleneck():
    cfg = ReverseHalfUNetConfig(n_levels=3)

    result = cfg.dilate_bottleneck(3)

    assert result is cfg
    assert cfg.decoder_block_cfg[-1].conv_cfg.kwargs["dilation"] == 3


def test_double_bottleneck():
    cfg = ReverseHalfUNetConfig(n_levels=3)

    original_depth = cfg.decoder_block_cfg[-1].depth

    result = cfg.double_bottleneck()

    assert result is cfg
    assert cfg.decoder_block_cfg[-1].depth == original_depth * 2


def test_reconstruction_mode():
    cfg = ReverseHalfUNetConfig(
        out_activation="relu",
    )

    result = cfg.reconstruction_mode()

    assert result is cfg
    assert cfg.out_activation is None
    assert cfg.out_block_cfg.out_activation is None


def test_segmentation_mode():
    cfg = ReverseHalfUNetConfig()

    result = cfg.segmentation_mode()

    assert result is cfg
    assert cfg.out_block_cfg.out_activation is None


def test_single_class_segmentation_mode():
    cfg = ReverseHalfUNetConfig()

    result = cfg.single_class_segmentation_mode()

    assert result is cfg
    assert cfg.out_activation == "sigmoid"
    assert cfg.out_block_cfg.out_activation == "sigmoid"


def test_multiclass_segmentation_mode():
    cfg = ReverseHalfUNetConfig()

    result = cfg.multiclass_segmentation_mode()

    assert result is cfg
    assert cfg.out_activation == "softmax"
    assert cfg.out_block_cfg.out_activation == "softmax"


def test_to_depthwise_separable():
    cfg = ReverseHalfUNetConfig(n_levels=3)

    result = cfg.to_depthwise_separable()

    assert result is cfg
    assert cfg.stem_block_cfg.conv_cfg.name == "DepthwiseSeparableConv"

    for decoder_cfg in cfg.decoder_block_cfg:
        assert decoder_cfg.conv_cfg.name == "DepthwiseSeparableConv"


def test_to_ghost_depthwise():
    cfg = ReverseHalfUNetConfig(n_levels=3)

    result = cfg.to_ghost_depthwise()

    assert result is cfg
    assert cfg.stem_block_cfg.conv_cfg.name == "GhostConv"

    for decoder_cfg in cfg.decoder_block_cfg:
        assert decoder_cfg.conv_cfg.name == "GhostConv"


def test_to_ghost_depthwise_separable():
    cfg = ReverseHalfUNetConfig(n_levels=3)

    result = cfg.to_ghost_depthwise_separable()

    assert result is cfg
    assert cfg.stem_block_cfg.conv_cfg.name == "GhostConv"

    for decoder_cfg in cfg.decoder_block_cfg:
        assert decoder_cfg.conv_cfg.name == "GhostConv"
        assert decoder_cfg.conv_cfg.kwargs["separable"] is True


def test_add_eca():
    cfg = ReverseHalfUNetConfig(n_levels=3)

    result = cfg.add_eca()

    assert result is cfg

    for decoder_cfg in cfg.decoder_block_cfg:
        assert "EfficientChannelAttn" in decoder_cfg.attn_cfg.name


def test_add_se():
    cfg = ReverseHalfUNetConfig(n_levels=3)

    result = cfg.add_se()

    assert result is cfg

    for decoder_cfg in cfg.decoder_block_cfg:
        assert "SqueezeExcite" in decoder_cfg.attn_cfg.name


# ===========================================================================
# ReverseHalfUNet construction
# ===========================================================================


def test_model_has_expected_number_of_levels(model, default_cfg):
    assert model.n_levels == default_cfg.n_levels

    assert len(model.pools) == default_cfg.n_levels - 1
    assert len(model.ups) == default_cfg.n_levels - 1
    assert len(model.decoders) == default_cfg.n_levels


def test_model_has_stem(model):
    assert model.stem is not None


def test_model_has_output_blocks(model):
    # Default supervision level is 0.
    assert len(model.out_blocks) == 1


@pytest.mark.parametrize("n_levels", [1, 2, 3, 4])
def test_model_constructs_for_different_numbers_of_levels(
    n_levels,
    default_upsample_cfg,
):
    cfg = ReverseHalfUNetConfig(
        hidden_channels=8,
        n_levels=n_levels,
        upsample_cfg=default_upsample_cfg,
    )

    model = ReverseHalfUNet(
        in_channels=3,
        out_channels=1,
        rank=2,
        cfg=cfg,
    )

    assert len(model.pools) == n_levels - 1
    assert len(model.ups) == n_levels - 1
    assert len(model.decoders) == n_levels


# ===========================================================================
# Forward pass
# ===========================================================================


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
def test_forward_different_spatial_sizes(
    spatial_size,
    default_upsample_cfg,
):
    cfg = ReverseHalfUNetConfig(
        hidden_channels=8,
        upsample_cfg=default_upsample_cfg,
    )

    model = ReverseHalfUNet(
        in_channels=3,
        out_channels=1,
        rank=2,
        cfg=cfg,
    )

    x = torch.randn(1, 3, spatial_size, spatial_size)

    y = model(x)

    assert y.shape == (1, 1, spatial_size, spatial_size)


# ===========================================================================
# Fusion
# ===========================================================================


@pytest.mark.parametrize("fusion_type", ["add", "concat"])
def test_forward_fusion_types(
    fusion_type,
    default_upsample_cfg,
):
    cfg = ReverseHalfUNetConfig(
        hidden_channels=8,
        fusion_type=fusion_type,
        upsample_cfg=default_upsample_cfg,
    )

    model = ReverseHalfUNet(
        in_channels=3,
        out_channels=1,
        rank=2,
        cfg=cfg,
    )

    x = torch.randn(1, 3, 64, 64)

    y = model(x)

    assert y.shape == (1, 1, 64, 64)


def test_fusion_type_is_case_insensitive():
    cfg = ReverseHalfUNetConfig(
        hidden_channels=8,
        fusion_type="ADD",
        upsample_cfg=LayerConfig(
            name="Upsample",
            kwargs={
                "scale_factor": 2,
                "mode": "bilinear",
            },
        ),
    )

    model = ReverseHalfUNet(
        in_channels=3,
        out_channels=1,
        rank=2,
        cfg=cfg,
    )

    x = torch.randn(1, 3, 64, 64)

    y = model(x)

    assert y.shape == (1, 1, 64, 64)


# ===========================================================================
# Deep supervision
# ===========================================================================


def test_integer_supervision_level_is_converted_to_list(
    default_cfg,
):
    model = ReverseHalfUNet(
        in_channels=3,
        out_channels=1,
        rank=2,
        cfg=default_cfg,
        supervision_levels=1,
    )

    assert model.supervision_levels == [1]


def test_list_supervision_levels_is_preserved(
    default_cfg,
):
    levels = [0, 1]

    model = ReverseHalfUNet(
        in_channels=3,
        out_channels=1,
        rank=2,
        cfg=default_cfg,
        supervision_levels=levels,
    )

    assert model.supervision_levels == levels


@pytest.mark.parametrize(
    "supervision_level",
    [0, 1, 2],
)
def test_single_deep_supervision_output(
    supervision_level,
    default_upsample_cfg,
):
    cfg = ReverseHalfUNetConfig(
        hidden_channels=8,
        n_levels=3,
        upsample_cfg=default_upsample_cfg,
    )

    model = ReverseHalfUNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
        supervision_levels=supervision_level,
    )
    print(model)
    x = torch.randn(1, 3, 64, 64)

    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape[1] == 2


def test_multiple_supervision_levels_return_list(
    default_upsample_cfg,
):
    cfg = ReverseHalfUNetConfig(
        hidden_channels=8,
        n_levels=3,
        upsample_cfg=default_upsample_cfg,
    )

    model = ReverseHalfUNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
        supervision_levels=[0, 1],
    )

    x = torch.randn(1, 3, 64, 64)

    y = model(x)

    assert isinstance(y, list)
    assert len(y) == 2

    for output in y:
        assert output.shape[0] == 1
        assert output.shape[1] == 2


def test_output_blocks_match_supervision_levels(
    default_upsample_cfg,
):
    cfg = ReverseHalfUNetConfig(
        hidden_channels=8,
        n_levels=4,
        upsample_cfg=default_upsample_cfg,
    )

    model = ReverseHalfUNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
        supervision_levels=[0, 2],
    )

    assert len(model.out_blocks) == 2


# ===========================================================================
# Segmentation modes
# ===========================================================================


def test_single_class_segmentation_forward(
    default_upsample_cfg,
):
    cfg = ReverseHalfUNetConfig(
        hidden_channels=8,
        upsample_cfg=default_upsample_cfg,
    ).single_class_segmentation_mode()

    model = ReverseHalfUNet(
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



# ===========================================================================
# Backward / gradients
# ===========================================================================


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


def test_gradients_with_deep_supervision(default_upsample_cfg):
    cfg = ReverseHalfUNetConfig(
        hidden_channels=8,
        n_levels=3,
        upsample_cfg=default_upsample_cfg,
    )

    model = ReverseHalfUNet(
        in_channels=3,
        out_channels=1,
        rank=2,
        cfg=cfg,
        supervision_levels=[0, 1],
    )

    x = torch.randn(
        1,
        3,
        64,
        64,
        requires_grad=True,
    )

    outputs = model(x)

    assert isinstance(outputs, list)

    loss = sum(output.sum() for output in outputs)
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, (
                f"Missing gradient for parameter: {name}"
            )
            assert torch.isfinite(parameter.grad).all()


# ===========================================================================
# Evaluation determinism
# ===========================================================================


def test_forward_is_deterministic_in_eval_mode(model):
    model.eval()

    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)

    assert torch.equal(y1, y2)