import pytest
import torch

from im2sim.src.layers.image_conv_blocks import ImageConvBlockConfig
from im2sim.src.layers.module_config import LayerConfig
from im2sim.src.layers.unet import UNet, UNetConfig


# ---------------------------------------------------------------------------
# UNetConfig construction
# ---------------------------------------------------------------------------


def test_default_config():
    cfg = UNetConfig()

    assert cfg.filters == [64, 128, 256]
    assert len(cfg.encoder_block_cfg) == 3
    assert len(cfg.decoder_block_cfg) == 3
    assert len(cfg.pool_cfg) == 2
    assert len(cfg.upsample_cfg) == 2


def test_custom_filters():
    cfg = UNetConfig(filters=[16, 32, 64, 128])

    assert cfg.filters == [16, 32, 64, 128]
    assert len(cfg.encoder_block_cfg) == 4
    assert len(cfg.decoder_block_cfg) == 4
    assert len(cfg.pool_cfg) == 3
    assert len(cfg.upsample_cfg) == 3


def test_encoder_block_config_is_replicated():
    block_cfg = ImageConvBlockConfig(
        depth=3,
        activation="GELU",
    )

    cfg = UNetConfig(
        filters=[16, 32, 64],
        encoder_block_cfg=block_cfg,
    )

    assert len(cfg.encoder_block_cfg) == 3
    assert all(block is block_cfg for block in cfg.encoder_block_cfg)


def test_decoder_block_config_is_replicated():
    block_cfg = ImageConvBlockConfig(
        depth=3,
        activation="GELU",
    )

    cfg = UNetConfig(
        filters=[16, 32, 64],
        decoder_block_cfg=block_cfg,
    )

    assert len(cfg.decoder_block_cfg) == 3
    assert all(block is block_cfg for block in cfg.decoder_block_cfg)


def test_default_encoder_configs_are_independent():
    cfg = UNetConfig(filters=[16, 32, 64])

    cfg.encoder_block_cfg[0].depth = 5

    assert cfg.encoder_block_cfg[1].depth != 5
    assert cfg.encoder_block_cfg[2].depth != 5


def test_default_decoder_configs_are_independent():
    cfg = UNetConfig(filters=[16, 32, 64])

    cfg.decoder_block_cfg[0].depth = 5

    assert cfg.decoder_block_cfg[1].depth != 5
    assert cfg.decoder_block_cfg[2].depth != 5


def test_encoder_block_config_length_must_match_filters():
    with pytest.raises(
        AssertionError,
        match="Length of encoder_block_cfg",
    ):
        UNetConfig(
            filters=[16, 32, 64],
            encoder_block_cfg=[
                ImageConvBlockConfig(),
                ImageConvBlockConfig(),
            ],
        )


def test_decoder_block_config_length_must_match_filters():
    with pytest.raises(
        AssertionError,
        match="Length of decoder_block_cfg",
    ):
        UNetConfig(
            filters=[16, 32, 64],
            decoder_block_cfg=[
                ImageConvBlockConfig(),
                ImageConvBlockConfig(),
            ],
        )


def test_pool_config_is_replicated():
    pool_cfg = LayerConfig(
        name="MaxPool",
        kwargs={"kernel_size": 2},
    )

    cfg = UNetConfig(
        filters=[16, 32, 64],
        pool_cfg=pool_cfg,
    )

    assert len(cfg.pool_cfg) == 2
    assert all(pool is pool_cfg for pool in cfg.pool_cfg)


def test_upsample_config_is_replicated():
    upsample_cfg = LayerConfig(
        name="Upsample",
        kwargs={"scale_factor": 2, "mode": "bilinear"},
    )

    cfg = UNetConfig(
        filters=[16, 32, 64],
        upsample_cfg=upsample_cfg,
    )

    assert len(cfg.upsample_cfg) == 2
    assert all(up is upsample_cfg for up in cfg.upsample_cfg)


def test_pool_config_length_must_match_levels_minus_one():
    with pytest.raises(
        AssertionError,
        match="Length of pool_cfg",
    ):
        UNetConfig(
            filters=[16, 32, 64],
            pool_cfg=[
                LayerConfig(name="MaxPool"),
            ],
        )


def test_upsample_config_length_must_match_levels_minus_one():
    with pytest.raises(
        AssertionError,
        match="Length of upsample_cfg",
    ):
        UNetConfig(
            filters=[16, 32, 64],
            upsample_cfg=[
                LayerConfig(name="Upsample"),
            ],
        )


# ---------------------------------------------------------------------------
# UNetConfig derived configurations
# ---------------------------------------------------------------------------


def test_skip_connection_defaults_to_nullified_block():
    cfg = UNetConfig(filters=[16, 32, 64])

    assert cfg.skip_connection_cfg is not None
    assert cfg.skip_connection_cfg.conv_cfg.name == None
    assert cfg.skip_connection_cfg.norm_cfg.name == None
    assert cfg.skip_connection_cfg.activation == None
    assert cfg.skip_connection_cfg.dropout_cfg.name == None
    assert cfg.skip_connection_cfg.attn_cfg.name == None


def test_output_block_defaults_to_single_conv():
    cfg = UNetConfig(filters=[16, 32, 64])

    assert cfg.out_block_cfg is not None
    assert cfg.out_block_cfg.depth == 1
    assert cfg.out_block_cfg.conv_cfg.kwargs["kernel_size"] == 1
    assert cfg.out_block_cfg.conv_cfg.kwargs["stride"] == 1
    assert cfg.out_block_cfg.conv_cfg.kwargs["padding"] == 0


def test_add_input_residual():
    cfg = UNetConfig(filters=[32, 32, 32])

    result = cfg.add_input_residual()

    assert result is cfg

    for block in cfg.encoder_block_cfg[1:]:
        assert block.residual_connections is not None

    for block in cfg.decoder_block_cfg:
        assert block.residual_connections is not None


def test_add_input_residual_requires_equal_filters():
    cfg = UNetConfig(filters=[16, 32, 64])

    with pytest.raises(
        AssertionError,
        match="All filters must be the same",
    ):
        cfg.add_input_residual()


def test_add_conv1_residual():
    cfg = UNetConfig(filters=[16, 32, 64])

    result = cfg.add_conv1_residual()

    assert result is cfg

    for block in cfg.encoder_block_cfg:
        assert block.residual_connections is not None

    for block in cfg.decoder_block_cfg:
        assert block.residual_connections is not None


def test_dilate_bottleneck():
    cfg = UNetConfig(filters=[16, 32, 64])

    original = cfg.encoder_block_cfg[-1].conv_cfg.kwargs.copy()

    result = cfg.dilate_bottleneck(dilation=3)

    assert result is cfg
    assert cfg.encoder_block_cfg[-1].conv_cfg.kwargs["dilation"] == 3
    assert cfg.encoder_block_cfg[-1].conv_cfg.kwargs != original


def test_double_bottleneck():
    cfg = UNetConfig(filters=[16, 32, 64])

    original_depth = cfg.encoder_block_cfg[-1].depth

    result = cfg.double_bottleneck()

    assert result is cfg
    assert cfg.encoder_block_cfg[-1].depth == original_depth * 2


def test_reconstruction_mode():
    cfg = UNetConfig(filters=[16, 32, 64])

    result = cfg.reconstruction_mode()

    assert result is cfg
    assert cfg.out_activation is None
    assert cfg.out_block_cfg.out_activation is None


def test_segmentation_mode():
    cfg = UNetConfig(filters=[16, 32, 64])

    result = cfg.segmentation_mode()

    assert result is cfg
    assert cfg.out_block_cfg.out_activation is None


def test_single_class_segmentation_mode():
    cfg = UNetConfig(filters=[16, 32, 64])

    result = cfg.single_class_segmentation_mode()

    assert result is cfg
    assert cfg.out_activation == "sigmoid"
    assert cfg.out_block_cfg.out_activation == "sigmoid"


def test_multiclass_segmentation_mode():
    cfg = UNetConfig(filters=[16, 32, 64])

    result = cfg.multiclass_segmentation_mode()

    assert result is cfg
    assert cfg.out_activation == "softmax"
    assert cfg.out_block_cfg.out_activation == "softmax"


def test_to_depthwise_separable():
    cfg = UNetConfig(filters=[16, 32, 64])

    result = cfg.to_depthwise_separable()

    assert result is cfg

    for block in cfg.encoder_block_cfg:
        assert block.conv_cfg.name == "DepthwiseSeparableConv"

    for block in cfg.decoder_block_cfg:
        assert block.conv_cfg.name == "DepthwiseSeparableConv"


def test_to_ghost_depthwise():
    cfg = UNetConfig(filters=[16, 32, 64])

    result = cfg.to_ghost_depthwise()

    assert result is cfg

    for block in cfg.encoder_block_cfg:
        assert block.conv_cfg.name == "GhostConv"

    for block in cfg.decoder_block_cfg:
        assert block.conv_cfg.name == "GhostConv"


def test_to_ghost_depthwise_separable():
    cfg = UNetConfig(filters=[16, 32, 64])

    result = cfg.to_ghost_depthwise_separable()

    assert result is cfg

    for block in cfg.encoder_block_cfg:
        assert block.conv_cfg.name == "GhostConv"
        assert block.conv_cfg.kwargs["separable"] is True

    for block in cfg.decoder_block_cfg:
        assert block.conv_cfg.name == "GhostConv"
        assert block.conv_cfg.kwargs["separable"] is True


def test_add_eca():
    cfg = UNetConfig(filters=[16, 32, 64])

    result = cfg.add_eca()

    assert result is cfg


def test_add_se():
    cfg = UNetConfig(filters=[16, 32, 64])

    result = cfg.add_se()

    assert result is cfg


def test_add_skip_eca():
    cfg = UNetConfig(filters=[16, 32, 64])

    result = cfg.add_skip_eca()

    assert result is cfg


def test_add_skip_se():
    cfg = UNetConfig(filters=[16, 32, 64])

    result = cfg.add_skip_se()

    assert result is cfg


# ---------------------------------------------------------------------------
# UNet construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rank,input_shape",
    [
        (1, (2, 3, 32)),
        (2, (2, 3, 32, 32)),
        (3, (2, 3, 16, 16, 16)),
    ],
)
def test_unet_forward_rank(rank, input_shape):
    cfg = UNetConfig(filters=[8, 16, 32])

    model = UNet(
        in_channels=3,
        out_channels=2,
        rank=rank,
        cfg=cfg,
    )

    x = torch.randn(*input_shape)
    y = model(x)

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == 2
    assert y.shape[2:] == x.shape[2:]


def test_unet_forward_2d():
    cfg = UNetConfig(filters=[8, 16, 32])

    model = UNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
    )

    x = torch.randn(2, 3, 64, 64)
    y = model(x)

    assert y.shape == (2, 2, 64, 64)




def test_unet_add_fusion():
    cfg = UNetConfig(
        filters=[16, 16, 16],
        fusion_type="add",
    )

    model = UNet(
        in_channels=16,
        out_channels=2,
        rank=2,
        cfg=cfg,
    )

    x = torch.randn(2, 16, 64, 64)
    y = model(x)

    assert y.shape == (2, 2, 64, 64)


def test_unet_fusion_type_is_case_insensitive():
    cfg = UNetConfig(
        filters=[16, 32, 64],
        fusion_type="  CONCAT ",
    )

    model = UNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
    )

    x = torch.randn(2, 3, 64, 64)
    y = model(x)

    assert y.shape == (2, 2, 64, 64)


# ---------------------------------------------------------------------------
# Supervision
# ---------------------------------------------------------------------------


def test_no_deep_supervision_returns_tensor():
    cfg = UNetConfig(filters=[8, 16, 32])

    model = UNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
        supervision_levels=0,
    )

    x = torch.randn(2, 3, 64, 64)
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, 2, 64, 64)


def test_supervision_level_is_normalised_to_list():
    cfg = UNetConfig(filters=[8, 16, 32])

    model = UNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
        supervision_levels=0,
    )

    assert model.supervision_levels == [0]


def test_multiple_supervision_levels_return_list():
    cfg = UNetConfig(filters=[8, 16, 32])

    model = UNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
        supervision_levels=[0, 1],
    )

    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)

    assert isinstance(outputs, list)
    assert len(outputs) == 2

    for output in outputs:
        assert output.shape[0] == 2
        assert output.shape[1] == 2


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_unet_gradients():
    cfg = UNetConfig(filters=[8, 16, 32])

    model = UNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
    )

    x = torch.randn(
        2,
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

    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_unet_train_eval_modes():
    cfg = UNetConfig(filters=[8, 16, 32])

    model = UNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
    )

    model.train()
    assert model.training

    model.eval()
    assert not model.training


# ---------------------------------------------------------------------------
# Model structure
# ---------------------------------------------------------------------------


def test_number_of_encoder_blocks():
    cfg = UNetConfig(filters=[8, 16, 32, 64])

    model = UNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
    )

    assert len(model.encoders) == 4


def test_number_of_pools():
    cfg = UNetConfig(filters=[8, 16, 32, 64])

    model = UNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
    )

    assert len(model.pools) == 3


def test_number_of_upsamples():
    cfg = UNetConfig(filters=[8, 16, 32, 64])

    model = UNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
    )

    assert len(model.ups) == 3


def test_number_of_decoders():
    cfg = UNetConfig(filters=[8, 16, 32, 64])

    model = UNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
    )

    assert len(model.decoders) == 3


def test_model_preserves_filter_configuration():
    cfg = UNetConfig(filters=[8, 16, 32])

    model = UNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
    )

    assert model.L == 3
    assert model.fusion_type == "concat"
    

def test_deep_supervision_outputs_have_matching_spatial_shapes():
    cfg = UNetConfig(filters=[8, 16, 32])

    model = UNet(
        in_channels=3,
        out_channels=2,
        rank=2,
        cfg=cfg,
        supervision_levels=[0, 1],
    )

    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)

    assert isinstance(outputs, list)
    assert len(outputs) == 2

    spatial_shapes = [output.shape[2:] for output in outputs]

    assert spatial_shapes[0] == spatial_shapes[1]