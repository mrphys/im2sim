from dataclasses import dataclass, field, fields

import torch
from im2sim.src.layers.image_conv_blocks import ImageConvBlock, ImageConvBlockConfig
from im2sim.src.layers.layer_util import (
    ResidualConnectionType,
    apply_residual_connection,
    get_image_layer,
)
from im2sim.src.layers.module_config import Config, ConfigurableModule, LayerConfig, register_config

@register_config
@dataclass
class ReverseHalfUNetConfig(Config):
    hidden_channels: int = 64
    num_downsamples: int = 4
    pool_spec: LayerConfig | list[LayerConfig] = field(
        default_factory=lambda: LayerConfig(name="MaxPool", kwargs={"kernel_size": 2})
    )
    upsample_spec: LayerConfig | list[LayerConfig] = field(
        default_factory=lambda: LayerConfig(
            name="Upsample", kwargs={"scale_factor": 2, "mode": "trilinear"}
        )
    )
    block_cfg: ImageConvBlockConfig = field(default_factory=lambda: ImageConvBlockConfig())
    blocks_per_level: int = 2
    out_activation: str | None = None
    stem_block_cfg: ImageConvBlockConfig | None = None
    decoder_block_cfg: list[ImageConvBlockConfig] | ImageConvBlockConfig | None = None
    out_block_cfg: ImageConvBlockConfig | None = None
    fusion_type: ResidualConnectionType = ResidualConnectionType.ADD

    def __post_init__(self):
        if self.stem_block_cfg is None:
            self.stem_block_cfg = self.block_cfg.apply_presets(["single_block"])


        if self.decoder_block_cfg is None:
            self.decoder_block_cfg = [self.block_cfg] * self.num_downsamples
        elif isinstance(self.decoder_block_cfg, ImageConvBlockConfig):
            self.decoder_block_cfg = [self.decoder_block_cfg] * self.num_downsamples

        if self.out_block_cfg is None:
            self.out_block_cfg = self.block_cfg.apply_presets(["single_conv"])
            self.out_block_cfg.out_activation = self.out_activation


class ReverseHalfUNet(torch.nn.Module, ConfigurableModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        hidden_channels: int = 64,
        num_downsamples: int = 4,
        pool_spec: LayerConfig | list[LayerConfig] | None = None,
        upsample_spec: LayerConfig | list[LayerConfig] | None = None,
        block_cfg: ImageConvBlockConfig | None = None,
        blocks_per_level: int = 2,
        out_activation: str | None = None,
        stem_block_cfg: ImageConvBlockConfig | None = None,
        decoder_block_cfg: list[ImageConvBlockConfig] | ImageConvBlockConfig | None = None,
        out_block_cfg: ImageConvBlockConfig | None = None,
        fusion_type: ResidualConnectionType = ResidualConnectionType.ADD,
    ):
        super().__init__()

        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_downsamples = num_downsamples

        self.fusion_type = fusion_type
        fusion_channels = (
            hidden_channels * 2
            if self.fusion_type is ResidualConnectionType.CONCAT
            else hidden_channels
        )

        if pool_spec is None:
            pool_spec = LayerConfig(name="MaxPool", kwargs={"kernel_size": 2})
        if upsample_spec is None:
            mode = "nearest" if rank == 1 else "bilinear" if rank == 2 else "trilinear"
            upsample_spec = LayerConfig(name="Upsample", kwargs={"scale_factor": 2, "mode": mode})
        if block_cfg is None:
            block_cfg = ImageConvBlockConfig()

        if isinstance(pool_spec, LayerConfig):
            pool_spec = [pool_spec] * num_downsamples
        if isinstance(upsample_spec, LayerConfig):
            upsample_spec = [upsample_spec] * num_downsamples

        for p, u in zip(pool_spec, upsample_spec, strict=True):
            assert p.name.lower() in ["maxpool", "averagepool"], f"pool type {p.name} not supported"
            assert u.name.lower() in ["upsample", "pixelshuffle"], (
                f"upsample type {u.name} not supported"
            )

        self.pools = torch.nn.ModuleList(
            [get_image_layer(pool.name, rank)(**pool.kwargs) for pool in pool_spec]
        )
        self.ups = torch.nn.ModuleList(
            [get_image_layer(upsample.name, rank)(**upsample.kwargs) for upsample in upsample_spec]
        )

        if stem_block_cfg is None:
            stem_block_cfg = block_cfg.apply_presets(["single_block"])

        self.stem = ImageConvBlock.build(rank, in_channels, hidden_channels, stem_block_cfg)
        print(type(self.stem))

        if decoder_block_cfg is None:
            decoder_block_cfg = [block_cfg] * num_downsamples
        elif isinstance(decoder_block_cfg, ImageConvBlockConfig):
            decoder_block_cfg = [decoder_block_cfg] * num_downsamples

        self.decoder_blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    *[ImageConvBlock.build(rank, fusion_channels, hidden_channels, cfg)]
                    * blocks_per_level
                )
                for cfg in decoder_block_cfg
            ]
        )

        if out_block_cfg is None:
            out_block_cfg = block_cfg.apply_presets(["single_conv"])
            out_block_cfg.out_activation = out_activation

        self.out_block = ImageConvBlock.build(
            rank, hidden_channels, out_channels, out_block_cfg
        )

    def forward(self, x):
        x = self.stem(x)

        pooled_features = [x]
        for pool in self.pools:
            x = pool(x)
            pooled_features.append(x)

        pooled_features = pooled_features[::-1]  # Reverse the order for decoding

        fused = None
        for pf, block, up in zip(pooled_features[:-1], self.decoder_blocks, self.ups, strict=True):
            if fused is None:
                fused = block(pf)
            else:
                fused = block(
                    apply_residual_connection(fused, pf, connection_type=self.fusion_type)
                )

            fused = up(fused)

        out = self.out_block(
            apply_residual_connection(fused, pooled_features[-1], connection_type=self.fusion_type)
        )
        return out




@ReverseHalfUNetConfig.register_preset("residual")
def half_unet_residual_type(cfg: ReverseHalfUNetConfig):
    cfg.decoder_block_cfg = [
        c.apply_presets(["0_residual"]) for c in cfg.decoder_block_cfg
    ]
    return cfg


@ReverseHalfUNetConfig.register_preset("dilated_bottleneck")
def unet_residual_type(cfg: ReverseHalfUNetConfig):
    cfg.decoder_block_cfg[-1] = cfg.decoder_block_cfg[-1].apply_presets(cfg.block_cfg, presets=["dilated_convs"])
    return cfg


def _apply_preset_to_all_blocks(cfg: ReverseHalfUNetConfig, preset_name: str):
    cfg.stem_block_cfg = cfg.stem_block_cfg.apply_presets([preset_name])
    cfg.decoder_block_cfg = [
        c.apply_presets([preset_name]) for c in cfg.decoder_block_cfg
    ]
    cfg.out_block_cfg = cfg.out_block_cfg.apply_presets([preset_name])
    return cfg


@ReverseHalfUNetConfig.register_preset("recon")
def reconstruction_config(cfg: ReverseHalfUNetConfig):
    _apply_preset_to_all_blocks(cfg, "recon")
    cfg.out_activation = None
    return cfg


@ReverseHalfUNetConfig.register_preset("single_class_segmentation")
def segmentation_config(cfg: ReverseHalfUNetConfig):
    _apply_preset_to_all_blocks(cfg, "segmentation")
    cfg.out_block_cfg.out_activation = "sigmoid"
    return cfg

@ReverseHalfUNetConfig.register_preset("multiclass_segmentation")
def segmentation_config(cfg: ReverseHalfUNetConfig):
    _apply_preset_to_all_blocks(cfg, "segmentation")
    cfg.out_block_cfg.out_activation = "softmax"
    return cfg


@ReverseHalfUNetConfig.register_preset("depthwise_separable")
def depthwise_separable_config(cfg: ReverseHalfUNetConfig):
    cfg.decoder_block_cfg = [
        c.apply_presets(["depthwise_separable"])
        for c in cfg.decoder_block_cfg
    ]
    return cfg

@ReverseHalfUNetConfig.register_preset("ghost_depthwise")
def ghost_dw_config(cfg: ReverseHalfUNetConfig):
    cfg.decoder_block_cfg = [
        c.apply_presets(["ghost_depthwise"])
        for c in cfg.decoder_block_cfg
    ]
    return cfg


@ReverseHalfUNetConfig.register_preset("ghost_depthwise_separable")
def ghost_dws_config(cfg: ReverseHalfUNetConfig):
    cfg.decoder_block_cfg = [
        c.apply_presets(["ghost_depthwise_separable"])
        for c in cfg.decoder_block_cfg
    ]
    return cfg


@ReverseHalfUNetConfig.register_preset("ECA")
def eca_config(cfg: ReverseHalfUNetConfig):
    cfg.decoder_block_cfg = [
        c.apply_presets(["ECA"]) for c in cfg.decoder_block_cfg
    ]
    return cfg


@ReverseHalfUNetConfig.register_preset("SE")
def squeeze_excitation_config(cfg: ReverseHalfUNetConfig):
    cfg.decoder_block_cfg = [
        c.apply_presets(["SE"]) for c in cfg.decoder_block_cfg
    ]
    return cfg


if __name__ == "__main__":

    def cfg_print(cfg):
        print("###################")
        for field in fields(cfg):
            value = getattr(cfg, field.name)
            print(f"  {field.name}: {value}")

    cfg = ImageConvBlockConfig()
    cfg = ReverseHalfUNetConfig(num_downsamples=3)
    cfg = ReverseHalfUNetConfig.apply_presets(cfg, ["ghost_depthwise", "residual"])

    model = ReverseHalfUNet.build(rank=3, in_channels=1, out_channels=1, cfg=cfg)

    print(model)

    x = torch.randn(1, 1, 128, 128, 128)  # Example input for a 3D tensor

    def check_gradients(model, x):
        model.train()

        # Ensure input tracks gradients if you want to test input gradients
        x = x.requires_grad_(True)

        # Forward
        y = model(x)

        # Use a scalar loss
        loss = y.sum()

        # Backward
        loss.backward()

        # Check input gradient
        assert x.grad is not None, "Input gradient is None"
        assert torch.isfinite(x.grad).all(), "Input gradient contains NaN/Inf"

        # Check parameter gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} gradient is None"
                assert torch.isfinite(param.grad).all(), f"{name} gradient contains NaN/Inf"
                assert param.grad.abs().sum() > 0, f"{name} gradient is zero"

        return True

    # # Check gradients
    # if check_gradients(model, x):
    #     print("Gradient check passed.")

    x = torch.randn(1, 1, 128, 128, 128)
    y = model(x)
    print(y.shape)
