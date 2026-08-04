import torch 
from dataclasses import dataclass, fields

import custom_image_layers
from layer_util import get_activation, get_image_layer, register_with_ranks, ResidualConnectionType, LayerSpec, ConfigurableModule, apply_residual_connection, ModuleSpec
from image_conv_blocks import ImageConvBlock, ImageConvBlockConfig, ImageConvBlockSpec

@dataclass
class HalfUNetConfig:
    hidden_channels: int = 64
    num_downsamples: int = 4
    pool_spec: LayerSpec | list[LayerSpec] = LayerSpec(name="MaxPool", kwargs={'kernel_size': 2})
    upsample_spec: LayerSpec | list[LayerSpec] = LayerSpec(name="Upsample", kwargs={'scale_factor': 2, 'mode': 'trilinear'})
    block_cfg: ImageConvBlockConfig = ImageConvBlockConfig()
    blocks_per_level: int = 2
    out_activation: str|None = None
    stem_block_cfg: ImageConvBlockConfig|None = None
    encoder_block_cfg: list[ImageConvBlockConfig]|ImageConvBlockConfig|None = None
    out_block_cfg: ImageConvBlockConfig|None = None
    fusion_type: ResidualConnectionType = ResidualConnectionType.ADD

    def __post_init__(self):
        if self.stem_block_cfg is None:
            self.stem_block_cfg = ImageConvBlockSpec.apply_presets(self.block_cfg, presets=["single_block"])

        if self.encoder_block_cfg is None:
            self.encoder_block_cfg = [self.block_cfg] * self.num_downsamples
        elif isinstance(self.encoder_block_cfg, ImageConvBlockConfig):
            self.encoder_block_cfg = [self.encoder_block_cfg] * self.num_downsamples

        if self.out_block_cfg is None:
            self.out_block_cfg = ImageConvBlockSpec.apply_presets(self.block_cfg, presets=["single_conv"])
            self.out_block_cfg.out_activation = self.out_activation


class HalfUNet(torch.nn.Module, ConfigurableModule):

    def __init__(self, 
                in_channels: int,
                out_channels: int,
                rank:int,
                hidden_channels: int = 64,
                num_downsamples: int = 4,
                pool_spec: LayerSpec | list[LayerSpec] = LayerSpec(name="MaxPool", kwargs={'kernel_size': 2}),
                upsample_spec: LayerSpec | list[LayerSpec] = LayerSpec(name="Upsample", kwargs={'scale_factor': 2, 'mode': 'trilinear'}),
                block_cfg: ImageConvBlockConfig = ImageConvBlockConfig(),
                blocks_per_level: int = 2,
                out_activation: str|None = None,
                stem_block_cfg: ImageConvBlockConfig|None = None,
                encoder_block_cfg: list[ImageConvBlockConfig]|ImageConvBlockConfig|None = None,
                out_block_cfg: ImageConvBlockConfig|None = None,
                fusion_type: ResidualConnectionType = ResidualConnectionType.ADD):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_downsamples = num_downsamples

        if isinstance(pool_spec, LayerSpec):
            pool_spec = [pool_spec] * num_downsamples
        if isinstance(upsample_spec, LayerSpec):
            upsample_spec = [upsample_spec] * num_downsamples
        
        for p, u in zip(pool_spec, upsample_spec):
            assert p.name.lower() in ["maxpool", "averagepool"], f"pool type {p.name} not supported"
            assert u.name.lower() in ["upsample", "pixelshuffle"], f"upsample type {u.name} not supported"

        self.pools = torch.nn.ModuleList([get_image_layer(pool.name, rank)(**pool.kwargs) for pool in pool_spec])
        self.ups = torch.nn.ModuleList([get_image_layer(upsample.name, rank)(**upsample.kwargs) for upsample in upsample_spec])

        if stem_block_cfg is None:
            stem_block_cfg = ImageConvBlockSpec.apply_presets(block_cfg, presets=["single_block"])

        self.stem = ImageConvBlock.from_config(rank, in_channels, hidden_channels, stem_block_cfg)
        print(type(self.stem))

        if encoder_block_cfg is None:
            encoder_block_cfg = [block_cfg] * num_downsamples
        elif isinstance(encoder_block_cfg, ImageConvBlockConfig):
            encoder_block_cfg = [encoder_block_cfg] * num_downsamples
        
        self.encoder_blocks = torch.nn.ModuleList([
            torch.nn.Sequential(*[ImageConvBlock.from_config(rank, hidden_channels, hidden_channels, cfg)] * blocks_per_level)
            for cfg in encoder_block_cfg
        ])

        if out_block_cfg is None:
            out_block_cfg = ImageConvBlockSpec.apply_presets(block_cfg, presets=["single_conv"])
            out_block_cfg.out_activation = out_activation
        
        self.fusion_type = fusion_type
        fusion_channels = hidden_channels * (num_downsamples+1) if self.fusion_type is ResidualConnectionType.CONCAT else hidden_channels
        self.out_block = ImageConvBlock.from_config(rank, fusion_channels, out_channels, out_block_cfg)

    
    def forward(self, x):
        x = self.stem(x)

        residuals = []
        for pool, block, up in zip(self.pools, self.encoder_blocks, self.ups):
            residuals.append(x)
            x = pool(x)
            x = block(x)
            
        
        fused = x
        for res, up in zip(reversed(residuals),self.ups):
            fused = up(fused)
            fused = apply_residual_connection(fused, res, connection_type=self.fusion_type)

        out = self.out_block(fused)
        return out
    

HalfUNetSpec = ModuleSpec(HalfUNet, HalfUNetConfig)


@HalfUNetSpec.register_config("residual")
def half_unet_residual_config(cfg: HalfUNetConfig):
    cfg.encoder_block_cfg = [ImageConvBlockSpec.apply_presets(c, presets=["0_residual"]) for c in cfg.encoder_block_cfg]
    return cfg

@HalfUNetSpec.register_config("dilated_bottleneck")
def unet_residual_config(cfg: HalfUNetConfig):
    cfg.encoder_block_cfg[-1] = ImageConvBlockSpec.apply_presets(cfg.block_cfg, presets=["dilated_convs"])
    return cfg

def _apply_preset_to_all_blocks(cfg: HalfUNetConfig, preset_name: str):
    cfg.stem_block_cfg= ImageConvBlockSpec.apply_presets(cfg.stem_block_cfg, presets=[preset_name])
    cfg.encoder_block_cfg= [ImageConvBlockSpec.apply_presets(c, presets=[preset_name]) for c in cfg.encoder_block_cfg]
    cfg.out_block_cfg = ImageConvBlockSpec.apply_presets(cfg.out_block_cfg, presets=[preset_name])
    return cfg


@HalfUNetSpec.register_config("recon")
def reconstruction_config(cfg: HalfUNetConfig):
    _apply_preset_to_all_blocks(cfg, "reconstruction")
    cfg.out_activation = None
    return cfg

@HalfUNetSpec.register_config("segmentation")
def segmentation_config(cfg: HalfUNetConfig):
    _apply_preset_to_all_blocks(cfg, "segmentation")
    return cfg


@HalfUNetSpec.register_config("depthwise_separable")
def depthwise_separable_config(cfg: HalfUNetConfig):
    _apply_preset_to_all_blocks(cfg, "depthwise_separable")
    return cfg

@HalfUNetSpec.register_config("ghost_depthwise")
def ghost_dw_config(cfg: HalfUNetConfig):
    _apply_preset_to_all_blocks(cfg, "ghost_depthwise")
    return cfg

@HalfUNetSpec.register_config("ghost_depthwise_separable")
def ghost_dws_config(cfg: HalfUNetConfig):
    _apply_preset_to_all_blocks(cfg, "ghost_depthwise_separable")
    return cfg


if __name__ == "__main__":

    def cfg_print(cfg):
        print("###################")
        for field in fields(cfg):
            value = getattr(cfg, field.name)
            print(f"  {field.name}: {value}")
    
    cfg = ImageConvBlockConfig()
    cfg_print(cfg)

    cfg = HalfUNetConfig(num_downsamples=2)
    cfg_print(cfg)
    cfg = HalfUNetSpec.apply_presets(cfg, presets=["ghost_depthwise", "residual"])
    cfg_print(cfg)

    model = HalfUNetSpec.build(rank=3, in_channels=1, out_channels=1, cfg=cfg)

    print(model)

    x = torch.randn(1, 1, 64, 64, 64)  # Example input for a 3D tensor
    y = model(x)
    print(y.shape)  # Should print the shape of the output tensor

    