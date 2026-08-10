from dataclasses import dataclass, field, fields
from copy import deepcopy
import torch
from im2sim.src.layers.image_conv_blocks import ImageConvBlock, ImageConvBlockConfig
from im2sim.src.layers.layer_util import (
    ResidualConnectionType,
    apply_residual_connection,
    get_image_layer,
)
from im2sim.src.layers.module_config import Config, ConfigurableModule, LayerConfig, register_config
from im2sim.src.utils import api_util


@api_util.export('configs.HalfUNetConfig')
@register_config
@dataclass
class HalfUNetConfig(Config):
    """
    Configuration class for defining the parameters of a half U-Net architecture (see im2sim.models.HalfUNet).
    
    Attributes can either be set directly when creating an instance of the class or modified later.

    Configuration presets can be applied to quickly set up common configurations for different use cases.

    The configuration can also be saved to and loaded from a YAML file.

    Args:

        hidden_channels (int): 
            Number of channels in the hidden layers. Default is 64.

        num_downsamples (int): 
            Number of downsampling operations in the encoder. Default is 4.

        pool_spec (LayerConfig | list[LayerConfig]): 
            Specification for the pooling layers. Default is a MaxPool layer with kernel size 2 for all levels.

        upsample_spec (LayerConfig | list[LayerConfig]): 
            Specification for the upsampling layers. Default is an Upsample layer with scale factor 2 and mode 'trilinear' for all levels. 
            The mode is automatically changed to 'bilinear' for 2D data and 'nearest' for 1D.

        block_cfg (ImageConvBlockConfig): 
            Configuration for the convolutional blocks. Default is a standard convolutional block with 2 layers, ReLU activation, and batch normalization.

        blocks_per_level (int): 
            Number of convolutional blocks per level in the encoder. Default is 2.

        out_activation (str | None): 
            Activation function for the output layer. Default is None, which means no activation is applied.

        stem_block_cfg (ImageConvBlockConfig | None): 
            Configuration for the stem block. If None, it defaults to a single convolutional block with the same configuration as `block_cfg`.

        encoder_block_cfg (list[ImageConvBlockConfig] | ImageConvBlockConfig | None): 
            Configuration for the encoder blocks. If None, it defaults to a list of `block_cfg` repeated for each downsampling level.

        out_block_cfg (ImageConvBlockConfig | None): 
            Configuration for the output block. If None, it defaults to a single convolutional block with the same configuration as `block_cfg` and the specified `out_activation`.

        fusion_type (ResidualConnectionType): 
            Type of residual connection to use in the network. It can be either 'add' (default), 'concat' or 'average'. 
            This determines how the encoder features are fused.

    Examples:

        To create a HalfUNet model for single class segmentation, you can do the following:

        >>> cfg = HalfUNetConfig(num_downsamples=3, hidden_channels=64)
        >>> cfg = cfg.apply_presets(["single_class_segmentation", "residual", "SE", "ghost_depthwise_separable"])
        >>> model = HalfUNet.build(in_channles=20, out_channels=1, rank=3, cfg=cfg)

        For more presets, see the Preset Library below.

    """
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
    encoder_block_cfg: list[ImageConvBlockConfig] | ImageConvBlockConfig | None = None
    out_block_cfg: ImageConvBlockConfig | None = None
    fusion_type: ResidualConnectionType = ResidualConnectionType.ADD

    def __post_init__(self):
        print("HalfUNetConfig __post_init__ called")
        if self.stem_block_cfg is None:
            self.stem_block_cfg = self.block_cfg.apply_presets(["single_block"])
            if self.stem_block_cfg.out_activation is None:
                self.stem_block_cfg.out_activation = self.stem_block_cfg.activation
        
        if self.encoder_block_cfg is None:
            self.encoder_block_cfg = deepcopy(self.block_cfg)
            if self.encoder_block_cfg.out_activation is None:
                self.encoder_block_cfg.out_activation = self.encoder_block_cfg.activation
            self.encoder_block_cfg = [self.encoder_block_cfg] * self.num_downsamples

        elif isinstance(self.encoder_block_cfg, ImageConvBlockConfig):
            self.encoder_block_cfg = [self.encoder_block_cfg] * self.num_downsamples

        if self.out_block_cfg is None:
            self.out_block_cfg = self.block_cfg.apply_presets(["single_conv"])
            self.out_block_cfg.out_activation = self.out_activation


@api_util.export('models.HalfUNet')
class HalfUNet(torch.nn.Module, ConfigurableModule):
    """
    A Half-UNet[1] architecture for image processing tasks.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rank (int): Dimensionality of the input data (1 for 1D, 2 for 2D, 3 for 3D).
        hidden_channels (int): Number of channels in the hidden layers. Default is 64.
        num_downsamples (int): Number of downsampling operations in the encoder. Default is 4.
        pool_spec (LayerConfig | list[LayerConfig]): Specification for the pooling layers. Default is a MaxPool layer with kernel size 2 for all levels.
        upsample_spec (LayerConfig | list[LayerConfig]): Specification for the upsampling layers. Default is an Upsample layer with scale factor 2 and mode 'trilinear' for all levels.
        block_cfg (ImageConvBlockConfig): Configuration for the convolutional blocks. Default is a standard convolutional block with 2 layers, ReLU activation, and batch normalization.
        blocks_per_level (int): Number of convolutional blocks per level in the encoder. Default is 2.
        out_activation (str | None): Activation function for the output layer. Default is None, which means no activation is applied.
        stem_block_cfg (ImageConvBlockConfig | None): Configuration for the stem block. If None, it defaults to a single convolutional block with the same configuration as `block_cfg`.
        encoder_block_cfg (list[ImageConvBlockConfig] | ImageConvBlockConfig | None): Configuration for the encoder blocks. If None, it defaults to a list of `block_cfg` repeated for each downsampling level.
        out_block_cfg (ImageConvBlockConfig | None): Configuration for the output block. If None, it defaults to a single convolutional block with the same configuration as `block_cfg` and the specified `out_activation`.
        fusion_type (ResidualConnectionType): Type of residual connection to use in the network. It can be either 'add' (default), 'concat' or 'average'. This determines how the encoder features are fused.
    
    The best way to build a HalfUNet is to use the `im2sim.configs.HalfUNetConfig` class to define the configuration and then call the `build` method.

    Example:

    To create a HalfUNet model for single class segmentation, you can do the following:

    >>> cfg = HalfUNetConfig(num_downsamples=3, hidden_channels=64)
    >>> cfg = cfg.apply_presets(["single_class_segmentation", "residual", "SE", "ghost_depthwise_separable"])
    >>> model = HalfUNet.build(in_channles=20, out_channels=1, rank=3, cfg=cfg)

    References:
    .. [1] H. Lu, Y. She, J. Tie, and S. Xu, Half-UNet: A Simplified U-Net Architecture for Medical Image Segmentation, 
        Front. Neuroinformatics, vol. 16, Jun. 2022, doi: 10.3389/fninf.2022.911679.
    """
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
        encoder_block_cfg: list[ImageConvBlockConfig] | ImageConvBlockConfig | None = None,
        out_block_cfg: ImageConvBlockConfig | None = None,
        fusion_type: ResidualConnectionType = ResidualConnectionType.ADD,
    ):
        """ """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_downsamples = num_downsamples

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

        if encoder_block_cfg is None:
            encoder_block_cfg = [block_cfg] * num_downsamples
        elif isinstance(encoder_block_cfg, ImageConvBlockConfig):
            encoder_block_cfg = [encoder_block_cfg] * num_downsamples

        self.encoder_blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    *[ImageConvBlock.build(rank, hidden_channels, hidden_channels, cfg)]
                    * blocks_per_level
                )
                for cfg in encoder_block_cfg
            ]
        )

        if out_block_cfg is None:
            out_block_cfg = block_cfg.apply_presets(["single_conv"])
            out_block_cfg.out_activation = out_activation

        self.fusion_type = fusion_type
        fusion_channels = (
            hidden_channels * (num_downsamples + 1)
            if self.fusion_type is ResidualConnectionType.CONCAT
            else hidden_channels
        )
        self.out_block = ImageConvBlock.build(
            rank, fusion_channels, out_channels, out_block_cfg
        )

    def forward(self, x):
        """ 
        Forward pass of the HalfUNet model.
        """
        x = self.stem(x)

        residuals = []
        for pool, block in zip(self.pools, self.encoder_blocks, strict=True):
            residuals.append(x)
            x = pool(x)
            x = block(x)

        fused = x
        for res, up in zip(reversed(residuals), self.ups, strict=True):
            fused = up(fused)
            fused = apply_residual_connection(fused, res, connection_type=self.fusion_type)

        out = self.out_block(fused)
        return out




@HalfUNetConfig.register_preset("residual")
def half_unet_residual_type(cfg: HalfUNetConfig):
    """
    Apply a residual connection to all encoder blocks in the HalfUNet configuration.
    
    The residual connection type is set to "add" for all encoder blocks, 
    which means that the output of each encoder block will be added to its input before being passed to the next layer. 
    This can help with gradient flow and improve training stability.
    """
    cfg.encoder_block_cfg = [
        c.apply_presets(["0_residual"]) for c in cfg.encoder_block_cfg
    ]
    return cfg


@HalfUNetConfig.register_preset("dilated_bottleneck")
def unet_residual_type(cfg: HalfUNetConfig):
    """
    Apply dilated convolutions to the bottleneck (lowest resolution) block in the HalfUNet configuration.

    This change modifies the last encoder block to use dilated convolutions, which can help increase the receptive field without increasing the number of parameters.
    """
    cfg.encoder_block_cfg[-1] = cfg.encoder_block_cfg[-1].apply_presets(["dilated_convs"]
    )
    return cfg


def _apply_preset_to_all_blocks(cfg: HalfUNetConfig, preset_name: str):
    """
    Apply a given preset to all blocks in the HalfUNet configuration.
    """
    cfg.stem_block_cfg = cfg.stem_block_cfg.apply_presets([preset_name])
    cfg.encoder_block_cfg = [c.apply_presets([preset_name]) for c in cfg.encoder_block_cfg]
    cfg.out_block_cfg = cfg.out_block_cfg.apply_presets([preset_name])
    return cfg


@HalfUNetConfig.register_preset("recon")
def reconstruction_config(cfg: HalfUNetConfig):
    """
    Apply a reconstruction preset to all blocks in the HalfUNet configuration.

    This preset is typically used for image reconstruction or superresolution tasks, 
    where the output is expected to be a continuous value (e.g., pixel intensity).
    """
    _apply_preset_to_all_blocks(cfg, "recon")
    cfg.out_activation = None
    return cfg


@HalfUNetConfig.register_preset("single_class_segmentation")
def segmentation_config(cfg: HalfUNetConfig):
    """
    Apply a single-class segmentation preset to all blocks in the HalfUNet configuration.

    Use this preset for binary segmentation tasks, where the output is expected to be a probability map for a single class.
    """
    _apply_preset_to_all_blocks(cfg, "segmentation")
    cfg.out_block_cfg.out_activation = "sigmoid"
    return cfg

@HalfUNetConfig.register_preset("multiclass_segmentation")
def segmentation_config(cfg: HalfUNetConfig):
    """
    Apply a multi-class segmentation preset to all blocks in the HalfUNet configuration.

    Use this preset for multi-class segmentation tasks, where the output is expected to be a probability map for multiple classes.
    """
    _apply_preset_to_all_blocks(cfg, "segmentation")
    cfg.out_block_cfg.out_activation = "softmax"
    return cfg


@HalfUNetConfig.register_preset("depthwise_separable")
def depthwise_separable_config(cfg: HalfUNetConfig):
    """
    Apply a depthwise separable convolution (see `im2sim.layers.DepthwiseSeparableConv`) preset to all encoder blocks in the HalfUNet configuration.
    """
    _apply_preset_to_all_blocks(cfg, "depthwise_separable")
    return cfg


@HalfUNetConfig.register_preset("ghost_depthwise")
def ghost_dw_config(cfg: HalfUNetConfig):
    """
    Apply a ghost depthwise convolution (see `im2sim.layers.GhostConv`) preset to all encoder blocks in the HalfUNet configuration.
    """
    _apply_preset_to_all_blocks(cfg, "ghost_depthwise")
    return cfg


@HalfUNetConfig.register_preset("ghost_depthwise_separable")
def ghost_dws_config(cfg: HalfUNetConfig):
    """
    Apply a ghost depthwise separable convolution (see `im2sim.layers.GhostConv`) preset to all encoder blocks in the HalfUNet configuration.
    """
    _apply_preset_to_all_blocks(cfg, "ghost_depthwise_separable")
    return cfg


@HalfUNetConfig.register_preset("ECA")
def eca_config(cfg: HalfUNetConfig):
    """
    Apply an Efficient Channel Attention (ECA) (see `im2sim.layers.EfficientChannelAttn`) preset to all encoder blocks in the HalfUNet configuration.
    """
    cfg.encoder_block_cfg = [
        c.apply_presets(["ECA"]) for c in cfg.encoder_block_cfg
    ]
    return cfg


@HalfUNetConfig.register_preset("SE")
def squeeze_excitation_config(cfg: HalfUNetConfig):
    """
    Apply a Squeeze-and-Excitation (SE) (see `im2sim.layers.SqueezeExcite`) preset to all encoder blocks in the HalfUNet configuration.
    """
    cfg.encoder_block_cfg = [
        c.apply_presets(["SE"]) for c in cfg.encoder_block_cfg
    ]
    return cfg


if __name__ == "__main__":

    def cfg_print(cfg):
        print("###################")
        for field in fields(cfg):
            value = getattr(cfg, field.name)
            print(f"  {field.name}: {value}")

    cfg = HalfUNetConfig(num_downsamples=3, hidden_channels=64)
    print(cfg.generate_preset_docs())
    # cfg = cfg.apply_presets(["single_class_segmentation", "residual", "SE", "ghost_depthwise_separable"])
    # model = HalfUNet.build(
    #     rank=3,
    #     in_channels=20,
    #     out_channels=1,
    #     cfg=cfg,
    # )
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Parameters: {total_params:,}")

    # print(model)

    # x = torch.randn(1, 20, 64, 64, 64)  # Example input for a 3D tensor

    # def check_gradients(model, x):
    #     model.train()

    #     # Ensure input tracks gradients if you want to test input gradients
    #     x = x.requires_grad_(True)

    #     # Forward
    #     y = model(x)

    #     # Use a scalar loss
    #     loss = y.sum()

    #     # Backward
    #     loss.backward()

    #     # Check input gradient
    #     assert x.grad is not None, "Input gradient is None"
    #     assert torch.isfinite(x.grad).all(), "Input gradient contains NaN/Inf"

    #     # Check parameter gradients
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             assert param.grad is not None, f"{name} gradient is None"
    #             assert torch.isfinite(param.grad).all(), f"{name} gradient contains NaN/Inf"
    #             assert param.grad.abs().sum() > 0, f"{name} gradient is zero"

    #     return True

    # # Check gradients
    # if check_gradients(model, x):
    #     print("Gradient check passed.")
