from copy import deepcopy
from dataclasses import dataclass, field

from im2sim.configs.core import Config, LayerConfig, register_config
from im2sim.configs.imageconv import ImageConvBlockConfig


@register_config
@dataclass
class ReverseHalfUNetConfig(Config):
    """
    Configuration class for the ReverseHalfUNet model (see `im2sim.models.ReverseHalfUNet`).

    For features like serialisation and saving/loading see `im2sim._internal.Config`.

    Args:
        hidden_channels (int):
            Number of hidden channels. Default is `64`.

        n_levels (int):
            Number of levels in the Reverse ReverseHalfUNet. Default is `3`.

        pool_cfg (LayerConfig | list[LayerConfig]):
            Configuration for pooling layers. Can be a single LayerConfig or a list of LayerConfigs for each level starting with the highest resolution.
            Default is `MaxPool` with kernel size `2`.

        upsample_cfg (LayerConfig | list[LayerConfig]):
            Configuration for upsampling layers. Can be a single LayerConfig or a list of LayerConfigs for each level starting with the lowest resolution.
            Upsamples are applied sequentially from the lowest resolution to the highest resolution. So at each stage, the upsample should only increase the resolution to the next higher level.
            Default is `Upsample` with scale factor `2` and mode is adjusted based on rank (`bilinear` for 2D, `trilinear` for 3D).

        block_cfg (ImageConvBlockConfig):
            Configuration for convolutional blocks. Used as a default for decoder blocks.
            Default is a standard convolutional block with `3x3` kernels, stride `1`, padding `1`, and `ReLU` activation.

        stem_block_cfg (ImageConvBlockConfig | None):
            Configuration for the stem block. Can only be a single ImageConvBlockConfig.
            If None, uses block_cfg for the stem block.

        decoder_block_cfg (list[ImageConvBlockConfig] | ImageConvBlockConfig | None):
            Configuration for decoder blocks. Can be a single ImageConvBlockConfig or a list of ImageConvBlockConfigs for each level starting with the lowest resolution.
            If None, uses block_cfg for all levels.

        out_block_cfg (ImageConvBlockConfig | None):
            Configuration for the output block. Can only be a single ImageConvBlockConfig.
            If None, uses a single convolution operation taken from block_cfg.

        decoder_blocks_per_level (int):
            Number of convolutional blocks per level in the decoder.

        fusion_type (str):
            Type of residual connection for skip connections. Options are `'add'` or `'concat'`. Default is `'add'`.

        out_activation (str | None):
            Activation function for the output block. If None, no activation is applied.

    Examples:

        To create a customised configuration for a ReverseHalfUNet, you can create a preferred ImageConvBlockConfig and use it for the decoder blocks:

        >>> block_cfg = ImageConvBlockConfig(
        >>>                         depth=4,
        >>>                         activation="LeakyReLU",
        >>>                         out_activation="sigmoid",
        >>>                         conv_cfg=LayerConfig(name="Conv", kwargs={"kernel_size": 5, "padding": "same"}),
        >>>                         norm_cfg=LayerConfig(name="BatchNorm", kwargs={"affine": True}),
        >>>                         dropout_cfg=LayerConfig(name="Dropout", kwargs={"p": 0.5}),
        >>>                         attn_cfg=LayerConfig(name="SqueezeExcite", kwargs={}),
        >>>                         dropout_position=[1, 3],
        >>>                         residual_connections={3: [0, 1]},
        >>>                         residual_type="concat"
        >>>                         )
        >>> cfg = ReverseHalfUNetConfig(
        >>>     hidden_channels=32,
        >>>     n_levels=4,
        >>>     pool_cfg=LayerConfig(name="MaxPool", kwargs={"kernel_size": 2}),
        >>>     upsample_cfg=LayerConfig(name="Upsample", kwargs={"scale_factor": 2, "mode": "bilinear"}),
        >>>     block_cfg=block_cfg)

        If you want more flexibility you can also specify different configurations for the decoder blocks:


        >>> decoder_block_cfg = [
        >>>     ImageConvBlockConfig(depth=5, activation="GELU", out_activation="sigmoid"),
        >>>     ImageConvBlockConfig(depth=4, activation="ELU", out_activation="sigmoid"),
        >>>     ImageConvBlockConfig(depth=3, activation="LeakyReLU", out_activation="sigmoid"),
        >>>     ImageConvBlockConfig(depth=2, activation="ReLU", out_activation="sigmoid")
        >>> ]
        >>> cfg = ReverseHalfUNetConfig(
        >>>    hidden_channels=32,
        >>>    n_levels=4,
        >>>    decoder_block_cfg=decoder_block_cfg
        >>> )

        Similarly, if you want to specify different configurations for the stem block and output block, you can do so:

        >>> stem_cfg = ImageConvBlockConfig(depth=1, activation="ReLU", out_activation="sigmoid")
        >>> out_block_cfg = ImageConvBlockConfig(depth=1, activation="ReLU", out_activation="sigmoid")
        >>> cfg = ReverseHalfUNetConfig(
        >>>    hidden_channels=32,
        >>>    n_levels=4,
        >>>    stem_block_cfg=stem_cfg,
        >>>    out_block_cfg=out_block_cfg
        >>> )


        The `mod()` method can be especially useful for making simple modifications to the block configurations for different types of blocks.

        >>> block_cfg = ImageConvBlockConfig(depth=3, activation="ReLU", out_activation="sigmoid")
        >>> decoder_block_cfg = [block_cfg.mod(depth=4) for _ in range(4)]
        >>> cfg = ReverseHalfUNetConfig(
        >>>    hidden_channels=32,
        >>>    n_levels=4,
        >>>    decoder_block_cfg=decoder_block_cfg
        >>> )

        You can also use the `mod()` method to modify a ReverseHalfUNetConfig object directly, which will apply the modification to all blocks of that type:
        >>> cfg = ReverseHalfUNetConfig(
        >>>    hidden_channels=32,
        >>>    n_levels=4,
        >>>    block_cfg=ImageConvBlockConfig(depth=3, activation="ReLU", out_activation="sigmoid")
        >>>    )
        >>> cfg = cfg.mod(decoder_block_cfg=ImageConvBlockConfig(depth=4))

        For heterogeneous pooling and upsampling layers, you can specify a list of LayerConfig objects for each level:

        >>> pool_cfg = [
        >>>    LayerConfig(name="MaxPool", kwargs={"kernel_size": (1,2)}),
        >>>    LayerConfig(name="AvgPool", kwargs={"kernel_size": 2}),
        >>>    LayerConfig(name="MaxPool", kwargs={"kernel_size": 2}),
        >>> ]
        >>> upsample_cfg = [
        >>>    LayerConfig(name="Upsample", kwargs={"scale_factor": 2, "mode": "bilinear"}),
        >>>    LayerConfig(name="Upsample", kwargs={"scale_factor": 2, "mode": "bilinear"}),
        >>>    LayerConfig(name="Upsample", kwargs={"scale_factor": (1,2), "mode": "bilinear"}),
        >>> ]
        >>> cfg = ReverseHalfUNetConfig(
        >>>    hidden_channels=32,
        >>>    n_levels=4,
        >>>    pool_cfg=pool_cfg,
        >>>    upsample_cfg=upsample_cfg
        >>> )

        To make simple modifications to the default configuration, you can modify a subset of attributes:

        >>> cfg = ReverseHalfUNetConfig(fusion_type='add', out_activation='sigmoid')


        Different presets exist to perform common transformations on the configuration.
        For example, to convert the convolutional blocks to use depthwise separable convolutions and add residual connections to the input of each block, you can do:

        >>> cfg = ReverseHalfUNetConfig().to_depthwise_separable().add_residual()

        To save the configuration to a YAML file and load it back, you can use:

        >>> cfg.save("my_config.yaml")
        >>> loaded_cfg = ReverseHalfUNetConfig.load("my_config.yaml")

        Refer to the methods below to see all available transformations that can be applied to the configuration.
    """

    hidden_channels: int = 64

    n_levels: int = 3

    pool_cfg: LayerConfig | list[LayerConfig] = field(
        default_factory=lambda: LayerConfig(name="MaxPool", kwargs={"kernel_size": 2})
    )
    upsample_cfg: LayerConfig | list[LayerConfig] = field(
        default_factory=lambda: LayerConfig(
            name="Upsample", kwargs={"scale_factor": 2, "mode": "trilinear"}
        )
    )

    block_cfg: ImageConvBlockConfig = field(default_factory=lambda: ImageConvBlockConfig())

    stem_block_cfg: ImageConvBlockConfig | None = None
    decoder_block_cfg: list[ImageConvBlockConfig] | ImageConvBlockConfig | None = None
    out_block_cfg: ImageConvBlockConfig | None = None

    decoder_blocks_per_level: int = 1

    fusion_type: str = "add"
    out_activation: str | None = None

    def __post_init__(self):

        L = self.n_levels

        # stem block configs
        if self.stem_block_cfg is None:
            self.stem_block_cfg = deepcopy(self.block_cfg)
            self.stem_block_cfg = self.stem_block_cfg.to_single_block()

        # Decoder configs
        if self.decoder_block_cfg is None:
            self.decoder_block_cfg = [deepcopy(self.block_cfg) for _ in range(L)]
        elif isinstance(self.decoder_block_cfg, ImageConvBlockConfig):
            self.decoder_block_cfg = [self.decoder_block_cfg] * L
        else:
            assert len(self.decoder_block_cfg) == L, (
                f"Length of decoder_block_cfg ({len(self.decoder_block_cfg)}) must be equal to number of levels ({L})"
            )
        self.decoder_block_cfg = list(reversed(self.decoder_block_cfg))

        if self.out_block_cfg is None:
            self.out_block_cfg = self.block_cfg.to_single_conv()
            self.out_block_cfg.conv_cfg.kwargs.update({"kernel_size": 1, "stride": 1, "padding": 0})
            self.out_block_cfg.out_activation = self.out_activation

        if isinstance(self.pool_cfg, LayerConfig):
            self.pool_cfg = [self.pool_cfg] * (L - 1)
        else:
            assert len(self.pool_cfg) == L - 1, (
                f"Length of pool_cfg ({len(self.pool_cfg)}) must be equal to levels - 1 ({L - 1})"
            )

        if isinstance(self.upsample_cfg, LayerConfig):
            self.upsample_cfg = [self.upsample_cfg] * (L - 1)
        else:
            assert len(self.upsample_cfg) == L - 1, (
                f"Length of upsample_cfg ({len(self.upsample_cfg)}) must be equal to levels - 1 ({L - 1})"
            )
        self.upsample_cfg = list(reversed(self.upsample_cfg))

    def add_residual(self):
        """
        Apply a residual connection to all decoder blocks in the ReverseHalfUNet configuration.

        The residual connection type is set to "add" for all decoder blocks,
        which means that the output of the first convolutional layer in each block will be added to the output of the last convolutional layer in that block.
        This can help with gradient flow and improve training stability.
        """

        for d in self.decoder_block_cfg[1:]:
            d.add_input_residual()

        return self

    def dilate_bottleneck(self, dilation: int = 2):
        """
        Apply dilated convolutions to the bottleneck (lowest resolution) block in the ReverseHalfUNet configuration.

        This change modifies the first decoder block to use dilated convolutions, which can help increase the receptive field without increasing the number of parameters.

        Args:
            dilation (int): the dilation rate for the convolutions in the bottleneck block. Default is 2.
        """
        self.decoder_block_cfg[-1] = self.decoder_block_cfg[-1].dilate_convs(dilation)
        return self

    def double_bottleneck(self):
        """
        Apply a double bottleneck configuration to the ReverseHalfUNet.

        This change modifies the first decoder block to have two convolutional layers instead of one, which can help increase the capacity of the model.
        """
        self.decoder_block_cfg[-1].depth = self.decoder_block_cfg[-1].depth * 2
        return self

    def reconstruction_mode(self):
        """
        Apply a reconstruction preset to all blocks in the ReverseHalfUNet configuration.

        This preset is typically used for image reconstruction or superresolution tasks,
        where the output is expected to be a continuous value (e.g., pixel intensity).
        """
        self.out_activation = None
        self.out_block_cfg.out_activation = None
        self.out_block_cfg = self.out_block_cfg.reconstruction_mode()
        for d in self.decoder_block_cfg:
            d = d.reconstruction_mode()
        return self

    def segmentation_mode(self):
        self.out_block_cfg.out_activation = None
        self.out_block_cfg = self.out_block_cfg.segmentation_mode()
        for d in self.decoder_block_cfg:
            d = d.segmentation_mode()
        return self

    def single_class_segmentation_mode(self):
        """
        Apply a single-class segmentation preset to all blocks in the ReverseHalfUNet configuration.

        Use this preset for binary segmentation tasks, where the output is expected to be a probability map for a single class.
        """
        self.segmentation_mode()
        self.out_activation = "sigmoid"
        self.out_block_cfg.out_activation = "sigmoid"
        return self

    def multiclass_segmentation_mode(self):
        """
        Apply a multi-class segmentation preset to all blocks in the ReverseHalfUNet configuration.

        Use this preset for multi-class segmentation tasks, where the output is expected to be a probability map for multiple classes.
        """
        self.segmentation_mode()
        self.out_activation = "softmax"
        self.out_block_cfg.out_activation = "softmax"
        return self

    def to_depthwise_separable(self):
        """
        Apply a depthwise separable convolution (see `im2sim.layers.DepthwiseSeparableConv`) preset to stem + all decoder blocks in the ReverseHalfUNet configuration.
        """
        self.stem_block_cfg.conv_cfg.name = "DepthwiseSeparableConv"
        for d in self.decoder_block_cfg:
            d.conv_cfg.name = "DepthwiseSeparableConv"
        return self

    def to_ghost_depthwise(self):
        """
        Apply a ghost depthwise convolution (see `im2sim.layers.GhostConv`) preset to stem + all decoder blocks in the ReverseHalfUNet configuration.
        """
        self.stem_block_cfg.conv_cfg.name = "GhostConv"
        for d in self.decoder_block_cfg:
            d.conv_cfg.name = "GhostConv"
        return self

    def to_ghost_depthwise_separable(self):
        """
        Apply a ghost depthwise separable convolution (see `im2sim.layers.GhostConv`) preset to stem + all decoder blocks in the ReverseHalfUNet configuration.
        """
        self.stem_block_cfg.conv_cfg.name = "GhostConv"
        for d in self.decoder_block_cfg:
            d.conv_cfg.name = "GhostConv"
            d.conv_cfg.kwargs["separable"] = True
        return self

    def add_eca(self):
        """
        Apply an Efficient Channel Attention (ECA) (see `im2sim.layers.EfficientChannelAttn`) preset to all decoder blocks in the ReverseHalfUNet configuration.
        """
        for d in self.decoder_block_cfg:
            d.add_eca()

        return self

    def add_se(self):
        """
        Apply a Squeeze-and-Excitation (SE) (see `im2sim.layers.SqueezeExcite`) preset to all decoder blocks in the ReverseHalfUNet configuration.
        """
        for d in self.decoder_block_cfg:
            d.add_se()
        return self
