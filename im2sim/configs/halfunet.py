from dataclasses import dataclass, field
from copy import deepcopy

from im2sim.configs.imageconv import ImageConvBlockConfig
from im2sim.configs.core import Config, LayerConfig, register_config


@register_config
@dataclass
class HalfUNetConfig(Config):
    """
    Configuration class for the HalfUNet model (see `im2sim.models.HalfUNet`).

    For features like serialisation and saving/loading see `im2sim._internal.Config`.

    Args:
        hidden_channels (int):
            Number of hidden channels in the HalfUNet. Default is `64`.

        n_levels (int):
            Number of levels in the HalfUNet. Default is `3`.

        pool_cfg (LayerConfig | list[LayerConfig]):
            Configuration for pooling layers. Can be a single LayerConfig or a list of LayerConfigs for each level starting with the highest resolution.
            Default is `MaxPool` with kernel size `2`.

        upsample_cfg (LayerConfig | list[LayerConfig]):
            Configuration for upsampling layers. Can be a single LayerConfig or a list of LayerConfigs for each level starting with the lowest resolution.
            Upsamples are applied sequentially from the lowest resolution to the highest resolution. So at each stage, the upsample should only increase the resolution to the next higher level.
            Default is `Upsample` with scale factor `2` and mode is adjusted based on rank (`bilinear` for 2D, `trilinear` for 3D).

        block_cfg (ImageConvBlockConfig):
            Configuration for convolutional blocks. Used as a default for encoder blocks.
            Default is a standard convolutional block with `3x3` kernels, stride `1`, padding `1`, and `ReLU` activation.

        stem_block_cfg (ImageConvBlockConfig | None):
            Configuration for the stem block. Can only be a single ImageConvBlockConfig.
            If None, uses block_cfg for the stem block.

        encoder_block_cfg (list[ImageConvBlockConfig] | ImageConvBlockConfig | None):
            Configuration for encoder blocks. Can be a single ImageConvBlockConfig or a list of ImageConvBlockConfigs for each level starting with the highest resolution.
            If None, uses block_cfg for all levels.

        out_block_cfg (ImageConvBlockConfig | None):
            Configuration for the output block. Can only be a single ImageConvBlockConfig.
            If None, uses a single convolution operation taken from block_cfg.

        encoder_blocks_per_level (int):
             Number of convolutional blocks per level in the encoder. Default is `1`.

        fusion_type (str):
            Type of residual connection for skip connections. Options are `'add'` or `'concat'`. Default is `'add'`.

        out_activation (str | None):
            Activation function for the output block. If None, no activation is applied.

    Examples:

        To create a customised configuration for a HalfUNet, you can create a preferred ImageConvBlockConfig and use it for the encoder blocks:

        ..  code-block:: python

            block_cfg = ImageConvBlockConfig(
                                    depth=4,
                                    activation="LeakyReLU",
                                    out_activation="sigmoid",
                                    conv_cfg=LayerConfig(name="Conv", kwargs={"kernel_size": 5, "padding": "same"}),
                                    norm_cfg=LayerConfig(name="BatchNorm", kwargs={"affine": True}),
                                    dropout_cfg=LayerConfig(name="Dropout", kwargs={"p": 0.5}),
                                    attn_cfg=LayerConfig(name="SqueezeExcite", kwargs={}),
                                    dropout_position=[1, 3],
                                    residual_connections={3: [0, 1]},
                                    residual_type="concat"
                                    )
            cfg = HalfUNetConfig(
                hidden_channels=32,
                n_levels=4,
                pool_cfg=LayerConfig(name="MaxPool", kwargs={"kernel_size": 2}),
                upsample_cfg=LayerConfig(name="Upsample", kwargs={"scale_factor": 2, "mode": "bilinear"}),
                block_cfg=block_cfg)

        If you want more flexibility you can also specify different configurations for the encoder blocks:

        ..  code-block:: python

            encoder_block_cfg = [
                ImageConvBlockConfig(depth=5, activation="GELU", out_activation="sigmoid"),
                ImageConvBlockConfig(depth=4, activation="ELU", out_activation="sigmoid"),
                ImageConvBlockConfig(depth=3, activation="LeakyReLU", out_activation="sigmoid"),
                ImageConvBlockConfig(depth=2, activation="ReLU", out_activation="sigmoid")
            ]
            cfg = HalfUNetConfig(
               hidden_channels=32,
               n_levels=4,
               encoder_block_cfg=encoder_block_cfg
            )

        Similarly, if you want to specify different configurations for the stem block and output block, you can do so:

        ..  code-block:: python

            stem_cfg = ImageConvBlockConfig(depth=1, activation="ReLU", out_activation="sigmoid")
            out_block_cfg = ImageConvBlockConfig(depth=1, activation="ReLU", out_activation="sigmoid")
            cfg = HalfUNetConfig(
               hidden_channels=32,
               n_levels=4,
               stem_block_cfg=stem_cfg,
               out_block_cfg=out_block_cfg
            )


        The `mod()` method can be especially useful for making simple modifications to the block configurations for different types of blocks.

        ..  code-block:: python

            block_cfg = ImageConvBlockConfig(depth=3, activation="ReLU", out_activation="sigmoid")
            encoder_block_cfg = [block_cfg.mod(depth=4) for _ in range(4)]
            cfg = HalfUNetConfig(
               hidden_channels=32,
               n_levels=4,
               encoder_block_cfg=encoder_block_cfg
            )

        You can also use the `mod()` method to modify a HalfUNetConfig object directly, which will apply the modification to all blocks of that type:

        ..  code-block:: python

            cfg = HalfUNetConfig(
               hidden_channels=32,
               n_levels=4,
               block_cfg=ImageConvBlockConfig(depth=3, activation="ReLU", out_activation="sigmoid")
               )
            cfg = cfg.mod(encoder_block_cfg=ImageConvBlockConfig(depth=4))

        For heterogeneous pooling and upsampling layers, you can specify a list of LayerConfig objects for each level:

        ..  code-block:: python

            pool_cfg = [
               LayerConfig(name="MaxPool", kwargs={"kernel_size": (1,2)}),
               LayerConfig(name="AvgPool", kwargs={"kernel_size": 2}),
               LayerConfig(name="MaxPool", kwargs={"kernel_size": 2}),
            ]
            upsample_cfg = [
               LayerConfig(name="Upsample", kwargs={"scale_factor": 2, "mode": "bilinear"}),
               LayerConfig(name="Upsample", kwargs={"scale_factor": 2, "mode": "bilinear"}),
               LayerConfig(name="Upsample", kwargs={"scale_factor": (1,2), "mode": "bilinear"}),
            ]
            cfg = HalfUNetConfig(
               hidden_channels=32,
               n_levels=4,
               pool_cfg=pool_cfg,
               upsample_cfg=upsample_cfg
            )

        To make simple modifications to the default configuration, you can modify a subset of attributes:

            cfg = HalfUNetConfig(fusion_type='add', out_activation='sigmoid')


        Different presets exist to perform common transformations on the configuration.
        For example, to convert the convolutional blocks to use depthwise separable convolutions and add residual connections to the input of each block, you can do:

            cfg = HalfUNetConfig().to_depthwise_separable().add_residual()

        To save the configuration to a YAML file and load it back, you can use:

            cfg.save("my_config.yaml")
            loaded_cfg = HalfUNetConfig.load("my_config.yaml")

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
    encoder_block_cfg: list[ImageConvBlockConfig] | ImageConvBlockConfig | None = None

    out_block_cfg: ImageConvBlockConfig | None = None
    encoder_blocks_per_level: int = 1
    fusion_type: str = "add"
    out_activation: str | None = None

    def __post_init__(self):

        L = self.n_levels

        # stem block configs
        if self.stem_block_cfg is None:
            self.stem_block_cfg = deepcopy(self.block_cfg)
            self.stem_block_cfg = self.stem_block_cfg.to_single_block()

        # Encoder configs
        if self.encoder_block_cfg is None:
            self.encoder_block_cfg = [deepcopy(self.block_cfg) for _ in range(L)]
        elif isinstance(self.encoder_block_cfg, ImageConvBlockConfig):
            self.encoder_block_cfg = [self.encoder_block_cfg] * L
        else:
            assert len(self.encoder_block_cfg) == L, (
                f"Length of encoder_block_cfg ({len(self.encoder_block_cfg)}) must be equal to number of levels ({L})"
            )

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
        Apply a residual connection to all encoder blocks in the HalfUNet configuration.

        The residual connection type is set to "add" for all encoder blocks,
        which means that the output of the first convolutional layer in each block will be added to the output of the last convolutional layer in that block.
        This can help with gradient flow and improve training stability.
        """

        for e in self.encoder_block_cfg[1:]:
            e.add_input_residual()

        return self

    def dilate_bottleneck(self, dilation: int = 2):
        """
        Apply dilated convolutions to the bottleneck (lowest resolution) block in the HalfUNet configuration.

        This change modifies the last encoder block to use dilated convolutions, which can help increase the receptive field without increasing the number of parameters.

        Args:
            dilation (int): the dilation rate to use for the convolutions in the bottleneck block. Default is 2.
        """
        self.encoder_block_cfg[-1] = self.encoder_block_cfg[-1].dilate_convs(dilation)
        return self

    def double_bottleneck(self):
        """
        Apply a double bottleneck configuration to the HalfUNet.

        This change modifies the last encoder block to have two convolutional layers instead of one, which can help increase the capacity of the model.
        """
        self.encoder_block_cfg[-1].depth = self.encoder_block_cfg[-1].depth * 2
        return self

    def reconstruction_mode(self):
        """
        Apply a reconstruction preset to all blocks in the HalfUNet configuration.

        This preset is typically used for image reconstruction or superresolution tasks,
        where the output is expected to be a continuous value (e.g., pixel intensity).
        """
        self.out_activation = None
        self.out_block_cfg.out_activation = None
        self.out_block_cfg = self.out_block_cfg.reconstruction_mode()
        for e in self.encoder_block_cfg:
            e = e.reconstruction_mode()
        return self

    def segmentation_mode(self):
        self.out_block_cfg.out_activation = None
        self.out_block_cfg = self.out_block_cfg.segmentation_mode()
        for e in self.encoder_block_cfg:
            e = e.segmentation_mode()
        return self

    def single_class_segmentation_mode(self):
        """
        Apply a single-class segmentation preset to all blocks in the HalfUNet configuration.

        Use this preset for binary segmentation tasks, where the output is expected to be a probability map for a single class.
        """
        self.segmentation_mode()
        self.out_activation = "sigmoid"
        self.out_block_cfg.out_activation = "sigmoid"
        return self

    def multiclass_segmentation_mode(self):
        """
        Apply a multi-class segmentation preset to all blocks in the HalfUNet configuration.

        Use this preset for multi-class segmentation tasks, where the output is expected to be a probability map for multiple classes.
        """
        self.segmentation_mode()
        self.out_activation = "softmax"
        self.out_block_cfg.out_activation = "softmax"
        return self

    def to_depthwise_separable(self):
        """
        Apply a depthwise separable convolution (see `im2sim.layers.DepthwiseSeparableConv`) preset to stem + all encoder blocks in the HalfUNet configuration.
        """
        self.stem_block_cfg.conv_cfg.name = "DepthwiseSeparableConv"
        for e in self.encoder_block_cfg:
            e.conv_cfg.name = "DepthwiseSeparableConv"
        return self

    def to_ghost_depthwise(self):
        """
        Apply a ghost depthwise convolution (see `im2sim.layers.GhostConv`) preset to stem + all encoder blocks in the HalfUNet configuration.
        """
        self.stem_block_cfg.conv_cfg.name = "GhostConv"
        for e in self.encoder_block_cfg:
            e.conv_cfg.name = "GhostConv"
        return self

    def to_ghost_depthwise_separable(self):
        """
        Apply a ghost depthwise separable convolution (see `im2sim.layers.GhostConv`) preset to stem + all encoder blocks in the HalfUNet configuration.
        """
        self.stem_block_cfg.conv_cfg.name = "GhostConv"
        for e in self.encoder_block_cfg:
            e.conv_cfg.name = "GhostConv"
            e.conv_cfg.kwargs["separable"] = True
        return self

    def add_eca(self):
        """
        Apply an Efficient Channel Attention (ECA) (see `im2sim.layers.EfficientChannelAttn`) preset to all encoder blocks in the HalfUNet configuration.
        """
        for e in self.encoder_block_cfg:
            e.add_eca()

        return self

    def add_se(self):
        """
        Apply a Squeeze-and-Excitation (SE) (see `im2sim.layers.SqueezeExcite`) preset to all encoder blocks in the HalfUNet configuration.
        """
        for e in self.encoder_block_cfg:
            e.add_se()
<<<<<<< HEAD:im2sim/src/layers/halfunet.py
        return self


@api_util.export("models.HalfUNet")
class HalfUNet(torch.nn.Module):
    """
    A flexible HalfUNet [1] implementation for image segmentation and reconstruction tasks.

    Args:
        in_channels (int):
            Number of input channels.

        out_channels (int):
            Number of output channels.

        rank (int):
            Spatial rank (1D, 2D, 3D).

        cfg (HalfUNetConfig):
            Configuration object for the HalfUNet.

    Examples:

        To create a HalfUNet model with a specific configuration, you can first create a HalfUNetConfig object and then pass it to the HalfUNet constructor.
        For example, to create a HalfUNet with 3 levels of depth, ReLU activation, and softmax output activation:

        ..  code-block:: python

            cfg = HalfUNetConfig(
                       hidden_channels=32,
                       n_levels=3,
                       encoder_block_cfg=ImageConvBlockConfig(depth=3, activation="ReLU"),
                       out_block_cfg=ImageConvBlockConfig(depth=1, activation="ReLU", out_activation="softmax")
                  )

        Since the configs are rankless, you could use the same config for a 1D, 2D, or 3D convolutional block by changing the rank parameter when creating the HalfUNet instance.

        ..  code-block:: python
        
            model1D = HalfUNet(
                   rank=1,
                   in_channels=32,
                   out_channels=32,
                   cfg=cfg,
               )
            model2D = HalfUNet(
                   rank=2,
                   in_channels=32,
                   out_channels=32,
                   cfg=cfg,
               )
            model3D = HalfUNet(
                   rank=3,
                   in_channels=32,
                   out_channels=32,
                   cfg=cfg,
               )

        The HalfUNet model can be used for both segmentation and reconstruction tasks.
        For segmentation, you can use the `single_class_segmentation_mode()` or `multiclass_segmentation_mode()` methods of the HalfUNetConfig to set the appropriate output activation function (sigmoid for single-class, softmax for multi-class).
        For reconstruction tasks, you can use the `reconstruction_mode()` method to set the output activation to None.

        ..  code-block:: python
        
            cfg_segmentation = HalfUNetConfig().single_class_segmentation_mode()
            model_segmentation = HalfUNet(
                   rank=2,
                   in_channels=32,
                   out_channels=1,
                   cfg=cfg_segmentation,
               )

            cfg_reconstruction = HalfUNetConfig().reconstruction_mode()
            model_reconstruction = HalfUNet(
                   rank=2,
                   in_channels=32,
                   out_channels=1,
                   cfg=cfg_reconstruction,
               )


        Models can be saved and loaded using the standard PyTorch methods:

        ..  code-block:: python

            torch.save(model.state_dict(), "model.pth")
            model.load_state_dict(torch.load("model.pth"))

        Configs can also be saved and loaded using the methods provided in the `im2sim.configs.UNetConfig` class:

        ..  code-block:: python

            cfg.save("my_config.yaml")
            loaded_cfg = HalfUNetConfig.load("my_config.yaml")
            model = HalfUNet(
                rank=2,
                in_channels=32,
                out_channels=32,
                cfg=loaded_cfg,
            )



    References:
        .. [1] H. Lu, Y. She, J. Tie, and S. Xu, Half-UNet: A Simplified HalfUNet Architecture for Medical Image Segmentation,
            Front. Neuroinformatics, vol. 16, Jun. 2022, doi: 10.3389/fninf.2022.911679.


    """

    def __init__(self, in_channels: int, out_channels: int, rank: int, cfg: HalfUNetConfig):
        """ """
        super().__init__()

        self.n_levels = cfg.n_levels
        self.hidden_channels = cfg.hidden_channels

        # ---- pooling / upsampling ----
        pool_cfg = cfg.pool_cfg
        upsample_cfg = cfg.upsample_cfg

        self.pools = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()

        # ---- stem ----

        self.stem = ImageConvBlock(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            rank=rank,
            cfg=cfg.stem_block_cfg,
        )

        # ---- encoder ----
        self.encoders = torch.nn.ModuleList()
        for i in range(self.n_levels):
            block = torch.nn.Sequential(
                *[
                    ImageConvBlock(
                        in_channels=self.hidden_channels,
                        out_channels=self.hidden_channels,
                        rank=rank,
                        cfg=cfg.encoder_block_cfg[i],
                    )
                    for _ in range(cfg.encoder_blocks_per_level)
                ]
            )

            self.encoders.append(block)

            if i < self.n_levels - 1:
                # we need to pass in the in_channels and out_channels to the pooling layer, as some pooling layers (e.g., strided conv) require them
                pool = call_with_supported_kwargs(
                    get_image_layer(pool_cfg[i].name, rank),
                    {
                        "in_channels": self.hidden_channels,
                        "out_channels": self.hidden_channels,
                        **pool_cfg[i].kwargs,
                    },
                )
                self.pools.append(pool)

        # ---- upsampling ----
        for i in reversed(range(self.n_levels - 1)):
            channels = (
                self.hidden_channels
                if cfg.fusion_type == "add"
                else self.hidden_channels * (self.n_levels - i - 1)
            )
            up = call_with_supported_kwargs(
                get_image_layer(upsample_cfg[i].name, rank),
                {"in_channels": channels, "out_channels": channels, **upsample_cfg[i].kwargs},
            )
            self.ups.append(up)

        # ---- out_conv ----
        self.fusion_type = cfg.fusion_type

        self.out_block = ImageConvBlock(
            in_channels=self.hidden_channels
            if self.fusion_type == "add"
            else self.hidden_channels * self.n_levels,
            out_channels=out_channels,
            rank=rank,
            cfg=cfg.out_block_cfg,
        )

    def forward(self, x):
        """
        Forward pass through the HalfUNet.
        """

        x = self.stem(x)
        enc_outputs = []
        # ---- encoder ----
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            enc_outputs.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)

        # ---- fusion ----
        for up, enc in zip(self.ups, reversed(enc_outputs[:-1]), strict=True):
            x = up(x)
            x = apply_residual_connection(x, enc, connection_type=self.fusion_type)

        output = self.out_block(x)
        return output


if __name__ == "__main__":

    def cfg_print(cfg):
        print("###################")
        for field in fields(cfg):
            value = getattr(cfg, field.name)
            print(f"  {field.name}: {value}")

    cfg = HalfUNetConfig(
        upsample_cfg=LayerConfig(name="ConvTranspose", kwargs={"kernel_size": 2, "stride": 2}),
        fusion_type="concat",
    )

    # cfg = cfg.apply_presets(["single_class_segmentation", "residual", "SE", "ghost_depthwise_separable"])
    model = HalfUNet(rank=2, in_channels=20, out_channels=1, cfg=cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    print(model)

    x = torch.randn(1, 20, 64, 64)  # Example input for a 3D tensor

    def check_gradients(model, x):
        model.train()

        # Ensure input tracks gradients if you want to test input gradients
        x = x.requires_grad_(True)

        print(x.shape)
        # Forward
        y = model(x)
        print(y.shape)

        # Use a scalar loss
        loss = sum([output.sum() for output in y]) if isinstance(y, list) else y.sum()

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

    # Check gradients
    if check_gradients(model, x):
        print("Gradient check passed.")
=======
        return self
>>>>>>> origin/restructure:im2sim/configs/halfunet.py
