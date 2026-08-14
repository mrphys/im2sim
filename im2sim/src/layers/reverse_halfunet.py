from copy import deepcopy
from dataclasses import dataclass, field, fields

import torch

from im2sim.src.layers.image_conv_blocks import ImageConvBlock, ImageConvBlockConfig
from im2sim.src.layers.layer_util import (
    apply_residual_connection,
    call_with_supported_kwargs,
    get_image_layer,
)
from im2sim.src.layers.module_config import Config, LayerConfig, register_config
from im2sim.src.utils import api_util


@api_util.export("configs.ReverseHalfUNetConfig")
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


@api_util.export("models.ReverseHalfUNet")
class ReverseHalfUNet(torch.nn.Module):
    """
    A flexible implementation of a 'Reverse HalfUNet', which is inspired by a HalfUNet[1] architecture but
    where the HalfUNet does away with the decoders of a conventional UNet[2], this model instead removes the encoders.
    Can be used for image segmentation and reconstruction tasks where blurring caused by additive fusion as in a HalfUNet is undesirable and/or deep supervision is required.

    Args:
        in_channels (int):
            Number of input channels.

        out_channels (int):
            Number of output channels.

        rank (int):
            Spatial rank (1D, 2D, 3D).

        cfg (ReverseHalfUNetConfig):
            Configuration object for the Reverse ReverseHalfUNet.

        supervision_levels (int | list[int]):
            Levels at which to apply deep supervision. `0` corresponds to the highest resolution output, `1` to the next lower resolution, and so on.
            Default is `0` (no deep supervision).

    Examples:

        To create a ReverseHalfUNet model with a specific configuration, you can first create a ReverseHalfUNetConfig object and then pass it to the ReverseHalfUNet constructor.
        For example, to create a ReverseHalfUNet with 3 levels of depth, ReLU activation, and softmax output activation:

        >>> cfg = ReverseHalfUNetConfig(
        >>>            hidden_channels=32,
        >>>            n_levels=3,
        >>>            decoder_block_cfg=ImageConvBlockConfig(depth=3, activation="ReLU"),
        >>>            out_block_cfg=ImageConvBlockConfig(depth=1, activation="ReLU", out_activation="softmax")
        >>>       )

        Since the configs are rankless, you could use the same config for a 1D, 2D, or 3D convolutional block by changing the rank parameter when creating the ReverseHalfUNet instance.

        >>> model1D = ReverseHalfUNet(
        >>>        rank=1,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )
        >>> model2D = ReverseHalfUNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )
        >>> model3D = ReverseHalfUNet(
        >>>        rank=3,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )

        The ReverseHalfUNet model can be used for both segmentation and reconstruction tasks.
        For segmentation, you can use the `single_class_segmentation_mode()` or `multiclass_segmentation_mode()` methods of the ReverseHalfUNetConfig to set the appropriate output activation function (sigmoid for single-class, softmax for multi-class).
        For reconstruction tasks, you can use the `reconstruction_mode()` method to set the output activation to None.

        >>> cfg_segmentation = ReverseHalfUNetConfig().single_class_segmentation_mode()
        >>> model_segmentation = ReverseHalfUNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=1,
        >>>        cfg=cfg_segmentation,
        >>>    )

        >>> cfg_reconstruction = ReverseHalfUNetConfig().reconstruction_mode()
        >>> model_reconstruction = ReverseHalfUNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=1,
        >>>        cfg=cfg_reconstruction,
        >>>    )

        If deep supervision is desired, you can specify the levels at which to apply it using the `supervision_levels` argument.

        >>> model_deep_supervision = ReverseHalfUNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>        supervision_levels=[0, 1],  # Apply deep supervision at top 2 levels
        >>>    )

        Models can be saved and loaded using the standard PyTorch methods:

        >>> torch.save(model.state_dict(), "model.pth")
        >>> model.load_state_dict(torch.load("model.pth"))

        Configs can also be saved and loaded using the methods provided in the `im2sim.configs.UNetConfig` class:

        >>> cfg.save("my_config.yaml")
        >>> loaded_cfg = ReverseHalfUNetConfig.load("my_config.yaml")
        >>> model = ReverseHalfUNet(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=loaded_cfg,
        >>>    )

    References:
        .. [1] H. Lu, Y. She, J. Tie, and S. Xu, Half-UNet: A Simplified ReverseHalfUNet Architecture for Medical Image Segmentation,
            Front. Neuroinformatics, vol. 16, Jun. 2022, doi: 10.3389/fninf.2022.911679.

        .. [2] O. Ronneberger, P. Fischer, and T. Brox, Reverse ReverseHalfUNet: Convolutional Networks for Biomedical Image Segmentation,
            May 18, 2015, arXiv: arXiv:1505.04597. doi: 10.48550/arXiv.1505.04597.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        cfg: ReverseHalfUNetConfig,
        supervision_levels: int | list[int] = 0,
    ):
        """ """
        super().__init__()

        self.hidden_channels = cfg.hidden_channels
        self.n_levels = cfg.n_levels

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

        # ---- pooling ----
        for i in range(self.n_levels - 1):
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

        # ---- deep supervision ----
        if isinstance(supervision_levels, int):
            supervision_levels = [supervision_levels]

        self.supervision_levels = supervision_levels

        self.fusion_type = cfg.fusion_type

        # ---- decoder ----
        self.decoders = torch.nn.ModuleList()
        self.out_blocks = torch.nn.ModuleList()

        for i in reversed(range(self.n_levels)):
            # in channels is hidden_channels if fusion_type is 'add' or if we are at the last level, otherwise it is hidden_channels * 2 (for concatenation)
            in_ch = (
                self.hidden_channels
                if self.fusion_type.strip().lower() == "add" or i == (self.n_levels - 1)
                else self.hidden_channels * 2
            )

            # upsample layer needs to know the number of channels in the input and output, as some upsampling layers (e.g., transposed conv) require them
            if i > 0:
                up = call_with_supported_kwargs(
                    get_image_layer(upsample_cfg[i - 1].name, rank),
                    {
                        "in_channels": self.hidden_channels,
                        "out_channels": self.hidden_channels,
                        **upsample_cfg[i - 1].kwargs,
                    },
                )
                self.ups.append(up)

            block = torch.nn.Sequential(
                *[
                    ImageConvBlock(
                        in_channels=in_ch if j == 0 else self.hidden_channels,
                        out_channels=self.hidden_channels,
                        rank=rank,
                        cfg=cfg.decoder_block_cfg[i],
                    )
                    for j in range(cfg.decoder_blocks_per_level)
                ]
            )

            self.decoders.append(block)

            # We only add output blocks for the levels specified in supervision_levels
            if i in supervision_levels:
                self.out_blocks.append(
                    ImageConvBlock(
                        in_channels=self.hidden_channels,
                        out_channels=out_channels,
                        rank=rank,
                        cfg=cfg.out_block_cfg,
                    )
                )

    def forward(self, x):
        """
        Forward pass through the Reverse ReverseHalfUNet.
        """
        x = self.stem(x)
        decoder_inputs = [x]

        # ---- pooling ----
        for pool in self.pools:
            x = pool(x)
            decoder_inputs.append(x)

        # ---- decoder ----
        decoder_outputs = []
        ctr = 0
        x = self.decoders[0](x)

        for i, (up, inp, dec) in enumerate(
            zip(self.ups, reversed(decoder_inputs[:-1]), self.decoders[1:], strict=True)
        ):
            if (self.n_levels - i - 1) in self.supervision_levels:
                decoder_outputs.append(self.out_blocks[ctr](x))
                ctr += 1

            x = up(x)

            # dynamic shape alignment
            if x.shape[2:] != inp.shape[2:]:
                inp = torch.nn.functional.interpolate(inp, size=x.shape[2:])

            x = apply_residual_connection(x, inp, connection_type=self.fusion_type)
            x = dec(x)

        if 0 in self.supervision_levels:
            decoder_outputs.append(self.out_blocks[ctr](x))

        # get deep supervision outputs and upsample if required
        out_shape = decoder_outputs[-1].shape[2:]
        for i in range(len(decoder_outputs)):
            if decoder_outputs[i].shape[2:] != out_shape:
                decoder_outputs[i] = torch.nn.functional.interpolate(
                    decoder_outputs[i], size=out_shape
                )

        if len(decoder_outputs) > 1:
            return decoder_outputs

        # if only one output, return it directly
        return decoder_outputs[-1]


if __name__ == "__main__":

    def cfg_print(cfg):
        print("###################")
        for field in fields(cfg):
            value = getattr(cfg, field.name)
            print(f"  {field.name}: {value}")

    model_deep_supervision = ReverseHalfUNet(
        rank=2,
        in_channels=32,
        out_channels=32,
        supervision_levels=[0, 1],  # Apply deep supervision at top 2 levels
    )
    model = model_deep_supervision
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    print(model)

    x = torch.randn(1, 32, 64, 64)  # Example input for a 3D tensor

    def check_gradients(model, x):
        model.train()

        # Ensure input tracks gradients if you want to test input gradients
        x = x.requires_grad_(True)

        print(x.shape)
        # Forward
        y = model(x)
        if isinstance(y, list):
            for output in y:
                print(output.shape)
        else:
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
