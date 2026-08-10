from dataclasses import dataclass, field

import torch
from im2sim.src.layers.custom_image_layers import *
from im2sim.src.layers.layer_util import (
    ResidualConnectionType,
    apply_residual_connection,
    get_activation,
    get_image_layer,
)
from im2sim.src.layers.module_config import Config, ConfigurableModule, LayerConfig, register_config
from im2sim.src.utils import api_util


@api_util.export("configs.ImageConvBlockConfig")
@register_config
@dataclass
class ImageConvBlockConfig(Config):
    """
    Configuration class for defining the parameters of an image convolutional block.
    
    Attributes can either be set directly when creating an instance of the class or modified later.

    Configuration presets can be applied to quickly set up common configurations for different use cases.

    The configuration can also be saved to and loaded from a YAML file.

    Args:

        depth (int):
            The number of convolutional layers in the block. Default is 2.

        activation (str | None): 
            The activation function to use after each convolutional layer. Default is "ReLU".

        out_activation (str | None): 
            The activation function to use after the final layer. Default is None.

        conv_config (LayerConfig): 
            Configuration for the convolutional layers, including kernel size and padding.

        norm_config (LayerConfig): 
            Configuration for the normalization layers, such as InstanceNorm with affine set to True.

        dropout_config (LayerConfig): 
            Configuration for the dropout layers. Default is no dropout.

        attn_config (LayerConfig): 
            Configuration for the attention layers. Default is no attention.

        dropout_position (int | list[int]): 
            Specifies the position(s) of the dropout layers within the block. Default is 1.

        residual_connections ( dict [int, list[ int ]] | None): 
            Specifies the residual connections within the block. 
            The keys represent the target layers, and the values are lists of source layers. Default is None.
            Example: {1: [0]} means that the input to the block will be added to the output of layer 1.

        residual_type (str): 
            The type of residual connection to use (e.g., "add"). Default is ResidualConnectionType.ADD.

    Examples:

        To create a configuration for an image convolutional block with a depth of 3, ReLU activation, and softmax output activation, you can use the following code:

        >>> cfg = ImageConvBlockConfig(depth=3, activation="ReLU", out_activation="softmax")
        

        To apply a preset configuration for a single convolutional layer without normalization or dropout, you can use:

        >>> cfg = ImageConvBlockConfig.apply_presets(cfg, ["single_conv"])

        To save the configuration to a YAML file and load it back, you can use:

        >>> cfg.save("config.yaml")
        >>> loaded_cfg = ImageConvBlockConfig().load("config.yaml")

    """

    depth: int = 2
    activation: str | None = "ReLU"
    out_activation: str | None = None
    conv_config: LayerConfig = field(
        default_factory=lambda: LayerConfig(
            name="Conv", kwargs={"kernel_size": 3, "padding": "same"}
        )
    )
    norm_config: LayerConfig = field(
        default_factory=lambda: LayerConfig(name="InstanceNorm", kwargs={"affine": True})
    )
    dropout_config: LayerConfig = field(default_factory=lambda: LayerConfig(name=None, kwargs={}))
    attn_config: LayerConfig = field(default_factory=lambda: LayerConfig(name=None, kwargs={}))
    dropout_position: int | list[int] = 1
    residual_connections: dict[int, list[int]] = None
    residual_type: str = ResidualConnectionType.ADD


@api_util.export("layers.ImageConvBlock")
class ImageConvBlock(torch.nn.Module, ConfigurableModule):
    """
    A configurable image convolutional block that consists of a sequence of
    convolutional layers, normalization layers, dropout layers, and attention
    layers. The block supports residual connections and allows for flexible
    configuration of its components.

    It is best used by creating a configuration object of type
    :class:ImageConvBlockConfig and then calling the build method to create
    an instance of the block.

    Args:

    in_channels : int
        Number of input channels.

    out_channels : int
        Number of output channels.

    rank : int
        The rank of the convolutional layers (e.g., 2 for 2D convolutions).

    depth : int, default=2
        The number of convolutional layers in the block.

    activation : str | None, default=None
        Activation function applied after each convolutional layer.

    out_activation : str | None, default=None
        Activation function applied after the final layer.

    conv_config : LayerConfig | None, default=None
        Configuration for convolutional layers. If None, a default configuration is used.

    norm_config : LayerConfig | None, default=None
        Configuration for normalization layers. If None, a default configuration is used.

    attn_config : LayerConfig | None, default=None
        Configuration for attention layers. If None, a default configuration is used.

    dropout_config : LayerConfig | None, default=None
        Configuration for dropout layers. If None, a default configuration is used.

    dropout_position : int | list[int], default=1
        Position(s) of dropout layers within the block.

    residual_connections : dict[int, list[int]] | None, default=None
        Specifies residual connections within the block. Keys represent target layers,
        and values are lists of source layers.(e.g. {1: [0]} adds the block input to the output of layer 1.)

    residual_type : str, default=ResidualConnectionType.ADD
        Type of residual connection to use (e.g., "add").



    Example:

    To create an ImageConvBlock with a depth of 3, ReLU activation, and softmax output activation, you can use the following code:

    >>> cfg = ImageConvBlockConfig(depth=3, activation="ReLU", out_activation="softmax")
    >>> model = ImageConvBlock.build(
    >>>        rank=2,
    >>>        in_channels=32,
    >>>        out_channels=32,
    >>>        cfg=cfg,
    >>>    )
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        depth: int = 2,
        activation: str | None = None,
        out_activation: str | None = None,
        conv_config: LayerConfig | None = None,
        norm_config: LayerConfig | None = None,
        attn_config: LayerConfig | None = None,
        dropout_config: LayerConfig | None = None,
        dropout_position: int | list[int] = 1,
        residual_connections: dict[int, list[int]] = None,
        residual_type: str = ResidualConnectionType.ADD,
    ):
        super().__init__()

        print("IN INIT")
        print(conv_config)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.depth = depth
        self.activation = get_activation(activation)
        self.out_activation = get_activation(out_activation)
        self.conv_config = conv_config
        self.norm_config = norm_config
        self.attn_config = attn_config
        self.dropout_config = dropout_config
        self.dropout_position = (
            dropout_position if isinstance(dropout_position, list) else [dropout_position]
        )
        self.residual_connections = residual_connections if residual_connections is not None else {}
        self.residual_type = residual_type

        print(self.dropout_position, self.depth)

        self._set_default_configs()
        self._validate_configs()

        self.layers = torch.nn.ModuleList()
        in_channels_per_layer = [self.in_channels]

        for i in range(depth):
            conv = get_image_layer(self.conv_config.name, rank=self.rank)(
                in_channels=in_channels_per_layer[-1],
                out_channels=out_channels,
                **self.conv_config.kwargs,
            )

            norm = get_image_layer(self.norm_config.name, rank=self.rank)(
                self.out_channels, **self.norm_config.kwargs
            )

            dropout = (
                get_image_layer(self.dropout_config.name, rank=self.rank)(
                    **self.dropout_config.kwargs
                )
                if i in self.dropout_position
                else torch.nn.Identity()
            )

            pre_residual = self.attn_config.name is not None and i in self.residual_connections
            no_residual_final = len(self.residual_connections.keys()) == 0 and i == self.depth - 1
            if pre_residual or no_residual_final:
                attn = get_image_layer(self.attn_config.name, rank=self.rank)(
                    self.out_channels, **self.attn_config.kwargs
                )
            else:
                attn = torch.nn.Identity()

            block = torch.nn.Sequential(
                conv,
                norm,
                dropout,
                attn,
                self.activation if i < self.depth - 1 else torch.nn.Identity(),
            )
            self.layers.append(block)

            in_channels_current = self.out_channels
            if (
                i in self.residual_connections
                and self.residual_type == ResidualConnectionType.CONCAT
            ):
                for src in self.residual_connections[i]:
                    in_channels_current += in_channels_per_layer[src]
            in_channels_per_layer.append(in_channels_current)

    def _set_default_configs(self):
        if self.conv_config is None:
            self.conv_config = LayerConfig(
                name="Conv", kwargs={"kernel_size": 3, "padding": "same"}
            )
        if self.norm_config is None:
            self.norm_config = LayerConfig(name=None, kwargs={})
        if self.dropout_config is None:
            self.dropout_config = LayerConfig(name=None, kwargs={})
        if self.attn_config is None:
            self.attn_config = LayerConfig(name=None, kwargs={})

    def _validate_configs(self):
        assert self.norm_config.name in [None, "BatchNorm", "InstanceNorm"], (
            f"Unsupported norm type: {self.norm_config.name}"
        )
        assert self.attn_config.name in [None, "EfficientChannelAttn", "SqueezeExcite"], (
            f"Unsupported attention type: {self.attn_config.name}"
        )
        if self.dropout_config.name is not None:
            assert max(self.dropout_position) < self.depth, (
                "Dropout position must be less than depth"
            )

    def forward(self, x):
        outputs = [x]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            print(i)
            if i in self.residual_connections:
                for src in self.residual_connections[i]:
                    x = apply_residual_connection(
                        outputs[src], x, connection_type=self.residual_type
                    )
            outputs.append(x)

        x = self.out_activation(x)
        return x



@ImageConvBlockConfig.register_preset("single_conv")
def single_conv_config(cfg: ImageConvBlockConfig):
    """
    Converts the blcok into a single convolutional layer with no normalization, dropout, or residual connections.
    """
    cfg.depth = 1
    cfg.norm_config = None
    cfg.dropout_config = None
    cfg.residual_connections = None
    cfg.activation = None
    return cfg


@ImageConvBlockConfig.register_preset("single_block")
def single_block_config(cfg: ImageConvBlockConfig):
    """
    Sets the block depth to 1 and removes dropout and residual connections, but keeps normalization and activation.
    """
    cfg.depth = 1
    cfg.dropout_config = None
    cfg.residual_connections = None
    return cfg


@ImageConvBlockConfig.register_preset("0_residual")
def half_unet_residual_type(cfg: ImageConvBlockConfig):
    """
    Configures the block to have a residual connection from the input to the output of the last layer.
    """
    cfg.residual_connections = {cfg.depth - 1: [0]}
    cfg.residual_type = ResidualConnectionType.ADD
    print("in 0_residual config", cfg)
    return cfg


@ImageConvBlockConfig.register_preset("1_residual")
def unet_residual_type(cfg: ImageConvBlockConfig):
    """
    Configures the block to have a residual connection from the output of the first layer to the output of the last layer.
    """
    assert cfg.depth > 1, "Depth must be greater than 1 for 1-residual connections"
    cfg.residual_connections = {cfg.depth - 1: [1]}
    cfg.residual_type = ResidualConnectionType.ADD
    return cfg


@ImageConvBlockConfig.register_preset("concat_residual")
def concat_residual_type(cfg: ImageConvBlockConfig):
    """
    Configures the block to have a residual connection from the input to the output of the last layer, using concatenation instead of addition.
    """
    cfg.residual_connections = {cfg.depth - 1: [0]}
    cfg.residual_type = ResidualConnectionType.CONCAT
    print(cfg)
    return cfg


@ImageConvBlockConfig.register_preset("recon")
def reconstruction_config(cfg: ImageConvBlockConfig):
    """
    Configures the block for reconstruction tasks by removing normalization and dropout layers.
    """
    cfg.norm_config = None
    cfg.dropout_config = None
    return cfg


@ImageConvBlockConfig.register_preset("segmentation")
def segmentation_config(cfg: ImageConvBlockConfig):
    """
    Configures the block for segmentation tasks by using InstanceNorm with trainable parameters.
    """
    cfg.norm_config = LayerConfig(name="InstanceNorm", kwargs={"affine": True})
    return cfg


@ImageConvBlockConfig.register_preset("depthwise_separable")
def depthwise_separable_config(cfg: ImageConvBlockConfig):
    """
    Configures the block to use depthwise separable convolutions (see `im2sim.layers.DepthwiseSeparableConv`) instead of standard convolutions.
    """
    cfg.conv_config = LayerConfig(name="DepthwiseSeparableConv", kwargs={})
    return cfg


@ImageConvBlockConfig.register_preset("ghost_depthwise")
def ghost_depthwise_config(cfg: ImageConvBlockConfig):
    """
    Configures the block to use Ghost convolutions instead of standard convolutions.
    """
    cfg.conv_config = LayerConfig(name="GhostConv", kwargs={})
    print("in ghost config", cfg)
    return cfg


@ImageConvBlockConfig.register_preset("ghost_depthwise_separable")
def ghost_separable_config(cfg: ImageConvBlockConfig):
    """
    Configures the block to use Ghost depthwise separable convolutions instead of standard convolutions.
    """
    cfg.conv_config = LayerConfig(name="GhostConv", kwargs={"separable": True})
    return cfg


@ImageConvBlockConfig.register_preset("dilated_convs")
def dilated_convs_config(cfg: ImageConvBlockConfig):
    """
    Configures the block to use dilated convolutions with dilation of 2 instead of standard convolutions.
    """
    cfg.conv_config.kwargs["dilation"] = 2
    return cfg


@ImageConvBlockConfig.register_preset("ECA")
def eca_config(cfg: ImageConvBlockConfig):
    """
    Configures the block to use Efficient Channel Attention (ECA)


    """
    cfg.attn_config = LayerConfig(name="EfficientChannelAttn", kwargs={})
    return cfg


@ImageConvBlockConfig.register_preset("SE")
def se_config(cfg: ImageConvBlockConfig):
    """
    Configures the block to use Squeeze-and-Excitation (SE) attention
    """
    cfg.attn_config = LayerConfig(name="SqueezeExcite", kwargs={})
    print("in SE config", cfg)
    return cfg


if __name__ == "__main__":
    cfg = ImageConvBlockConfig(depth=3, activation="ReLU", out_activation="softmax")
    cfg = ImageConvBlockConfig.apply_presets(cfg, ["ghost_depthwise", "0_residual", "SE"])
    cfg.save("test_config.yaml")
    cfg2 = ImageConvBlockConfig().load("test_config.yaml")
    cfg2.save("test_config2.yaml")
    
#     model = ImageConvBlock.build(
#         rank=2,
#         in_channels=32,
#         out_channels=32,
#         cfg=cfg,
#     )
#     # # print(cfg)

#     # print(model)
#     # x = torch.randn(1, 32, 64, 64)
#     # y = model(x)
#     # print(y.shape)
