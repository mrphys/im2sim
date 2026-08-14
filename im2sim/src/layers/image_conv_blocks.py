from dataclasses import dataclass, field

import torch

from im2sim.src.layers.layer_util import (
    apply_residual_connection,
    get_activation,
    get_image_layer,
)
from im2sim.src.layers.module_config import Config, LayerConfig, register_config
from im2sim.src.utils import api_util


@api_util.export("configs.ImageConvBlockConfig")
@register_config
@dataclass
class ImageConvBlockConfig(Config):
    """
    Configuration class for defining the parameters of an image convolutional block.

    The default configuration consists of a single convolutional layer with ReLU activation, InstanceNorm normalization, and no dropout or attention layers.

    Attributes can either be set directly when creating an instance of the class or modified later.
    Configuration presets can be applied to quickly set up common configurations for different use cases.

    The configuration can also be saved to and loaded from a YAML file.

    Args:

        depth (int):
            The number of convolutional layers in the block. Default is `1`.

        activation (str | None):
            The activation function to use after each convolutional layer. Default is `"ReLU"`.

        out_activation (str | None):
            The activation function to use after the final layer. Default is `None`.

        conv_cfg (LayerConfig):
            Configuration for the convolutional layers, including kernel size and padding.
            Default is a 3x3 convolution with padding set to `"same"`.

        norm_cfg (LayerConfig):
            Configuration for the normalization layers, such as `InstanceNorm` with affine set to `True`.
            Default is `InstanceNorm` with affine set to `True`.

        dropout_cfg (LayerConfig):
            Configuration for the dropout layers. Default is no dropout.


        attn_cfg (LayerConfig):
            Configuration for the attention layers. Default is no attention.
            Attention is applied before the residual connection if the layer is a target of a residual connection, or after the last layer if there are no residual connections.

        dropout_position (int | list[int]):
            Specifies the position(s) of the dropout layers within the block.
            Dropout will be applied after the specified layer(s), where the layer numbers start from `1` and end at `depth`.
            If a list is provided, dropout will be applied after each specified layer.
            Default is `1`. (only applied if dropout_cfg is not `None`)

        residual_connections ( dict [int, list[ int ]] | None):
            Specifies the residual connections within the block.
            The keys represent the target layers (numbered from `1` to `depth`), and the values are lists of source layers. Default is `None`.
            Example: `{1: [0]}` means that the input(0) to the block will be added to the output of layer 1.

        residual_type (str):
            The type of residual connection to use (e.g., `"add"`, `"concat"`, `"average"`). Default is `"add"`.

    Examples:

        To create a highly customised configuration for an image convolutional block, you can specify all attributes of the configuration:

        >>> cfg = ImageConvBlockConfig(
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


        To make simple modifications to the default configuration, you can modify a subset of attributes:

        >>> cfg = ImageConvBlockConfig(depth=3, activation="ReLU", out_activation="softmax")

        To make modifications to an existing configuration, you can use the `mod()` method:

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
        >>> mini_block_cfg = block_cfg.mod(depth=2, dropout_position=[1], residual_connections={1: [0]})

        Different presets exist to perform common mods on the configuration.
        For example, to convert the block into a single convolutional layer with no normalization, dropout, or residual connections, you can use `to_single_conv()`
        or to add a residual connection from the input to the output of the last layer, you can use `add_input_residual()`:

        >>> cfg = cfg.to_single_conv()
        >>> cfg = cfg.add_input_residual()

        To save the configuration to a YAML file and load it back, you can use:

        >>> cfg.save("my_config.yaml")
        >>> loaded_cfg = ImageConvBlockConfig.load("my_config.yaml")

        Refer to the methods below to see all available transformations that can be applied to the configuration.

    """

    depth: int = 1
    activation: str | None = "ReLU"
    out_activation: str | None = None
    conv_cfg: LayerConfig = field(
        default_factory=lambda: LayerConfig(
            name="Conv", kwargs={"kernel_size": 3, "padding": "same"}
        )
    )
    norm_cfg: LayerConfig = field(
        default_factory=lambda: LayerConfig(name="InstanceNorm", kwargs={"affine": True})
    )
    dropout_cfg: LayerConfig = field(default_factory=lambda: LayerConfig(name=None, kwargs={}))
    attn_cfg: LayerConfig = field(default_factory=lambda: LayerConfig(name=None, kwargs={}))
    dropout_position: int | list[int] = 1
    residual_connections: dict[int, list[int]] = None
    residual_type: str = "add"

    def to_single_conv(self):
        """
        Converts the blcok into a single convolutional layer with no normalization, dropout, or residual connections.
        """
        self.depth = 1
        self.norm_cfg = LayerConfig(name=None, kwargs={})
        self.dropout_cfg = LayerConfig(name=None, kwargs={})
        self.attn_cfg = LayerConfig(name=None, kwargs={})
        self.residual_connections = None
        self.activation = None
        return self

    def to_single_block(self):
        """
        Sets the block depth to 1 and removes dropout and residual connections, but keeps normalization and activation.
        """
        self.depth = 1
        self.dropout_cfg = LayerConfig(name=None, kwargs={})
        self.residual_connections = None
        return self

    def add_input_residual(self):
        """
        Configures the block to have a residual connection from the input to the output of the last layer.
        """
        self.residual_connections = {self.depth - 1: [0]}
        self.residual_type = "add"
        return self

    def add_conv1_residual(self):
        """
        Configures the block to have a residual connection from the output of the first layer to the output of the last layer.
        """
        assert self.depth > 1, "Depth must be greater than 1 for 1-residual connections"
        self.residual_connections = {self.depth - 1: [1]}
        self.residual_type = "add"
        return self

    def add_input_concat_residual(self):
        """
        Configures the block to have a residual connection from the input to the output of the last layer, using concatenation instead of addition.
        """
        self.residual_connections = {self.depth - 1: [0]}
        self.residual_type = "concat"
        return self

    def reconstruction_mode(self):
        """
        Configures the block for reconstruction tasks by removing normalization and dropout layers.
        """
        self.norm_cfg = LayerConfig(name=None, kwargs={})
        self.dropout_cfg = LayerConfig(name=None, kwargs={})
        return self

    def segmentation_mode(self):
        """
        Configures the block for segmentation tasks by using InstanceNorm with trainable parameters.
        """
        self.norm_cfg = LayerConfig(name="InstanceNorm", kwargs={"affine": True})
        return self

    def dilate_convs(self, dilation: int = 2):
        """
        Configures the block to use dilated convolutions with dilation instead of standard convolutions.

        Args:
            dilation (int): The dilation factor to use for the convolutional layers. Default is `2`.

        """
        self.conv_cfg.kwargs["dilation"] = dilation
        return self

    def add_eca(self):
        """
        Configures the block to use Efficient Channel Attention (ECA)
        """
        self.attn_cfg = LayerConfig(name="EfficientChannelAttn", kwargs={})
        return self

    def add_se(self):
        """
        Configures the block to use Squeeze-and-Excitation (SE) attention
        """
        self.attn_cfg = LayerConfig(name="SqueezeExcite", kwargs={})
        return self

    def nullify(self):
        """
        Configures the block to have no normalization, dropout, or attention layers.
        """
        self.conv_cfg = LayerConfig(name=None, kwargs={})
        self.norm_cfg = LayerConfig(name=None, kwargs={})
        self.dropout_cfg = LayerConfig(name=None, kwargs={})
        self.attn_cfg = LayerConfig(name=None, kwargs={})
        self.residual_connections = None
        self.activation = None
        self.out_activation = None
        return self


@api_util.export("layers.ImageConvBlock")
class ImageConvBlock(torch.nn.Module):
    """
    A configurable image convolutional block that consists of a sequence of
    convolutional layers, normalization layers, dropout layers, and attention
    layers. The block supports residual connections and allows for flexible
    configuration of its components.

    Args:
        in_channels (int):
            Number of input channels.

        out_channels (int):
            Number of output channels.

        rank (int):
            The rank of the convolutional layers (e.g., `2` for 2D convolutions).

        cfg (ImageConvBlockConfig):
            Configuration object that defines the parameters of the block.

    Examples:

        To create an ImageConvBlock with a depth of 3, ReLU activation, and softmax output activation, you can use the following code:

        >>> cfg = ImageConvBlockConfig(depth=3, activation="ReLU", out_activation="softmax")
        >>> model = ImageConvBlock(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )

        Since the configs are rankless, you could use the same config for a 1D, 2D, or 3D convolutional block by changing the rank parameter when creating the ImageConvBlock instance.

        >>> cfg = ImageConvBlockConfig(depth=3, activation="ReLU", out_activation="softmax")
        >>> model1D = ImageConvBlock(
        >>>        rank=1,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )
        >>> model2D = ImageConvBlock(
        >>>        rank=2,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )
        >>> model3D = ImageConvBlock(
        >>>        rank=3,
        >>>        in_channels=32,
        >>>        out_channels=32,
        >>>        cfg=cfg,
        >>>    )

        Models can be saved and loaded using the standard PyTorch methods:

        >>> torch.save(model.state_dict(), "model.pth")
        >>> model.load_state_dict(torch.load("model.pth"))

        Configs can also be saved and loaded using the methods provided in the `im2sim.configs.ImageConvBlockConfig` class:
    """

    def __init__(self, in_channels: int, out_channels: int, rank: int, cfg: ImageConvBlockConfig):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.depth = cfg.depth
        self.activation = get_activation(cfg.activation)
        self.out_activation = get_activation(cfg.out_activation)
        self.conv_cfg = cfg.conv_cfg
        self.norm_cfg = cfg.norm_cfg
        self.attn_cfg = cfg.attn_cfg
        self.dropout_cfg = cfg.dropout_cfg
        self.dropout_position = (
            cfg.dropout_position
            if isinstance(cfg.dropout_position, list)
            else [cfg.dropout_position]
        )
        self.residual_connections = (
            cfg.residual_connections if cfg.residual_connections is not None else {}
        )
        self.residual_type = cfg.residual_type

        self._set_default_configs()
        self._validate_configs()

        self.layers = torch.nn.ModuleList()
        in_channels_per_layer = []
        in_channels_current = self.in_channels

        for i in range(self.depth):
            if i in self.residual_connections and self.residual_type.lower().strip() == "concat":
                for src in self.residual_connections[i]:
                    in_channels_current += in_channels_per_layer[src]

            in_channels_per_layer.append(in_channels_current)

            conv = get_image_layer(self.conv_cfg.name, rank=self.rank)(
                in_channels=in_channels_per_layer[-1],
                out_channels=out_channels,
                **self.conv_cfg.kwargs,
            )

            norm = get_image_layer(self.norm_cfg.name, rank=self.rank)(
                self.out_channels, **self.norm_cfg.kwargs
            )

            dropout = (
                get_image_layer(self.dropout_cfg.name, rank=self.rank)(**self.dropout_cfg.kwargs)
                if (i + 1) in self.dropout_position
                else torch.nn.Identity()
            )

            pre_residual = self.attn_cfg.name is not None and (i + 1) in self.residual_connections
            no_residual_final = len(self.residual_connections.keys()) == 0 and i == self.depth - 1
            if pre_residual or no_residual_final:
                attn = get_image_layer(self.attn_cfg.name, rank=self.rank)(
                    self.out_channels, **self.attn_cfg.kwargs
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

    def _set_default_configs(self):
        if self.conv_cfg is None:
            self.conv_cfg = LayerConfig(name="Conv", kwargs={"kernel_size": 3, "padding": "same"})
        if self.norm_cfg is None:
            self.norm_cfg = LayerConfig(name=None, kwargs={})
        if self.dropout_cfg is None:
            self.dropout_cfg = LayerConfig(name=None, kwargs={})
        if self.attn_cfg is None:
            self.attn_cfg = LayerConfig(name=None, kwargs={})

    def _validate_configs(self):
        assert self.norm_cfg.name in [None, "BatchNorm", "InstanceNorm"], (
            f"Unsupported norm type: {self.norm_cfg.name}"
        )
        assert self.attn_cfg.name in [None, "EfficientChannelAttn", "SqueezeExcite"], (
            f"Unsupported attention type: {self.attn_cfg.name}"
        )
        if self.dropout_cfg.name is not None:
            assert max(self.dropout_position) < self.depth, (
                "Dropout position must be less than depth"
            )

        if self.residual_type.lower().strip() == "concat" and any(
            dst >= self.depth for dst in self.residual_connections
        ):
            raise ValueError(
                "Residual connections with 'concat' type cannot be created on the last layer since it would change the output channels."
            )

    def forward(self, x):
        """ """
        outputs = [x]
        for i, layer in enumerate(self.layers):
            if i in self.residual_connections:
                for src in self.residual_connections[i]:
                    x = apply_residual_connection(
                        outputs[src], x, connection_type=self.residual_type
                    )

            x = layer(x)

            outputs.append(x)

        x = self.out_activation(x)
        return x


if __name__ == "__main__":
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
        residual_type="concat",
    )
    mini_block_cfg = block_cfg.mod(depth=2, dropout_position=[1], residual_connections={1: [0]})

    # block_cfg = block_cfg.mod(residual_connections={3: [1, 0]})
    model = ImageConvBlock(
        rank=2,
        in_channels=32,
        out_channels=32,
        cfg=mini_block_cfg,
    )

    print(model)
    x = torch.randn(1, 32, 64, 64)
    y = model(x)
    print(y.shape)
