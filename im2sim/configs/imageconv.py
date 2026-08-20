
from dataclasses import dataclass, field

from im2sim.configs.core import Config, LayerConfig, register_config


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
