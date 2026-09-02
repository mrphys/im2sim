import torch

from im2sim.configs.core import LayerConfig
from im2sim.configs.image_blocks import ImageConvBlockConfig
from im2sim.utils.layer_util import (
    apply_residual_connection,
    get_activation,
    get_image_layer,
)


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
