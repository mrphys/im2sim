import torch
from collections import defaultdict
from dataclasses import dataclass 

import custom_image_layers
from layer_util import get_activation, get_image_layer, LayerSpec, ResidualConnectionType, apply_residual_connection, ConfigurableModule, ModuleSpec



@dataclass
class ImageConvBlockConfig:
    depth: int = 2
    activation: str|None = "ReLU"
    out_activation: str|None = None
    conv_type: LayerSpec = LayerSpec(name="Conv", kwargs={"kernel_size": 3, "padding": "same"})
    norm_type: LayerSpec = LayerSpec(name="InstanceNorm", kwargs={"affine": True})
    dropout_type: LayerSpec = LayerSpec(name=None, kwargs={})
    dropout_position: int|list[int] = 1
    residual_connections: dict[int,list[int]]=None
    residual_type: str = ResidualConnectionType.ADD


class ImageConvBlock(torch.nn.Module, ConfigurableModule):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 rank: int, 
                 depth: int = 2,
                 activation: str|None = None, 
                 out_activation: str|None = None,
                 conv_type: LayerSpec|None = None,
                 norm_type: LayerSpec|None = None,
                 dropout_type: LayerSpec|None = None,
                 dropout_position: int|list[int] = 1,
                 residual_connections: dict[int,list[int]]=None,
                 residual_type: str = ResidualConnectionType.ADD,
                 ):
        super().__init__()

        if dropout_type is not None:
            assert dropout_position < depth, "Dropout position must be less than depth"
        

        if conv_type is None:
            conv_type = LayerSpec(name="Conv", rank=rank, kwargs={})
        if norm_type is None:
            norm_type = LayerSpec(name=None, kwargs={})
        if dropout_type is None:
            dropout_type = LayerSpec(name=None, kwargs={})

        if isinstance(dropout_position, int):
            dropout_position = [dropout_position]

        self.residual_connections = residual_connections if residual_connections is not None else {}
        self.residual_type = residual_type
               

        self.layers = torch.nn.ModuleList()
        self.activation = get_activation(activation)
        self.out_activation = get_activation(out_activation)

        in_channels_per_layer = [in_channels]
        for i in range(depth):
            conv = get_image_layer(conv_type.name, rank=rank)
            conv = get_image_layer(conv_type.name, rank=rank)(
                            in_channels=in_channels_per_layer[-1],
                            out_channels=out_channels,
                            **conv_type.kwargs)
            
            assert norm_type.name in [None, "BatchNorm", "InstanceNorm"], f"Unsupported norm type: {norm_type.name}"
            
            norm = get_image_layer(norm_type.name, rank=rank)(out_channels,**norm_type.kwargs)

            dropout = get_image_layer(dropout_type.name, rank=rank)(**dropout_type.kwargs) if i in dropout_position else torch.nn.Identity()
            
            block = torch.nn.Sequential(
                conv,
                norm,
                dropout,
                self.activation if i<depth-1 else torch.nn.Identity(),
            )
            self.layers.append(block)

            in_channels_current = out_channels
            if i+1 in self.residual_connections.keys() and self.residual_type == ResidualConnectionType.CONCAT:
                for src in self.residual_connections[i+1]:
                    in_channels_current += in_channels_per_layer[src]
            in_channels_per_layer.append(in_channels_current)

        
    def forward(self, x):
        outputs = [x]

        for i, layer in enumerate(self.layers):
            print(x.shape)
            x = layer(x)
            if i+1 in self.residual_connections.keys():
                for src in self.residual_connections[i+1]:
                    x = apply_residual_connection(outputs[src], x, connection_type=self.residual_type)
            outputs.append(x)

        x = self.out_activation(x)
        return x


ImageConvBlockSpec = ModuleSpec(ImageConvBlock, ImageConvBlockConfig)


@ImageConvBlockSpec.register_config("single_conv")
def half_unet_residual_config(cfg: ImageConvBlockConfig):
    cfg.depth = 1
    cfg.norm_type = None
    cfg.dropout_type = None
    cfg.residual_connections = None
    cfg.activation = None
    return cfg

@ImageConvBlockSpec.register_config("single_block")
def half_unet_residual_config(cfg: ImageConvBlockConfig):
    cfg.depth = 1
    cfg.dropout_type = None
    cfg.residual_connections = None
    return cfg

@ImageConvBlockSpec.register_config("0_residual")
def half_unet_residual_config(cfg: ImageConvBlockConfig):
    cfg.residual_connections = {cfg.depth-1: [0]}
    cfg.residual_type = ResidualConnectionType.ADD
    return cfg

@ImageConvBlockSpec.register_config("1_residual")
def unet_residual_config(cfg: ImageConvBlockConfig):
    assert cfg.depth > 1, "Depth must be greater than 1 for 1-residual connections"
    cfg.residual_connections = {cfg.depth-1: [1]}
    cfg.residual_type = ResidualConnectionType.ADD
    return cfg

@ImageConvBlockSpec.register_config("concat_residual")
def concat_residual_config(cfg: ImageConvBlockConfig):
    cfg.residual_connections = {cfg.depth-1: [0]}
    cfg.residual_type = ResidualConnectionType.CONCAT
    return cfg

@ImageConvBlockSpec.register_config("recon")
def reconstruction_config(cfg: ImageConvBlockConfig):
    cfg.norm_type = None
    cfg.dropout_type = None
    return cfg

@ImageConvBlockSpec.register_config("segmentation")
def segmentation_config(cfg: ImageConvBlockConfig):
    cfg.norm_type = LayerSpec(name="InstanceNorm", kwargs={"affine": True})
    return cfg


@ImageConvBlockSpec.register_config("depthwise_separable")
def depthwise_separable_config(cfg: ImageConvBlockConfig):
    cfg.conv_type = LayerSpec(name="DepthwiseSeparableConv", kwargs={})
    return cfg

@ImageConvBlockSpec.register_config("ghost_depthwise")
def ghost_config(cfg: ImageConvBlockConfig):
    cfg.conv_type = LayerSpec(name="GhostConv", kwargs={})
    return cfg

@ImageConvBlockSpec.register_config("ghost_depthwise_separable")
def ghost_config(cfg: ImageConvBlockConfig):
    cfg.conv_type = LayerSpec(name="GhostConv", kwargs={"separable":True})
    return cfg

@ImageConvBlockSpec.register_config("dilated_convs")
def dilated_convs_config(cfg: ImageConvBlockConfig):
    cfg.conv_type = LayerSpec(name="Conv", kwargs={"kernel_size": 3, "padding": "same", "dilation": 2})
    return cfg

if __name__ == "__main__":
    cfg = ImageConvBlockConfig(depth=3, activation="ReLU", out_activation="SoftPlus").apply_presets(["ghost", "0_residual"])
    model = ImageConvBlockSpec.build(cfg, presets=["ghost", "0_residual"])

    print(model)
    x = torch.randn(1, 3, 64, 64, 64)
    y = model(x)
    print(y.shape)