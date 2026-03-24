import torch
import torch.nn as nn
from ..layers.layer_util import get_image_layer, get_activation, Upsample4d

from ..layers import *
logger = logging.getLogger(__name__)
    
class UNet(nn.Module):
    """
    UNet for 2D or 3D images.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        filters (List[int]): Encoder filter sizes.
        kernel_size (int): Conv kernel size.
        conv_blocks_per_level (int): Depth per level.
        rank (int): Spatial rank.
        activation (str): Activation function.
        norm_type (str): Normalization type.
        dropout_rate (float): Dropout.
        final_activation (str): Output activation.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 filters=[16,32,64,128,256],
                 kernel_size=3,
                 conv_blocks_per_level=1,
                 rank=3,
                 activation="relu",
                 norm_type=None,
                 dropout_rate=None,
                 final_activation="linear"):

        super().__init__()

        self.encoder = ImageEncoder(
            in_channels=in_channels,
            filters=filters,
            kernel_size=kernel_size,
            conv_blocks_per_level=conv_blocks_per_level,
            rank=rank,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate
        )

        self.decoder = ImageDecoder(
            filters=filters,
            kernel_size=kernel_size,
            conv_blocks_per_level=conv_blocks_per_level,
            rank=rank,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate
        )

        conv = get_image_layer("Conv", rank)

        self.final_conv = conv(filters[0], out_channels, kernel_size=1)

        self.final_act = (
            get_activation(final_activation)()
            if final_activation.lower() != "linear"
            else nn.Identity()
        )

    def forward(self, x):
        enc_feats = self.encoder(x)
        x = self.decoder(enc_feats)
        x = self.final_conv(x)
        x = self.final_act(x)
        return x
    

