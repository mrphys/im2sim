import logging
import torch
from torch import nn
from .layer_util import get_image_layer, get_activation

logger = logging.getLogger(__name__)

class ImageConvBlock(nn.Module):
    """
    A convolutional block for image data

    Args:
        in_channels (int): The number of channels in the input to the layer.
        filters (int, optional): The number of filters in each convolutional layer (default: 32)
        kernel_size (int, optional): The kernel(filter) size for the convolutional layers (default: 3)
        depth (int, optional): The number of successive convolutional layers (default: 2)
        rank (int, optional): The number of spatial dimensions in the data i.e., 2D, 3D (default:2),
        activation (str, optional): The activation function applied after each convolution (default: "relu", options: "leakyrelu","gelu","sigmoid","linear")
        norm_type (str, optional): The normalization method to apply between convolutions (default:None, options: "BatchNorm", "InstanceNorm", "LayerNorm")
        dropout_rate (float, optional): The spatial dropout rate to be applied to the final convolution output (default:None)

    Returns:
        A `torch.nn.Module` object.
    
    """
    def __init__(self, 
                in_channels, 
                filters=32, 
                kernel_size=3,
                depth=1, 
                rank=3,
                activation='relu', 
                norm_type=None,
                dropout_rate=None):
        super().__init__() 

        conv = get_image_layer('Conv', rank)
        self.convs = nn.ModuleList([
            conv(in_channels if i==0 else filters, filters, kernel_size, padding=kernel_size//2)
            for i in range(depth)
        ])

        self.norms = nn.ModuleList([
            get_image_layer(norm_type, rank)(filters, affine=True, eps=1e-5) if norm_type else nn.Identity()
            for _ in range(depth)
        ])
        self.drop = get_image_layer('Dropout', rank)(p=dropout_rate) if dropout_rate else nn.Identity()

        self.act = get_activation(activation)(inplace=True) if activation.lower() == 'relu' else get_activation(activation)()
        

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature maps in image space [in_channels, ...] where the number of dims in ... corresponds to rank

        Returns:
            torch.Tensor: Output feature maps [out_channels, ...]
        """

        for conv, norm in zip(self.convs, self.norms):
            logger.debug("Image feature shape:%s", tuple(x.shape))
            x = self.act(norm(conv(x)))
        return self.drop(x)

class ImageConvResBlock(nn.Module):
    """
    A convolutional residual block for image data

    Args:
        in_channels (int): The number of channels in the input to the layer.
        filters (int, optional): The number of filters in each convolutional layer (default: 32)
        kernel_size (int, optional): The kernel(filter) size for the convolutional layers (default: 3)
        depth (int, optional): The number of successive convolutional layers (default: 3)
        rank (int, optional): The number of spatial dimensions in the data i.e., 2D, 3D (default:2),
        activation (str, optional): The activation function applied after each convolution (default: "relu", options: "leakyrelu","gelu","sigmoid","linear")
        norm_type (str, optional): The normalization method to apply between convolutions (default:None, options: "BatchNorm", "InstanceNorm", "LayerNorm")
        dropout_rate (float, optional): The spatial dropout rate to be applied to the convolution prior to residual connection (default:None)

    Returns:
        A `torch.nn.Module` object.
    
    """
    def __init__(self, 
                in_channels, 
                filters=32, 
                kernel_size=3,
                depth=3, 
                rank=3,
                activation='relu', 
                norm_type=None,
                dropout_rate=None):
        super().__init__() 


        self.initial_conv = ImageConvBlock(in_channels=in_channels,
                                            filters=filters,
                                            kernel_size=kernel_size,
                                            rank=rank,
                                            activation='linear',
                                            norm_type=norm_type,
                                            depth=1,
                                            dropout_rate=None)
        
        self.main_conv = ImageConvBlock(in_channels=filters,
                                            filters=filters,
                                            kernel_size=kernel_size,
                                            rank=rank,
                                            activation=activation,
                                            norm_type=norm_type,
                                            depth=depth-2,
                                            dropout_rate=dropout_rate)
        
        self.out_conv = ImageConvBlock(in_channels=filters,
                                            filters=filters,
                                            kernel_size=kernel_size,
                                            rank=rank,
                                            activation='linear',
                                            norm_type=norm_type,
                                            depth=1,
                                            dropout_rate=None)
        # self.drop =  get_image_layer('Dropout', rank)(p=dropout_rate) if dropout_rate else nn.Identity()
        self.out_act = get_activation(activation)(inplace=True) if activation.lower() == 'relu' else get_activation(activation)()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature maps in image space [in_channels, ...] where the number of dims in ... corresponds to rank

        Returns:
            torch.Tensor: Output feature maps [out_channels, ...]
        """
        x1 = self.initial_conv(x)
        x = self.main_conv(x1) 
        # x = self.drop(x)
        x = self.out_act(self.out_conv(x) + x1)
        return x
    
class ImageResEncoder(nn.Module):
    """
    A CNN encoder for images. Structured like the encoder of a ResUNet.

    Args:
        in_channels (int): The number of channels in the input image.
        filters (List[int], optional): The number of convolutional filters in each encoder level (default: [16,32,64,128,256])
        kernel_size (int, optional): The kernel(filter) size for the convolutional layers (default: 3)
        res_depth (int, optional): The number of successive convolutional layers in each residual block (default: 3)
        res_blocks_per_level (int, optional): The number of successive residual blocks per encoder level (default: 2)
        rank (int, optional): The number of spatial dimensions in the data i.e., 2D, 3D (default:3),
        activation (str, optional): The activation function applied after each convolution (default: "relu", options: "leakyrelu","gelu","sigmoid","linear")
        norm_type (str, optional): The normalization method to apply between convolutions (default:None, options: "BatchNorm", "InstanceNorm", "LayerNorm")
        dropout_rate (float, optional): The spatial dropout rate to be applied to each residual block prior to residual connection (default:None)

    Returns:
        A `torch.nn.Module` object.
    """

    def __init__(self,
                in_channels, 
                filters=[16,32,64,128,256],
                kernel_size=3,
                res_depth=3,
                res_blocks_per_level=2,
                rank=3,
                norm_type=None,
                pool_type='MaxPool',
                pool_size=2,
                activation='relu',
                dropout_rate=None):
        super().__init__()
        
        n_levels = len(filters)
        in_channels = [in_channels, *filters]
        self.conv_blocks = nn.ModuleList([
                nn.ModuleList([
                    ImageConvResBlock(in_channels=in_channels[i] if j==0 else in_channels[i+1], 
                                    filters=filters[i], 
                                    kernel_size=kernel_size, 
                                    depth=res_depth,
                                    rank=rank,
                                    activation=activation,
                                    norm_type=norm_type,
                                    dropout_rate=dropout_rate)
                    for j in range(res_blocks_per_level)
                    ])
            for i in range(n_levels)
        ])
        pool = get_image_layer(pool_type, rank)
        self.maxpools = nn.ModuleList([
            pool(pool_size) if i>0 else nn.Identity()
            for i in range(n_levels)
        ])

    def forward(self,x):
        """
        Args:
            x (torch.Tensor): Input image [in_channels, ...]

        Returns:
            List[torch.Tensor]: Output feature maps from each level ordered from top to bottom [Tensor([filters[0], ...], ..., Tensor([filters[N], ...])
        """
        logger.debug("IN ENCODER")
        outputs = []
        for pool, convs in zip(self.maxpools, self.conv_blocks):
            x = pool(x)
            for conv in convs: 
                x = conv(x)
            logger.debug("Conv output shape:%s", x.shape)
            outputs.append(x)
        return outputs
    

class ImageEncoder(nn.Module):
    """
    A CNN encoder for images. Structured like the encoder of a UNet.

    Args:
        in_channels (int): The number of channels in the input image.
        filters (List[int], optional): The number of convolutional filters in each encoder level (default: [16,32,64,128,256])
        kernel_size (int, optional): The kernel(filter) size for the convolutional layers (default: 3)
        conv_blocks_per_level (int, optional): The number of successive convolutional blocks per encoder level (default: 1)
        rank (int, optional): The number of spatial dimensions in the data i.e., 2D, 3D (default:3),
        activation (str, optional): The activation function applied after each convolution (default: "relu", options: "leakyrelu","gelu","sigmoid","linear")
        norm_type (str, optional): The normalization method to apply between convolutions (default:None, options: "BatchNorm", "InstanceNorm", "LayerNorm")
        dropout_rate (float, optional): The spatial dropout rate to be applied to each residual block prior to residual connection (default:None)

    Returns:
        A `torch.nn.Module` object.
    """

    def __init__(self,
                in_channels, 
                filters=[16,32,64,128,256],
                kernel_size=3,
                conv_blocks_per_level=1,
                rank=3,
                norm_type=None,
                pool_type='MaxPool',
                pool_size=2,
                activation='relu',
                dropout_rate=None):
        super().__init__()
        
        n_levels = len(filters)
        self.conv_blocks = nn.ModuleList([
                ImageConvBlock(in_channels=in_channels if i==0 else filters[i-1], 
                        filters=filters[i], 
                        kernel_size=kernel_size, 
                        depth=conv_blocks_per_level,
                        rank=rank,
                        activation=activation,
                        norm_type=norm_type,
                        dropout_rate=dropout_rate)
            for i in range(n_levels)
        ])
        pool = get_image_layer(pool_type, rank)
        self.maxpools = nn.ModuleList([
            pool(pool_size) if i>0 else nn.Identity()
            for i in range(n_levels)
        ])

    def forward(self,x):
        """
        Args:
            x (torch.Tensor): Input image [in_channels, ...]

        Returns:
            List[torch.Tensor]: Output feature maps from each level ordered from top to bottom [Tensor([filters[0], ...], ..., Tensor([filters[N], ...])
        """
        outputs = []
        for pool, conv in zip(self.maxpools, self.conv_blocks):
            x = conv(pool(x))
            outputs.append(x)
        return outputs





class ImageDecoder(nn.Module):
    """
    CNN decoder for images. Mirrors ImageEncoder like a UNet decoder.

    Args:
        filters (List[int]): Encoder filter sizes in top→bottom order.
        kernel_size (int): Convolution kernel size.
        conv_blocks_per_level (int): Number of conv blocks per level.
        rank (int): Spatial rank (2 or 3).
        upsample_type (str): "ConvTranspose" or "Upsample".
        activation (str): Activation name.
        norm_type (str): Normalization type.
        dropout_rate (float): Dropout rate.
        skip (bool): Use skip connections.
    """

    def __init__(self,
                 filters=[16,32,64,128,256],
                 kernel_size=3,
                 conv_blocks_per_level=1,
                 rank=3,
                 upsample_type="Upsample",
                 activation="relu",
                 norm_type=None,
                 dropout_rate=None,
                 skip=True):
        super().__init__()

        self.skip = skip
        n_levels = len(filters)

        rev_filters = filters[::-1]

        if upsample_type.lower() == 'upsample':
            # if rank == 4:
            #     self.ups = nn.ModuleList([
            #         Upsample4d(scale_factor=(1, 2, 2, 2))
            #         for _ in range(n_levels - 1)
            #     ])
            
            # else:
            self.ups = nn.ModuleList([
                nn.Upsample(scale_factor=2, mode='trilinear' if rank==3 else 'bilinear', align_corners=True)
                for _ in range(n_levels - 1)
            ])
        else:
            up_layer = get_image_layer(upsample_type, rank)
            self.ups = nn.ModuleList([
                up_layer(rev_filters[i], rev_filters[i+1], kernel_size=2, stride=2)
                for i in range(n_levels - 1)
            ])


        self.conv_blocks = nn.ModuleList([
            ImageConvBlock(
                in_channels=(
                    rev_filters[i] + rev_filters[i+1]
                    if skip else rev_filters[i]
                ),
                filters=rev_filters[i+1],
                kernel_size=kernel_size,
                depth=conv_blocks_per_level,
                rank=rank,
                activation=activation,
                norm_type=norm_type,
                dropout_rate=dropout_rate
            )
            for i in range(n_levels - 1)
        ])


    def _match_size(self, x, skip):
        if x.shape[2:] != skip.shape[2:]:
            # center crop skip to x
            diff = [s - t for s, t in zip(skip.shape[2:], x.shape[2:])]
            slices = [slice(d//2, d//2 + t) for d, t in zip(diff, x.shape[2:])]
            skip = skip[(..., *slices)]
        return skip


    def forward(self, encoder_outputs):
        """
        Args:
            encoder_outputs: List of tensors from encoder (top→bottom).

        Returns:
            Decoded tensor at highest resolution.
        """

        # Reverse the encoder outputs so we traverse from bottleneck to top
        rev_enc = encoder_outputs[::-1]

        # Start from bottleneck
        x = rev_enc[0]

        # Traverse decoder levels
        for i, (up, conv) in enumerate(zip(self.ups, self.conv_blocks)):

            x = up(x)

            if self.skip:
                skip_feat = rev_enc[i + 1]  # next encoder feature
                skip_feat = self._match_size(x, skip_feat)
                x = torch.cat([x, skip_feat], dim=1)

            x = conv(x)

        return x






# import torch
# import torch.nn as nn


class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # First branch (x1)
        self.conv1a = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=True
        )
        self.norm1a = nn.InstanceNorm3d(
            out_channels, affine=True, eps=1e-5
        )

        # Second branch
        self.conv1b = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=True
        )
        self.norm1b = nn.InstanceNorm3d(
            out_channels, affine=True, eps=1e-5
        )

        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=False)
        self.dropout = nn.Dropout3d(p=0.3)

        self.conv2b = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=True
        )
        self.norm2b = nn.InstanceNorm3d(
            out_channels, affine=True, eps=1e-5
        )

        self._init_weights()

    def _init_weights(self):
        # Match TF he_normal (fan_in, normal)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_in',
                    nonlinearity='leaky_relu'
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # First branch
        x1 = self.conv1a(x)
        x1 = self.norm1a(x1)
        # Second branch
        x2 = self.conv1b(x)
        x2 = self.norm1b(x2)
        x2 = self.act(x2)
        x2 = self.dropout(x2)

        x2 = self.conv2b(x2)
        x2 = self.norm2b(x2)

        # Residual add
        out = x1 + x2
        out = self.act(out)

        return out
    

class Encoder3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.block1_1 = ConvResBlock(in_channels, 16)
        self.block1_2 = ConvResBlock(16, 16)

        self.block2_1 = ConvResBlock(16, 32)
        self.block2_2 = ConvResBlock(32, 32)

        self.block3_1 = ConvResBlock(32, 64)
        self.block3_2 = ConvResBlock(64, 64)

        self.block4_1 = ConvResBlock(64, 128)
        self.block4_2 = ConvResBlock(128, 128)

        self.block5_1 = ConvResBlock(128, 256)
        self.block5_2 = ConvResBlock(256, 256)

        self.pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        X1 = self.block1_1(x)
        X1 = self.block1_2(X1)
        X1p = self.pool(X1)

        X2 = self.block2_1(X1p)
        X2 = self.block2_2(X2)
        X2p = self.pool(X2)

        X3 = self.block3_1(X2p)
        X3 = self.block3_2(X3)
        X3p = self.pool(X3)

        X4 = self.block4_1(X3p)
        X4 = self.block4_2(X4)
        X4p = self.pool(X4)

        X5 = self.block5_1(X4p)
        X5 = self.block5_2(X5)
        X5p = self.pool(X5)

        return [X1, X2, X3, X4, X5]