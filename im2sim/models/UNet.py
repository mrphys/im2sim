import torch
from torch import nn
import torch.nn.functional as F
from ..layers.layer_util import get_image_layer, get_activation, standardize_spatial_factors


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
        drop = get_image_layer('Dropout', rank)
        self.convs = nn.ModuleList([
            conv(in_channels if i==0 else filters, filters, kernel_size, padding=kernel_size//2)
            for i in range(depth)
        ])

        self.norms = nn.ModuleList([
            get_image_layer(norm_type, rank)(filters) if norm_type else nn.Identity()
            for _ in range(depth)
        ])
        self.drop = drop(p=dropout_rate) if dropout_rate else nn.Identity()

        self.act = get_activation(activation)(inplace=True) if activation.lower() == 'relu' else get_activation(activation)()
        

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature maps in image space [in_channels, ...] where the number of dims in ... corresponds to rank

        Returns:
            torch.Tensor: Output feature maps [out_channels, ...]
        """

        for conv, norm in zip(self.convs, self.norms):
            x = norm(self.act(conv(x)))
        return self.drop(x)
    

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
                pool_sizes = None,
                kernel_size=3,
                conv_blocks_per_level=1,
                rank=3,
                norm_type=None,
                pool_type='MaxPool',
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

        if pool_sizes is None:
            pool_sizes = 2

        pool_sizes_standard = standardize_spatial_factors(pool_sizes, rank)

        pool = get_image_layer(pool_type, rank)
        self.maxpools = nn.ModuleList([
            pool(pool_sizes_standard[i-1]) if i>0 else nn.Identity()
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
                 pool_sizes=None,
                 upsample_sizes = None,
                 conv_blocks_per_level=1,
                 rank=3,
                 upsample_type="ConvTranspose",
                 activation="relu",
                 norm_type=None,
                 dropout_rate=None,
                 skip=True):
        super().__init__()

        self.skip = skip
        self.rank = rank

        if pool_sizes is None:
            pool_sizes = 2

        pool_sizes_standard = standardize_spatial_factors(pool_sizes, rank)

        n_levels = len(filters)

        rev_filters = filters[::-1]
        rev_pool_sizes = pool_sizes_standard[::-1]

        if upsample_sizes is None:
            upsample_sizes = rev_pool_sizes
        else:
            upsample_sizes = standardize_spatial_factors(upsample_sizes, rank)


        if upsample_type.lower() == 'upsample':
            self.ups = nn.ModuleList([
                nn.Upsample(scale_factor=upsample_sizes[i], mode='trilinear' if rank==3 else 'bilinear', align_corners=True)
                for i in range(n_levels - 1)
            ])
        else:
            up_layer = get_image_layer(upsample_type, rank)
            self.ups = nn.ModuleList([
                up_layer(rev_filters[i], rev_filters[i+1], kernel_size=upsample_sizes[i], stride=upsample_sizes[i])
                for i in range(n_levels - 1)
            ])


        self.conv_blocks = nn.ModuleList([
            ImageConvBlock(
                in_channels=rev_filters[i+1] * (2 if skip else 1),
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
        """
        Match skip spatial size to x spatial size using:
        - center crop if skip is larger
        - interpolation if skip is smaller

        Args:
            x: decoder tensor, shape [B, C, ...]
            skip: encoder skip tensor, shape [B, C, ...]

        Returns:
            skip resized to have spatial shape x.shape[2:]
        """
        target_size = x.shape[2:]
        skip_size = skip.shape[2:]

        if skip_size == target_size:
            return skip

        # First crop any dimensions where skip is too large
        slices = [slice(None), slice(None)]
        needs_crop = False

        for s, t in zip(skip_size, target_size):
            if s > t:
                start = (s - t) // 2
                end = start + t
                slices.append(slice(start, end))
                needs_crop = True
            else:
                slices.append(slice(None))

        if needs_crop:
            skip = skip[tuple(slices)]

        # Then upsample if any dimensions are still too small
        if skip.shape[2:] != target_size:
            mode = "trilinear" if self.rank == 3 else "bilinear"
            skip = F.interpolate(skip, size=target_size, mode=mode, align_corners=True)

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

class UNet(nn.Module):
    """
    Flexible UNet for 2D or 3D images.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        filters (List[int]): Encoder filter sizes.
        kernel_size (int): Conv kernel size.
        pool_sizes (Tuple or list): pool sizes per level ,
        upsample_sizes (Tuple or list): upsample sizes per level,
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
                 pool_sizes = None,
                 upsample_sizes = None,
                 kernel_size=3,
                 conv_blocks_per_level=1,
                 rank=3,
                 activation="relu",
                 norm_type=None,
                 dropout_rate=None,
                 final_activation="linear"):
        
        pool_sizes_temp = []
        if pool_sizes is None:
            for i in range(len(filters)-1):
                pool_sizes_temp.append(2)
            pool_sizes = pool_sizes_temp
        
        if not isinstance(pool_sizes, (list, tuple)):
            raise TypeError("pool_sizes must be a list or tuple")
        
        for i, p in enumerate(pool_sizes):
            if isinstance(p, int) and not isinstance(p,bool):
                if p < 1:
                    raise ValueError(f"pool_sizes must be positive")
            elif isinstance(p, (tuple,list)):
                if len(p) != rank:
                    raise ValueError(f"pool_sizes tuple must be same length as rank")
                for j , ps in enumerate(p):
                    if not isinstance(ps, int) or isinstance(ps,bool):
                        raise TypeError(f"pool_sizes must be an int")
                    if ps < 1:
                        raise ValueError(f"pool_sizes must be positive")
            else:
                raise TypeError("each entry in pool_sizes must be either an int or a tuple or list")
        
        if len(filters) != (len(pool_sizes)+1):
            raise ValueError(f"pool_sizes do not match number of filters. For {len(filters)}, please input {len(filters)-1} number of pools.")
        

        if upsample_sizes is not None:
            if not isinstance(upsample_sizes, (list, tuple)):
                raise TypeError("upsample_sizes must be a list or tuple")
            
            for i, p in enumerate(upsample_sizes):
                if isinstance(p, int) and not isinstance(p,bool):
                    if p < 1:
                        raise ValueError(f"upsample_sizes must be positive")
                elif isinstance(p, (tuple,list)):
                    if len(p) != rank:
                        raise ValueError(f"upsample_sizes tuple must be same length as rank")
                    for j , ps in enumerate(p):
                        if not isinstance(ps, int) or isinstance(ps,bool):
                            raise TypeError(f"upsample_sizes must be an int")
                        if ps < 1:
                            raise ValueError(f"upsample_sizes must be positive")
                else:
                    raise TypeError("each entry in upsample_sizes must be either an int or a tuple or list")
            
            if len(filters) != (len(upsample_sizes)+1):
                raise ValueError(f"upsample_sizes do not match number of filters. For {len(filters)}, please input {len(filters)-1} number of upsamples.")
            

        super().__init__()

        self.encoder = ImageEncoder(
            in_channels=in_channels,
            filters=filters,
            pool_sizes=pool_sizes,
            kernel_size=kernel_size,
            conv_blocks_per_level=conv_blocks_per_level,
            rank=rank,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate
        )

        self.decoder = ImageDecoder(
            filters=filters,
            pool_sizes=pool_sizes,
            upsample_sizes = upsample_sizes,
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

class StandardUNet(UNet):
    """
    Standaed UNet for 2D or 3D images.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        filters (List[int]): Encoder filter sizes.
        kernel_size (int): Conv kernel size.
        pool_sizes (Tuple or list): pool sizes per dim ,
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
                 filters=[16, 32, 64, 128, 256],
                 pool_size = 2,
                 kernel_size=3,
                 conv_blocks_per_level=1,
                 rank=3,
                 activation="relu",
                 norm_type=None,
                 dropout_rate=None,
                 final_activation="linear"):
        
        pool_size_standard = []
        if isinstance(pool_size, int):
            pool_size_single = []
            for j in range(rank):
                pool_size_single.append(pool_size)
            pool_size_single = tuple(pool_size_single)
            for i in range(len(filters)-1):
                pool_size_standard.append(pool_size_single)
        elif isinstance(pool_size , (tuple,list)):
            if len(pool_size) == rank:
                for i in range(len(filters)-1):
                    pool_size_standard.append(pool_size)
            else:
                raise ValueError(f"pool_size must be an int or have same length as rank")

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            filters=filters,
            pool_sizes=pool_size_standard,
            upsample_sizes=None,
            kernel_size=kernel_size,
            conv_blocks_per_level=conv_blocks_per_level,
            rank=rank,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )