import torch
from torch import nn
import torch.nn.functional as F
from ..layers.layer_util import get_image_layer, get_activation, standardize_spatial_factors, _same_padding_time
from ..layers.layer_util import MaxPoolTime, UpsampleTime, ConvTime, BatchNormTime, ConvTransTime

class ImageConvBlockTime(nn.Module):
    """
    A convolutional block for image data with time.

    Expected input:
        [B, C, T, D, H, W]
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

        conv = ConvTime
        padding = _same_padding_time(kernel_size, rank=rank)

        self.convs = nn.ModuleList([
            conv(in_channels if i==0 else filters, filters, kernel_size, padding=padding, rank=rank)
            for i in range(depth)
        ])

        if norm_type is None:
            self.norms = nn.ModuleList([nn.Identity() for _ in range(depth)])
        elif norm_type == "BatchNorm":
            self.norms = nn.ModuleList([BatchNormTime(filters, rank=rank) for _ in range(depth)])
        else:
            raise ValueError(
                f"ImageConvBlockTime currently only supports norm_type=None or 'BatchNorm', got {norm_type}"
            )
        
        self.drop = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()

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
    

class ImageEncoderTime(nn.Module):
    """
    A CNN encoder for 3+1D image-time data.

    Expected input:
        x: [B, C, T, D, H, W]

    pool_sizes entries should be 4D factors:
        (T, D, H, W)
    """

    def __init__(self,
                in_channels, 
                filters=[16,32,64,128,256],
                pool_sizes = None,
                kernel_size=3,
                conv_blocks_per_level=1,
                rank=3,
                norm_type=None,
                activation='relu',
                dropout_rate=None):
        super().__init__()

        if rank not in (2, 3):
            raise ValueError(f"UNetTime currently only supports rank=2 or rank=3, got {rank}")
        
        n_levels = len(filters)

        if pool_sizes is None:
            pool_sizes = [tuple([1] + [2] * rank) for _ in range(n_levels - 1)]

        if not isinstance(pool_sizes, (list, tuple)):
            raise TypeError("pool_sizes must be a list or tuple")

        if len(pool_sizes) != n_levels - 1:
            raise ValueError(
                f"pool_sizes must have length {n_levels - 1} for filters of length {n_levels}, "
                f"got {len(pool_sizes)}"
            )

        pool_sizes_standard = standardize_spatial_factors(pool_sizes, rank+1)

        self.conv_blocks = nn.ModuleList([
                ImageConvBlockTime(in_channels=in_channels if i==0 else filters[i-1], 
                        filters=filters[i], 
                        kernel_size=kernel_size, 
                        depth=conv_blocks_per_level,
                        rank=rank,
                        activation=activation,
                        norm_type=norm_type,
                        dropout_rate=dropout_rate)
            for i in range(n_levels)
        ])

        self.maxpools = nn.ModuleList([
            MaxPoolTime(pool_sizes_standard[i-1],rank=rank) if i>0 else nn.Identity()
            for i in range(n_levels)
        ])

    def forward(self,x):
        outputs = []
        for pool, conv in zip(self.maxpools, self.conv_blocks):
            x = conv(pool(x))
            outputs.append(x)
        return outputs


class ImageDecoderTime(nn.Module):
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

        if rank not in (2, 3):
            raise ValueError(f"UNetTime currently only supports rank=2 or rank=3, got {rank}")

        n_levels = len(filters)

        if pool_sizes is None:
            pool_sizes = [tuple([1] + [2] * rank) for _ in range(n_levels - 1)]

        if not isinstance(pool_sizes, (list, tuple)):
            raise TypeError("pool_sizes must be a list or tuple")

        if len(pool_sizes) != n_levels - 1:
            raise ValueError(
                f"pool_sizes must have length {n_levels - 1} for filters of length {n_levels}, "
                f"got {len(pool_sizes)}"
            )

        pool_sizes_standard = standardize_spatial_factors(pool_sizes, rank+1)

        rev_filters = filters[::-1]
        rev_pool_sizes = pool_sizes_standard[::-1]

        if upsample_sizes is None:
            upsample_sizes = rev_pool_sizes
        else:
            if not isinstance(upsample_sizes, (list, tuple)):
                raise TypeError("upsample_sizes must be a list or tuple")

            if len(upsample_sizes) != n_levels - 1:
                raise ValueError(
                    f"upsample_sizes must have length {n_levels - 1} for filters of length {n_levels}, "
                    f"got {len(upsample_sizes)}"
                )

            upsample_sizes = standardize_spatial_factors(upsample_sizes, rank + 1)

        if upsample_type.lower() == 'upsample':
            self.ups = nn.ModuleList([
                UpsampleTime(scale_factor=upsample_sizes[i], mode='trilinear' if rank==3 else 'bilinear', align_corners=True, rank=rank)
                for i in range(n_levels - 1)
            ])
        elif upsample_type.lower() == 'convtranspose':
            self.ups = nn.ModuleList([
                ConvTransTime(rev_filters[i], rev_filters[i+1], kernel_size=upsample_sizes[i], stride=upsample_sizes[i],rank=rank)
                for i in range(n_levels - 1)
            ])
        else:
            raise ValueError(f"upsample_type must be 'Upsample' or 'ConvTranspose', got {upsample_type}")
        
        if upsample_type.lower() == "upsample":
            self.up_channel_adjust = nn.ModuleList([
                ConvTime(
                    in_channels=rev_filters[i],
                    out_channels=rev_filters[i + 1],
                    kernel_size=1,
                    padding=0,
                    rank=rank
                )
                for i in range(n_levels - 1)
            ])
        else:
            self.up_channel_adjust = nn.ModuleList([
                nn.Identity()
                for _ in range(n_levels - 1)
            ])


        self.conv_blocks = nn.ModuleList([
            ImageConvBlockTime(
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
        Match skip size to x size for 3+1D tensors [B, C, T, D, H, W].

        Strategy:
        - center crop dimensions where skip is too large
        - pure interpolation upsample where skip is too small
        """
        target_size = x.shape[2:]   # (T, D, H, W)
        skip_size = skip.shape[2:]

        if skip_size == target_size:
            return skip

        # First crop dimensions where skip is too large
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

        # Then non-learned upsample if any dimensions are still too small
        if skip.shape[2:] != target_size:
            up = UpsampleTime(size=target_size, mode="trilinear", align_corners=True,rank=self.rank)
            up = up.to(skip.device)
            skip = up(skip)

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
        for i, (up, ch_adjust, conv) in enumerate(zip(self.ups, self.up_channel_adjust, self.conv_blocks)):
            x = up(x)
            x = ch_adjust(x)

            if self.skip:
                skip_feat = rev_enc[i + 1]  # next encoder feature
                skip_feat = self._match_size(x, skip_feat)
                x = torch.cat([x, skip_feat], dim=1)

            x = conv(x)

        return x

class UNetTime(nn.Module):
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
                 upsample_type="ConvTranspose",
                 activation="relu",
                 norm_type=None,
                 dropout_rate=None,
                 final_activation="linear"):
        
        if rank not in (2, 3):
            raise ValueError(f"UNetTime currently only supports rank=2 or rank=3, got {rank}")
        
        if pool_sizes is None:
            pool_sizes = [tuple([1] + [2] * rank) for _ in range(len(filters) - 1)]
        
        if not isinstance(pool_sizes, (list, tuple)):
            raise TypeError("pool_sizes must be a list or tuple")
        
        for i, p in enumerate(pool_sizes):
            if isinstance(p, int) and not isinstance(p,bool):
                if p < 1:
                    raise ValueError(f"pool_sizes must be positive")
            elif isinstance(p, (tuple,list)):
                if len(p) != rank+1:
                    raise ValueError(f"pool_sizes tuple must be same length as rank + 1")
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
                    if len(p) != rank + 1:
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

        self.encoder = ImageEncoderTime(
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

        self.decoder = ImageDecoderTime(
            filters=filters,
            pool_sizes=pool_sizes,
            upsample_sizes = upsample_sizes,
            kernel_size=kernel_size,
            conv_blocks_per_level=conv_blocks_per_level,
            rank=rank,
            upsample_type = upsample_type,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate
        )

        self.final_conv = ConvTime(filters[0], out_channels, kernel_size=1, padding=0,rank=rank)

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
