from torch import nn
from layer_util import get_image_layer, get_activation


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

    Returns:
        A `torch.nn.Module` object.
    
    """
    def __init__(self, 
                in_channels, 
                filters=32, 
                kernel_size=3,
                depth=2, 
                rank=3,
                activation='relu', 
                norm_type=None):
        super().__init__() 

        conv = get_image_layer('Conv', rank)
        self.convs = nn.ModuleList([
            conv(in_channels if i==0 else filters, filters, kernel_size, padding=kernel_size//2)
            for i in range(depth)
        ])

        self.norms = nn.ModuleList([
            get_image_layer(norm_type, rank)(filters) if norm_type else nn.Identity()
            for _ in range(depth)
        ])

        self.act = get_activation(activation)(inplace=True) if activation.lower() == 'relu' else get_activation(activation)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature maps in image space [in_channels, ...] where the number of dims in ... corresponds to rank

        Returns:
            torch.Tensor: Output feature maps [out_channels, ...]
        """
        for conv, norm in zip(self.convs, self.norms):
            x = norm(self.act(conv(x)))
        return x
    
class ImageEncoder(nn.Module):
    """
    A CNN encoder for images. Structured like the encoder of a UNet.

    Args:
        in_channels (int): The number of channels in the input image.
        filters (List[int], optional): The number of convolutional filters in each encoder level (default: [16,32,64,128,256])
        kernel_size (int, optional): The kernel(filter) size for the convolutional layers (default: 3)
        depth (int, optional): The number of successive convolutional layers in each encoder level (default: 2)
        rank (int, optional): The number of spatial dimensions in the data i.e., 2D, 3D (default:2),
        activation (str, optional): The activation function applied after each convolution (default: "relu", options: "leakyrelu","gelu","sigmoid","linear")
        norm_type (str, optional): The normalization method to apply between convolutions (default:None, options: "BatchNorm", "InstanceNorm", "LayerNorm")

    Returns:
        A `torch.nn.Module` object.
    """

    def __init__(self,
                in_channels, 
                filters=[16,32,64,128,256],
                kernel_size=3,
                convs_per_layer=2,
                rank=3,
                norm_type=None,
                pool_type='MaxPool',
                activation='relu'):
        super().__init__()
        
        n_levels = len(filters)
        self.conv_blocks = nn.ModuleList([
            ImageConvBlock(in_channels=in_channels if i==0 else filters[i-1], 
                      filters=filters[i], 
                      kernel_size=kernel_size, 
                      depth=convs_per_layer,
                      rank=rank,
                      activation=activation,
                      norm_type=norm_type)
            for i in range(n_levels)
        ])
        pool = get_image_layer(pool_type, rank)
        self.maxpools = nn.ModuleList([
            pool() if i>0 else nn.Identity()
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





        