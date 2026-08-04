import torch

from layer_util import get_activation, get_image_layer, register_with_ranks, LayerSpec, ResidualConnectionType, apply_residual_connection



@register_with_ranks("DepthwiseConv", ranks=(1, 2, 3))
class DepthwiseConv(torch.nn.Module):

    def __init__(self, 
                 in_channels, 
                 rank, 
                 kernel_size=3, 
                 stride=1, 
                 padding='same', 
                 dilation=1, 
                 bias=True):
        super().__init__()
        self.conv = get_image_layer("Conv", rank)(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )

    def forward(self, x):
        return self.conv(x)



@register_with_ranks("DepthwiseSeparableConv", ranks=(1, 2, 3))
class DepthwiseSeparableConv(torch.nn.Module):

    def __init__(self, 
                 in_channels, 
                 out_channels,
                 rank, 
                 kernel_size=3, 
                 stride=1, 
                 padding='same', 
                 dilation=1, 
                 bias=True, 
                 activation=None):
        super().__init__()

        self.depthwise = get_image_layer("Conv", rank)(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )
        self.pointwise = get_image_layer("Conv", rank)(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        return x
    

@register_with_ranks("GhostConv", ranks=(1, 2, 3))
class GhostConv(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 rank,
                 kernel_size=3, 
                 ratio=2, 
                 dw_kernel_size=3, 
                 stride=1, 
                 padding='same',
                 separable=False,
                 bias=True):
        super().__init__()

        self.out_channels = out_channels
        self.init_channels = int(out_channels / ratio)
        self.new_channels = out_channels - self.init_channels


        self.primary_conv = get_image_layer("Conv", rank)(
                                in_channels=in_channels,
                                out_channels=self.init_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=bias
                            )  
        
        cheap_conv_type = "DepthwiseSeparableConv" if separable else "DepthwiseConv"

        self.cheap_operation = get_image_layer(cheap_conv_type, rank)(
                                in_channels=self.init_channels,
                                out_channels=self.new_channels,
                                kernel_size=dw_kernel_size,
                                stride=1,
                                padding='same',
                                bias=bias
                            )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)



    


                

            



    

