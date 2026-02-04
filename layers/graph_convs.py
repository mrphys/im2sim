import torch
from torch import nn
from layer_util import get_graph_layer, get_activation


class GraphConvBlock(nn.Module):
    """
    A convolutional block for graph data 

    Args:
        in_channels (int): The number of channels in the input to the layer.
        filters (int, optional): The number of filters in each convolutional layer (default: 32)
        depth (int, optional): The number of successive convolutional layers (default: 2)
        conv_type (str, optional): The type of graph convolution to apply (default: "ChebConv", options: "GraphConv", "GCNConv", "GATConv")
        conv_kwargs(dict, optional): Dictionary of keyword arguments for the chosen conv_type
        activation (str, optional): The activation function applied after each convolution (default: "relu", options: "leakyrelu","gelu","sigmoid","linear")
        norm_type (str, optional): The normalization method to apply between convolutions (default:"InstanceNorm", options: "BatchNorm",  "LayerNorm")

    Returns:
        A `torch.nn.Module` object.
    
    """
    def __init__(self, 
                in_channels, 
                filters, 
                depth=2, 
                conv_type='ChebConv',
                conv_kwargs={'K':3},
                activation='relu', 
                norm_type='InstanceNorm'):
        super().__init__() 

        conv = get_graph_layer(conv_type)
        self.convs = nn.ModuleList([
            conv(in_channels if i==0 else filters, filters, **conv_kwargs)
            for i in range(depth)
        ])

        self.norms = nn.ModuleList([
            get_graph_layer(norm_type)(filters) if norm_type else nn.Identity()
            for _ in range(depth)
        ])

        self.act = get_activation(activation)(inplace=True) if activation.lower() == 'relu' else get_activation(activation)

    def forward(self, x):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(self.act(conv(x)))
        return x

class GraphDecoder(nn.Module):
    """
    A cgraph convolutional decoder block with the same structure as MeshDeformNet and Image2Flow

    Args:
        encoder_channels (List[int]): The number of channels projected from the encoder to each decoder level (len=n_decoder_levels)
        out_channels (int): The number of output channels including node coordinates and features
        filters (List(List(int)), optional): The number of convolutional filters for each level (default:[[384,288], [144,96], [64,32]])
        convs_per_layer (int, optional): The number of successive convolutional layers in the deform convolution(default: 3)
        template_edge_index (torch.Tensor, optional): If template tensor is the fixed it can be passed (default: None)
        conv_type (str, optional): The type of graph convolution to apply (default: "ChebConv", options: "GraphConv", "GCNConv", "GATConv")
        conv_kwargs(dict, optional): Dictionary of keyword arguments for the chosen conv_type
        activation (str, optional): The activation function applied after each convolution (default: "relu", options: "leakyrelu","gelu","sigmoid","linear")
        out_activation (str, optional): The activation function applied after each convolution (default: "linear", options: "leakyrelu","gelu","sigmoid","relu","softmax")
        norm_type (str, optional): The normalization method to apply between convolutions (default:"InstanceNorm", options: "BatchNorm",  "LayerNorm")

    Returns:
        A `torch.nn.Module` object.
    
    """
    def __init__(self,
                 encoder_channels,
                 out_channels,
                 filters = [[384,288], [144,96], [64,32]],
                 convs_per_layer = 3,
                 template_edge_index=None,
                 conv_type="ChebConv",
                 conv_kwargs={'K':3},
                 activation="relu",
                 out_activation="linear",
                 norm_type="InstanceNorm"):
        super().__init__()

        self.n_levels = len(filters)

        self.process_convs = nn.ModuleList([
            GraphConvBlock(in_channels=out_channels if i==0 else filters[i-1][1], 
                            filters=filters[i][0], 
                            depth=1, 
                            conv_type=conv_type,
                            conv_kwargs=conv_kwargs,
                            activation=activation, 
                            norm_type=norm_type)
            for i in range(self.n_levels)
        ])

        self.deform_convs = nn.ModuleList([
            GraphConvBlock(in_channels=filters[i][0] + encoder_channels[i], 
                            filters=filters[i][1], 
                            depth=convs_per_layer, 
                            conv_type=conv_type,
                            conv_kwargs=conv_kwargs,
                            activation=activation, 
                            norm_type=norm_type)
            for i in range(self.n_levels)
        ])

        self.out_convs = nn.ModuleList([
            GraphConvBlock(in_channels=filters[i][1], 
                            filters=out_channels, 
                            depth=1, 
                            conv_type=conv_type,
                            conv_kwargs=conv_kwargs,
                            activation=out_activation, 
                            norm_type=None)
            for i in range(self.n_levels)
        ])

        self.edge_index=template_edge_index
    
    def forward(self, template, edge_index, encoder_projections):
        if edge_index is None:
            edge_index=self.edge_index
        
        outputs = []
        x=template.clone()
        for pconv, dconv, oconv, enc in zip(self.process_convs,self.deform_convs, self.out_convs, encoder_projections):
            x = dconv(torch.cat([pconv(x), enc], axis=-1))
            res = oconv(x) + template
            outputs.append(res)
            
        return outputs