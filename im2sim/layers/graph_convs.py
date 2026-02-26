import logging

import torch
from torch import nn
from .layer_util import get_graph_layer, get_activation

logger = logging.getLogger(__name__)

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
                depth=1, 
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

        self.act = get_activation(activation)(inplace=True) if activation.lower() == 'relu' else get_activation(activation)()

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            logger.debug("Graph features shape:%s", x.shape)
            x = norm(self.act(conv(x,edge_index)))
        return x

class GraphConvResBlock(nn.Module):
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
                depth=3, 
                conv_type='ChebConv',
                conv_kwargs={'K':3},
                activation='relu', 
                norm_type='InstanceNorm'):
        super().__init__() 

        self.convs=nn.ModuleList(
                        GraphConvBlock(in_channels=in_channels if i==0 else filters,
                                    filters=filters,
                                    conv_type=conv_type,
                                    conv_kwargs=conv_kwargs,
                                    activation=activation,
                                    norm_type=norm_type)
                        for i in range(depth)
        )
        

    def forward(self, x, edge_index):
        x1 = self.convs[0](x,edge_index)
        x = self.convs[1](x1,edge_index)
        for conv in self.convs[2:]:
            x = conv(x,edge_index)
        return x+x1


class GraphResDecoderBlock(nn.Module):
    """
    A graph convolutional decoder block with the same structure as MeshDeformNet and Image2Flow

    Args:
        encoder_channels (List[int]): The number of channels projected from the encoder to each decoder level (len=n_decoder_levels)
        out_channels (int): The number of output channels including node coordinates and features
        filters (List(List(int)), optional): The number of convolutional filters for each level (default:[[384,288], [144,96], [64,32]])
        res_block_depth (int, optional): The number of successive convolutions in each residual block (default: 3)
        n_process_blocks (int, optional): The number of residual blocks prior to projection(default: 1)
        n_deform_blocks (int, optional): The number of residual blocks after projection(default: 3)
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
                 projection_channels,
                 graph_channels,
                 out_channels,
                 filters,
                 res_depth = 3,
                 n_process_blocks = 1,
                 n_deform_blocks = 3,
                 template_edge_index=None,
                 conv_type="ChebConv",
                 conv_kwargs={'K':3},
                 activation="relu",
                 out_activation="linear",
                 norm_type="InstanceNorm"):
        super().__init__()

        conv_config = dict(depth=res_depth, 
                            conv_type=conv_type,
                            conv_kwargs=conv_kwargs,
                            activation=activation, 
                            norm_type=norm_type)
        
        self.process_conv = nn.ModuleList([
                GraphConvResBlock(in_channels=graph_channels if i==0 else filters[0],
                                filters=filters[0], 
                                **conv_config)
                for i in range(n_process_blocks)
        ])


        self.deform_conv = nn.ModuleList([
                GraphConvResBlock(in_channels=filters[0] + projection_channels if i==0 else filters[1], 
                                filters=filters[1], 
                                **conv_config)
                for i in range(n_deform_blocks)
        ])

        self.out_conv = GraphConvBlock(in_channels=filters[1], 
                                        filters=out_channels, 
                                        depth=1, 
                                        conv_type=conv_type,
                                        conv_kwargs=conv_kwargs,
                                        activation=out_activation, 
                                        norm_type=None)
        
        self.edge_index=template_edge_index
    
    def forward(self,graph_features,encoder_projection,prev_results,edge_index):
        logger.debug("IN GRAPH DECODER")
        if edge_index is None:
            edge_index=self.edge_index

        x=graph_features.clone()
        logger.debug("Process convs...")
        for pconv in self.process_conv: x=pconv(x,edge_index)
        x = torch.cat([x, encoder_projection], axis=-1)
        logger.debug("Decoder convs")
        for dconv in self.deform_conv: x=dconv(x, edge_index)
        res = self.out_conv(x, edge_index) + prev_results

        return x,res
