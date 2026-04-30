import logging

import torch
from torch import nn
import torch_geometric.nn as gnn

from .layer_util import get_graph_layer, get_activation
from .projections import TrilinearProjection
from ..data.mesh_utils import cluster_pool




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
                conv_kwargs={'K':1},
                activation='relu', 
                norm_type='InstanceNorm'):
        super().__init__() 

        conv = get_graph_layer(conv_type)
        self.convs = nn.ModuleList([
            conv(in_channels if i==0 else filters, filters, **conv_kwargs)
            for i in range(depth)
        ])

        # self.norms = nn.ModuleList([
        #     get_graph_layer(norm_type)(filters, affine=True) if norm_type else nn.Identity()
        #     for _ in range(depth)
        # ])

        self.norms = nn.ModuleList([
            # get_graph_layer(norm_type)(filters, affine=True) if norm_type else nn.Identity()
            torch.nn.InstanceNorm2d(1, affine=True, eps=1e-3) if norm_type else nn.Identity()
            for _ in range(depth)
        ])

        self.act = get_activation(activation)(inplace=True) if activation.lower() == 'relu' else get_activation(activation)()

    def forward(self, x, edge_index):
        # for conv, norm in zip(self.convs, self.norms):
        #     logger.debug("Graph features shape:%s", x.shape)
        #     x = self.act(norm(conv(x,edge_index)))
        # return x
        for conv, norm in zip(self.convs, self.norms):
            logger.debug("Graph features shape:%s", x.shape)
            x = conv(x,edge_index)
            # x = self.act(norm(x.permute(1,0).unsqueeze(0))).squeeze(0).permute(1,0)
            x = self.act(norm(x.unsqueeze(0).unsqueeze(0))).squeeze()
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
                conv_kwargs={'K':1},
                activation='relu', 
                norm_type='InstanceNorm'):
        super().__init__() 

        conv = get_graph_layer(conv_type)
        self.convs = nn.ModuleList([
            conv(in_channels if i==0 else filters, filters, **conv_kwargs)
            for i in range(depth)
        ])
        # self.norms = nn.ModuleList([
        #     get_graph_layer(norm_type)(filters, affine=True) if norm_type else nn.Identity()
        #     for _ in range(depth)
        # ])
        self.norms = nn.ModuleList([
            # get_graph_layer(norm_type)(filters, affine=True) if norm_type else nn.Identity()
            torch.nn.InstanceNorm2d(1, affine=True, eps=1e-3) if norm_type else nn.Identity()
            for _ in range(depth)
        ])
        self.act = get_activation(activation)(inplace=True) if activation.lower() == 'relu' else get_activation(activation)()
        

    def forward(self, x, edge_index):

        for i, (conv,norm) in enumerate(zip(self.convs, self.norms)):

            # x = norm(conv(x, edge_index))
            x = conv(x,edge_index)
            # x = norm(x.permute(1,0).unsqueeze(0)).squeeze(0).permute(1,0)
            x = norm(x.unsqueeze(0).unsqueeze(0)).squeeze()

            if i==0:
                x1 = x.clone()
            elif i<len(self.convs)-1:
                x = self.act(x)

        x = self.act((x+x1)/2)

        return x


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
                #  n_process_blocks = 1,
                 n_deform_blocks = 3,
                 template_edge_index=None,
                 conv_type="ChebConv",
                 conv_kwargs={'K':3},
                 activation="relu",
                 out_activation="linear",
                 norm_type="InstanceNorm"):
        super().__init__()
                            
        # self.process_conv = nn.ModuleList([
        #         GraphConvResBlock(in_channels=graph_channels if i==0 else filters[0],
        #                         filters=filters[0], 
        #                         **conv_config)
        #         for i in range(n_process_blocks)
        # ])
        self.process_conv = GraphConvBlock(in_channels=graph_channels, 
                                        filters=filters[0], 
                                        depth=1, 
                                        conv_type=conv_type,
                                        conv_kwargs=conv_kwargs,
                                        activation=activation, 
                                        norm_type=None)


        self.deform_conv = nn.ModuleList([
                GraphConvResBlock(in_channels=filters[0] + projection_channels+ out_channels if i==0 else filters[1], 
                                filters=filters[1], 
                                depth=res_depth, 
                                conv_type=conv_type,
                                conv_kwargs=conv_kwargs,
                                activation=activation, 
                                norm_type=norm_type)
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

        x = graph_features.clone()

        logger.debug("Process convs...")
        x = self.process_conv(x,edge_index)

        x = torch.cat([x, encoder_projection, prev_results], axis=-1)
        logger.debug("Decoder convs")
        for dconv in self.deform_conv: 
            x=dconv(x, edge_index)

        res = self.out_conv(x, edge_index) + prev_results
        return x,res


# class RecursiveTopKPooling(nn.Module):

#     def __init__(self,
#                  n_channels,
#                  n_levels = 5,
#                  compression_ratio = 0.5):
#         super().__init__()

#         self.pools = nn.ModuleList([
#             gnn.TopKPooling(in_channels=n_channels, ratio=compression_ratio)
#             for _ in range(n_levels-1)
#         ])

#     def forward(self, x, edge_index, edge_attr=None, batch=None):
#         x_list, edge_index_list, edge_attr_list, batch_list, perm_list, score_list = [x], [edge_index], [edge_attr], [batch], [], []
#         for pool in self.pools:
#             x, edge_index, edge_attr, batch, perm, score = pool(x, edge_index, edge_attr, batch)
#             x_list.append(x)
#             edge_index_list.append(edge_index)
#             edge_attr_list.append(edge_attr)
#             batch_list.append(batch)
#             perm_list.append(perm)
#             score_list.append(score)
#         perm_list.append(torch.ones(x.shape[0]).to(torch.bool))
#         score_list.append(torch.ones(x.shape[0]))
#         return  x_list, edge_index_list, edge_attr_list, batch_list, perm_list, score_list



class RecursiveClusterPooling(nn.Module):

    def __init__(self, n_levels = 5):
        super().__init__()
        self.n_levels = 5

    def forward(self, graph):
        multigraph = [graph.clone()]
        for _ in range(self.n_levels-1):
            graph = cluster_pool(graph)
            multigraph.append(graph.clone())
        return multigraph
 



class  GraphUNetDecoderBlock(nn.Module):

    def __init__(self,
                 #in_channels,
                 out_channels,
                 filters,
                 domain_size,
                 res_depth = 3,
                 n_align_blocks = 1,
                 n_deform_blocks = 3,
                 conv_type="ChebConv",
                 conv_kwargs={'K':3},
                 activation="relu",
                 out_activation="linear",
                 norm_type="InstanceNorm",
                 batched_ops = True):
        super().__init__()

        conv_config = dict(depth=res_depth, 
                            conv_type=conv_type,
                            conv_kwargs=conv_kwargs,
                            activation=activation, 
                            norm_type=norm_type)
        

        if n_align_blocks > 0:
            self.align=True
            self.align_conv = gnn.Sequential('x, edge_index, batch',[
                    (GraphConvResBlock(in_channels=out_channels*2 if i==0 else filters,
                                    filters=filters, 
                                    **conv_config), 'x, edge_index -> x')
                    for i in range(n_align_blocks)
            ])
        else:
            self.align=False


        self.deform_conv = gnn.Sequential('x, edge_index, batch',[
                (GraphConvResBlock(in_channels=out_channels+filters if i==0 else filters, 
                                filters=filters, 
                                **conv_config), 'x, edge_index -> x')
                for i in range(n_deform_blocks)
        ])

        self.convert_conv = GraphConvBlock(in_channels=filters, 
                                        filters=out_channels, 
                                        depth=1, 
                                        conv_type=conv_type,
                                        conv_kwargs=conv_kwargs,
                                        activation=out_activation, 
                                        norm_type=None)
        
        self.projection_args = {"domain_size":domain_size, "batch_ops":batched_ops}
        
    # INFO: removed graph features for now may want to add back 
    def forward(self,image_features,prev_deformation,template_x,edge_index,batch):

        # Move all zero points after unpooling
        if self.align:
            x = torch.cat([prev_deformation, template_x], axis=-1)
            x = self.align_conv(x, edge_index)
            x = self.convert_conv(x, edge_index)
            prev_deformation = prev_deformation+x

        # apply current deformation to template
        x = template_x + prev_deformation
        proj = TrilinearProjection(**self.projection_args)(image_features, x[:,:3], batch)
        x = torch.cat([x, proj], axis=-1)

        # get new deformations based on current position and projections
        x = self.deform_conv(x, edge_index)
        x = self.convert_conv(x, edge_index)
        return x+prev_deformation
        