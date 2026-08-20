import logging
from copy import copy

import torch
from torch import nn

from im2sim.utils.layer_util import get_activation, get_graph_layer, register_graph_layer
from im2sim.utils import api_util

logger = logging.getLogger(__name__)


@api_util.export("layers.DefaultGraphNorm")
@register_graph_layer(name="defaultnorm")
class DefaultGraphNorm(torch.nn.Module):
    """
    The default normalisation for im2sim graph blocks.
    Uses torch.nn.InstanceNorm2d applied to graph data, but all channels are normalised together.

    Args:
        None

    """

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.InstanceNorm2d(1, affine=True, eps=1e-3)

    def forward(self, x, batch):
        """
        Args:
            x (`torch.Tensor`): The input tensor of shape (N, C) where N is the number of nodes and C is the number of channels.
            batch (`torch.Tensor`): The batch tensor of shape (N,) indicating the batch index for each node.

        Returns:
            `torch.Tensor`: The normalized tensor of shape (N, C).
        """
        if x.dim() != 2:
            raise RuntimeError(f"Expected x.dim()==2, got {x.dim()}")
        shape = x.shape

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        for b in torch.unique(batch):
            x[batch == b] = self.norm(x[batch == b].unsqueeze(0).unsqueeze(0)).reshape(shape)
        return x


@api_util.export("layers.GraphConvBlock")
class GraphConvBlock(nn.Module):
    """
    A convolutional block for graph data

    Args:
        in_channels (int): The number of channels in the input to the layer.
        filters (int, optional): The number of filters in each convolutional layer (default: 32)
        depth (int, optional): The number of successive convolutional layers (default: 2)
        conv_type (str, optional): The type of graph convolution to apply (default: "GATConv", options: All PyG Convs)
        conv_kwargs(dict, optional): Dictionary of keyword arguments for the chosen conv_type
        activation (str, optional): The activation function applied after each convolution (default: "relu", options: All torch activations)
        norm_type (str, optional): The normalization method to apply between convolutions (default:"defaultnorm", options: All PyG Norms)

    Returns:
        A `torch.nn.Module` object.

    """

    def __init__(
        self,
        in_channels,
        filters,
        depth=1,
        conv_type="GATConv",
        conv_kwargs=None,
        activation="ReLU",
        norm_type="defaultnorm",
        norm_kwargs=None,
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                get_graph_layer(
                    name=conv_type,
                    args=[in_channels if i == 0 else filters, filters],
                    kwargs=conv_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norms = nn.ModuleList(
            [
                get_graph_layer(name=norm_type, kwargs=norm_kwargs) if norm_type else nn.Identity()
                for _ in range(depth)
            ]
        )

        self.act = get_activation(activation)

    def forward(self, in_graph):
        graph = copy(in_graph)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            graph = conv(graph)
            graph = norm(graph)
            graph.x = self.act(graph.x)
        return graph


@api_util.export("layers.GraphConvResBlock")
class GraphConvResBlock(nn.Module):
    """
    A convolutional block for graph data

    Args:
        in_channels (int): The number of channels in the input to the layer.
        filters (int, optional): The number of filters in each convolutional layer (default: 32)
        depth (int, optional): The number of successive convolutional layers (default: 2)
        conv_type (str, optional): The type of graph convolution to apply (default: "ChebConv", options: All PyG Convs)
        conv_kwargs(dict, optional): Dictionary of keyword arguments for the chosen conv_type
        activation (str, optional): The activation function applied after each convolution (default: "relu", options: All torch activations)
        norm_type (str, optional): The normalization method to apply between convolutions (default:"InstanceNorm", options: All PyG Norms)

    Returns:
        A `torch.nn.Module` object.

    """

    def __init__(
        self,
        in_channels,
        filters,
        depth=3,
        conv_type="GATConv",
        conv_kwargs=None,
        activation="ReLU",
        norm_type="defaultnorm",
        norm_kwargs=None,
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                get_graph_layer(
                    name=conv_type,
                    args=[in_channels if i == 0 else filters, filters],
                    kwargs=conv_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norms = nn.ModuleList(
            [
                get_graph_layer(name=norm_type, kwargs=norm_kwargs) if norm_type else nn.Identity()
                for _ in range(depth)
            ]
        )

        self.act = get_activation(activation)

    def forward(self, in_graph):
        graph = copy(in_graph)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            graph = norm(conv(graph))

            if i == 0:
                x1 = graph.x
            elif i < len(self.convs) - 1:
                graph.x = self.act(graph.x)

        graph.x = self.act((graph.x + x1) / 2)

        return graph


@api_util.export("layers.GraphResDecoderBlock")
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
        conv_type (str, optional): The type of graph convolution to apply (default: "ChebConv", options: All PyG Convs)
        conv_kwargs(dict, optional): Dictionary of keyword arguments for the chosen conv_type
        activation (str, optional): The activation function applied after each convolution (default: "relu", options: All torch activations)
        out_activation (str, optional): The activation function applied after each convolution (default: "linear", options: All torch activations)
        norm_type (str, optional): The normalization method to apply between convolutions (default:"InstanceNorm",  options: All PyG Norms)

    Returns:
        A `torch.nn.Module` object.

    """

    def __init__(
        self,
        projection_channels,
        graph_channels,
        out_channels,
        filters,
        res_depth=3,
        n_deform_blocks=3,
        template_edge_index=None,
        conv_type="GATConv",
        conv_kwargs=None,
        activation="relu",
        out_activation="linear",
        norm_type="defaultnorm",
    ):
        super().__init__()

        self.process_conv = GraphConvBlock(
            in_channels=graph_channels,
            filters=filters[0],
            depth=1,
            conv_type=conv_type,
            conv_kwargs=conv_kwargs,
            activation=activation,
            norm_type=None,
        )

        self.deform_conv = nn.ModuleList(
            [
                GraphConvResBlock(
                    in_channels=filters[0] + projection_channels + out_channels
                    if i == 0
                    else filters[1],
                    filters=filters[1],
                    depth=res_depth,
                    conv_type=conv_type,
                    conv_kwargs=conv_kwargs,
                    activation=activation,
                    norm_type=norm_type,
                )
                for i in range(n_deform_blocks)
            ]
        )

        self.out_conv = GraphConvBlock(
            in_channels=filters[1],
            filters=out_channels,
            depth=1,
            conv_type=conv_type,
            conv_kwargs=conv_kwargs,
            activation=out_activation,
            norm_type=None,
        )

        self.edge_index = template_edge_index

    def forward(self, in_graph, prev_results, encoder_projection):
        if in_graph.edge_index is None and self.edge_index is not None:
            in_graph.edge_index = self.edge_index

        graph = copy(in_graph)
        graph = self.process_conv(graph)
        graph.x = torch.cat([graph.x, encoder_projection, prev_results], axis=-1)

        for dconv in self.deform_conv:
            graph = dconv(graph)

        new_results = self.out_conv(graph).x + prev_results
        return graph, new_results


# class GraphUNetDecoderBlock(nn.Module):

#     def __init__(self,
#                  #in_channels,
#                  out_channels,
#                  filters,
#                  domain_size,
#                  res_depth = 3,
#                  n_align_blocks = 1,
#                  n_deform_blocks = 3,
#                  conv_type="ChebConv",
#                  conv_kwargs={'K':3},
#                  activation="relu",
#                  out_activation="linear",
#                  norm_type="InstanceNorm",
#                  batched_ops = True):
#         super().__init__()

#         conv_config = dict(depth=res_depth,
#                             conv_type=conv_type,
#                             conv_kwargs=conv_kwargs,
#                             activation=activation,
#                             norm_type=norm_type)


#         if n_align_blocks > 0:
#             self.align=True
#             self.align_conv = gnn.Sequential('x, edge_index, batch',[
#                     (GraphConvResBlock(in_channels=out_channels*2 if i==0 else filters,
#                                     filters=filters,
#                                     **conv_config), 'x, edge_index -> x')
#                     for i in range(n_align_blocks)
#             ])
#         else:
#             self.align=False


#         self.deform_conv = gnn.Sequential('x, edge_index, batch',[
#                 (GraphConvResBlock(in_channels=out_channels+filters if i==0 else filters,
#                                 filters=filters,
#                                 **conv_config), 'x, edge_index -> x')
#                 for i in range(n_deform_blocks)
#         ])

#         self.convert_conv = GraphConvBlock(in_channels=filters,
#                                         filters=out_channels,
#                                         depth=1,
#                                         conv_type=conv_type,
#                                         conv_kwargs=conv_kwargs,
#                                         activation=out_activation,
#                                         norm_type=None)

#         self.projection_args = {"domain_size":domain_size, "batch_ops":batched_ops}

#     # INFO: removed graph features for now may want to add back
#     def forward(self,image_features,prev_deformation,template_x,edge_index,batch):

#         # Move all zero points after unpooling
#         if self.align:
#             x = torch.cat([prev_deformation, template_x], axis=-1)
#             x = self.align_conv(x, edge_index)
#             x = self.convert_conv(x, edge_index)
#             prev_deformation = prev_deformation+x

#         # apply current deformation to template
#         x = template_x + prev_deformation
#         proj = TrilinearProjection(**self.projection_args)(image_features, x[:,:3], batch)
#         x = torch.cat([x, proj], axis=-1)

#         # get new deformations based on current position and projections
#         x = self.deform_conv(x, edge_index)
#         x = self.convert_conv(x, edge_index)
#         return x+prev_deformation
