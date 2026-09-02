import torch
import torch_geometric as pyg

from torch_geometric.utils import dropout_edge, dropout_node
from im2sim.utils.layer_util import register_graph_layer, register_pyg_layer, get_activation
from im2sim.layers.custom_image_layers import EfficientChannelAttn, SqueezeExcite



@register_graph_layer(name="GraphActivation")
class GraphActivation(torch.nn.Module):
    """
    A wrapper for activation functions to be used in graph neural networks.

    Args:
        activation_name (str): The name of the activation function to be used. Must be one of the activation functions supported by PyTorch.
    """

    def __init__(self, activation_name: str):
        super().__init__()
        self.activation = get_activation(activation_name)

    def forward(self, in_graph: pyg.data.Data) -> pyg.data.Data:
        """
        Args:
            x (torch.Tensor): The input tensor of shape (N, C) where N is the number of nodes and C is the number of channels.
            batch (torch.Tensor): The batch tensor of shape (N,) indicating the batch index for each node.

        Returns:
            torch.Tensor: The tensor after applying the activation function.
        """
        graph = in_graph.clone()
        graph.x = self.activation(graph.x)
        return graph



@register_pyg_layer(name="DefaultGraphNorm")
class DefaultGraphNorm(torch.nn.Module):
    """
    The default normalisation for im2sim graph blocks.
    Uses torch.nn.InstanceNorm2d applied to graph data, but all channels are normalised together.

    Args:
        None

    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = torch.nn.InstanceNorm2d(1, affine=True, eps=1e-3)

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor of shape (N, C) where N is the number of nodes and C is the number of channels.
            batch (torch.Tensor): The batch tensor of shape (N,) indicating the batch index for each node.

        Returns:
            torch.Tensor: The normalized tensor of shape (N, C).
        """
        if x.dim() != 2:
            raise RuntimeError(f"Expected x.dim()==2, got {x.dim()}")
        shape = x.shape

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        for b in torch.unique(batch):
            x[batch == b] = self.norm(x[batch == b].unsqueeze(0).unsqueeze(0)).reshape(shape)
        return x
    
@register_graph_layer(name="GraphDropout")
class GraphDropout(torch.nn.Module):
    """
    Dropout for graph data, including node dropout, edge dropout, and channel dropout.

    Args:
        p_node (float): The probability of dropping a node. Default is 0.0 (no dropout).
        p_edge (float): The probability of dropping an edge. Default is 0.0 (no dropout).
        p_channel (float): The probability of dropping a channel. Default is 0.0 (no dropout).
    """

    def __init__(self, p_node: float, p_edge: float, p_channel: float):
        super().__init__()
        self.p_node = p_node
        self.p_edge = p_edge
        self.p_channel = p_channel

        self.node_dropout = torch.nn.Dropout1d(p_node) if p_node > 0 else None
        self.channel_dropout = torch.nn.Dropout1d(p_channel) if p_channel > 0 else None

    def forward(self, in_graph: pyg.data.Data) -> pyg.data.Data:
        """
        Args:
            x (torch.Tensor): The input tensor of shape (N, C) where N is the number of nodes and C is the number of channels.
            edge_index (torch.Tensor): The edge index tensor of shape (2, E) where E is the number of edges.

        Returns:
            torch.Tensor: The tensor after applying dropout to nodes and edges.
        """
        graph = in_graph.clone()

        if self.p_node > 0:
            graph.x = self.node_dropout(graph.x)

        if self.p_channel > 0:
            graph.x = self.channel_dropout(graph.x.permute(1,0)).permute(1,0)


        if self.p_edge > 0:

            graph.edge_index, edge_mask = dropout_edge(graph.edge_index, p=self.p_edge, training=self.training)

            if graph.edge_attr is not None:
                graph.edge_attr = graph.edge_attr[edge_mask]

            if graph.edge_weight is not None:
                graph.edge_weight = graph.edge_weight[edge_mask]

        return graph

@register_graph_layer(name="EdgeDropout")
class EdgeDropout(GraphDropout):
    """
    Dropout for edges in a graph.

    Args:
        p (float): The probability of dropping an edge. Default is 0.1.
    """

    def __init__(self, p: float = 0.1):
        super().__init__(p_node=0.0, p_edge=p, p_channel=0.0)   


@register_graph_layer(name="NodeDropout")
class NodeDropout(GraphDropout):
    """
    Dropout for nodes in a graph.

    Args:
        p (float): The probability of dropping a node. Default is 0.1.
    """

    def __init__(self, p: float = 0.1):
        super().__init__(p_node=p, p_edge=0.0, p_channel=0.0)


@register_graph_layer(name="ChannelDropout")
class ChannelDropout(GraphDropout):
    """
    Dropout for channels in a graph.

    Args:
        p (float): The probability of dropping a channel. Default is 0.1.
    """

    def __init__(self, p: float = 0.1):
        super().__init__(p_node=0.0, p_edge=0.0, p_channel=p)


@register_pyg_layer(name="EfficientChannelAttn")
class GraphECA(torch.nn.Module):
    """
    Efficient Channel Attention (ECA) for graph data.

    Args:
        channels (int): The number of input channels.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.eca = EfficientChannelAttn(in_channels, rank=1)

    def forward(self, x: torch.Tensor, batch:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor of shape (N, C) where N is the number of nodes and C is the number of channels.

        Returns:
            torch.Tensor: The tensor after applying ECA.
        """

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        for b in torch.unique(batch):
            x[batch == b] = self.eca(x[batch == b].permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)

        return x
    
@register_pyg_layer(name="SqueezeExcite")
class GraphSE(torch.nn.Module):
    """
    Squeeze-and-Excitation (SE) for graph data.

    Args:
        channels (int): The number of input channels.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.se = SqueezeExcite(in_channels, rank=1)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor of shape (N, C) where N is the number of nodes and C is the number of channels.

        Returns:
            torch.Tensor: The tensor after applying SE.
        """

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        for b in torch.unique(batch):
            x[batch == b] = self.se(x[batch == b].permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)

        return x