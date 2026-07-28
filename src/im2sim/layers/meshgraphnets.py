# This code is from https://github.com/echowve/meshGraphNets_pytorch/tree/master

import torch
from torch import nn
from torch_geometric.data import Data
from torch_scatter import scatter_add


class MGNEdgeBlock(nn.Module):
    def __init__(self, custom_func: nn.Module):

        super().__init__()
        self.net = custom_func

    def forward(self, graph):

        node_attr = graph.x
        senders_idx, receivers_idx = graph.edge_index
        edge_attr = graph.edge_attr

        edges_to_collect = []

        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_attr)

        collected_edges = torch.cat(edges_to_collect, dim=1)

        edge_attr = self.net(collected_edges)  # Update

        return Data(x=node_attr, edge_attr=edge_attr, edge_index=graph.edge_index)


class MGNNodeBlock(nn.Module):
    def __init__(self, custom_func: nn.Module):
        super().__init__()
        self.net = custom_func

    def forward(self, graph):
        # Decompose graph
        edge_attr = graph.edge_attr
        nodes_to_collect = []

        _, receivers_idx = graph.edge_index
        num_nodes = graph.num_nodes
        agg_received_edges = scatter_add(
            edge_attr, receivers_idx, dim=0, dim_size=num_nodes
        )

        nodes_to_collect.append(graph.x)
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)

        x = self.net(collected_nodes)
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)


def build_mlp(in_size, hidden_size, out_size, lay_norm=True):

    module = nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out_size),
    )
    if lay_norm:
        return nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))
    return module


class MGNEncoder(nn.Module):
    def __init__(self, edge_input_size=128, node_input_size=128, hidden_size=128):
        super().__init__()

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)

    def forward(self, graph):

        node_attr, edge_attr = graph.x, graph.edge_attr
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)

        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)


class MGNGnBlock(nn.Module):
    def __init__(self, hidden_size=128):

        super().__init__()

        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)

        self.eb_module = MGNEdgeBlock(custom_func=eb_custom_func)
        self.nb_module = MGNNodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):

        x = graph.x.clone()
        edge_attr = graph.edge_attr.clone()

        graph = self.eb_module(graph)
        graph = self.nb_module(graph)

        x = x + graph.x
        edge_attr = edge_attr + graph.edge_attr

        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)


class MGNDecoder(nn.Module):
    def __init__(self, hidden_size=128, output_size=2):
        super().__init__()
        self.decode_module = build_mlp(
            hidden_size, hidden_size, output_size, lay_norm=False
        )

    def forward(self, graph):
        return self.decode_module(graph.x)


class MeshGraphNet(nn.Module):
    def __init__(
        self,
        message_passing_num,
        node_input_size,
        edge_input_size,
        output_size,
        hidden_size=128,
    ):

        super().__init__()

        self.encoder = MGNEncoder(
            edge_input_size=edge_input_size,
            node_input_size=node_input_size,
            hidden_size=hidden_size,
        )

        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(MGNGnBlock(hidden_size=hidden_size))
        self.processer_list = nn.ModuleList(processer_list)

        self.decoder = MGNDecoder(hidden_size=hidden_size, output_size=output_size)

    def forward(self, graph):

        graph = self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        decoded = self.decoder(graph)

        return decoded
