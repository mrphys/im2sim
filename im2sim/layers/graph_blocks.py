import torch
import torch_geometric as pyg
import sys
sys.path.append('/Users/anirudh/Documents/im2sim/im2sim')
from im2sim.configs.core import LayerConfig
from im2sim.layers import custom_graph_layers
from im2sim.configs.graph_blocks import GraphConvBlockConfig

from im2sim.utils.layer_util import (
    apply_residual_connection,
    get_graph_layer,
)

class GraphConvBlock(torch.nn.Module):
    """
    A configurable image convolutional block that consists of a sequence of
    convolutional layers, normalization layers, dropout layers, and attention
    layers. The block supports residual connections and allows for flexible
    configuration of its components.

    Args:
        in_channels (int):
            Number of input channels.

        out_channels (int):
            Number of output channels.

        cfg (GraphConvBlockConfig):
            Configuration object that defines the parameters of the block.

    Examples:

        To create an GraphConvBlock with a depth of 3, ReLU activation, and softmax output activation, you can use the following code:

        .. code-block:: python

            cfg = GraphConvBlockConfig(depth=3, activation="ReLU", out_activation="softmax")
            model = GraphConvBlock(
                   rank=2,
                   in_channels=32,
                   out_channels=32,
                   cfg=cfg,
               )

        Models can be saved and loaded using the standard PyTorch methods:
        
        .. code-block:: python

            torch.save(model.state_dict(), "model.pth")
            model.load_state_dict(torch.load("model.pth"))

        Configs can also be saved and loaded using the methods provided in the `im2sim.configs.GraphConvBlockConfig` class:
    """

    def __init__(self, in_channels: int, out_channels: int,  cfg: GraphConvBlockConfig):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = cfg.depth
        self.activation = custom_graph_layers.GraphActivation(cfg.activation)
        self.out_activation = custom_graph_layers.GraphActivation(cfg.out_activation)
        self.conv_cfg = cfg.conv_cfg
        self.norm_cfg = cfg.norm_cfg
        self.attn_cfg = cfg.attn_cfg
        self.dropout_cfg = cfg.dropout_cfg
        self.dropout_position = (
            cfg.dropout_position
            if isinstance(cfg.dropout_position, list)
            else [cfg.dropout_position]
        )
        self.residual_connections = (
            cfg.residual_connections if cfg.residual_connections is not None else {}
        )
        self.residual_type = cfg.residual_type

        self._set_default_configs()
        self._validate_configs()

        self.layers = torch.nn.ModuleList()
        in_channels_per_layer = []
        in_channels_current = self.in_channels

        for i in range(self.depth):
            if i in self.residual_connections and self.residual_type.lower().strip() == "concat":
                for src in self.residual_connections[i]:
                    in_channels_current += in_channels_per_layer[src]

            in_channels_per_layer.append(in_channels_current)

            conv = get_graph_layer(
                    name = self.conv_cfg.name, 
                    kwargs = {"in_channels": in_channels_per_layer[-1], 
                                "out_channels": self.out_channels, 
                                **self.conv_cfg.kwargs}
                    )

            norm = get_graph_layer(
                name = self.norm_cfg.name,
                kwargs = {"in_channels":self.out_channels, 
                          **self.norm_cfg.kwargs}
            )

            dropout = (
                get_graph_layer(self.dropout_cfg.name, kwargs=self.dropout_cfg.kwargs)
                if (i + 1) in self.dropout_position
                else torch.nn.Identity()
            )

            pre_residual = self.attn_cfg.name is not None and (i + 1) in self.residual_connections
            no_residual_final = len(self.residual_connections.keys()) == 0 and i == self.depth - 1
            if pre_residual or no_residual_final:
                attn = get_graph_layer(
                    name = self.attn_cfg.name,
                    kwargs = {"in_channels":self.out_channels, 
                              **self.attn_cfg.kwargs}
                )
            else:
                attn = torch.nn.Identity()

            block = torch.nn.Sequential(
                conv,
                norm,
                dropout,
                attn,
                self.activation if i < self.depth - 1 else torch.nn.Identity(),
            )
            self.layers.append(block)

            in_channels_current = self.out_channels

    def _set_default_configs(self):
        if self.conv_cfg is None:
            self.conv_cfg = LayerConfig(name="GCNConv", kwargs={})
        if self.norm_cfg is None:
            self.norm_cfg = LayerConfig(name=None, kwargs={})
        if self.dropout_cfg is None:
            self.dropout_cfg = LayerConfig(name=None, kwargs={})
        if self.attn_cfg is None:
            self.attn_cfg = LayerConfig(name=None, kwargs={})

    def _validate_configs(self):
        assert self.attn_cfg.name in [None, "EfficientChannelAttn", "SqueezeExcite"], (
            f"Unsupported attention type: {self.attn_cfg.name}"
        )
        if self.dropout_cfg.name is not None:
            assert max(self.dropout_position) < self.depth, (
                "Dropout position must be less than depth"
            )

        if self.residual_type.lower().strip() == "concat" and any(
            dst >= self.depth for dst in self.residual_connections
        ):
            raise ValueError(
                "Residual connections with 'concat' type cannot be created on the last layer since it would change the output channels."
            )

    def forward(self, in_graph: pyg.data.Data) -> pyg.data.Data:
        """ """
        graph = in_graph.clone()
        outputs = [graph.x]
        for i, layer in enumerate(self.layers):
            if i in self.residual_connections:
                for src in self.residual_connections[i]:
                    graph.x = apply_residual_connection(
                        outputs[src], graph.x, connection_type=self.residual_type
                    )

            graph = layer(graph)
            outputs.append(graph.x)

        graph = self.out_activation(graph)
        graph.edge_index = in_graph.edge_index
        return graph


if __name__ == "__main__":
    block_cfg = GraphConvBlockConfig(
        depth=4,
        activation="LeakyReLU",
        out_activation="sigmoid",
        conv_cfg=LayerConfig(name="GATConv", kwargs={}),
        norm_cfg=LayerConfig(name="DefaultGraphNorm", kwargs={}),
        dropout_cfg=LayerConfig(name="EdgeDropout", kwargs={"p": 0.5}),
        attn_cfg=LayerConfig(name="SqueezeExcite", kwargs={}),
        # dropout_position=[1, 3],
        residual_connections={3: [1,0]},
        residual_type="concat"
    )
    block = GraphConvBlock(in_channels=16, out_channels=32, cfg=block_cfg)
    print(block)

    graph = pyg.data.Data(
        x=torch.randn(10, 16),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long),
    )

    graph.x = graph.x.requires_grad_(True)
    output_graph = block(graph)

    # Make a scalar loss
    loss = output_graph.x.sum()

    # Backward pass
    loss.backward()

    # Check input gradient
    assert graph.x.grad is not None, "Input gradient is None"
    assert torch.isfinite(graph.x.grad).all(), "Input gradient contains NaN/Inf"

    # Check parameter gradients
    for name, param in block.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} gradient is None"
            assert torch.isfinite(param.grad).all(), f"{name} gradient contains NaN/Inf"
            assert param.grad.abs().sum() > 0, f"{name} gradient is zero"

    print(graph.x.shape, graph.edge_index.shape)
    print(output_graph.x.shape, output_graph.edge_index.shape)
    print("All checks passed.")



