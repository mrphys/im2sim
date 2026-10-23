# ==============================================================================
# Copyright 2026 University College London.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import copy

import torch
import torch_geometric as pyg
from torch.utils.checkpoint import checkpoint

from im2sim.configs.virtual_node import StructVNConfig
from im2sim.layers.virtual_node_layers import (
    DirectionalBlock,
    GraphContext,
    StructVirtualNode,
)
from im2sim.models.graph_decoders import SimpleGraphDecoder
from im2sim.utils.layer_util import get_activation


class StructVN(torch.nn.Module):
    """
    Graph network that propagates dense boundary conditions with one virtual node per structure.

    ``encode -> [local layers -> virtual-node exchange] x n_blocks -> decode``. The local layers
    are sparse-matmul layers on ``[h, mean, d_x h, d_y h, d_z h]`` (see
    :class:`~im2sim.layers.virtual_node_layers.SparseOps`), and each virtual-node exchange pools
    the nodes of every structure (``cfg.structure_keys``) into a virtual node, lets the virtual
    nodes interact, and broadcasts the result back with attention. Geometry enters through the
    node coordinates (``coords`` or ``pos``), normalised per graph and Fourier encoded, and through
    the directional differences.

    Args:
        in_channels (int):
            Number of channels of `graph.x`, e.g. the dense boundary condition on the mesh.

        out_channels (int):
            Number of predicted channels per node.

        cfg (StructVNConfig):
            Configuration object that defines the parameters of the network.

    Examples:

        .. code-block:: python

            model = StructVN(in_channels=8, out_channels=4, cfg=StructVNConfig())
            out_graph = model(graph)  # out_graph.x has shape (N, 4)

        The graph must have `x`, `edge_index`, `coords` (or `pos`) and one index attribute per
        entry of `cfg.structure_keys`. Batches of graphs are supported.
    """

    def __init__(self, in_channels: int, out_channels: int, cfg: StructVNConfig):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cfg = cfg

        c = cfg.hidden_channels
        pos_channels = 3 + 6 * cfg.n_fourier
        self.encode = torch.nn.Sequential(
            torch.nn.Linear(in_channels + pos_channels, c),
            get_activation(cfg.activation),
            torch.nn.Linear(c, c),
        )
        self.local = torch.nn.ModuleList(
            torch.nn.ModuleList(DirectionalBlock(c, cfg.activation) for _ in range(cfg.n_local))
            for _ in range(cfg.n_blocks)
        )
        self.far = torch.nn.ModuleList(
            StructVirtualNode(c, len(cfg.structure_keys), cfg.heads, cfg.activation)
            for _ in range(cfg.n_blocks)
        )
        self.decode = torch.nn.Sequential(
            torch.nn.Linear(c, c),
            get_activation(cfg.activation),
            torch.nn.Linear(c, out_channels),
        )

    def forward(self, in_graph: pyg.data.Data) -> pyg.data.Data:
        """
        Args:
            in_graph (pyg.data.Data): The input graph.

        Returns:
            pyg.data.Data: A shallow copy of the graph with the prediction in `x`,
            of shape (N, out_channels).
        """
        ctx = GraphContext(in_graph, self.cfg.structure_keys)
        h = self.encode(torch.cat([in_graph.x, ctx.pos_features(self.cfg.n_fourier)], dim=1))

        # Checkpointing keeps training memory small (a real mesh has ~10^4 nodes x 5C features)
        if self.training and torch.is_grad_enabled() and self.cfg.checkpointing:

            def run(f, *args):
                return checkpoint(f, *args, use_reentrant=False)

        else:

            def run(f, *args):
                return f(*args)

        for local, far in zip(self.local, self.far, strict=True):
            for block in local:
                h = run(block, h, ctx.ops)
            h = run(far, h, ctx)

        graph = copy.copy(in_graph)
        graph.x = self.decode(h)
        return graph


class StructVNDecoder(SimpleGraphDecoder):
    """
    A graph decoder that uses a :class:`StructVN` in place of the graph convolution blocks of
    :class:`SimpleGraphDecoder`.

    It has the same forward signature, `(in_graph, projected_features)`, so it can be used as the
    `graph_decoder` of :class:`Im2SimBase`. The protocol options of the config
    (`protocol`, `pred_feature_key`, ...) work as for :class:`SimpleGraphDecoder`.

    Args:
        in_channels (int):
            Number of input channels for the network. This should match the number of channels in
            `graph.x` plus the number of channels in `projected_features`.
            If `cfg.pred_feature_key` is not `'x'`, `out_channels` is added automatically to account for the
            predicted/updated features that are concatenated to the input.

        out_channels (int):
            Number of output channels. This should match the number of channels in
            `graph.<pred_feature_key>` or `pred_feature_channels` if they are specified.

        cfg (StructVNConfig):
            Configuration for the network and the protocol.

        graph_channels (int, optional):
            See :class:`SimpleGraphDecoder`.

    Examples:

        .. code-block:: python

            cfg = StructVNConfig(pred_feature_key="cfd")
            decoder = StructVNDecoder(in_channels=8, out_channels=4, cfg=cfg)
            out_graph = decoder(graph)  # out_graph.cfd has shape (N, 4)
    """

    def _build_module(
        self, in_channels: int, out_channels: int, cfg: StructVNConfig
    ) -> torch.nn.Module:
        return StructVN(in_channels, out_channels, cfg)
