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

from copy import deepcopy

import torch
import torch_geometric as pyg

from im2sim.configs.graph_decoder import SimpleGraphDecoderConfig
from im2sim.layers.graph_blocks import GraphConvBlock
from im2sim.models.gnn_wrappers import GNN_PROTOCOLS


class SimpleGraphDecoder(torch.nn.Module):
    """
    A graph decoder that takes a graph and projected image features and produces an updated graph.

    Args:
        in_channels (int):
            Number of input channels for the graph convolution blocks.
            This should match the number of channels in `graph.x` plus the number of channels in `projected_features`.
            If `cfg.pred_feature_key` is not `'x'`, `out_channels` is added automatically to account for the
            predicted/updated features that are concatenated to the input.

        out_channels (int):
            Number of output channels for the graph convolution blocks.
            This should match the number of channels in `graph.<pred_feature_key>` or `pred_feature_channels` if they are specified.

        cfg (SimpleGraphDecoderConfig):
            Configuration for the graph decoder.

        graph_channels (int, optional):
            Number of channels in `graph.x` before the projected features are concatenated.
            Only used if `cfg.pred_feature_key` is not `'x'`: `graph.x` is then not written by the
            decoder, so an MLP maps the concatenated `in_channels` features back to `graph_channels`
            to give the output `graph.x`. This keeps the shape of `graph.x` fixed, so the decoder can
            be applied iteratively. If None, it defaults to `in_channels` (no projected features).

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cfg: SimpleGraphDecoderConfig,
        graph_channels: int | None = None,
    ):
        super().__init__()

        self.pred_feature_key = cfg.pred_feature_key

        # Maps graph.x with the projected features concatenated back to its original channels
        self.x_mlp = None
        if cfg.pred_feature_key != "x":
            graph_channels = graph_channels if graph_channels is not None else in_channels
            self.x_mlp = torch.nn.Sequential(
                torch.nn.Linear(in_channels, in_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(in_channels, graph_channels),
            )

        in_channels += out_channels if cfg.pred_feature_key != "x" else 0

        hidden_channels = (
            cfg.block_cfg.hidden_channels
            if cfg.block_cfg.hidden_channels is not None
            else in_channels
        )

        process_blocks = [
            GraphConvBlock(
                in_channels=in_channels if i == 0 else hidden_channels,
                out_channels=hidden_channels,
                cfg=cfg.block_cfg,
            )
            for i in range(cfg.n_blocks)
        ]

        out_conv = GraphConvBlock(
            in_channels=hidden_channels,
            out_channels=out_channels,
            cfg=deepcopy(cfg.block_cfg).to_single_conv(),
        )

        module = torch.nn.Sequential(*process_blocks, out_conv)

        self.decoder = GNN_PROTOCOLS[cfg.protocol](
            module=module,
            in_channels=in_channels,
            out_channels=out_channels,
            pred_feature_key=cfg.pred_feature_key,
            pred_feature_channels=cfg.pred_feature_channels,
            include_ids=cfg.include_ids,
            exclude_ids=cfg.exclude_ids,
        )

    def forward(
        self, in_graph: pyg.data.Data, projected_features: torch.Tensor = None
    ) -> pyg.data.Data:
        """
        Args:
            in_graph (pyg.data.Data): The input graph.
            projected_features (torch.Tensor, optional): Image features projected onto the graph nodes,
                of shape (N, C). If given, they are concatenated to `graph.x` before decoding.

        Returns:
            pyg.data.Data: The updated graph. `graph.x` keeps its original number of channels.
        """

        graph = in_graph.clone()
        init_channels = graph.x.shape[-1]

        if projected_features is not None:
            # Concatenate the image features to the node features
            graph.x = torch.cat([graph.x, projected_features], dim=-1)

        # Apply the process blocks
        graph = self.decoder(graph)

        if self.x_mlp is not None:
            # graph.x was not written by the decoder, so map it back to the original channels
            graph.x = self.x_mlp(graph.x)
            if graph.x.shape[-1] != init_channels:
                raise ValueError(
                    f"graph.x has {init_channels} channels but the decoder maps it to "
                    f"{graph.x.shape[-1]}. Set `graph_channels` to the number of channels in graph.x."
                )
        else:
            # The prediction is written to the first channels of graph.x. Drop the projected
            # features so graph.x keeps its original number of channels.
            written_channels = (
                self.decoder.pred_feature_channels
                if self.decoder.pred_feature_channels is not None
                else range(self.decoder.out_channels)
            )
            if max(written_channels) >= init_channels:
                raise ValueError(
                    f"The decoder writes to channel {max(written_channels)} of graph.x, which only "
                    f"has {init_channels} channels before the projected features are added."
                )
            graph.x = graph.x[:, :init_channels]

        return graph
