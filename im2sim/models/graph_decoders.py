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


import torch
import torch_geometric as pyg

from im2sim.configs.graph_decoder import SimpleGraphDecoderConfig
from im2sim.layers.graph_blocks import GraphConvBlock
from im2sim.models.gnn_wrappers import GNN_PROTOCOLS


class SimpleGraphDecoder(torch.nn.Module):
    """
    A graph decoder that takes a graph and image features and produces an updated graph.

    Args:
        in_channels (int):
            Number of input channels for the graph convolution block.
            This should match the number of channels in `graph.x` plus the number of channels in `image_features` if they are concatenated
            and the number of predicted/updated features if they are pred_feature_key is not `'x'`.

        out_channels (int):
            Number of output channels for the graph convolution block.
            This should match the number of channels in `graph.<pred_feature_key>` or `pred_feature_channels` if they are specified.

        cfg (GraphConvBlockConfig):
            Configuration for the graph convolution block.


    """

    def __init__(self, in_channels: int, out_channels: int, cfg: SimpleGraphDecoderConfig):
        super().__init__()

        in_channels += out_channels if cfg.pred_feature_key != 'x' else 0

        hidden_channels = cfg.block_cfg.hidden_channels if cfg.block_cfg.hidden_channels is not None else in_channels

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
            cfg=cfg.block_cfg.to_single_conv(),
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

        graph = in_graph.clone()
        init_channels = graph.x.shape[-1]

        if projected_features is not None:
            # Concatenate the image features to the node features
            graph.x = torch.cat([graph.x, projected_features], dim=-1)

        # Apply the process blocks
        graph = self.decoder(graph)

        # Keep only the original number of channels
        graph.x = graph.x[:, :init_channels]

        return graph
