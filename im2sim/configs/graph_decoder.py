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


from dataclasses import dataclass, field

from im2sim.configs.graph_blocks import GraphConvBlockConfig


@dataclass
class SimpleGraphDecoderConfig:
    """
    Configuration class for defining the parameters of a graph decoder.

    Args:
        block_cfg (GraphConvBlockConfig):
            Configuration for the graph convolution block.

        protocol (str):
            The GNN protocol to use (`'update'` or `'predict'`). Default is `'update'`.

        pred_feature_key (str):
            The key in the graph data where the predicted features will be stored. Default is `'x'`.

        pred_feature_channels (list[int]):
            List of channel indices to predict. If `None`, all channels will be predicted. Default is `None`.

        include_ids (list[str]):
            List of graph attributes that contain node IDs to include in the prediction. If `None`, all nodes will be included. Default is `None`.

        exclude_ids (list[str]):
            List of graph attributes that contain node IDs to exclude from the prediction. If `None`, no nodes will be excluded. Default is `None`.
    """

    block_cfg: GraphConvBlockConfig = field(default_factory=GraphConvBlockConfig)
    protocol: str = "update"
    pred_feature_key: str = "x"
    pred_feature_channels: list[int] = None
    include_ids: list[str] = None
    exclude_ids: list[str] = None
