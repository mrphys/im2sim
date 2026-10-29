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

from im2sim.configs.core import Config, LayerConfig, register_config


@register_config
@dataclass
class StructVNConfig(Config):
    """
    Configuration class for the structure virtual-node network (`im2sim.models.StructVN`) and
    its decoder (`im2sim.models.StructVNDecoder`).

    The network encodes the node features, then applies `n_blocks` times `n_local` directional
    local layers followed by one virtual-node exchange with a virtual node per mesh structure (the structures are given to the network, not the config).
    The virtual nodes carry a summary of each structure (e.g. the dense boundary condition on the
    wall and on each cap) across the whole mesh in a single step.

    The last five attributes are the graph-protocol options of
    `im2sim.configs.SimpleGraphDecoderConfig`, and are only used by `StructVNDecoder`.

    Args:

        hidden_channels (int):
            Width of the network. Default is `32`.

        n_blocks (int):
            Number of (local layers, virtual-node exchange) blocks. Default is `3`.

        n_local (int):
            Number of directional local layers per block. Default is `2`.

        heads (int):
            Attention heads of the virtual-node self-attention. Must divide `hidden_channels`.
            Default is `4`.

        conv_cfg (LayerConfig):
            The graph convolution of the local layers, built as
            `conv(hidden_channels, hidden_channels, **kwargs)`. Any PyG layer taking
            `(x, edge_index)` can be used (e.g. `GATConv`, `GCNConv`, `SAGEConv`, `DirectionalConv`), and it must
            output `hidden_channels` channels (for multi-head layers use e.g.
            `kwargs={"heads": 4, "concat": False}`). Default is `GATConv` with one head.

        n_fourier (int):
            Number of Fourier frequencies of the position encoding, which has `3 + 6 * n_fourier`
            channels. Default is `4`.

        activation (str | None):
            The activation function used inside the network. Default is `"GELU"`.

        checkpointing (bool):
            Recompute layer activations in the backward pass to save memory during training.
            Default is `False`.

        protocol (str):
            The GNN protocol to use (`'update'` or `'predict'`). Default is `'predict'`.

        pred_feature_key (str):
            The key in the graph data where the predicted features will be stored. Default is `'x'`.

        pred_feature_channels (list[int]):
            List of channel indices to predict. If `None`, all channels will be predicted. Default is `None`.

        include_ids (list[str]):
            List of graph attributes that contain node IDs to include in the prediction. If `None`, all nodes will be included. Default is `None`.

        exclude_ids (list[str]):
            List of graph attributes that contain node IDs to exclude from the prediction. If `None`, no nodes will be excluded. Default is `None`.

    Examples:

        .. code-block:: python

            cfg = StructVNConfig(hidden_channels=64)
            gat_cfg = cfg.mod(conv_cfg=LayerConfig(name="GATConv", kwargs={}))
            cfg.save("my_config.json")
            loaded_cfg = StructVNConfig.load("my_config.json")
    """

    hidden_channels: int = 32
    n_blocks: int = 3
    n_local: int = 2
    heads: int = 4
    conv_cfg: LayerConfig = field(default_factory=lambda: LayerConfig(name="GATConv", kwargs={}))
    n_fourier: int = 4
    activation: str | None = "GELU"
    checkpointing: bool = False
    protocol: str = "predict"
    pred_feature_key: str = "x"
    pred_feature_channels: list[int] | None = None
    include_ids: list[str] | None = None
    exclude_ids: list[str] | None = None
