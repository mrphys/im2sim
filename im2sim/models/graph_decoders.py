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

    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 cfg: SimpleGraphDecoderConfig):
        super().__init__()
        block = GraphConvBlock(in_channels=in_channels, 
                               out_channels=out_channels, 
                               cfg=cfg.block_cfg)
        self.decoder_block = GNN_PROTOCOLS[cfg.protocol](module=block, 
                                                pred_feature_key=cfg.pred_feature_key, 
                                                pred_feature_channels=cfg.pred_feature_channels,
                                                include_ids=cfg.include_ids,
                                                exclude_ids=cfg.exclude_ids)

    def forward(self, in_graph: pyg.data.Data, projected_features: torch.Tensor = None) -> pyg.data.Data:
        graph = in_graph.clone()
        init_channels = graph.x.shape[-1]
        if projected_features is not None:
            # Concatenate the image features to the node features
            graph.x = torch.cat([graph.x, projected_features], dim=-1)
        # Apply the graph convolution block
        graph = self.decoder_block(graph)
        graph.x = graph.x[:, :init_channels]  # Keep only the original number of channels
        return graph
    

