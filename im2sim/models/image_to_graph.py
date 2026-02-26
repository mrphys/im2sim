import logging

import torch
from torch import nn

from ..layers import ImageResEncoder,GraphResDecoderBlock,TrilinearProjection

logger = logging.getLogger(__name__)



class SimpleI2G(nn.Module):

    def __init__(self,
                in_channels, 
                out_channels,
                cnn_filters=[16,32,64,128,256],
                cnn_kernel_size=3,
                cnn_res_depth=3,
                cnn_res_blocks_per_level=2,
                cnn_rank=3,
                cnn_norm_type=None,
                cnn_pool_type='MaxPool',
                cnn_pool_size=2,
                cnn_activation='relu',
                projection_ids = [[3,4],[1,2],[0,1]],
                gnn_filters = [[384,288], [144,96], [64,32]],
                gnn_res_depth = 3,
                gnn_n_process_blocks = 1,
                gnn_n_deform_blocks = 3,
                template_edge_index=None,
                gnn_conv_type="ChebConv",
                gnn_conv_kwargs={'K':3},
                gnn_activation="relu",
                out_activation="linear",
                gnn_norm_type="InstanceNorm",
                batched_ops=True):
        super().__init__()

        logger.debug("Defining SimpleI2G layers...")
        self.batched_ops = batched_ops

        self.encoder = ImageResEncoder(in_channels=in_channels,
                               filters=cnn_filters,
                               kernel_size=cnn_kernel_size,
                               res_depth=cnn_res_depth,
                               res_blocks_per_level=cnn_res_blocks_per_level,
                               rank=cnn_rank,
                               norm_type=cnn_norm_type,
                               pool_type=cnn_pool_type,
                               pool_size=cnn_pool_size,
                               activation=cnn_activation)
        
        # self.projection_layers = nn.ModuleList([TrilinearProjection() for _ in range(len(cnn_filters))])
        self.projection_ids = projection_ids

        projection_channels = _get_projection_channels(cnn_filters, self.projection_ids)
        self.decoder_blocks = nn.ModuleList([
                            GraphResDecoderBlock(projection_channels=projection_channels[i], 
                                            graph_channels=out_channels if i==0 else gnn_filters[i-1][1],
                                            out_channels=out_channels,
                                            filters=gnn_filters[i],
                                            res_depth=gnn_res_depth,
                                            n_process_blocks = gnn_n_process_blocks,
                                            n_deform_blocks = gnn_n_deform_blocks,
                                            template_edge_index=template_edge_index,
                                            conv_type=gnn_conv_type,
                                            conv_kwargs=gnn_conv_kwargs,
                                            activation=gnn_activation,
                                            out_activation=out_activation,
                                            norm_type=gnn_norm_type)
                            for i in range(len(gnn_filters))

        ])
        logger.debug("Done")

    def forward(self, x, template):
        logger.debug("In model forward pass...")
        encoder_outputs = self.encoder(x)
        outputs = []
        graph_features = template.x.clone()
        curr_mesh = template.x.clone()
        for dec, ids in zip(self.decoder_blocks, self.projection_ids):
            proj_inp = torch.cat([TrilinearProjection(domain_size=x.shape[-3:], batch_ops=self.batched_ops)(encoder_outputs[id], curr_mesh[:,:3], template.batch)
                                  for id in ids], dim=-1)
            graph_features, curr_mesh = dec(graph_features,proj_inp,curr_mesh,template.edge_index)
            out_graph = template.clone()
            out_graph.x = curr_mesh
            outputs.append(out_graph)
        return outputs




def _get_projection_channels(filters, ids):
    channels = []
    for id_list in ids:
        sum = 0
        for id in id_list:
            sum += filters[id]
        channels.append(sum)
    return channels
