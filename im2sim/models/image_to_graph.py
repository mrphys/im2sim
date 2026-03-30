import logging

import torch
from torch import nn
import torch_geometric.nn as gnn 
# from torch_geometric.nn import TopKPooling
from torch_geometric.data import Data

from ..layers import *

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


class I2GUNet(nn.Module):

    def __init__(self,
                in_channels, 
                out_channels,
                domain_size,
                filters=[16,32,64,128,256],
                cnn_kernel_size=3,
                cnn_res_depth=3,
                cnn_res_blocks_per_level=2,
                cnn_rank=3,
                cnn_norm_type='InstanceNorm',
                cnn_pool_type='MaxPool',
                cnn_pool_size=2,
                cnn_activation='leaky_relu',
                gnn_res_depth = 3,
                gnn_n_align_blocks = 1,
                gnn_n_deform_blocks = 3,
                gnn_conv_type="ChebConv",
                gnn_conv_kwargs={'K':3},
                gnn_activation="leaky_relu",
                gnn_norm_type="InstanceNorm",
                batched_ops=True):
        super().__init__()

        logger.debug("Defining I2G layers...")
        self.batched_ops = batched_ops
        self.n_levels = len(filters)
        self.n_channels = out_channels
        self.encoder = ImageResEncoder(in_channels=in_channels,
                               filters=filters,
                               kernel_size=cnn_kernel_size,
                               res_depth=cnn_res_depth,
                               res_blocks_per_level=cnn_res_blocks_per_level,
                               rank=cnn_rank,
                               norm_type=cnn_norm_type,
                               pool_type=cnn_pool_type,
                               pool_size=cnn_pool_size,
                               activation=cnn_activation)
        
       
        self.gpool = RecursiveClusterPooling(n_levels=self.n_levels)
        
        self.decoder = nn.ModuleList([
            GraphUNetDecoderBlock(#in_channels = out_channels if i==1 else filters[-i+1],
                                out_channels = out_channels,
                                filters = filters[-i],
                                domain_size = domain_size,
                                res_depth = gnn_res_depth,
                                n_align_blocks = 0 if i==1 else gnn_n_align_blocks,
                                n_deform_blocks = gnn_n_deform_blocks,
                                conv_type=gnn_conv_type,
                                conv_kwargs=gnn_conv_kwargs,
                                activation=gnn_activation, 
                                norm_type=gnn_norm_type,
                                batched_ops=batched_ops)
            for i in range(1,self.n_levels+1)
        ])
        
       
        logger.debug("Done")


    def forward(self, img, template):
        encoder_features = self.encoder(img)
        encoder_features.reverse()
        
        multi_template = self.gpool(template)
        multi_template.reverse()

        outputs = []
        deformation = torch.zeros_like(multi_template[0].x)

        for i, (dec_layer, enc) in enumerate(zip(self.decoder, encoder_features)):
            t = multi_template[i]
            deformation = dec_layer(enc,deformation,t.x, t.edge_index, t.batch)
            if i < self.n_levels-1:
                out_graph = Data(x=t.x+deformation, edge_index=t.edge_index, batch=t.batch)
                outputs.append(out_graph)
                deformation = gnn.unpool.knn_interpolate(deformation, t.x, multi_template[i+1].x)

        out_graph = template.clone()
        out_graph.x = template.x+deformation
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
