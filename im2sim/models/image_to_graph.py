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


# class I2GUNet(nn.Module):

#     def __init__(self,
#                 in_channels, 
#                 out_channels,
#                 filters=[16,32,64,128,256],
#                 cnn_kernel_size=3,
#                 cnn_res_depth=3,
#                 cnn_res_blocks_per_level=2,
#                 cnn_rank=3,
#                 cnn_norm_type=None,
#                 cnn_pool_type='MaxPool',
#                 cnn_pool_size=2,
#                 cnn_activation='relu',
#                 gnn_res_depth = 3,
#                 gnn_n_process_blocks = 1,
#                 gnn_n_deform_blocks = 3,
#                 template_edge_index=None,
#                 gnn_conv_type="ChebConv",
#                 gnn_conv_kwargs={'K':3},
#                 gnn_activation="relu",
#                 out_activation="linear",
#                 gnn_norm_type="InstanceNorm",
#                 batched_ops=True):
#         super().__init__()

#         logger.debug("Defining I2G layers...")
#         self.batched_ops = batched_ops
#         self.n_levels = len(filters)

#         self.encoder = ImageResEncoder(in_channels=in_channels,
#                                filters=filters,
#                                kernel_size=cnn_kernel_size,
#                                res_depth=cnn_res_depth,
#                                res_blocks_per_level=cnn_res_blocks_per_level,
#                                rank=cnn_rank,
#                                norm_type=cnn_norm_type,
#                                pool_type=cnn_pool_type,
#                                pool_size=cnn_pool_size,
#                                activation=cnn_activation)
        
#         gnn_config = dict(depth=gnn_res_depth, 
#                             conv_type=gnn_conv_type,
#                             conv_kwargs=gnn_conv_kwargs,
#                             activation=gnn_activation, 
#                             norm_type=gnn_norm_type)
       
#         self.align_conv = nn.ModuleList([
#                 gnn.Sequential('x, edge_index, batch',[
#                     (GraphConvResBlock(in_channels=out_channels*2 if i==0 else filt,
#                                 filters=filt, 
#                                 **gnn_config),'x, edge_index -> x')
#                     for i in range(gnn_n_process_blocks)
#                 ])
#                 for filt in reversed(filters)
#         ])

#         self.deform_conv = nn.ModuleList([
#                 gnn.Sequential('x, edge_index, batch',[
#                     (GraphConvResBlock(in_channels=filt*2 + out_channels if i==0 else filt,
#                                 filters=filt, 
#                                 **gnn_config), 'x, edge_index -> x')
#                     for i in range(gnn_n_deform_blocks)
#                 ])
#                 for filt in reversed(filters)
#         ])

#         self.convert_convs = nn.ModuleList([
#                 GraphConvBlock(in_channels=filt, 
#                                 filters=out_channels, 
#                                 depth=1, 
#                                 conv_type=gnn_conv_type,
#                                 conv_kwargs=gnn_conv_kwargs,
#                                 activation=out_activation, 
#                                 norm_type=None)
#                 for filt in reversed(filters)
#         ])
        
#         self.pools = nn.ModuleList([gnn.TopKPooling(in_channels=3) for _ in range(len(filters)-1)])
        
       
#         logger.debug("Done")

#     def forward(self, img, template):
#         logger.debug("In model forward pass...")
#         # Get encoder outputs
#         encoder_outputs = self.encoder(img)
#         encoder_outputs.reverse()
#         # Get pooled coordinates of the template 
        
#         x, edge_index, batch = template.x, template.edge_index, template.batch
#         feats, edges, batches,perms = [x],[edge_index],[batch],[]
#         # layer = TopKPooling(in_channels=3)
#         # logger.debug("device: x:%s, edge:%s, batch:%s, layer:%s", x.device, edge_index.device, batch.device, next(layer.parameters()).device)
        
#         for pool in self.pools:
#             logger.debug("Pooling the template...")
            
#             x, edge_index, _, batch, perm, _ = pool(x, edge_index, batch=batch)
#             feats.append(x)
#             edges.append(edge_index)
#             batches.append(batch)
#             logger.debug("shape - x:%s", tuple(x.shape))
#             perms.append(perm)
#         feats.reverse() # [10k, 5k, 2.5k, 1.25k, 600].reverse()
#         edges.reverse()
#         batches.reverse()
#         perms.append(torch.ones(feats[0].shape[0]).to(torch.bool)) # add a selector that does nothing for the bottom level
#         perms.reverse() # [5k, 2.5k,1.24k,600].reverse()

#         curr_mesh = feats[0]
#         outputs=[]
#         for enc, feat, edge, batch, perm, aconv, cconv, dconv in zip(encoder_outputs, feats, edges, batches, perms, self.align_conv, self.convert_convs, self.deform_conv):
#             logger.debug("In graph decoder...")
#             temp = torch.zeros_like(feat)
#             temp[perm] = curr_mesh
#             curr_mesh = torch.cat([temp, feat], axis=-1) # [N,6]
#             logger.debug("curr_mesh:%s, edge:%s", tuple(curr_mesh.shape), tuple(edge.shape))
#             graph_features = aconv(curr_mesh, edge) # [N,6] -> [N,F]
#             logger.debug("curr_mesh after align conv:%s", tuple(graph_features.shape))
#             proc_mesh = temp + cconv(graph_features,edge) # [N,F] -> [N,3]
#             logger.debug("curr_mesh after convert conv:%s", tuple(proc_mesh.shape))
#             proj = TrilinearProjection(domain_size=img.shape[-3:], batch_ops=self.batched_ops)(enc, proc_mesh[:,:3], batch)
#             logger.debug("projection shape:%s", tuple(proj.shape))
#             curr_mesh = torch.cat([proc_mesh, graph_features, proj], axis=-1) # [N, 3+F+C]
#             logger.debug("deform conv input shape:%s", tuple(curr_mesh.shape))
#             curr_mesh = dconv(curr_mesh, edge) # [N,F]
#             logger.debug("deform conv output shape:%s", tuple(curr_mesh.shape))
#             curr_mesh = proc_mesh + cconv(curr_mesh, edge) # [N,3]
#             logger.debug("convert conv output shape:%s", tuple(curr_mesh.shape))
#             out_graph = Data()
#             out_graph.x = curr_mesh
#             out_graph.edge_index = edge
#             out_graph.batch = batch
#             outputs.append(out_graph)
#         outputs[-1] = template.clone()
#         outputs[-1].x = curr_mesh
#         return outputs

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
        # features = multi_template[0].x.clone()
        for i, (dec_layer, enc) in enumerate(zip(self.decoder, encoder_features)):
            t = multi_template[i]
            deformation = dec_layer(enc,deformation,t.x, t.edge_index, t.batch)
            out_graph = Data(x=t.x+deformation, edge_index=t.edge_index, batch=t.batch)
            outputs.append(out_graph)
            if i < self.n_levels-1:
                deformation = gnn.unpool.knn_interpolate(deformation, t.x, multi_template[i+1].x)
                
               

        # outputs[-1] = template.clone()
        # outputs[-1].x = template.x+deformation+prev_deformation
        return outputs
            



def _get_projection_channels(filters, ids):
    channels = []
    for id_list in ids:
        sum = 0
        for id in id_list:
            sum += filters[id]
        channels.append(sum)
    return channels
