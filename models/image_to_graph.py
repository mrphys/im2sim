from layers import ImageEncoder,GraphDecoderBlock,TrilinearProjection
import torch
from torch import nn



class SimpleI2G(nn.Module):

    def __init__(self,
                in_channels, 
                out_channels,
                cnn_filters=[16,32,64,128,256],
                cnn_kernel_size=3,
                cnn_convs_per_layer=2,
                cnn_rank=3,
                cnn_norm_type=None,
                cnn_pool_type='MaxPool',
                cnn_pool_size=2,
                cnn_activation='relu',
                projection_ids = [[3,4],[1,2],[0,1]],
                gnn_filters = [[384,288], [144,96], [64,32]],
                gnn_n_process_convs = 1,
                gnn_n_deform_convs = 3,
                template_edge_index=None,
                gnn_conv_type="ChebConv",
                gnn_conv_kwargs={'K':3},
                gnn_activation="relu",
                out_activation="linear",
                gnn_norm_type="InstanceNorm"):
        super().__init__()

        self.encoder = ImageEncoder(in_channels=in_channels,
                               filters=cnn_filters,
                               kernel_size=cnn_kernel_size,
                               convs_per_layer=cnn_convs_per_layer,
                               rank=cnn_rank,
                               norm_type=cnn_norm_type,
                               pool_type=cnn_pool_type,
                               pool_size=cnn_pool_size,
                               activation=cnn_activation)
        
        # self.projection_layers = nn.ModuleList([TrilinearProjection() for _ in range(len(cnn_filters))])
        self.projection_ids = projection_ids

        projection_channels = _get_projection_channels(cnn_filters, self.projection_ids)
        print(f"Projection channels: {projection_channels}")
        self.decoder_blocks = nn.ModuleList([
                            GraphDecoderBlock(projection_channels=projection_channels[i], 
                                            graph_channels=out_channels if i==0 else gnn_filters[i-1][1],
                                            out_channels=out_channels,
                                            filters=gnn_filters[i],
                                            n_process_convs = gnn_n_process_convs,
                                            n_deform_convs = gnn_n_deform_convs,
                                            template_edge_index=template_edge_index,
                                            conv_type=gnn_conv_type,
                                            conv_kwargs=gnn_conv_kwargs,
                                            activation=gnn_activation,
                                            out_activation=out_activation,
                                            norm_type=gnn_norm_type)
                            for i in range(len(gnn_filters))

        ])

    def forward(self, x, template):
        encoder_outputs = self.encoder(x)
        outputs = []
        graph = template.x.clone()
        out = template.x.clone()
        for dec, ids in zip(self.decoder_blocks, self.projection_ids):
            proj_inp = torch.cat([TrilinearProjection()(encoder_outputs[id], out[:,:3])
                                  for id in ids], dim=-1)
            print(graph.shape, proj_inp.shape, out.shape)
            graph, out = dec(graph,proj_inp,out,template.edge_index)
            outputs.append(out)
        return outputs




def _get_projection_channels(filters, ids):
    channels = []
    for id_list in ids:
        sum = 0
        for id in id_list:
            sum += filters[id]
        channels.append(sum)
    return channels
