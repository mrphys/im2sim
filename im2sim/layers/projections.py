import torch
import torch.nn.functional as F
from torch import nn
from ..data_ops.utils import make_padded_batch


class TrilinearProjection(nn.Module):

    def __init__(self, domain_size, batch_ops=True):
        super().__init__()
        self.domain_size = domain_size
        self.batch_ops = batch_ops

    def forward(self, encoder_outputs, graph_coords, batch):
        if self.batch_ops:
            # make a padded batched tensor and get the padding mask 
            # do the projection
            # extract the graph features using the padding mask 
            padded_coords, padding_mask = make_padded_batch(graph_coords, batch)
            grid = torch.stack([(2*padded_coords[:,j]/(d-1)) - 1
                                for j,d in enumerate(self.domain_size)], axis=-1) # normalise coords [-1,1] and divide by scale 
            grid = grid.unsqueeze(-2).unsqueeze(-2) # [B,N,3]->[B,N,1,1,3]
            grid = grid.type_as(encoder_outputs)
            projections = F.grid_sample(encoder_outputs[i],grid).squeeze().permute(0,2,1) # [B,C,N,1,1] -> [B,N,C]
            projections = projections[padding_mask]
        else:
            # loop through the batches and concatenate the projections
            projections = []
            for i in torch.unique(batch):
                coords = graph_coords[batch==i]
                grid = torch.stack([(2*coords[:,j]/(d-1)) - 1
                                for j,d in enumerate(self.domain_size)], axis=-1) # normalise coords [-1,1] and divide by scale 
                grid = grid.unsqueeze(0).unsqueeze(-2).unsqueeze(-2) # [N,3]->[1,N,1,1,3]
                grid = grid.type_as(encoder_outputs)
                projections.append(F.grid_sample(encoder_outputs[i].unsqueeze(0),grid).squeeze().permute(1,0)) # [1,C,N,1,1] -> [1,C,N] -> [N,C]
            projections = torch.cat(projections, dim=0)
        return projections
