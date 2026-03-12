import logging

import torch
import torch.nn.functional as F
from torch import nn
from ..data.mesh_utils import make_padded_batch

logger = logging.getLogger(__name__)

class TrilinearProjection(nn.Module):

    def __init__(self, domain_size, batch_ops=True):
        super().__init__()
        self.domain_size = domain_size
        self.batch_ops = batch_ops

    def forward(self, encoder_outputs, graph_coords, batch):
        logger.debug("IN PROJECTION")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Domain size: %s", tuple(self.domain_size))
            logger.debug("Encoder outputs -  shape:%s,  max:%.2f, min:%.2f", tuple(encoder_outputs.shape), encoder_outputs.max(), encoder_outputs.min())
            logger.debug("Graph coords -  shape:%s,  max:%.2f, min:%.2f", tuple(graph_coords.shape), graph_coords.max(), graph_coords.min())
            logger.debug("Batch -  shape:%s,  vals:%s", tuple(batch.shape), tuple(torch.unique(batch)))
        if self.batch_ops:
            # make a padded batched tensor and get the padding mask 
            # do the projection
            # extract the graph features using the padding mask 
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("batch: %s", tuple(torch.unique(batch)))
            padded_coords, padding_mask = make_padded_batch(graph_coords, batch)
            logger.debug("padded_shape: %s, mask_shape:%s",
                         tuple(padded_coords.shape), tuple(padding_mask.shape))
            grid = torch.stack([(2*padded_coords[...,j]/(d-1)) - 1
                                for j,d in enumerate(self.domain_size)], axis=-1) # normalise coords [-1,1] and divide by scale 
            grid = grid.unsqueeze(-2).unsqueeze(-2) # [B,N,3]->[B,N,1,1,3]]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("grid_shape: %s, grid_vals: %.2f-%.2f, encoder_outputs_shape:%s",tuple(grid.shape), grid.max(), grid.min(), tuple(encoder_outputs.shape))
            grid = grid.type_as(encoder_outputs)
            projections = F.grid_sample(encoder_outputs,grid, align_corners=True).squeeze(-1).squeeze(-1).permute(0,2,1) # [B,C,N,1,1] -> [B,N,C]
            logger.debug("projection_shape:%s",tuple(projections.shape))
            projections = projections[padding_mask]
            logger.debug("masked_projection_shape:%s",tuple(projections.shape))
            
        else:
            # loop through the batches and concatenate the projections
            projections = []
            for i in torch.unique(batch).to(torch.int16):
                coords = graph_coords[batch==i]
                logger.debug('coords shape: %s', tuple(coords.shape))
                grid = torch.stack([(2*coords[:,j]/(d-1)) - 1
                                for j,d in enumerate(self.domain_size)], axis=-1) # normalise coords [-1,1] and divide by scale 
                grid = grid.unsqueeze(0).unsqueeze(-2).unsqueeze(-2) # [N,3]->[1,N,1,1,3]
                if logger.isEnabledFor(logging.DEBUG):
                    # logger.debug("i: %s, %s",type(i), i.dtype)
                    logger.debug("grid_shape: %s, grid_vals: %.2f-%.2f, encoder_outputs_shape:%s",tuple(grid.shape), grid.max(), grid.min(), tuple(encoder_outputs[i].shape))
                grid = grid.type_as(encoder_outputs)
                projections.append(F.grid_sample(encoder_outputs[i].unsqueeze(0),grid, align_corners=True).squeeze().permute(1,0)) # [1,C,N,1,1] -> [1,C,N] -> [N,C]
                logger.debug("projection %d shape:%s",i,tuple(projections[-1].shape))
            projections = torch.cat(projections, dim=0)
            logger.debug("final projection shape:%s",tuple(projections.shape))
        return projections
