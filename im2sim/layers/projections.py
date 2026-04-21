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
            projections = F.grid_sample(encoder_outputs,grid, align_corners=True, padding_mode='border').squeeze(-1).squeeze(-1).permute(0,2,1) # [B,C,N,1,1] -> [B,N,C]
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
                projections.append(F.grid_sample(encoder_outputs[i].unsqueeze(0),grid, align_corners=True, padding_mode='border').squeeze(0).squeeze(-1).squeeze(-1).permute(1,0)) # [1,C,N,1,1] -> [1,C,N] -> [N,C]
                logger.debug("projection %d shape:%s",i,tuple(projections[-1].shape))
            projections = torch.cat(projections, dim=0)
            logger.debug("final projection shape:%s",tuple(projections.shape))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("final projection vals: %.2f-%.2f",projections.min(), projections.max())
        return projections



class OGProjection(nn.Module):
    def __init__(self, image_dim):
        super().__init__()
        self.image_dim = image_dim

    def forward(self, image_features, graph_features, batch):
        projections = []
        for i in torch.unique(batch).to(torch.int16):
            # TensorFlow tf.shape equivalents
            h = image_features[i].shape[-3]
            w = image_features[i].shape[-2]
            d = image_features[i].shape[-1]

            # Last 3 coords
            x = graph_features[batch==i, -3]
            y = graph_features[batch==i, -2]
            z = graph_features[batch==i, -1]

            factor = torch.tensor(self.image_dim / h, dtype=x.dtype, device=x.device)

            x = x / factor
            y = y / factor
            z = z / factor

            # floor / ceil with clamp
            x1 = torch.minimum(torch.floor(x), torch.tensor(h - 1, dtype=x.dtype, device=x.device))
            x2 = torch.minimum(torch.ceil(x),  torch.tensor(h - 1, dtype=x.dtype, device=x.device))
            y1 = torch.minimum(torch.floor(y), torch.tensor(w - 1, dtype=x.dtype, device=x.device))
            y2 = torch.minimum(torch.ceil(y),  torch.tensor(w - 1, dtype=x.dtype, device=x.device))
            z1 = torch.minimum(torch.floor(z), torch.tensor(d - 1, dtype=x.dtype, device=x.device))
            z2 = torch.minimum(torch.ceil(z),  torch.tensor(d - 1, dtype=x.dtype, device=x.device))

            # cast to int for indexing
            x1 = x1.long()
            x2 = x2.long()
            y1 = y1.long()
            y2 = y2.long()
            z1 = z1.long()
            z2 = z2.long()

            # mimic tf.gather_nd(image_features[0], ...)
            img0 = image_features[i]

            def gather(xi, yi, zi):
                return img0[...,xi, yi, zi]

            # --- z1 plane ---
            q11 = gather(x1, y1, z1)
            q21 = gather(x2, y1, z1)
            q12 = gather(x1, y2, z1)
            q22 = gather(x2, y2, z1)

            wx  = (x - x1.float()).unsqueeze(0)
            wx2 = (x2.float() - x).unsqueeze(0)

            lerp_x1 = q21 * wx + q11 * wx2
            lerp_x2 = q22 * wx + q12 * wx2

            wy  = (y - y1.float()).unsqueeze(0)
            wy2 = (y2.float() - y).unsqueeze(0)

            lerp_y1 = lerp_x2 * wy + lerp_x1 * wy2

            # --- z2 plane ---
            q11 = gather(x1, y1, z2)
            q21 = gather(x2, y1, z2)
            q12 = gather(x1, y2, z2)
            q22 = gather(x2, y2, z2)

            lerp_x1 = q21 * wx + q11 * wx2
            lerp_x2 = q22 * wx + q12 * wx2

            lerp_y2 = lerp_x2 * wy + lerp_x1 * wy2

            # --- z interpolation ---
            wz  = (z - z1.float()).unsqueeze(0)
            wz2 = (z2.float() - z).unsqueeze(0)

            lerp_z = lerp_y2 * wz + lerp_y1 * wz2
            projections.append(lerp_z)

        projections = torch.cat(projections, dim=0).permute(1,0)
        return projections