import logging

import torch
import torch.nn.functional as F
from torch import nn

from ..data.mesh_utils import make_padded_batch

logger = logging.getLogger(__name__)



class TrilinearProjection(nn.Module):
    def __init__(self, domain_size):
        super().__init__()
        self.domain_size = domain_size

    def forward(self, encoder_outputs, graph_coords, batch):
        projections = []
        
        n_dims = graph_coords.shape[1]
        for i in torch.unique(batch).to(torch.int16):

            coords = graph_coords[batch == i]
            n_nodes = coords.shape[0]

            grid = torch.stack(
                [
                    (2 * coords[:, j] / (d - 1)) - 1
                    for j, d in enumerate(self.domain_size)
                ],
                axis=-1,
            )  # normalise coords [-1,1] and divide by scale

            grid = grid.reshape(1,n_nodes, 1, 1, n_dims) # [N,3]->[1,N,1,1,3]

            grid = grid.type_as(encoder_outputs)

            projections.append(
                F.grid_sample(
                    encoder_outputs[i].unsqueeze(0),
                    grid,
                    align_corners=True,
                    padding_mode="border",
                )
                .reshape(encoder_outputs.shape[1], -1)
                .permute(1, 0)
            )  # [1,C,N,1,1] -> [1,C,N] -> [N,C]

            projections = torch.cat(projections, dim=0)

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
            x = graph_features[batch == i, -3]
            y = graph_features[batch == i, -2]
            z = graph_features[batch == i, -1]

            factor = torch.tensor(self.image_dim / h, dtype=x.dtype, device=x.device)

            x = x / factor
            y = y / factor
            z = z / factor

            # floor / ceil with clamp
            x1 = torch.minimum(
                torch.floor(x), torch.tensor(h - 1, dtype=x.dtype, device=x.device)
            )
            x2 = torch.minimum(
                torch.ceil(x), torch.tensor(h - 1, dtype=x.dtype, device=x.device)
            )
            y1 = torch.minimum(
                torch.floor(y), torch.tensor(w - 1, dtype=x.dtype, device=x.device)
            )
            y2 = torch.minimum(
                torch.ceil(y), torch.tensor(w - 1, dtype=x.dtype, device=x.device)
            )
            z1 = torch.minimum(
                torch.floor(z), torch.tensor(d - 1, dtype=x.dtype, device=x.device)
            )
            z2 = torch.minimum(
                torch.ceil(z), torch.tensor(d - 1, dtype=x.dtype, device=x.device)
            )

            # cast to int for indexing
            x1 = x1.long()
            x2 = x2.long()
            y1 = y1.long()
            y2 = y2.long()
            z1 = z1.long()
            z2 = z2.long()

            # mimic tf.gather_nd(image_features[0], ...)
            img0 = image_features[i]

            def gather(img, xi, yi, zi):
                return img[..., xi, yi, zi]

            # --- z1 plane ---
            q11 = gather(img0, x1, y1, z1)
            q21 = gather(img0, x2, y1, z1)
            q12 = gather(img0, x1, y2, z1)
            q22 = gather(img0, x2, y2, z1)

            wx = (x - x1.float()).unsqueeze(0)
            wx2 = (x2.float() - x).unsqueeze(0)

            lerp_x1 = q21 * wx + q11 * wx2
            lerp_x2 = q22 * wx + q12 * wx2

            wy = (y - y1.float()).unsqueeze(0)
            wy2 = (y2.float() - y).unsqueeze(0)

            lerp_y1 = lerp_x2 * wy + lerp_x1 * wy2

            # --- z2 plane ---
            q11 = gather(img0, x1, y1, z2)
            q21 = gather(img0, x2, y1, z2)
            q12 = gather(img0, x1, y2, z2)
            q22 = gather(img0, x2, y2, z2)

            lerp_x1 = q21 * wx + q11 * wx2
            lerp_x2 = q22 * wx + q12 * wx2

            lerp_y2 = lerp_x2 * wy + lerp_x1 * wy2

            # --- z interpolation ---
            wz = (z - z1.float()).unsqueeze(0)
            wz2 = (z2.float() - z).unsqueeze(0)

            lerp_z = lerp_y2 * wz + lerp_y1 * wz2
            projections.append(lerp_z)

        projections = torch.cat(projections, dim=0).permute(1, 0)
        return projections
