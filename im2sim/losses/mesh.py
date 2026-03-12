from itertools import combinations
import logging
from ..data.mesh_utils import _compute_edge_lengths

import torch

logger = logging.getLogger(__name__)

def edge_length_deviation_loss(gr1, gr2):
    return _edge_length_deviation(gr2.x[:,:3], gr2.edge_index)

    
def _edge_length_deviation(points, edges):
    lengths = _compute_edge_lengths(points, edges)
    std_dev = lengths.sum(-1).std()
    return std_dev



def _aspect_ratio(tet_vertices):
    vert_ids = list(combinations(range(4),2))
    edge_coords = tet_vertices[...,vert_ids,:]
    distances = torch.linalg.norm(edge_coords[...,0,:]-edge_coords[...,1,:], dim=-1)
    aspect_ratio = distances.max(-1).values/distances.mean(-1)
    return aspect_ratio.mean()
