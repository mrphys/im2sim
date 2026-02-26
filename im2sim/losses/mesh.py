from itertools import combinations
import logging

import torch

logger = logging.getLogger(__name__)

def edge_length_deviation_loss(gr1, gr2):
    return _edge_length_deviation(gr2.x[:,:3], gr2.edge_index)

    
def _edge_length_deviation(points, edges):
    lengths = _compute_edge_lengths(points, edges)
    std_dev = lengths.sum(-1).std()
    return std_dev

def _compute_edge_lengths(points, edges):
    logger.debug('points_shape:%s, max_edge_index:%d', tuple(points.shape), edges.max())
    coords = points[edges]
    logger.debug('coords_shape:%s', tuple(coords.shape))
    distances = (coords[0] - coords[1])**2
    return distances


def _aspect_ratio(tet_vertices):
    vert_ids = list(combinations(range(4),2))
    edge_coords = tet_vertices[...,vert_ids,:]
    distances = torch.linalg.norm(edge_coords[...,0,:]-edge_coords[...,1,:], dim=-1)
    aspect_ratio = distances.max(-1).values/distances.mean(-1)
    return aspect_ratio.mean()
