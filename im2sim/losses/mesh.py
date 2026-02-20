import torch
from itertools import combinations


def edge_length_deviation(points, edges):
    lengths = _compute_edge_lengths(points, edges)
    std_dev = lengths.sum(-1).std()
    return std_dev

def _compute_edge_lengths(points, edges):
    coords = points[:, edges]
    distances = (coords[:,0] - coords[:,1])**2
    return distances


def aspect_ratio(tet_vertices):
    vert_ids = list(combinations(range(4),2))
    edge_coords = tet_vertices[...,vert_ids,:]
    distances = torch.linalg.norm(edge_coords[...,0,:]-edge_coords[...,1,:], dim=-1)
    aspect_ratio = distances.max(-1).values/distances.mean(-1)
    return aspect_ratio.mean()

def edge_length_change(points1, points2, edges):
    original_edge_lengths = _compute_edge_lengths(points1, edges)
    new_edge_lengths = _compute_edge_lengths(points2, edges)
    movement = new_edge_lengths - original_edge_lengths
    return movement.abs().sum()