import torch
# import torch.nn.functional as F 
from kaolin.metrics.tetmesh import amips, tetrahedron_volume
from kaolin.ops.mesh.tetmesh import _validate_tet_vertices
from itertools import combinations
from kaolin.ops.mesh import face_areas
def equivolume(tet_vertices, tetrahedrons_mean=None, pow=4):
    r"""Compute the EquiVolume loss as devised by *Gao et al.* in `Learning Deformable Tetrahedral Meshes for 3D
    Reconstruction <https://nv-tlabs.github.io/DefTet/>`_ NeurIPS 2020.
    See `supplementary material <https://nv-tlabs.github.io/DefTet/files/supplement.pdf>`_ for the definition of the loss function.

    Args:
        tet_vertices (torch.Tensor):
            Batched tetrahedrons, of shape
            :math:`(\text{batch_size}, \text{num_tetrahedrons}, 4, 3)`.
        tetrahedrons_mean (torch.Tensor):
            Mean volume of all tetrahedrons in a grid,
            of shape :math:`(\text{batch_size})` or :math:`(1,)` (broadcasting).
            Default: Compute ``torch.mean(tet_vertices, dim=-1)``.
        pow (int):
            Power for the equivolume loss.
            Increasing power puts more emphasis on the larger tetrahedron deformation.
            Default: 4.

    Returns:
        (torch.Tensor):
            EquiVolume loss for each mesh, of shape :math:`(\text{batch_size})`.

    Example:
        >>> tet_vertices = torch.tensor([[[[0.5000, 0.5000, 0.7500],
        ...                                [0.4500, 0.8000, 0.6000],
        ...                                [0.4750, 0.4500, 0.2500],
        ...                                [0.5000, 0.3000, 0.3000]],
        ...                               [[0.4750, 0.4500, 0.2500],
        ...                                [0.5000, 0.9000, 0.3000],
        ...                                [0.4500, 0.4000, 0.9000],
        ...                                [0.4500, 0.4500, 0.7000]]],
        ...                              [[[0.7000, 0.3000, 0.4500],
        ...                                [0.4800, 0.2000, 0.3000],
        ...                                [0.9000, 0.4500, 0.4500],
        ...                                [0.2000, 0.5000, 0.1000]],
        ...                               [[0.3750, 0.4500, 0.2500],
        ...                                [0.9000, 0.8000, 0.7000],
        ...                                [0.6000, 0.9000, 0.3000],
        ...                                [0.5500, 0.3500, 0.9000]]]])
        >>> equivolume(tet_vertices, pow=4)
        tensor([[2.2961e-10],
                [7.7704e-10]])
    """
    _validate_tet_vertices(tet_vertices)

    # compute the volume of each tetrahedron
    volumes = tetrahedron_volume(tet_vertices)

    if tetrahedrons_mean is None:
        # finding the mean volume of all tetrahedrons in the tetrahedron grid
        tetrahedrons_mean = torch.mean(volumes, dim=-1).unsqueeze(-1)
    # tetrahedrons_mean = tetrahedrons_mean.reshape(1, -1)
    # compute EquiVolume loss
    equivolume_loss = torch.mean(torch.pow(
        torch.abs(volumes - tetrahedrons_mean), exponent=pow),
        dim=-1, keepdim=True)

    return equivolume_loss

def edge_length_deviation(points, edges):
    lengths = _compute_edge_lengths(points, edges)
    std_dev = lengths.sum(-1).std()
    return std_dev

def _compute_edge_lengths(points, edges):
    coords = points[:, edges]
    distances = (coords[:,0] - coords[:,1])**2
    return distances

def volume_loss(tet_vertices1, tet_vertices2):
    volumes1 = tetrahedron_volume(tet_vertices1).sum()
    volumes2 = tetrahedron_volume(tet_vertices2).sum()
    return torch.abs(volumes1 - volumes2)

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

def surface_to_volume(tet_vertices, tri_vertices, faces):
    volume = tetrahedron_volume(tet_vertices)
    area = face_areas(tri_vertices, faces)
    return volume.abs().sum()/area.abs().sum()