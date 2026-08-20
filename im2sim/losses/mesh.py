from itertools import combinations

import torch
import torch.nn.functional as F

from im2sim.data.mesh_utils import compute_edge_lengths


def edge_length_deviation_loss(gr1, gr2):
    """
    Computes the edge length deviation loss between two graphs.
    The loss is calculated as the squared difference between the edge length deviations of the two graphs, with a ReLU activation to ensure non-negativity.

    Args:
        gr1 (torch_geometric.data.Data): The ground truth graph, containing node features and edge indices.
        gr2 (torch_geometric.data.Data): The predicted graph, containing node features and edge indices.
    """
    ed1 = _edge_length_deviation(gr1.x[:, :3], gr1.edge_index)
    ed2 = _edge_length_deviation(gr2.x[:, :3], gr2.edge_index)
    return F.relu(ed2 - ed1) ** 2


def _edge_length_deviation(points, edges):
    lengths = compute_edge_lengths(points, edges)
    dev = lengths.std() / (lengths.mean() + 1e-8)
    return dev


def _aspect_ratio(x, cells):
    tet_vertices = x[cells, :]
    vert_ids = list(combinations(range(4), 2))
    edge_coords = tet_vertices[vert_ids, :]
    distances = torch.linalg.norm(edge_coords[:, 0, :, :] - edge_coords[:, 1, :, :], dim=-1)
    aspect_ratio = distances.max(0).values / distances.mean(0)
    return aspect_ratio.mean()


class AspectRatioLoss(torch.nn.Module):
    """
    Computes the aspect ratio difference between two graphs.
    Higher aspect ratios indicate more elongated tetrahedra, which can be undesirable in mesh generation.
    This loss can therefore be used as a regularizer to avoid generating meshes with poor quality tetrahedra.

    Args:
        cell_key (str): The key in the graph data object that corresponds to the tetrahedral cells.
                        This is used to select the appropriate cells for computing the aspect ratio.
    """

    def __init__(self, cell_key):
        super().__init__()
        if isinstance(cell_key, str):
            self.select = lambda obj: getattr(obj, cell_key)
        else:
            raise TypeError(f"face_key must be a graph attribute but is {cell_key}")

    def forward(self, gr1, gr2):
        """
        Computes the aspect ratio loss between two graphs.

        Args:
            gr1 (torch_geometric.data.Data): The ground truth graph, containing node features and cell indices.
            gr2 (torch_geometric.data.Data): The predicted graph, containing node features and cell indices.
        """
        ar1 = _aspect_ratio(x=gr1.x[:, :3], cells=self.select(gr1))
        ar2 = _aspect_ratio(x=gr2.x[:, :3], cells=self.select(gr2))
        return F.relu(ar2 - ar1) ** 2


def _face_norm(face_verts):
    side1 = face_verts[1] - face_verts[0]
    side2 = face_verts[2] - face_verts[0]

    norm_vec = torch.cross(side1, side2, dim=-1)
    unit_norm = norm_vec / (torch.norm(norm_vec, dim=-1, keepdim=True) + 1e-8)
    return unit_norm


def face_norm_loss(x1, x2, b1, b2, f1, f2):
    # x:[N,3], f:[3,M], norm: [3,M,3]
    norm1 = _face_norm(x1[f1, :])
    norm2 = _face_norm(x2[f2, :])

    batch1 = b1[f1[0]]
    batch2 = b2[f2[0]]

    consistency = torch.Tensor([0.0]).to(norm1.device)
    similarity = torch.Tensor([0.0]).to(norm1.device)

    for b in torch.unique(b1).tolist():
        mask1 = batch1 == b
        mask2 = batch2 == b
        consistency += torch.norm(norm2[mask2].std(0))
        similarity += torch.norm(norm1[mask1].mean(0) - norm2[mask2].mean(0))

    return consistency + similarity


class FaceNormalLoss(torch.nn.Module):
    """
    Computes the face normal consistency loss between two graphs.
    This loss encourages the predicted graph to have face normals that are consistent with those of the ground truth graph.
    This is useful for structures like inlet and outlet caps.

    Args:
        face_key (str): The key in the graph data object that corresponds to the face indices.
                        This is used to select the appropriate faces for computing the face normals.
    """

    def __init__(self, face_key=None):
        super().__init__()
        if isinstance(face_key, str):
            self.select = lambda obj: getattr(obj, face_key)
        else:
            raise TypeError(f"face_key must be a graph attribute but is {face_key}")

    def forward(self, gr1, gr2):
        """
        Computes the face normal loss between two graphs.

        Args:
            gr1 (torch_geometric.data.Data): The ground truth graph, containing node features and face indices.
            gr2 (torch_geometric.data.Data): The predicted graph, containing node features and face indices.
        """
        faces1 = self.select(gr1)
        faces2 = self.select(gr2)
        loss = face_norm_loss(
            x1=gr1.x[:, :3],
            x2=gr2.x[:, :3],
            b1=gr1.batch,
            b2=gr2.batch,
            f1=faces1,
            f2=faces2,
        )
        return loss


def tet_det(x, cells):
    """Return signed 6*volume per tet (scalar triple product)."""

    a = x[cells[0]]
    b = x[cells[1]]
    c = x[cells[2]]
    d = x[cells[3]]

    e1 = b - a
    e2 = c - a
    e3 = d - a

    det = (torch.cross(e1, e2) * e3).sum(-1)  # signed det(D) = signed 6V
    return det


def inversion_loss(x, cells, min_vol=1e-3):

    det6 = tet_det(x, cells)

    vol = det6 / 6.0

    return torch.maximum(torch.zeros(1).to(vol.device), min_vol - vol).mean()


class InversionLoss(torch.nn.Module):
    """
    Computes the inversion loss for tetrahedral meshes.
    This loss penalizes inverted tetrahedra in the predicted mesh.
    A tetrahedron is considered inverted if its volume is less than a specified minimum volume threshold.

    Args:
        cell_key (str): The key in the graph data object that corresponds to the tetrahedral cells.
                        This is used to select the appropriate cells for computing the inversion loss.
        min_vol (float): The minimum volume threshold below which a tetrahedron is considered inverted. Default is 1e-3.
    """

    def __init__(self, cell_key, min_vol=1e-3):
        super().__init__()
        if isinstance(cell_key, str):
            self.select = lambda obj: getattr(obj, cell_key)
        else:
            raise TypeError(f"face_key must be a graph attribute but is {cell_key}")
        self.min_vol = min_vol

    def forward(self, gr1, gr2):
        """
        Computes the inversion loss between two graphs.

        Args:
            gr1 (torch_geometric.data.Data):
            The ground truth graph, containing node features and cell indices.
            This is not used in the computation but is included for consistency with other loss functions.

            gr2 (torch_geometric.data.Data): The predicted graph, containing node features and cell indices.
        """
        cells = self.select(gr2)
        loss = inversion_loss(x=gr2.x[:, :3], cells=cells, min_vol=self.min_vol)
        return loss
