# ==============================================================================
# Copyright 2026 University College London.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from abc import ABC, abstractmethod
from itertools import combinations

import torch
import torch.nn.functional as F

from im2sim.mesh_ops import compute_edge_lengths


class MeshLoss(torch.nn.Module, ABC):
    """
    Base class for mesh loss computation.

    Args:
        required_attrs (list[str] | None): List of attributes that must be present in the input graphs for loss computation. Default is `None`.
        supervised (bool): If `True`, both true and predicted graphs are required to have the specified attributes. If `False`, only the predicted graph is required to have the specified attributes. Default is `True`.
    """

    def __init__(self, required_attrs=None, supervised=True):
        super().__init__()
        self.supervised = supervised
        self.required_attrs = required_attrs + ["coords"]

    def forward(self, true_graph, pred_graph):
        """
        Computes the loss between the ground truth and predicted graphs.

        Args:
            true_graph (torch_geometric.data.Data): The ground truth graph.
            pred_graph (torch_geometric.data.Data): The predicted graph.
        """
        self._prepare_graphs(true_graph, pred_graph)
        return self._compute_loss(true_graph, pred_graph)

    def _prepare_graphs(self, true_graph, pred_graph):
        """
        Prepares the input graphs for loss computation.
        This method can be overridden in subclasses to perform any necessary preprocessing on the graphs before computing the loss.

        Args:
            true_graph (torch_geometric.data.Data): The ground truth graph.
            pred_graph (torch_geometric.data.Data): The predicted graph.
        """
        for attr in self.required_attrs:
            if self.supervised and (not hasattr(true_graph, attr) or not hasattr(pred_graph, attr)):
                raise ValueError(
                    f"Both true_graph and pred_graph must have '{attr}' attribute for supervised loss computation."
                )
            elif not self.supervised and not hasattr(pred_graph, attr):
                raise ValueError(f"pred_graph must have '{attr}' attribute.")

            if self.supervised and (
                getattr(true_graph, attr) is None or getattr(pred_graph, attr) is None
            ):
                raise ValueError(
                    f"Both true_graph and pred_graph must have '{attr}' attribute for supervised loss computation."
                )
            elif not self.supervised and getattr(pred_graph, attr) is None:
                raise ValueError(f"pred_graph must have '{attr}' attribute.")

    @abstractmethod
    def _compute_loss(self, true_graph, pred_graph):
        """
        Computes the loss between the ground truth and predicted graphs.
        This method should be implemented in subclasses to define the specific loss computation.

        Args:
            true_graph (torch_geometric.data.Data): The ground truth graph.
            pred_graph (torch_geometric.data.Data): The predicted graph.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class EdgeLengthDeviationLoss(MeshLoss):
    """
    Computes the edge length deviation loss between two graphs.
    The edge deviation is defined as the standard deviation of the edge lenghths divided by the mean edge length.

    The loss is calculated as the squared difference between the edge length deviations of the two graphs, with a ReLU activation to ensure non-negativity.

    Args:
        supervised (bool):
            If `True`, the difference in edge length deviations between the true and predicted graphs is computed.
            If `False`, only the edge length deviation of the predicted graph is computed. Default is `True`.

    .. note:
        The forward() method accepts 2 PyG graphs: true graph and pred_graph (see `im2sim.losses.MeshLoss`). 
        If supervised is False, the true graph is not used for loss computation.

        For the edge length deviation loss, the graph objects only need to have the `coords` and `edge_index` attributes.

    Example:

        For supervised training: 

        .. code-block:: python
        
            edge_criterion = EdgeLengthDeviationLoss()
            edge_loss = edge_criterion(true_graph, pred_graph)

        For unsupervised training:

        .. code-block:: python

            edge_criterion = EdgeLengthDeviationLoss(supervised=False)
            edge_loss = edge_criterion(None, pred_graph)

    """

    def __init__(self, supervised=True):
        """"""
        super().__init__(required_attrs=["edge_index"], supervised=supervised)

    def _compute_loss(self, true_graph, pred_graph):
        if self.supervised:
            return edge_length_deviation_loss(true_graph, pred_graph)
        else:
            return _edge_length_deviation(pred_graph.coords, pred_graph.edge_index)


def edge_length_deviation_loss(gr1, gr2):
    """
    Computes the edge length deviation loss between two graphs.
    The loss is calculated as the squared difference between the edge length deviations of the two graphs, with a ReLU activation to ensure non-negativity.

    Args:
        gr1 (torch_geometric.data.Data): The ground truth graph, containing node features and edge indices.
        gr2 (torch_geometric.data.Data): The predicted graph, containing node features and edge indices.

    .. note:
        The forward() method accepts 2 PyG graphs: true graph and pred_graph (see `im2sim.losses.MeshLoss`). 

        If supervised is False, the loss is the mean edge length deviation of the predicted graph.
    """
    ed1 = _edge_length_deviation(gr1.coords, gr1.edge_index)
    ed2 = _edge_length_deviation(gr2.coords, gr2.edge_index)
    return F.relu(ed2 - ed1) ** 2


def _edge_length_deviation(points, edges):
    lengths = compute_edge_lengths(points, edges)
    dev = lengths.std() / (lengths.mean() + 1e-8)
    return dev


class AspectRatioLoss(MeshLoss):
    """
    Computes the aspect ratio loss for tetrahedral meshes, which is a measure of skewness or distortion of the tetrahedra in the mesh.
    The aspect ratio is defined as the mean ratio of the maximum edge length to the mean edge length of all tetrahedra in the mesh.

    The loss is calculated as the squared difference between the aspect ratios of the true and predicted graphs, with a ReLU activation to ensure non-negativity.

    Args:
        cell_key (str): The key in the graph data object that corresponds to the tetrahedral cells.
                        This is used to select the appropriate cells for computing the aspect ratio loss.
                        The cell ids should be in the shape of [4, num_cells] for tetrahedral meshes.
        supervised (bool):
            If `True`, the difference in aspect ratios between the true and predicted graphs is computed.
            If `False`, only the aspect ratio of the predicted graph is computed. Default is `True`.

    .. note:
        The forward() method accepts 2 PyG graphs: true graph and pred_graph (see `im2sim.losses.MeshLoss`). 
        If supervised is False, the loss is the mean aspect ratio of the predicted graph.

        For the aspect ratio loss, the graph objects need to have `coords` and a cell_index attribute corresponding to `cell_key` of shape [4, n_cells] to store the ids of each tetrahedron.
        See `im2sim.mesh_ops.get_structure_cells` to generate the cell_index attribute from a tetrahedral mesh.

    Example:

        

        For supervised training: 

        .. code-block:: python

            skewness_criterion = AspectRatioLoss(cell_key = 'volume_cell_index')
            skew_loss = skewness_criterion(true_graph, pred_graph)

        For unsupervised training:

        .. code-block:: python

            skewness_criterion = AspectRatioLoss(cell_key = 'volume_cell_index', supervised=False)
            skew_loss = skewness_criterion(None, pred_graph)
    """

    def __init__(self, cell_key, supervised=True):
        super().__init__(required_attrs=[cell_key], supervised=supervised)
        self.cell_key = cell_key

    def _compute_loss(self, true_graph, pred_graph):
        if self.supervised:
            return aspect_ratio_loss(true_graph, pred_graph, self.cell_key)
        else:
            return _aspect_ratio(pred_graph.coords, pred_graph[self.cell_key])


def aspect_ratio_loss(gr1, gr2, cell_key):
    ar1 = _aspect_ratio(x=gr1.coords, cells=gr1[cell_key])
    ar2 = _aspect_ratio(x=gr2.coords, cells=gr2[cell_key])
    return F.relu(ar2 - ar1) ** 2


def _aspect_ratio(x, cells):
    tet_vertices = x[cells, :]
    vert_ids = list(combinations(range(4), 2))
    edge_coords = tet_vertices[vert_ids, :]
    distances = torch.linalg.norm(edge_coords[:, 0, :, :] - edge_coords[:, 1, :, :], dim=-1)
    aspect_ratio = distances.max(0).values / distances.mean(0)
    return aspect_ratio.mean()


class FaceNormalLoss(MeshLoss):
    """
    Computes the face normal loss for triangular meshes. The loss is a combination of two components: consistency and similarity.
    The consistency component measures the standard deviation of the face normals within each batch to enforce a flat surface,
    while the similarity component measures the difference between the mean face normals of the true and predicted graphs.

    Args:
        face_key (str): The key in the graph data object that corresponds to the triangular faces.
                        This is used to select the appropriate faces for computing the face normal loss.
                        The face ids should be in the shape of [3, num_faces] for triangular meshes.
        supervised (bool):
            If `True`, the difference in face normals between the true and predicted graphs is computed.
            If `False`, only the face normal consistency of the predicted graph is computed. Default is `True`.

    .. note:
        The forward() method accepts 2 PyG graphs: true graph and pred_graph (see `im2sim.losses.MeshLoss`). 
        If supervised is False, the loss is the face normal consistency of the predicted graph.

        For the face normal loss, the graph objects need to have `coords` and a face_index attribute corresponding to `face_key` of shape [3, n_faces] to store the ids of each triangle face.
        See `im2sim.mesh_ops.get_structure_cells` to generate the face_index attribute from a mesh.
    
    """

    def __init__(self, face_key, supervised=True):
        super().__init__(required_attrs=[face_key, "batch"], supervised=supervised)
        self.face_key = face_key

    def _compute_loss(self, true_graph, pred_graph):
        if self.supervised:
            return face_norm_loss(true_graph, pred_graph, self.face_key)
        else:
            return face_norm_consistency(pred_graph, self.face_key)


def _face_norm(face_verts):
    side1 = face_verts[1] - face_verts[0]
    side2 = face_verts[2] - face_verts[0]

    norm_vec = torch.cross(side1, side2, dim=-1)
    unit_norm = norm_vec / (torch.norm(norm_vec, dim=-1, keepdim=True) + 1e-8)
    return unit_norm


def face_norm_consistency(graph, face_key):
    norm = _face_norm(graph.coords[graph[face_key], :])
    batch = graph.batch[graph[face_key][0]]
    consistency = torch.Tensor([0.0]).to(norm.device)
    for b in torch.unique(batch).tolist():
        mask = batch == b
        consistency += torch.norm(norm[mask].std(0))
    return consistency


def face_norm_loss(true_graph, pred_graph, face_key):
    # x:[N,3], f:[3,M], norm: [3,M,3]
    norm1 = _face_norm(true_graph.coords[true_graph[face_key], :])
    norm2 = _face_norm(pred_graph.coords[pred_graph[face_key], :])

    batch1 = true_graph.batch[true_graph[face_key][0]]
    batch2 = pred_graph.batch[pred_graph[face_key][0]]

    consistency = torch.Tensor([0.0]).to(norm1.device)
    similarity = torch.Tensor([0.0]).to(norm1.device)

    for b in torch.unique(true_graph.batch).tolist():
        mask1 = batch1 == b
        mask2 = batch2 == b
        consistency += torch.norm(norm2[mask2].std(0))
        similarity += torch.norm(norm1[mask1].mean(0) - norm2[mask2].mean(0))

    return consistency + similarity


class InversionLoss(MeshLoss):
    """
    Computes the inversion loss for tetrahedral meshes.
    The loss is calculated based on the signed volume of the tetrahedra formed by the predicted coordinates and the specified cells.
    The forward() method accepts a true graph, but doesn't use it for loss computation. The loss is computed solely based on the predicted graph.

    Args:
        cell_key (str): The key in the graph data object that corresponds to the tetrahedral cells.
                        This is used to select the appropriate cells for computing the inversion loss.
                        The cell ids should be in the shape of [4, num_cells] for tetrahedral meshes.
        min_vol (float): The minimum volume threshold. Tetrahedra with volumes below this threshold will contribute to the loss. Default is 1e-3.

    .. note:
        The forward() method accepts 2 PyG graphs: true graph and pred_graph (see `im2sim.losses.MeshLoss`) but only uses the pred_graph for this loss.

        For the aspect ratio loss, the graph objects need to have `coords` and a cell_index attribute corresponding to `cell_key` of shape [4, n_cells] to store the ids of each tetrahedron.
        See `im2sim.mesh_ops.get_structure_cells` to generate the cell_index attribute from a tetrahedral mesh.
    """

    def __init__(self, cell_key: str, min_vol=1e-3):
        super().__init__(required_attrs=[cell_key], supervised=False)
        self.cell_key = cell_key
        self.min_vol = min_vol

    def _compute_loss(self, true_graph, pred_graph):
        cells = getattr(pred_graph, self.cell_key)
        loss = inversion_loss(x=pred_graph.coords, cells=cells, min_vol=self.min_vol)
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

    det = (torch.linalg.cross(e1, e2) * e3).sum(-1)  # signed det(D) = signed 6V
    return det


def inversion_loss(x, cells, min_vol=1e-3):

    det6 = tet_det(x, cells)

    vol = det6 / 6.0

    return torch.maximum(torch.zeros(1).to(vol.device), min_vol - vol).mean()


if __name__ == "__main__":
    from torch_geometric.data import Data

    # Create a simple tetrahedral mesh graph
    coords = torch.tensor([[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.5, 1.0, 0.0],
                           [0.5, 0.5, 1.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2, 3],
                               [1, 2, 3, 0]], dtype=torch.long)
    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long).T

    graph = Data(coords=coords, edge_index=edge_index, cells=cells)

    # Create a predicted graph with inverted tetrahedron
    pred_coords = torch.tensor([[0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [0.5, -1.0, 0.0],
                                [0.5, -0.5, -1.0]], dtype=torch.float32)
    pred_graph = Data(coords=pred_coords, edge_index=edge_index, cells=cells)

    # Compute inversion loss
    inv_loss = EdgeLengthDeviationLoss(supervised=True)
    loss_value = inv_loss(graph, pred_graph=pred_graph)

    print(loss_value)
