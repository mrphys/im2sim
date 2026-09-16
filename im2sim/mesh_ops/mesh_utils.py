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

import numpy as np
import torch
import torch_geometric.nn as gnn
from pyvista.core.pointset import PointGrid
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


class InputMeshError(ValueError):
    pass


def get_structure_ids(mesh: PointGrid, structure_dict: dict[int, str]) -> dict[str, torch.Tensor]:
    """
    Extracts node ids for different substructures in a pyvista PointGrid object.

    Args:
        mesh (pyvista.core.pointset.PointGrid): A pyvista mesh object.
        structure_dict (Dict[int, str]): A dictionary that maps pyvista 'CellEntityIds' to structure names.

    Returns:
        ids (Dict[str, torch.Tensor]): A dictionary with items in the format 'structurename_index': torch.Tensor(N), where N is the number of nodes in the structure.

    Example:

        .. code-block:: python

            # load in vtu mesh 
            mesh = pv.read('example_mesh.vtu')

            # if CellEntityIds are 0 for volume and 1 for surface, we can create a structure_dict like this:
            structure_dict = {0: "vol", 1: "surf"}

            ids = get_structure_ids(mesh, structure_dict)

            # ids will be a dictionary like {'vol_index': tensor([...]), 'surf_index': tensor([...])}
    """
    cells = get_structure_edges(mesh, structure_dict)
    ids = {f"{k.split('_edge_index')[0]}_index": torch.unique(v) for k, v in cells.items()}
    return ids


def get_structure_edges(mesh: PointGrid, structure_dict: dict[int, str]) -> dict[str, torch.Tensor]:
    """
    Extracts edges for different substructures in a pyvista PointGrid object.

    Args:
        mesh (pyvista.core.pointset.PointGrid): A pyvista mesh object.
        structure_dict (Dict[int, str]): A dictionary that maps pyvista 'CellEntityIds' to structure names.

    Returns:
        edges (Dict[str, torch.Tensor]): A dictionary with items in the format 'structurename_index': torch.Tensor(2, N),
        where N is the number of edges in the structure.

    Example:

        .. code-block:: python

            # load in vtu mesh
            mesh = pv.read('example_mesh.vtu')

            # if CellEntityIds are 0 for volume and 1 for surface, we can create a structure_dict like this:
            structure_dict = {0: "vol", 1: "surf"}

            edges = get_structure_edges(mesh, structure_dict)

            # edges will be a dictionary like {'vol_edge_index': tensor([[...], [...]]), 'surf_edge_index': tensor([[...], [...]])}
    """

    if _has_missing_ids(mesh, structure_dict):
        raise InputMeshError("Mesh has missing ids")

    edges = {}

    for i, key in structure_dict.items():
        edges[f"{key}_edge_index"] = get_edges(mesh, i)

    return edges


def get_edges(mesh: PointGrid, structure_id: int) -> torch.Tensor:
    """
    Extracts edges for a single structure in a pyvista PointGrid object.

    Args:
        mesh (pyvista.core.pointset.PointGrid): A pyvista mesh object.
        structure_id (int): An integer corresponding to the structure id

    Returns:
        edges (torch.Tensor):  Tensor of shape (2, N) where N is the number of edges in the structure.
    """
    submesh = mesh.extract_cells(np.where(mesh["CellEntityIds"] == structure_id))
    edges = submesh.extract_all_edges().lines.reshape(-1, 3)[:, 1:].T
    edges = torch.from_numpy(submesh["vtkOriginalPointIds"][edges]).long()
    edges = to_undirected(edges)
    return edges


def get_structure_cells(mesh: PointGrid, structure_dict: dict[int, str]) -> dict[str, torch.Tensor]:
    """
    Extracts cells for different substructures in a pyvista PointGrid object.

    Args:
        mesh (pyvista.core.pointset.PointGrid): A pyvista mesh object.
        structure_dict (Dict[int, str]): A dictionary that maps pyvista 'CellEntityIds' to structure names.

    Returns:
        cells (Dict[str, torch.Tensor]): A dictionary with items in the format 'structurename_index': torch.Tensor(m, N), where m is 3 for triangles and 4 for tetrahedrons
        and N is the number of cells in the structure.

    Example:

        .. code-block:: python

            # load in vtu mesh
            mesh = pv.read('example_mesh.vtu')

            # if CellEntityIds are 0 for volume and 1 for surface, we can create a structure_dict like this:
            structure_dict = {0: "vol", 1: "surf"}

            cells = get_structure_cells(mesh, structure_dict)

            # cells will be a dictionary like {'vol_cell_index': tensor([[...], [...], [...], [...]]), 'surf_cell_index': tensor([[...], [...], [...]])}

    """
    if _has_missing_ids(mesh, structure_dict):
        raise InputMeshError("Mesh has missing ids")

    out_dict = {k: None for k in structure_dict.values()}

    for id, k in structure_dict.items():
        submesh = mesh.extract_cells(np.where(mesh["CellEntityIds"] == id)[0])
        subcells = submesh.cells.reshape(-1, submesh.cells[0] + 1)[:, 1:]
        cells = submesh["vtkOriginalPointIds"][subcells]
        out_dict[k] = torch.from_numpy(cells).permute(1, 0).to(torch.long)

    out_dict = {f"{k}_cell_index": v for k, v in out_dict.items()}

    return out_dict


def _has_missing_ids(mesh: PointGrid, structure_dict: dict[int, str]) -> bool:
    ids = np.unique(mesh["CellEntityIds"])

    missing_ids = set(structure_dict.keys()) - set(ids.tolist())

    return len(missing_ids) != 0


def set_attrs(data: Data, attrs: dict[str, torch.Tensor]) -> None:
    """
    A helper function to set multiple attributes of a PyG Data object with keys and values from a dictionary.

    Args:
        data (torch_geometric.data.Data): Data object to be modified
        attrs (Dict[str, torch.Tensor]): a dictionary of attribute names and values to be set in the Data object

    Returns:
        None

    Examples:

        .. code-block:: python

            # load in vtu mesh
            mesh = pv.read('example_mesh.vtu')

            # create a PyG Data object
            data = Data(coords=torch.from_numpy(mesh.points).float())

            # get node ids for different structures
            structure_dict = {0: "vol", 1: "surf"}
            ids = get_structure_ids(mesh, structure_dict)

            # set the ids as attributes in the Data object
            set_attrs(data, ids)

            # now data will have attributes 'vol_index' and 'surf_index' with the corresponding node ids
    """
    for k, v in attrs.items():
        setattr(data, k, v)


def get_edges_tet(mesh: PointGrid) -> torch.Tensor:
    """
    A function to get the edge index for training from a tetrahedral pyvista mesh

    Args:
        mesh (pyvista.core.pointset.PointGrid): A pyvista mesh object.

    Returns:
        edges (torch.Tensor): A tensor of shape [2,M] where M is the number of edges and the values are the node ids

    Example:

        .. code-block:: python

            # load in vtu mesh
            mesh = pv.read('example_mesh.vtu')

            # get the edge index for the volume structure (assuming CellEntityIds 0 corresponds to volume)
            edges = get_edges_tet(mesh)

            # edges will be a tensor of shape [2, M] where M is the number of edges in the volume structure
    """
    edges = get_structure_edges(mesh, {0: "vol"})["vol_edge_index"]
    return edges


def get_edges_surf(mesh: PointGrid) -> torch.Tensor:
    """
    A function to get the edge index for training from a pyvista surface mesh. Use this function for meshes that are not tetrahedral, e.g. .vtk files.

    Args:
        mesh (pyvista.core.pointset.PointGrid): A pyvista mesh object.

    Returns:
        edges (torch.Tensor): A tensor of shape [2,M] where M is the number of edges and the values are the node ids

    Example:

        .. code-block:: python

            # load in vtk mesh
            mesh = pv.read('example_mesh.vtk')

            # get the edge index for the surface structure 
            edges = get_edges_surf(mesh)


    """
    edges = mesh.extract_all_edges().lines.reshape(-1, 3)[:, 1:]
    edges = torch.Tensor(edges).T.long()
    edges = to_undirected(edges)
    return edges


def get_node_features(mesh: PointGrid, feature_names: list[str]) -> torch.Tensor:
    """
    Extracts the node features from a pyvista mesh object based on the feature names provided.

    Args:
        mesh (pyvista.core.pointset.PointGrid): A pyvista mesh object.
        feature_names (List[str]): A list of feature names in the mesh pointdata.

    Returns:
        features (torch.Tensor): A tensor of shape [N,C] where N is the number of nodes and C is len(feature_names).

    Example:

        If each node in the mesh has features like pressure, x-velocity, y-velocity, and z-velocity, you can extract these features as follows:

        .. code-block:: python

            # load in vtu mesh
            mesh = pv.read('example_mesh.vtu')

            # get the node features for the specified feature names
            feature_names = ['pressure', 'x-velocity', 'y-velocity', 'z-velocity']
            features = get_node_features(mesh, feature_names)

            # features will be a tensor of shape [N, 4] where N is the number of nodes in the mesh and the columns correspond to the specified features
            
    """
    features = torch.from_numpy(np.array([mesh.point_data[name] for name in feature_names]).T)
    return features


def make_padded_batch(x: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A helper function to pad a batch of data to the same size.

    Takes a flat concatenated batch of variable-length graphs/pointclouds and
    pads them to a uniform length so they can be stacked into a dense tensor.

    Args:
        x (torch.Tensor): A tensor of features of shape (N, C) where N is the
            total number of nodes across all instances in the batch, and C is
            the number of features per node.
        batch (torch.Tensor): A tensor of shape (N,) containing integer indices
            in the range [0, B-1] where B is the batch size. Each value
            indicates which instance in the batch the corresponding node
            belongs to.

    Returns:
        padded_x (torch.Tensor): A dense tensor of shape (B, L, C) where B is
            the batch size and L is the length of the longest instance. Shorter
            instances are zero-padded to length L.
        mask (torch.BoolTensor): A boolean tensor of shape (B, L) where
            mask[i, j] is True if position j in instance i is a real node,
            and False if it is padding. Suitable for use as an attention mask
            or for zeroing out padded positions in a loss function.

    Example:

        .. code-block:: python
            # 5 nodes total, 3 instances: instance 0 has 3 nodes, instances 1 and 2 have 1 node each
            x = torch.randn(5, 8)
            batch = torch.tensor([0, 0, 0, 1, 2])
            padded_x, mask = make_padded_batch(x, batch)
            padded_x.shape  # (3, 3, 8)
            mask.shape      # (3, 3)
            mask
            tensor([[ True,  True,  True],
                    [ True, False, False],
                    [ True, False, False]])
    """
    jagged_x = [x[batch == i] for i in torch.unique(batch)]
    padded_x = torch.nn.utils.rnn.pad_sequence(jagged_x, batch_first=True)
    lengths = torch.tensor([len(s) for s in jagged_x])
    mask = torch.arange(padded_x.size(1))[None, :] < lengths[:, None]
    return padded_x, mask


def compute_edge_lengths(points: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """
    Computes the squared Euclidean distance for each edge in a mesh.

    Args:
        points (torch.Tensor): Node coordinate tensor of shape (N, D) where N
            is the number of nodes and D is the spatial dimensionality
            (e.g. 3 for 3D meshes).
        edges (torch.Tensor): Edge index tensor of shape (2, E) where E is the
            number of edges. Each column represents an edge as a pair of node
            indices [src, dst].

    Returns:
        distances (torch.Tensor): A tensor of shape (E, D) containing the
            per-dimension squared differences between the endpoints of each
            edge. Sum over the last dimension to get scalar squared edge
            lengths.

    Example:

        .. code-block:: python

            # 4 nodes in 3D space
            points = torch.tensor([[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0],
                                   [1.0, 1.0, 0.0],
                                   [0.0, 1.0, 1.0]])
            
            # 3 edges connecting the nodes
            edges = torch.tensor([[0, 1, 2],
                                  [1, 2, 3]])
            
            distances = compute_edge_lengths(points, edges)

            # distances will be a tensor of shape (3,) containing the lengths of the edges
    """
    coords = points[edges]
    distances = torch.linalg.norm(coords[0] - coords[1], dim=-1)
    return distances


def cluster_pool(mesh: Data) -> Data:
    """
    Performs Graclus clustering-based pooling on a mesh graph, coarsening it
    by merging nodes into clusters weighted by inverse edge length.

    Shorter edges produce higher weights, encouraging spatially close nodes to
    be merged together. This preserves the overall geometry of the mesh while
    reducing its resolution.

    Args:
        mesh (torch_geometric.data.Data): A PyTorch Geometric Data object with
            the following required attributes:
                - x (torch.Tensor): Node feature matrix of shape (N, C).
                - edge_index (torch.Tensor): Edge index tensor of shape (2, E).

    Returns:
        pooled_mesh (torch_geometric.data.Data): A coarsened PyTorch Geometric
            Data object with fewer nodes, where each node represents the
            average of the nodes in its cluster. Has the same structure as the
            input mesh with updated x and edge_index.

    Notes:
        - Edge weights are computed as 1 / (squared_length + 1e-8), where the
          epsilon prevents division by zero for degenerate zero-length edges.
        - Pooling is performed using torch_geometric.nn.avg_pool, so node
          features in each cluster are averaged.

    Example:

        .. code-block:: python
            # Assume mesh is a PyG Data object with x and edge_index
            pooled_mesh = cluster_pool(mesh)

            # pooled_mesh will have fewer nodes and edges, with features averaged over clusters
    """
    distances = compute_edge_lengths(mesh.x, mesh.edge_index)
    weights = 1 / (distances + 1e-8)
    clusters = gnn.graclus(mesh.edge_index, weights, mesh.x.shape[0])
    pooled_mesh = gnn.avg_pool(clusters, mesh)
    return pooled_mesh


def rasterize(points: torch.Tensor, im_shape: list[int], vox_sizes: list[float]) -> torch.Tensor:
    """
    Computes the squared Euclidean distance between voxel centroids in a grid to a pointcloud

    Args:
        points (torch.Tensor): Node coordinate tensor of shape (N, D) where N
            is the number of nodes and D is the spatial dimensionality
            (e.g. 3 for 3D meshes).
        im_shape (torch.Tensor): A list of dim sizes for the image/mask corresponding
            to the point cloud.
        vox_sizes (torch.Tensor): A list of voxel sizes for each dimension

    Returns:
        distances (torch.Tensor): A tensor of shape specified by im_shape where each voxel
            is the distance of the voxel centroid to the pointcloud.

    Example:
        .. code-block:: python

            # 4 nodes in 3D space
            points = torch.tensor([[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0],
                                   [1.0, 1.0, 0.0],
                                   [0.0, 1.0, 1.0]])
            
            # image shape and voxel sizes
            im_shape = [128, 128, 128]
            vox_sizes = [1.0, 1.0, 1.0]
            
            distances = rasterize(points, im_shape, vox_sizes)

            # distances will be a tensor of shape (128, 128, 128) containing the distance from each voxel centroid to the nearest point in the point cloud
    """
    im_coords = [
        torch.arange(size / 2, n, size) for n, size in zip(im_shape, vox_sizes, strict=True)
    ]
    grids = torch.meshgrid(*im_coords, indexing="ij")  # three [128,128,128] tensors
    coord_tensor = torch.stack(grids, dim=-1).reshape(-1, 3)

    nns = gnn.pool.knn(x=points, y=coord_tensor, k=1)
    dists = torch.linalg.norm(coord_tensor - points[nns[1]], dim=-1)
    return dists.reshape(im_shape)


def hard_threshold(y: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """
    Thresholds a Tensor y according to a specified float threshold. Every value less than the threshold
    is assigned 1.0 and values greater are assigned 0.0

    Args:
        y (torch.Tensor): tensor containing the raw values
        threshold (float): threshold value

    Returns:
        y_thresh (torch.Tensor): thresholded input tensor

    """
    return (y < threshold).float()


def soft_threshold(y, threshold=1.5, sharpness=10.0):
    """
    Thresholds a Tensor y according to a specified float threshold. Every value less than the threshold
    is assigned 1.0 and values greater are assigned 0.0

    Args:
        y (torch.Tensor): tensor containing the raw values
        threshold (float): threshold value

    Returns:
        y_thresh (torch.Tensor): thresholded input tensor

    """
    return torch.sigmoid(sharpness * (threshold - y))
