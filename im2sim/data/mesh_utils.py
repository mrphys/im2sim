import numpy as np
import torch
from itertools import combinations
from torch_geometric.utils import to_undirected
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from pyvista.core.pointset import PointGrid
from typing import Dict, List, Tuple


def get_structure_ids(mesh: PointGrid, structure_dict: Dict[int, str]) -> Dict[str, torch.Tensor]:
    """
    Extracts node ids for different substructures in a pyvista PointGrid object.

    Args:
        mesh (pyvista.core.pointset.PointGrid): A pyvista mesh object.
        structure_dict (Dict[int, str]): A dictionary that maps pyvista 'CellEntityIds' to structure names.
    
    Returns:
        ids (Dict[str, torch.Tensor]): A dictionary with items in the format 'structurename_index': torch.Tensor(N), where N is the number of nodes in the structure.
    """
    cells = get_structure_cells(mesh, structure_dict)

    if cells == -1:
        return -1

    ids = {f'{k.split("_cell_index")[0]}_index':torch.unique(v) for k, v in cells.items()}
    return ids

def get_structure_edges(mesh: PointGrid, structure_dict: Dict[int, str]) -> Dict[str, torch.Tensor]:
    cells = get_structure_cells(mesh, structure_dict)

    if cells == -1:
        return -1

    edges = {}

    for k, v in cells.items():
        edge_list = []

        for i in range(v.shape[1]):  # each cell
            nodes = v[:, i].tolist()
            edge_list += list(combinations(nodes, 2))

        edge_tensor = torch.tensor(edge_list, dtype=torch.long).t()
        edges[f'{k.split("_cell_index")[0]}_edge_index'] = edge_tensor

    return edges


def get_structure_edges(mesh: PointGrid, structure_dict: Dict[int, str]) -> Dict[str, torch.Tensor]:
    """
    Extracts edges for different substructures in a pyvista PointGrid object.

    Args:
        mesh (pyvista.core.pointset.PointGrid): A pyvista mesh object.
        structure_dict (Dict[int, str]): A dictionary that maps pyvista 'CellEntityIds' to structure names.
    
    Returns:
        edges (Dict[str, torch.Tensor]): A dictionary with items in the format 'structurename_index': torch.Tensor(2, N), 
        where N is the number of edges in the structure.
    """
    cells = get_structure_cells(mesh, structure_dict)

    if cells == -1:
        return -1

    edges =  {k:None for k in structure_dict}

    for k, v in cells.items():
        e_ids =  list(combinations(range(v.shape[0]),2))
        cell_edges = v[e_ids,:]
        edges[f'{k.split("_cell_index")[0]}'] =  cell_edges.permute(1,0,2).reshape(2,-1)

    edges = {f'{k}_edge_index':v for k,v in edges.items()}
    return edges
    
def get_structure_edges2(mesh: PointGrid, structure_dict: Dict[int, str]) -> Dict[str, torch.Tensor]:
    cells = get_structure_cells(mesh, structure_dict)

    edges = {}

    for k, v in cells.items():
        edge_list = []

        for i in range(v.shape[1]):  # each cell
            nodes = v[:, i]
            edge_list.append(torch.combinations(nodes, r=2))

        edge_index = torch.cat(edge_list, dim=0).T
        edge_index = torch.unique(edge_index, dim=1)

        edges[f'{k.split("_cell_index")[0]}_edge_index'] = edge_index

    return edges

def get_structure_cells(mesh: PointGrid, structure_dict: Dict[int, str]) -> Dict[str, torch.Tensor]:
    """
    Extracts cells for different substructures in a pyvista PointGrid object.

    Args:
        mesh (pyvista.core.pointset.PointGrid): A pyvista mesh object.
        structure_dict (Dict[int, str]): A dictionary that maps pyvista 'CellEntityIds' to structure names.
    
    Returns:
        cells (Dict[str, torch.Tensor]): A dictionary with items in the format 'structurename_index': torch.Tensor(m, N), where m is 3 for triangles and 4 for tetrahedrons
        and N is the number of cells in the structure.
    """
    ids = np.unique(mesh['CellEntityIds'])

    missing_ids = set(structure_dict.keys()) - set(ids.tolist())
    if len(missing_ids) != 0:
        return -1

    out_dict = {k:None for k in structure_dict.values()}

    
    for id, k in structure_dict.items():
        submesh = mesh.extract_cells(np.where(mesh['CellEntityIds'] == id)[0])
        subcells = submesh.cells.reshape(-1, submesh.cells[0]+1)[:,1:]
        cells = submesh['vtkOriginalPointIds'][subcells]
        out_dict[k] = torch.from_numpy(cells).permute(1,0).to(torch.int)
    
    out_dict = {f'{k}_cell_index':v for k,v in out_dict.items()}

    return out_dict


def set_attrs(data:Data, attrs:Dict[str, torch.Tensor]) -> None:
    """
    A helper function to set multiple attributes of a PyG Data object with keys and values from a dictionary.

    Args:
        data (torch_geometric.data.Data): Data object to be modified
        attrs (Dict[str, torch.Tensor]): a dictionary of attribute names and values to be set in the Data object 

    Returns:
        None
    """
    for k,v in attrs.items():
        setattr(data, k, v)


def get_edges_tet(mesh: PointGrid) -> torch.Tensor:
    """
    A function to get the edge index for training from a tetrahedral pyvista mesh 

    Args:
        mesh (pyvista.core.pointset.PointGrid): A pyvista mesh object.

    Returns:
        edges (torch.Tensor): A tensor of shape [2,M] where M is the number of edges and the values are the node ids
    """
    edges = get_structure_edges(mesh, {0:'vol'})['vol_edge_index']
    edges = to_undirected(edges)
    return edges

def get_edges_tet2(mesh: PointGrid) -> torch.Tensor:
    """
    A function to get the edge index for training from a tetrahedral pyvista mesh 

    Args:
        mesh (pyvista.core.pointset.PointGrid): A pyvista mesh object.

    Returns:
        edges (torch.Tensor): A tensor of shape [2,M] where M is the number of edges and the values are the node ids
    """
    edges = get_structure_edges2(mesh, {0:'vol'})['vol_edge_index']
    edges = to_undirected(edges)
    return edges

def get_node_features(mesh:PointGrid, feature_names: List[str]) -> torch.Tensor:
    """
    Extracts the node features from a pyvista mesh object based on the feature names provided.

    Args:
        mesh (pyvista.core.pointset.PointGrid): A pyvista mesh object.
        feature_names (List[str]): A list of feature names in the mesh pointdata.

    Returns:
        features (torch.Tensor): A tensor of shape [N,C] where N is the number of nodes and C is len(feature_names).
    """
    features = torch.from_numpy(np.array([mesh.point_data[name] for name in feature_names]).T)
    return features



def make_padded_batch(x: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        >>> # 5 nodes total, 3 instances: instance 0 has 3 nodes, instances 1 and 2 have 1 node each
        >>> x = torch.randn(5, 8)
        >>> batch = torch.tensor([0, 0, 0, 1, 2])
        >>> padded_x, mask = make_padded_batch(x, batch)
        >>> padded_x.shape  # (3, 3, 8)
        >>> mask.shape      # (3, 3)
        >>> mask
        tensor([[ True,  True,  True],
                [ True, False, False],
                [ True, False, False]])
    """
    jagged_x = [x[batch == i] for i in torch.unique(batch)]
    padded_x = torch.nn.utils.rnn.pad_sequence(jagged_x, batch_first=True)
    lengths = torch.tensor([len(s) for s in jagged_x])
    mask = torch.arange(padded_x.size(1))[None, :] < lengths[:, None]
    return padded_x, mask

def _compute_edge_lengths(points: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
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
    """
    distances = _compute_edge_lengths(mesh.x, mesh.edge_index).sum(-1)
    weights = 1 / (distances + 1e-8)
    clusters = gnn.graclus(mesh.edge_index, weights, mesh.x.shape[0])
    pooled_mesh = gnn.avg_pool(clusters, mesh)
    return pooled_mesh

