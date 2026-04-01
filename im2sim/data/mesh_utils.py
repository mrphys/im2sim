import numpy as np
import torch
from itertools import combinations
from torch_geometric.utils import to_undirected
import torch_geometric.nn as gnn


def get_structure_ids(mesh, structure_dict):
    cells = get_structure_cells(mesh, structure_dict)

    if cells == -1:
        return -1

    ids = {f'{k.split("_cell_index")[0]}_index':torch.unique(v) for k, v in cells.items()}
    return ids


def get_structure_edges(mesh, structure_dict):
    cells = get_structure_cells(mesh, structure_dict)

    if cells == -1:
        return -1

    edges =  {k:None for k in structure_dict}

    for k, v in cells.items():
        e_ids =  list(combinations(range(v.shape[0]),2))
        cell_edges = v[e_ids,:]
        edges[f'{k.split("_cell_index")[0]}': cell_edges.permute(1,0,2).reshape(2,-1)] 

    edges = {f'{k}_edge_index':v for k,v in edges.items()}
    return edges
    

def get_structure_cells(mesh, structure_dict):
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


def set_attrs(data, attrs):
    for k,v in attrs.items():
        setattr(data, k, v)


     

def get_edges_tet(mesh):
    tet_cells = mesh.extract_cells(np.where(mesh['CellEntityIds'] == 0)[0])
    tet_cells = tet_cells.cells.reshape(-1, 5)[:, 1:]
    edges = np.reshape(np.array([list(combinations(cell,2)) for cell in tet_cells]), [-1,2])
    edges = torch.from_numpy(np.unique(edges, axis=0).T)
    edges = to_undirected(edges)
    return edges

def get_node_features(mesh, feature_names):
    features = torch.from_numpy(np.array([mesh.point_data[name] for name in feature_names]).T)
    return features

# def set_structure_masks(data, mesh, structure_list):
#     node_ids = get_node_ids(mesh)
#     for id, structure in enumerate(structure_list):
#             setattr(data, f"is_{structure}", torch.from_numpy(node_ids==id))
#     return data

def get_tet_cells(mesh):
    tet_cells = mesh.extract_cells(np.where(mesh['CellEntityIds'] == 0)[0])
    tet_cells = tet_cells.cells.reshape(-1, 5)[:, 1:]
    tet_cells = torch.from_numpy(tet_cells).permute(1,0)
    return tet_cells

def make_padded_batch(x, batch):
    jagged_x = [x[batch==i] for i in torch.unique(batch)]
    padded_x= torch.nn.utils.rnn.pad_sequence(jagged_x, batch_first=True)
    lengths = torch.tensor([len(s) for s in jagged_x])
    mask = torch.arange(padded_x.size(1))[None, :] < lengths[:, None]
    return padded_x, mask


def _compute_edge_lengths(points, edges):
    coords = points[edges]
    distances = (coords[0] - coords[1])**2
    return distances

def cluster_pool(mesh):
    distances = _compute_edge_lengths(mesh.x, mesh.edge_index).sum(-1)
    weights = 1/(distances + 1e-8)
    clusters = gnn.graclus(mesh.edge_index,weights, mesh.x.shape[0])
    pooled_mesh = gnn.avg_pool(clusters, mesh)
    return pooled_mesh

def extract_features(mesh, fnames):
    out = []
    for name in fnames:
        out.append(torch.from_numpy(mesh.point_data[name]))
    return torch.stack(out, dim=-1)
