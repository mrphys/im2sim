import numpy as np
import torch
from itertools import combinations
from torch_geometric.utils import to_undirected


# def get_node_ids(mesh):
#     cell_ids = np.zeros((len(mesh.points),))
#     for id in np.unique(mesh['CellEntityIds']):
#         cells = mesh.extract_cells(np.where(mesh['CellEntityIds'] == id)[0])['vtkOriginalPointIds']
#         nodes = np.unique(cells)
#         cell_ids[nodes] = id
#     return cell_ids

def add_structure_masks(data, mesh, structure_list):
    ids = np.unique(mesh['CellEntityIds'])
    missing_ids = set(range(len(structure_list))) - set(ids.tolist())
    for id in missing_ids:
        setattr(data, f'{structure_list[id]}_mask', torch.zeros((len(mesh.points))).to(torch.bool))
    for id, name in zip(ids, structure_list):
        cell_ids = torch.zeros((len(mesh.points)))
        cells = mesh.extract_cells(np.where(mesh['CellEntityIds'] == id)[0])['vtkOriginalPointIds']
        nodes = np.unique(cells)
        cell_ids[nodes] = 1
        setattr(data, f'{name}_mask', cell_ids.to(torch.bool))
     

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
