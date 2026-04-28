from itertools import combinations
import logging
from ..data.mesh_utils import _compute_edge_lengths


import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def edge_length_deviation_loss(gr1, gr2):
    ed1 = _edge_length_deviation(gr1.x[:,:3], gr1.edge_index)
    ed2 = _edge_length_deviation(gr2.x[:,:3], gr2.edge_index)
    return F.relu(ed2-ed1)**2
    
def _edge_length_deviation(points, edges):
    lengths = _compute_edge_lengths(points, edges)
    std_dev = lengths.sum(-1).std()
    return std_dev


def _aspect_ratio(x, cells):
    tet_vertices = x[cells,:]
    vert_ids = list(combinations(range(4),2))
    edge_coords = tet_vertices[vert_ids,:]
    distances = torch.linalg.norm(edge_coords[:,0,:,:]-edge_coords[:,1,:,:], dim=-1)
    aspect_ratio = distances.max(0).values/distances.mean(0)
    return aspect_ratio.mean()

class AspectRatioLoss(torch.nn.Module):
    
    def __init__(self, cell_key):
        super().__init__()
        if isinstance(cell_key, str):
            self.select = lambda obj: getattr(obj, cell_key)
        else:
            raise ValueError(f"face_key must be a graph attribute but is {cell_key}")
        

    def forward(self, gr1, gr2):
        ar1 = _aspect_ratio(x = gr1.x[:,:3],
                              cells = self.select(gr1))
        ar2 = _aspect_ratio(x = gr2.x[:,:3],
                              cells = self.select(gr2))
        return F.relu(ar2-ar1)**2


def _face_norm(face_verts):
    side1 = face_verts[1] - face_verts[0]
    side2 = face_verts[2] - face_verts[0]

    norm_vec = torch.cross(side1,side2, dim=-1)
    unit_norm = norm_vec/(torch.norm(norm_vec, dim=-1, keepdim=True)+1e-8)
    return unit_norm


def face_norm_loss(x1,x2,b1,b2,f1,f2):
    # x:[N,3], f:[3,M], norm: [3,M,3]
    norm1 = _face_norm(x1[f1,:])
    norm2 = _face_norm(x2[f2,:])

    batch1 = b1[f1[0]]
    batch2 = b2[f2[0]]

    consistency = torch.Tensor([0.0]).to(norm1.device)
    similarity = torch.Tensor([0.0]).to(norm1.device)

    for b in torch.unique(b1).tolist():
        mask1 = (batch1 == b)
        mask2 = (batch2 == b)
        consistency += torch.norm(norm2[mask2].std(0))
        similarity += torch.norm(norm1[mask1].mean(0) - norm2[mask2].mean(0))


    return consistency + similarity

 

class FaceNormalLoss(torch.nn.Module):
    
    def __init__(self, face_key = None):
        super().__init__()
        if isinstance(face_key, str):
            self.select = lambda obj: getattr(obj, face_key)
        else:
            raise ValueError(f"face_key must be a graph attribute but is {face_key}")


    def forward(self, gr1, gr2):
        faces1 = self.select(gr1)
        faces2 = self.select(gr2)
        loss = face_norm_loss(x1 = gr1.x[:,:3],
                              x2 = gr2.x[:,:3],
                              b1 = gr1.batch,
                              b2 = gr2.batch,
                              f1 = faces1,
                              f2 = faces2)
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
    
    def __init__(self, cell_key, min_vol = 1e-3):
        super().__init__()
        if isinstance(cell_key, str):
            self.select = lambda obj: getattr(obj, cell_key)
        else:
            raise ValueError(f"face_key must be a graph attribute but is {cell_key}")
        self.min_vol = min_vol


    def forward(self, gr1, gr2):
        cells = self.select(gr2)
        loss = inversion_loss(x = gr2.x[:,:3],
                              cells = cells, 
                              min_vol=self.min_vol)
        return loss