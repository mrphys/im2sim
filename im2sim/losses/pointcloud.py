import torch
import torch_geometric.nn as gnn


def chamfer_loss(y1, b1, y2, b2):
    nns1 = gnn.pool.knn(x=y2, y=y1, batch_x=b2, batch_y=b1, k=1)
    d1 = torch.linalg.norm(y1 - y2[nns1[1]], dim=-1).mean()
    nns2 = gnn.pool.knn(x=y1, y=y2, batch_x=b1, batch_y=b2, k=1)
    d2 = torch.linalg.norm(y2 - y1[nns2[1]], dim=-1).mean()
    return d1 + d2
        
    