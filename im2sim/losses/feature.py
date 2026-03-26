
import logging

import torch

from torch_geometric.nn import knn_interpolate

logger = logging.getLogger(__name__)

# def mse(gr1, gr2):
#     return torch.mean((gr1.x[...,3:] - gr2.x[...,3:])**2)

def mse(x1, x2):
    return torch.mean((x1 - x2)**2)

class KnnMSE(torch.nn.Module):

    def __init__(self, k=3):
        super().__init__()
        self.k = k 

    def forward(self, true_graph, pred_graph):
        c1 = true_graph.x[:,:3]
        c2 = pred_graph.x[:,:3]
        print(c1.shape, c2.shape, c1.max(), c2.max(), c1.min(), c2.min())
        f1 = true_graph.x[:,3:]
        f2 = pred_graph.x[:,3:]
        print(f1.shape, f2.shape, f1.max(), f2.max(), f1.min(), f2.min())
        b1 = true_graph.batch
        b2 = pred_graph.batch


        f1_interp = knn_interpolate(f1, c1, c2, b1, b2, k=self.k)
        print(f1_interp.shape, f2.shape, f1_interp.max(), f2.max(), f1_interp.min(), f2.min())

        return mse(f1_interp, f2)
        



    
