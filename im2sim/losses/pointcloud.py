import logging
import inspect

import torch
import torch_geometric.nn as gnn

logger = logging.getLogger(__name__)


def _compute_batch_chamfer(y1, y2, b1=None, b2=None):
    if b1==None:
        b1 = torch.zeros(y1.shape[0])
    if b2==None:
        b2 = torch.zeros(y2.shape[0])
    logging.debug("shapes - y1:%s, y2:%s, b1:%s, b2%s", 
                  tuple(y1.shape),tuple(y2.shape),tuple(b1.shape),tuple(b2.shape))
    nns1 = gnn.pool.knn(x=y2, y=y1, batch_x=b2, batch_y=b1, k=1)
    logging.debug("nn shape: %s", nns1.shape)
    d1 = torch.linalg.norm(y1 - y2[nns1[1]], dim=-1).mean()
    nns2 = gnn.pool.knn(x=y1, y=y2, batch_x=b1, batch_y=b2, k=1)
    logging.debug("nn shape: %s", nns2.shape)
    d2 = torch.linalg.norm(y2 - y1[nns2[1]], dim=-1).mean()
    return d1 + d2


class ChamferLoss(torch.nn.Module):
    
    def __init__(self, mask = None):
        super().__init__()
        if isinstance(mask, str):
            self.mask = lambda obj: getattr(obj, mask)
        elif inspect.isfunction(mask):
            self.mask = mask
        else:
            raise ValueError("mask must be either a graph attribute or a function")


    def forward(self, gr1, gr2):
        mask1 = self.mask(gr1)
        mask2 = self.mask(gr2)
        logging.debug("mask_type - %s",type(mask1))
        loss = _compute_batch_chamfer(y1 = gr1.x[mask1,:3],
                                      y2 = gr2.x[mask2,:3],
                                      b1 = gr1.batch[mask1],
                                      b2 = gr2.batch[mask2])
        return loss
