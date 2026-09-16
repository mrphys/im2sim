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

import logging

import torch
import torch_geometric.nn as gnn

from im2sim.losses.mesh import MeshLoss

logger = logging.getLogger(__name__)


def _compute_batch_chamfer(y1, y2, b1=None, b2=None):
    if b1 is None:
        b1 = torch.zeros(y1.shape[0])
    if b2 is None:
        b2 = torch.zeros(y2.shape[0])
    logger.debug(
        "shapes - y1:%s, y2:%s, b1:%s, b2%s",
        tuple(y1.shape),
        tuple(y2.shape),
        tuple(b1.shape),
        tuple(b2.shape),
    )
    nns1 = gnn.pool.knn(x=y2, y=y1, batch_x=b2, batch_y=b1, k=1)
    logger.debug("nn shape: %s", nns1.shape)
    if nns1.shape[-1] == 0:
        return (y2 * 0).sum()

    d1 = torch.linalg.norm(y1 - y2[nns1[1]], dim=-1).mean()
    nns2 = gnn.pool.knn(x=y1, y=y2, batch_x=b1, batch_y=b2, k=1)

    if nns2.shape[-1] == 0:
        return (y2 * 0).sum()
    logger.debug("nn shape: %s", nns2.shape)
    d2 = torch.linalg.norm(y2 - y1[nns2[1]], dim=-1).mean()
    return d1 + d2


class ChamferLoss(MeshLoss):
    """
    Computes the Chamfer loss between two point clouds represented as graphs.

    Args:
        id_key (str, optional):
            The key in the graph data that contains ids for points corresponding a specific structure.
            Set this if you want to compute the loss only for the surface or inlet etc.
            If None, the loss will be computed for all points in both graphs. Default is None.
    """

    def __init__(self, id_key: str = None):
        required_attrs = ["coords", "batch"]
        if id_key is not None:
            required_attrs.append(id_key)
        super().__init__(required_attrs=required_attrs, supervised=True)
        self.id_key = id_key

    def _compute_loss(self, true_graph, pred_graph):

        if self.id_key is not None:
            true_ids = true_graph[self.id_key]
            pred_ids = pred_graph[self.id_key]
        else:
            true_ids = torch.arange(true_graph.coords.shape[0], device=true_graph.coords.device)
            pred_ids = torch.arange(pred_graph.coords.shape[0], device=pred_graph.coords.device)

        return _compute_batch_chamfer(
            y1=true_graph.coords[true_ids],
            y2=pred_graph.coords[pred_ids],
            b1=true_graph.batch[true_ids],
            b2=pred_graph.batch[pred_ids],
        )
