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
"""
Version-robust wrappers around the compiled PyTorch Geometric ops used by im2sim.

PyTorch Geometric dispatches ``knn`` and ``graclus`` to different compiled backends depending on
its version:

- ``torch_geometric < 2.8``: ``torch_cluster``
- ``torch_geometric >= 2.8``: ``pyg-lib >= 0.6.0`` (``torch_cluster`` is no longer used)

Neither backend is installed with ``torch_geometric`` itself, and wheels are not available for
every PyTorch version (e.g. ``pyg-lib >= 0.6.0`` requires ``torch >= 2.8`` and ``torch_cluster``
has no wheels for ``torch >= 2.12``). The functions in this module use the compiled backend when
PyTorch Geometric can find one and otherwise fall back to a pure-PyTorch implementation, so im2sim
works with any combination. Run ``im2sim-install-pyg-addons`` to install the compiled backend.

Scatter ops are not wrapped here: in every version PyTorch Geometric itself uses ``torch_scatter``
when it is installed (e.g. for min/max/mul reductions on CUDA, segment reductions and softmax in
message passing) and native PyTorch otherwise. ``im2sim-install-pyg-addons`` installs it too.
"""

import warnings

import torch
import torch_geometric.nn as gnn
import torch_geometric.typing as pyg_typing
from torch_geometric.utils import scatter


def _has_backend(pyg_lib_flag: str) -> bool:
    # torch_geometric >= 2.8 exposes one flag per pyg-lib op, older versions only torch_cluster.
    if hasattr(pyg_typing, pyg_lib_flag):
        return bool(getattr(pyg_typing, pyg_lib_flag))
    return bool(getattr(pyg_typing, "WITH_TORCH_CLUSTER", False))


HAS_COMPILED_KNN = _has_backend("WITH_KNN")
HAS_COMPILED_GRACLUS = _has_backend("WITH_GRACLUS")
HAS_TORCH_SCATTER = bool(getattr(pyg_typing, "WITH_TORCH_SCATTER", False))

_FALLBACK_MSG = (
    "No compiled backend for `{op}` found (torch_geometric {pyg} needs {backend}); "
    "using a slower pure-PyTorch fallback. Run `im2sim-install-pyg-addons` to install it."
)
_warned = set()


def _warn_fallback(op: str):
    if op in _warned:
        return
    _warned.add(op)
    import torch_geometric

    backend = "`pyg-lib>=0.6.0`" if hasattr(pyg_typing, "WITH_KNN") else "`torch-cluster`"
    warnings.warn(
        _FALLBACK_MSG.format(op=op, pyg=torch_geometric.__version__, backend=backend),
        stacklevel=3,
    )


def _knn_torch(x, y, k, batch_x=None, batch_y=None, max_chunk_elements=2**24):
    num_x, num_y = x.size(0), y.size(0)
    if num_x == 0 or num_y == 0 or k <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=y.device)

    k = min(k, num_x)
    chunk = max(1, max_chunk_elements // num_x)
    rows, cols = [], []
    for start in range(0, num_y, chunk):
        y_chunk = y[start : start + chunk]
        dist = torch.cdist(y_chunk.to(x.dtype), x)
        if batch_x is not None and batch_y is not None:
            other_batch = batch_y[start : start + chunk, None] != batch_x[None, :]
            dist.masked_fill_(other_batch, float("inf"))
        dist, col = dist.topk(k, dim=1, largest=False)
        row = torch.arange(start, start + y_chunk.size(0), device=y.device)
        row = row[:, None].expand_as(col)
        # Examples with fewer than k points in x produce inf distances, drop them like PyG does.
        valid = torch.isfinite(dist)
        rows.append(row[valid])
        cols.append(col[valid])
    return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)


@torch.no_grad()
def knn(x, y, k, batch_x=None, batch_y=None):
    """
    Finds for each point in ``y`` the ``k`` nearest points in ``x``.

    Drop-in replacement for ``torch_geometric.nn.knn`` that works without ``pyg-lib`` /
    ``torch_cluster`` installed.

    Args:
        x (torch.Tensor): Points to search in, shape (N, D).
        y (torch.Tensor): Query points, shape (M, D).
        k (int): The number of neighbours.
        batch_x (torch.Tensor, optional): Batch vector for ``x`` of shape (N,).
        batch_y (torch.Tensor, optional): Batch vector for ``y`` of shape (M,).

    Returns:
        torch.Tensor: Assignment index of shape (2, M * k), where row 0 indexes ``y`` and row 1
            indexes ``x``.
    """
    if batch_x is not None:
        batch_x = batch_x.long()
    if batch_y is not None:
        batch_y = batch_y.long()

    if HAS_COMPILED_KNN:
        return gnn.knn(x, y, k, batch_x=batch_x, batch_y=batch_y)

    _warn_fallback("knn")
    if (batch_x is None) != (batch_y is None):
        # Match PyG, a missing batch vector means all points belong to example 0.
        if batch_x is None:
            batch_x = x.new_zeros(x.size(0), dtype=torch.long)
        else:
            batch_y = y.new_zeros(y.size(0), dtype=torch.long)
    return _knn_torch(x, y, k, batch_x, batch_y)


def knn_interpolate(x, pos_x, pos_y, batch_x=None, batch_y=None, k=3):
    """
    Inverse squared distance weighted k-nearest neighbour interpolation of ``x`` from ``pos_x``
    onto ``pos_y``.

    Drop-in replacement for ``torch_geometric.nn.knn_interpolate`` that works without
    ``pyg-lib`` / ``torch_cluster`` installed.

    Args:
        x (torch.Tensor): Features at ``pos_x``, shape (N, C).
        pos_x (torch.Tensor): Source positions, shape (N, D).
        pos_y (torch.Tensor): Target positions, shape (M, D).
        batch_x (torch.Tensor, optional): Batch vector for ``pos_x`` of shape (N,).
        batch_y (torch.Tensor, optional): Batch vector for ``pos_y`` of shape (M,).
        k (int): The number of neighbours. Default is 3.

    Returns:
        torch.Tensor: Interpolated features at ``pos_y``, shape (M, C).
    """
    with torch.no_grad():
        y_idx, x_idx = knn(pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y)
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

    y = scatter(x[x_idx] * weights, y_idx, 0, pos_y.size(0), reduce="sum")
    return y / scatter(weights, y_idx, 0, pos_y.size(0), reduce="sum")


def _graclus_python(edge_index, weight, num_nodes):
    # Same greedy matching as torch_cluster/pyg-lib: visit nodes in random order and pair each
    # unmatched node with the unmatched neighbour connected by the heaviest edge.
    row, col = edge_index.cpu()
    mask = row != col
    row, col = row[mask], col[mask]
    weight = None if weight is None else weight.detach().cpu()[mask]

    perm = torch.argsort(row, stable=True)
    row, col = row[perm], col[perm]
    weight = None if weight is None else weight[perm].tolist()
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    rowptr[1:] = torch.bincount(row, minlength=num_nodes).cumsum(0)
    rowptr, col = rowptr.tolist(), col.tolist()

    cluster = [-1] * num_nodes
    for u in torch.randperm(num_nodes).tolist():
        if cluster[u] >= 0:
            continue
        cluster[u] = u
        best, best_weight = -1, float("-inf")
        for e in range(rowptr[u], rowptr[u + 1]):
            v = col[e]
            if cluster[v] >= 0:
                continue
            if weight is None:
                best = v
                break
            if weight[e] > best_weight:
                best, best_weight = v, weight[e]
        if best >= 0:
            cluster[best] = u

    return torch.tensor(cluster, dtype=torch.long, device=edge_index.device)


def graclus(edge_index, weight=None, num_nodes=None):
    """
    Greedy graph clustering that matches each node with at most one neighbour, maximising the
    edge weight.

    Drop-in replacement for ``torch_geometric.nn.graclus`` that works without ``pyg-lib`` /
    ``torch_cluster`` installed.

    Args:
        edge_index (torch.Tensor): Edge indices of shape (2, E).
        weight (torch.Tensor, optional): Edge weights of shape (E,).
        num_nodes (int, optional): The number of nodes.

    Returns:
        torch.Tensor: Cluster assignment of shape (num_nodes,).
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0

    if HAS_COMPILED_GRACLUS:
        return gnn.graclus(edge_index, weight, num_nodes)

    _warn_fallback("graclus")
    return _graclus_python(edge_index, weight, num_nodes)
