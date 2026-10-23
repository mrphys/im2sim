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

"""Toy graph builders and graph utilities for lab experiments.

Every builder returns a ``torch_geometric.data.Data`` with node positions stored under both
``pos`` (PyG convention) and ``coords`` (im2sim convention), an undirected ``edge_index``, a
placeholder feature matrix ``x`` and (where meaningful) a boolean ``boundary`` mask.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data
from torch_geometric.nn import knn_interpolate
from torch_geometric.utils import (
    degree,
    get_laplacian,
    grid,
    remove_self_loops,
    to_scipy_sparse_matrix,
    to_undirected,
)

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def make_graph(pos: torch.Tensor, edge_index: torch.Tensor, **attrs) -> Data:
    """Assemble a Data object with ``pos``/``coords`` aliases and an undirected edge set."""
    pos = pos.float()
    edge_index = to_undirected(remove_self_loops(edge_index)[0], num_nodes=pos.size(0))
    data = Data(x=torch.ones(pos.size(0), 1), edge_index=edge_index, pos=pos, coords=pos)
    for key, value in attrs.items():
        data[key] = value
    return data


def hop_distances(graph: Data, source: int) -> np.ndarray:
    """Unweighted shortest-path (hop) distance from ``source`` to every node (inf if unreachable)."""
    adj = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes).tocsr()
    return shortest_path(adj, unweighted=True, indices=source, directed=False)


def mean_edge_length(graph: Data) -> float:
    """Mean Euclidean edge length."""
    src, dst = graph.edge_index
    return float((graph.pos[src] - graph.pos[dst]).norm(dim=-1).mean())


def laplacian_eigs(graph: Data, k: int = 8, normalization: str | None = "sym"):
    """Smallest ``k`` eigenpairs of the graph Laplacian.

    Args:
        graph (Data): The graph.
        k (int): Number of eigenpairs.
        normalization (str | None): ``None`` (combinatorial), ``"sym"`` or ``"rw"``, as in
            :func:`torch_geometric.utils.get_laplacian`.

    Returns:
        tuple[np.ndarray, np.ndarray]: eigenvalues ``[k]`` and unit-norm eigenvectors ``[N, k]``.
    """
    n = graph.num_nodes
    ei, ew = get_laplacian(graph.edge_index, normalization=normalization, num_nodes=n)
    lap = to_scipy_sparse_matrix(ei, ew, n).tocsr()
    k = min(k, n - 1)
    if n <= 1500:
        vals, vecs = np.linalg.eigh(lap.toarray())
        return vals[:k], vecs[:, :k]
    vals, vecs = sp.linalg.eigsh(lap, k=k, sigma=-1e-3, which="LM")
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def dirichlet_solve(
    graph: Data, boundary: torch.Tensor, boundary_values: torch.Tensor
) -> torch.Tensor:
    """Solve the graph Laplace equation with fixed boundary values.

    Interior nodes satisfy ``u_i = mean(u_j for j in neighbours(i))``.

    Args:
        graph (Data): The graph.
        boundary (torch.Tensor): Bool mask ``[N]`` of fixed nodes.
        boundary_values (torch.Tensor): Values ``[N]`` (only boundary entries are used).

    Returns:
        torch.Tensor: The harmonic extension ``[N]``.
    """
    n = graph.num_nodes
    ei, ew = get_laplacian(graph.edge_index, normalization=None, num_nodes=n)
    lap = to_scipy_sparse_matrix(ei, ew, n).tocsr()
    b = boundary.numpy().astype(bool)
    inner = ~b
    u = np.zeros(n)
    u[b] = boundary_values.numpy()[b]
    if inner.any():
        rhs = -lap[inner][:, b] @ u[b]
        u[inner] = sp.linalg.spsolve(lap[inner][:, inner].tocsc(), rhs)
    return torch.from_numpy(u).float()


# --------------------------------------------------------------------------------------
# toy graphs
# --------------------------------------------------------------------------------------


def path_graph(n: int) -> Data:
    """Chain of ``n`` nodes along the x axis (unit spacing)."""
    idx = torch.arange(n - 1)
    ei = torch.stack([idx, idx + 1])
    pos = torch.stack([torch.arange(n).float(), torch.zeros(n), torch.zeros(n)], dim=1)
    boundary = torch.zeros(n, dtype=torch.bool)
    boundary[[0, n - 1]] = True
    return make_graph(pos, ei, boundary=boundary)


def ring_graph(n: int) -> Data:
    """Cycle of ``n`` nodes on a unit circle in the xy plane."""
    idx = torch.arange(n)
    ei = torch.stack([idx, (idx + 1) % n])
    ang = 2 * math.pi * idx.float() / n
    pos = torch.stack([ang.cos(), ang.sin(), torch.zeros(n)], dim=1)
    return make_graph(pos, ei, boundary=torch.zeros(n, dtype=torch.bool))


def grid_graph(height: int, width: int | None = None, diagonals: bool = False) -> Data:
    """Regular grid in the z=0 plane (built from :func:`torch_geometric.utils.grid`).

    Args:
        height (int): Number of rows.
        width (int | None): Number of columns (default: ``height``).
        diagonals (bool): Also connect diagonal neighbours (PyG's native 8-neighbourhood);
            default is 4-connected.
    """
    width = width or height
    ei, pos2 = grid(height, width)
    if not diagonals:
        step = (pos2[ei[0]] - pos2[ei[1]]).abs().sum(1)
        ei = ei[:, step == 1]
    pos = torch.cat([pos2, torch.zeros(pos2.size(0), 1)], dim=1)
    data = make_graph(pos, ei)
    p = pos2
    data.boundary = (p == p.min(0).values).any(1) | (p == p.max(0).values).any(1)
    return data


def tree_graph(depth: int, branching: int = 2) -> Data:
    """Complete ``branching``-ary tree of the given depth, laid out by level."""
    edges, level_of, nodes_at = [], [0], [[0]]
    count = 1
    for d in range(1, depth + 1):
        cur = []
        for parent in nodes_at[-1]:
            for _ in range(branching):
                edges.append((parent, count))
                level_of.append(d)
                cur.append(count)
                count += 1
        nodes_at.append(cur)
    pos = torch.zeros(count, 3)
    for d, nodes in enumerate(nodes_at):
        m = len(nodes)
        pos[nodes, 0] = torch.linspace(-1, 1, m) * (m > 1)
        pos[nodes, 1] = -float(d)
    boundary = torch.zeros(count, dtype=torch.bool)
    boundary[nodes_at[-1]] = True
    return make_graph(pos, torch.tensor(edges).t(), boundary=boundary, level=torch.tensor(level_of))


def barbell_graph(n_clique: int = 5, n_path: int = 6) -> Data:
    """Two cliques joined by a path: a classic oversquashing bottleneck."""
    edges = []
    for off in (0, n_clique + n_path):
        edges += [(off + i, off + j) for i in range(n_clique) for j in range(i + 1, n_clique)]
    chain = [n_clique - 1, *range(n_clique, n_clique + n_path), n_clique + n_path]
    edges += list(zip(chain[:-1], chain[1:]))
    n = 2 * n_clique + n_path
    pos = torch.zeros(n, 3)
    pos[:, 0] = torch.arange(n).float()
    pos[:n_clique, 1] = torch.linspace(-1, 1, n_clique)
    pos[n_clique + n_path :, 1] = torch.linspace(-1, 1, n_clique)
    return make_graph(pos, torch.tensor(edges).t())


def tube_graph(
    radius: float = 1.0,
    length: float = 4.0,
    n_rings: int = 16,
    n_per_ring: int = 12,
    n_radial: int = 4,
) -> Data:
    """Vessel-like solid cylinder with structured connectivity.

    Axial stations each hold a centre-line node plus ``n_radial`` concentric rings of
    ``n_per_ring`` nodes. Extra attributes: ``wall_dist`` (distance to the wall),
    ``radial`` (distance to the axis), ``axial`` (position along the axis), ``poiseuille``
    (analytic parabolic profile), ``is_wall``, ``is_cap``, ``boundary`` (wall or cap) and the
    scalar ``radial_spacing``.
    """
    per_station = 1 + n_radial * n_per_ring

    def nid(i, k, a):  # station, radial level (0 = axis), angle index
        return i * per_station if k == 0 else i * per_station + 1 + (k - 1) * n_per_ring + a

    pos, edges, radial, axial, wall, cap = [], [], [], [], [], []
    zs = np.linspace(0, length, n_rings)
    for i, z in enumerate(zs):
        pos.append([0.0, 0.0, z])
        radial.append(0.0)
        for k in range(1, n_radial + 1):
            r = radius * k / n_radial
            for a in range(n_per_ring):
                th = 2 * math.pi * a / n_per_ring
                pos.append([r * math.cos(th), r * math.sin(th), z])
                radial.append(r)
        for k in range(0, n_radial + 1):
            for a in range(n_per_ring if k else 1):
                u = nid(i, k, a)
                axial.append(z)
                wall.append(k == n_radial)
                cap.append(i in (0, n_rings - 1))
                if i + 1 < n_rings:
                    edges.append((u, nid(i + 1, k, a)))
                if k == 0:
                    edges += [(u, nid(i, 1, b)) for b in range(n_per_ring)]
                else:
                    edges.append((u, nid(i, k, (a + 1) % n_per_ring)))
                    if k < n_radial:
                        edges.append((u, nid(i, k + 1, a)))
    pos = torch.tensor(pos)
    radial = torch.tensor(radial).float()
    wall = torch.tensor(wall)
    cap = torch.tensor(cap)
    return make_graph(
        pos,
        torch.tensor(edges).t(),
        radial=radial,
        axial=torch.tensor(axial).float(),
        wall_dist=radius - radial,
        poiseuille=1 - (radial / radius) ** 2,
        is_wall=wall,
        is_cap=cap,
        boundary=wall | cap,
        radial_spacing=radius / n_radial,
    )


def u_tube_graph(
    n_per_limb: int = 30, limb_length: float = 6.0, gap: float = 0.4, cross_edges: bool = False
) -> Data:
    """Two parallel limbs joined at one end: close in space, far apart along the graph.

    ``limb`` is 0 for the first limb and 1 for the second; ``mirror[i]`` is the node facing
    node ``i`` across the gap (at the same x, hence close in space but far along the graph). With
    ``cross_edges=True`` facing nodes are also connected (a geometry-aware graph that
    short-cuts the U).
    """
    n = n_per_limb
    h = limb_length / (n - 1)
    xs = torch.arange(n).float() * h
    pos = torch.zeros(2 * n, 3)
    pos[:n, 0] = xs
    pos[n:, 0] = xs.flip(0)
    pos[n:, 1] = gap
    idx = torch.arange(2 * n - 1)
    edges = [torch.stack([idx, idx + 1])]
    mirror = torch.cat([torch.arange(2 * n - 1, n - 1, -1), torch.arange(n - 1, -1, -1)])
    if cross_edges:
        edges.append(torch.stack([torch.arange(n), mirror[:n]]))
    limb = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    return make_graph(pos, torch.cat(edges, dim=1), limb=limb, mirror=mirror, gap=float(gap))


# --------------------------------------------------------------------------------------
# resampling / real data
# --------------------------------------------------------------------------------------


def subsample(
    graph: Data, idx: torch.Tensor, connectivity: str = "knn", k: int | None = None, r: float | None = None
) -> Data:
    """Keep nodes ``idx`` and rebuild the edges from their positions.

    Args:
        graph (Data): Source graph.
        idx (torch.Tensor): Indices of the nodes to keep.
        connectivity (str): ``"knn"``, ``"radius"`` or ``"delaunay"`` (2D point sets only).
        k (int | None): Neighbours for ``"knn"``; defaults to the source mean degree.
        r (float | None): Radius for ``"radius"``; defaults to 1.5x the mean new edge length.

    Returns:
        Data: A new graph; every node-level tensor of the source is indexed by ``idx``.
    """
    n = graph.num_nodes
    out = Data()
    for key, val in graph.items():
        if key == "edge_index":
            continue
        if torch.is_tensor(val) and val.dim() > 0 and val.size(0) == n:
            out[key] = val[idx]
        else:
            out[key] = val
    pos = out.pos
    if connectivity == "knn":
        k = k or max(2, round(graph.edge_index.size(1) / n))
        out = T.KNNGraph(k=k)(out)
    elif connectivity == "radius":
        if r is None:
            tmp = T.KNNGraph(k=3)(out.clone())
            r = 1.5 * mean_edge_length(tmp)
        out = T.RadiusGraph(r=r)(out)
    elif connectivity == "delaunay":
        if pos.size(1) != 2 and float(pos[:, 2:].abs().max()) > 0:
            raise ValueError("delaunay connectivity needs 2D positions")
        out.pos = pos[:, :2]
        out = T.FaceToEdge()(T.Delaunay()(out))
        out.pos = pos
    else:
        raise ValueError(f"unknown connectivity '{connectivity}'")
    out.edge_index = to_undirected(out.edge_index, num_nodes=out.num_nodes)
    if "coords" in out:
        out.coords = out.pos
    return out


def resample(
    graph: Data, fraction: float, connectivity: str = "knn", seed: int = 0, **kwargs
) -> tuple[Data, torch.Tensor]:
    """Randomly keep ``fraction`` of the nodes and rebuild connectivity.

    Returns:
        tuple[Data, torch.Tensor]: the new graph and the kept source-node indices.
    """
    g = torch.Generator().manual_seed(seed)
    n = graph.num_nodes
    idx = torch.randperm(n, generator=g)[: max(4, int(round(fraction * n)))].sort().values
    return subsample(graph, idx, connectivity, **kwargs), idx


def from_mesh_graph(graph: Data) -> Data:
    """Wrap a real (e.g. im2sim) graph so lab experiments can run on it.

    Ensures ``pos`` and ``coords`` are both present (taking whichever exists), an undirected
    edge set, a feature matrix ``x`` and a ``boundary`` mask (degree-based if absent).
    """
    out = graph.clone()
    pos = out.pos if "pos" in out else out.coords
    out.pos = out.coords = pos.float()
    out.edge_index = to_undirected(remove_self_loops(out.edge_index)[0], num_nodes=pos.size(0))
    if "x" not in out or out.x is None:
        out.x = torch.ones(pos.size(0), 1)
    if "boundary" not in out:
        deg = degree(out.edge_index[0], pos.size(0))
        out.boundary = deg < deg.float().quantile(0.25)
    return out


__all__ = [
    "barbell_graph",
    "dirichlet_solve",
    "from_mesh_graph",
    "grid_graph",
    "hop_distances",
    "knn_interpolate",
    "laplacian_eigs",
    "make_graph",
    "mean_edge_length",
    "path_graph",
    "resample",
    "ring_graph",
    "subsample",
    "tree_graph",
    "tube_graph",
    "u_tube_graph",
]
