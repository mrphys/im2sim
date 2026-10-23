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

import numpy as np
import pytest
import torch

from im2sim.lab import graphs


@pytest.mark.parametrize(
    "g",
    [
        graphs.path_graph(8),
        graphs.ring_graph(8),
        graphs.grid_graph(4, 5),
        graphs.tree_graph(3),
        graphs.barbell_graph(),
        graphs.tube_graph(n_rings=5),
        graphs.u_tube_graph(10),
    ],
)
def test_builders_are_consistent(g):
    n = g.num_nodes
    assert g.pos.shape == (n, 3) and torch.equal(g.pos, g.coords)
    assert g.x.shape[0] == n
    ei = g.edge_index
    assert ei.max() < n and (ei[0] != ei[1]).all()
    # undirected: every edge has its reverse
    fwd = set(map(tuple, ei.t().tolist()))
    assert all((b, a) in fwd for a, b in fwd)
    assert np.isfinite(graphs.hop_distances(g, 0)).all(), "graph must be connected"


def test_hop_distances_on_path():
    assert graphs.hop_distances(graphs.path_graph(6), 0).tolist() == [0, 1, 2, 3, 4, 5]


def test_tube_attributes():
    g = graphs.tube_graph(radius=2.0, n_rings=6, n_per_ring=8, n_radial=3)
    assert g.num_nodes == 6 * (1 + 3 * 8)
    assert torch.allclose(g.wall_dist, 2.0 - g.radial)
    assert g.poiseuille.max() == pytest.approx(1.0) and g.poiseuille.min() == pytest.approx(0.0)
    assert g.boundary[g.is_wall].all() and g.boundary[g.is_cap].all()
    assert g.radial_spacing == pytest.approx(2.0 / 3)


def test_u_tube_close_in_space_far_in_graph():
    g = graphs.u_tube_graph(20, limb_length=6.0, gap=0.3)
    i = 10
    j = int(g.mirror[i])
    assert (g.pos[i] - g.pos[j]).norm() == pytest.approx(0.3, abs=1e-5)
    assert graphs.hop_distances(g, i)[j] > 15
    short = graphs.u_tube_graph(20, 6.0, 0.3, cross_edges=True)
    assert graphs.hop_distances(short, i)[j] == 1


def test_dirichlet_solution_is_harmonic():
    g = graphs.grid_graph(7)
    vals = torch.randn(g.num_nodes)
    u = graphs.dirichlet_solve(g, g.boundary, vals)
    assert torch.allclose(u[g.boundary], vals[g.boundary])
    src, dst = g.edge_index
    mean_nb = torch.zeros_like(u).index_add_(0, src, u[dst]) / torch.bincount(src).float()
    assert torch.allclose(u[~g.boundary], mean_nb[~g.boundary], atol=1e-5)


def test_laplacian_eigs_constant_mode():
    vals, vecs = graphs.laplacian_eigs(graphs.ring_graph(10), k=3, normalization=None)
    assert vals[0] == pytest.approx(0, abs=1e-8)
    assert np.allclose(np.abs(vecs[:, 0]), 1 / np.sqrt(10))


@pytest.mark.parametrize("conn", ["knn", "radius", "delaunay"])
def test_resample_keeps_attributes(conn):
    g = graphs.grid_graph(8)
    sub, idx = graphs.resample(g, 0.5, conn, seed=1)
    assert sub.num_nodes == len(idx) == 32
    assert torch.equal(sub.pos, g.pos[idx]) and torch.equal(sub.coords, sub.pos)
    assert torch.equal(sub.boundary, g.boundary[idx])
    assert sub.edge_index.numel() > 0


def test_from_mesh_graph_accepts_coords_only():
    from torch_geometric.data import Data

    raw = Data(coords=torch.rand(20, 3), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
    g = graphs.from_mesh_graph(raw)
    assert torch.equal(g.pos, g.coords) and g.x.shape == (20, 1) and g.boundary.shape == (20,)


def test_grid_is_four_connected_with_border_boundary():
    g = graphs.grid_graph(5)
    deg = torch.bincount(g.edge_index[0])
    assert deg.max() == 4 and deg.min() == 2
    assert int(g.boundary.sum()) == 16
    assert graphs.hop_distances(g, 0)[-1] == 8
    assert graphs.grid_graph(5, diagonals=True).edge_index.size(1) > g.edge_index.size(1)
