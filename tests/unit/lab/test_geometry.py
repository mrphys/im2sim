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
import torch_geometric.nn as gnn

from im2sim.lab import (
    ModelAdapter,
    equivariance_error,
    geometric_targets,
    grid_graph,
    linear_probe,
    proximity_leakage,
    resolution_invariance,
    tube_graph,
    u_tube_graph,
)


class RelPos(torch.nn.Module):
    """Message passing on relative positions only: translation-invariant, not rotation-invariant."""

    def __init__(self):
        super().__init__()
        self.conv = gnn.PointNetConv(
            local_nn=torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
        )

    def forward(self, data):
        return self.conv(data.x, data.pos, data.edge_index)


class DistNet(torch.nn.Module):
    """Uses only edge lengths: invariant to rotation and translation, not to scale."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 4)

    def forward(self, data):
        src, dst = data.edge_index
        d = (data.pos[src] - data.pos[dst]).norm(dim=-1, keepdim=True)
        out = torch.zeros(data.num_nodes, 4).index_add_(0, dst, self.lin(d))
        return out


@pytest.fixture
def tube():
    return tube_graph(n_rings=6, n_per_ring=8, n_radial=2)


def test_translation_invariant_relpos(tube):
    torch.manual_seed(0)
    r = equivariance_error(ModelAdapter(RelPos()), tube, n_trials=3)
    assert r.scalars["translation_error"] < 1e-4
    assert r.scalars["rotation_error"] > 1e-3
    assert r.arrays["rotation_trials"].shape == (3,)


def test_edge_length_model_rotation_translation_invariant(tube):
    torch.manual_seed(0)
    r = equivariance_error(ModelAdapter(DistNet()), tube, n_trials=3)
    assert r.scalars["rotation_error"] < 1e-4 and r.scalars["translation_error"] < 1e-4
    assert r.scalars["scale_error"] > 1e-3


def test_coordinate_input_is_not_invariant(tube):
    torch.manual_seed(0)
    a = ModelAdapter(gnn.GCN(3, 8, 2, out_channels=2))
    r = equivariance_error(a, tube, n_trials=3, input_fn=lambda g: g.pos)
    assert r.scalars["translation_error"] > 1e-3


def test_vector_output_equivariance():
    """A linear map of coordinates, x -> x @ W with W = I, is rotation-equivariant."""

    class Identity(torch.nn.Module):
        def forward(self, x, edge_index):
            return x

    g = grid_graph(4)
    a = ModelAdapter(Identity(), kind="pyg_xe")
    r = equivariance_error(a, g, input_fn=lambda g: g.pos, output_kind="vector", n_trials=3)
    assert r.scalars["rotation_error"] < 1e-5
    s = equivariance_error(a, g, input_fn=lambda g: g.pos, output_kind="scalar", n_trials=3)
    assert s.scalars["rotation_error"] > 1e-3


def test_geometric_targets_shapes(tube):
    t = geometric_targets(tube)
    n = tube.num_nodes
    assert t["coords"].shape == (n, 3) and t["mean_curvature"].shape == (n, 1)
    assert "wall_dist" in t and "eig1" in t


def test_linear_probe_recovers_input_geometry(tube):
    """Probing 'input' when the input is the coordinates must give R^2 ~ 1 for coords."""
    torch.manual_seed(0)
    a = ModelAdapter(gnn.GCN(3, 16, 2, out_channels=2))
    r = linear_probe(a, tube, input_fn=lambda g: g.pos)
    targets = list(r.arrays["targets"])
    assert r.arrays["layers"][0] == "input"
    assert r.arrays["r2"][0, targets.index("coords")] > 0.99
    assert r.arrays["r2"].shape == (3, len(targets))


def test_linear_probe_constant_input_has_no_geometry(tube):
    a = ModelAdapter(gnn.GCN(1, 16, 2, out_channels=2))
    r = linear_probe(a, tube)
    assert r.arrays["r2"][0].max() < 0.05


def test_resolution_invariance_pointwise_model_is_exact():
    """A per-node MLP ignores edges, so resampling cannot change its outputs."""

    class Pointwise(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 2)

        def forward(self, x, edge_index):
            return self.lin(x)

    g = grid_graph(8)
    r = resolution_invariance(
        ModelAdapter(Pointwise()), g, fractions=(1.0, 0.5), input_fn=lambda g: g.pos
    )
    assert r.scalars["max_error"] < 1e-6


def test_resolution_invariance_gcn_changes():
    torch.manual_seed(0)
    g = grid_graph(10)
    a = ModelAdapter(gnn.GCN(1, 8, 2, out_channels=2))
    r = resolution_invariance(a, g, fractions=(1.0, 0.4), input_fn=lambda g: g.pos[:, :1])
    assert r.arrays["error"][1] > r.arrays["error"][0] or r.arrays["error"][0] < 1e-5


def test_proximity_leakage_topology_matters():
    torch.manual_seed(0)
    a = ModelAdapter(gnn.GCN(1, 8, 3, out_channels=2))
    mesh = proximity_leakage(a, u_tube_graph(30, gap=0.3))
    knn = proximity_leakage(a, u_tube_graph(30, gap=0.3, cross_edges=True))
    assert mesh.scalars["leakage_ratio"] == 0.0 and mesh.scalars["hop_gap"] > 20
    assert knn.scalars["leakage_ratio"] > 0.1 and knn.scalars["hop_gap"] == 1
    assert np.isfinite(mesh.arrays["delta"]).all()
