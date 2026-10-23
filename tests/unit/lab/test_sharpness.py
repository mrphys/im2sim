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

from im2sim.lab import ModelAdapter, grid_graph, path_graph, ring_graph, spectral_gain
from im2sim.lab.sharpness import rise_width


def test_rise_width_of_known_sigmoid():
    x = np.linspace(-10, 10, 2001)
    w = 2.0
    got = rise_width(x, 1 / (1 + np.exp(-x / w)))
    assert got == pytest.approx(2 * np.log(9) * w, rel=1e-2)


def test_rise_width_hard_step_is_one_sample():
    x = np.arange(20.0)
    assert rise_width(x, (x > 9.5).astype(float)) <= 1.0


class Smoother(torch.nn.Module):
    """K steps of mean aggregation: a low-pass filter."""

    def __init__(self, k):
        super().__init__()
        self.layers = torch.nn.ModuleList(gnn.SimpleConv("mean") for _ in range(k))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class Identity(torch.nn.Module):
    def forward(self, x, edge_index):
        return x


def test_identity_has_flat_response():
    r = spectral_gain(ModelAdapter(Identity(), kind="pyg_xe"), ring_graph(24), k=10)
    assert np.allclose(r.arrays["gain"], 1.0, atol=1e-5)
    assert np.allclose(r.arrays["dirichlet_ratio"][1:], 1.0, atol=1e-4)
    assert r.scalars["cutoff_eigval"] == float("inf")


def test_mean_aggregation_is_lowpass():
    g = ring_graph(40)
    r = spectral_gain(ModelAdapter(Smoother(3), kind="pyg_xe"), g, k=12, normalization="rw")
    assert r.arrays["gain"][0] == pytest.approx(1.0, abs=1e-5)
    # a circulant filter shares the Laplacian eigenbasis, so mean frequency is unchanged
    assert np.allclose(r.arrays["dirichlet_ratio"][1:], 1.0, atol=1e-4)
    assert r.arrays["gain"][-1] < r.arrays["gain"][1] < r.arrays["gain"][0]
    assert np.isfinite(r.scalars["cutoff_eigval"])


def test_deeper_smoothing_has_lower_cutoff():
    g = path_graph(60)
    cut = [
        spectral_gain(ModelAdapter(Smoother(k), kind="pyg_xe"), g, k=20, normalization="rw")
        for k in (1, 6)
    ]
    assert cut[1].scalars["cutoff_eigval"] <= cut[0].scalars["cutoff_eigval"]
    assert cut[1].arrays["gain"][-1] < cut[0].arrays["gain"][-1]


def test_spectral_gain_runs_on_grid():
    r = spectral_gain(ModelAdapter(gnn.GCN(1, 4, 2, out_channels=1)), grid_graph(6), k=6)
    assert r.arrays["eigvals"].shape == (6,)
