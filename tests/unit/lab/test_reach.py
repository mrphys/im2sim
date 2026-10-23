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

import pytest
import torch
import torch_geometric.nn as gnn

from im2sim.lab import ModelAdapter, grid_graph, jacobian_reach, path_graph


class SimpleStack(torch.nn.Module):
    """K parameter-free mean-aggregation layers: receptive field is exactly K hops."""

    def __init__(self, k):
        super().__init__()
        self.layers = torch.nn.ModuleList(gnn.SimpleConv("mean") for _ in range(k))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 2)

    def forward(self, x, edge_index):
        return self.lin(x)


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_simpleconv_reach_equals_depth(k):
    r = jacobian_reach(ModelAdapter(SimpleStack(k)), path_graph(21), centre=10)
    assert r.scalars["receptive_field"] == k


def test_mlp_has_zero_reach():
    r = jacobian_reach(ModelAdapter(MLP()), path_graph(9), centre=4)
    assert r.scalars["receptive_field"] == 0 and r.scalars["effective_radius"] == 0


def test_gcn_receptive_field_bounded_by_depth():
    torch.manual_seed(0)
    model = gnn.GCN(1, 8, 3, out_channels=1)
    r = jacobian_reach(ModelAdapter(model), grid_graph(9), centre=40)
    assert r.scalars["receptive_field"] <= 3
    assert r.arrays["node_sens"].shape == (81,)


def test_default_centre_and_threshold_effect():
    r = jacobian_reach(ModelAdapter(SimpleStack(2)), path_graph(11))
    assert r.meta["centre"] == 5
    strict = jacobian_reach(ModelAdapter(SimpleStack(2)), path_graph(11), threshold=0.9)
    assert strict.scalars["effective_radius"] <= r.scalars["effective_radius"]
