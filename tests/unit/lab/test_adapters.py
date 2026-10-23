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

from im2sim.lab import ModelAdapter, path_graph


class XEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = gnn.GCNConv(1, 4), gnn.GCNConv(4, 2)

    def forward(self, x, edge_index):
        return self.c2(self.c1(x, edge_index).relu(), edge_index)


class DataModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = gnn.GCNConv(1, 2)

    def forward(self, data):
        return self.c(data.x, data.edge_index)


def test_kind_detection():
    assert ModelAdapter(XEModel()).kind == "pyg_xe"
    assert ModelAdapter(DataModel()).kind == "pyg_data"
    assert ModelAdapter(gnn.GCN(1, 4, 2, out_channels=2)).kind == "pyg_xe"


@pytest.mark.parametrize("model", [XEModel(), DataModel()])
def test_call_and_input_override(model):
    g = path_graph(6)
    a = ModelAdapter(model)
    out = a(g)
    assert out.shape == (6, 2)
    out2 = a(g, torch.zeros(6, 1))
    assert not torch.allclose(out, out2)
    assert torch.equal(g.x, torch.ones(6, 1)), "input graph must not be modified"


def test_im2sim_graph_block_returns_x():
    from im2sim.configs import GraphConvBlockConfig
    from im2sim.layers import GraphConvBlock

    block = GraphConvBlock(1, 3, GraphConvBlockConfig(depth=2))
    a = ModelAdapter(block)
    assert a.kind == "graph"
    assert a(path_graph(5)).shape == (5, 3)


def test_hidden_states_and_set_coords():
    g = path_graph(5)
    a = ModelAdapter(XEModel())
    hs = a.hidden_states(g)
    assert list(hs)[0] == "input" and len(hs) == 3
    assert hs["c1"].shape == (5, 4)
    g2 = ModelAdapter.set_coords(g, g.pos + 1)
    assert torch.equal(g2.pos, g2.coords) and not torch.equal(g.pos, g2.pos)


def test_evaluating_restores_mode():
    a = ModelAdapter(XEModel())
    a.model.train()
    with a.evaluating():
        assert not a.model.training
    assert a.model.training
