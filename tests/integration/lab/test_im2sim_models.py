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

"""The lab must treat im2sim models and plain PyG models identically."""

import pytest
import torch

from im2sim.configs import GraphConvBlockConfig, SimpleGraphDecoderConfig
from im2sim.lab import (
    ModelAdapter,
    equivariance_error,
    jacobian_reach,
    linear_probe,
    run_suite,
    tube_graph,
)
from im2sim.models import SimpleGraphDecoder


@pytest.fixture
def tube():
    return tube_graph(n_rings=6, n_per_ring=8, n_radial=2)


def decoder(depth, in_ch=1, out_ch=1, norm=None):
    # with pred_feature_key='x' the decoder writes into the existing x channels, so out == in
    cfg = SimpleGraphDecoderConfig(
        protocol="predict",
        pred_feature_key="x",
        n_blocks=2,
        block_cfg=GraphConvBlockConfig(depth=depth, activation="ReLU", norm_cfg=norm),
    )
    return SimpleGraphDecoder(in_channels=in_ch, out_channels=out_ch, cfg=cfg)


def test_simple_graph_decoder_through_adapter(tube):
    a = ModelAdapter(decoder(1))
    assert a.kind == "graph"
    r = jacobian_reach(a, tube, centre=100)
    # 2 blocks of one GCN layer each, plus the output conv: exactly 3 hops, nothing global
    assert r.scalars["receptive_field"] == 3 and r.scalars["global_coupling"] == 0.0
    assert equivariance_error(a, tube, n_trials=2).scalars["rotation_error"] == 0.0


def test_default_graph_norm_couples_every_node(tube):
    """DefaultGraphNorm normalises over the whole graph, which the Jacobian probe exposes."""
    from im2sim.configs.core import LayerConfig

    norm = LayerConfig(name="DefaultGraphNorm", kwargs={})
    r = jacobian_reach(ModelAdapter(decoder(1, norm=norm)), tube, centre=100)
    assert r.scalars["global_coupling"] > 0.01
    assert r.scalars["receptive_field"] == r.arrays["hops"].max()


def test_hidden_states_of_im2sim_decoder(tube):
    a = ModelAdapter(decoder(2))
    hs = a.hidden_states(tube)
    assert len(hs) > 2
    res = linear_probe(a, tube, input_fn=lambda g: g.pos[:, :1])
    assert res.arrays["r2"].shape[0] == len(hs)


def test_run_suite_mixes_pyg_and_im2sim(tube):
    from functools import partial

    import torch_geometric.nn as gnn

    models = {
        "pyg_gcn": lambda i, o: ModelAdapter(gnn.GCN(i, 8, 3, out_channels=o)),
        "im2sim_decoder": lambda i, o: ModelAdapter(decoder(1, i, i)),
    }
    res = run_suite(models, {"reach": partial(jacobian_reach, graph=tube)}, untrained_channels=(1, 1), verbose=False)
    assert [r.model for r in res] == ["pyg_gcn", "im2sim_decoder"]
    assert all(r.scalars["receptive_field"] >= 1 for r in res)


@pytest.mark.slow
def test_im2sim_gen2_with_rasterizer():
    """Full image+graph model: the adapter feeds the graph; the image comes from the rasterizer."""
    from torch_geometric.data import Data

    from im2sim.configs import HalfUNetConfig
    from im2sim.layers import MaskRasterizer, TrilinearProjection
    from im2sim.models import Im2SimGen2

    size = 16
    cfd_cfg = SimpleGraphDecoderConfig(pred_feature_key="cfd")
    model = Im2SimGen2(
        image_shape=(size,) * 3,
        image_channels=1,
        projection_channels=8,
        graph_channels=4,
        out_channels=2,
        encoder_cfg=HalfUNetConfig(),
        decoder_cfg=cfd_cfg,
        projection=TrilinearProjection(size),
        rasterizer=MaskRasterizer(),
        n_iters=1,
    )
    g = Data(
        x=torch.randn(30, 4),
        coords=torch.rand(30, 3) * (size - 1),
        edge_index=torch.stack([torch.arange(29), torch.arange(1, 30)]),
    )
    a = ModelAdapter(model, kind="im2sim", output_key="cfd", image=None)
    out = a(g)
    assert out.shape == (30, 2)
    r = jacobian_reach(a, g, centre=15)
    assert r.arrays["node_sens"].shape == (30,)
