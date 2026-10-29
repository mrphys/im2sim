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

"""Trained experiments. Small, but they fit models, so they are marked slow."""

import numpy as np
import pytest
import torch_geometric.nn as gnn

from im2sim.lab import (
    ModelAdapter,
    boundary_layer,
    copy_task,
    grid_graph,
    harmonic_extension,
    step_sweep,
    tube_graph,
)

pytestmark = pytest.mark.slow


def gcn_factory(layers, hidden=32):
    return lambda i, o: ModelAdapter(gnn.GCN(i, hidden, layers, out_channels=o))


def test_copy_task_reach_limits_accuracy():
    """A 2-layer GCN solves distance 1-2 but cannot beat chance at distance 6."""
    r = copy_task(gcn_factory(2), distances=(1, 2, 6), steps=250, seed=0)
    acc = dict(zip(r.arrays["distances"], r.arrays["accuracy"]))
    assert acc[1] > 0.9
    assert acc[6] < 0.7
    assert r.scalars["max_distance_solved"] >= 1


def test_harmonic_extension_runs_and_reports_baseline():
    r = harmonic_extension(gcn_factory(4), sizes=(5, 7), steps=150, n_train=6, n_test=4)
    assert r.arrays["error"].shape == (2,) and r.arrays["diameters"].tolist() == [8, 12]
    assert np.isfinite(r.arrays["error"]).all() and (r.arrays["baseline_error"] > 0).all()


def test_step_sweep_outputs_and_identity_mode():
    g = grid_graph(10)
    r = step_sweep(gcn_factory(2, 16), g, widths=(0.0, 2.0), steps=120, mode="coords")
    assert r.arrays["rmse"].shape == (2,)
    assert r.arrays["rise_true"][1] > r.arrays["rise_true"][0]
    assert r.arrays["profile_pred_1"].shape == (100,)
    ident = step_sweep(gcn_factory(2, 16), g, widths=(1.0,), steps=120, mode="identity")
    assert ident.scalars["mean_rmse"] < 0.3


def test_boundary_layer_profiles():
    g = tube_graph(n_rings=6, n_per_ring=8, n_radial=3)
    r = boundary_layer(gcn_factory(2, 16), g, deltas=(2.0, 0.5), steps=120)
    assert r.arrays["profile_r"].shape == (4,)
    assert r.arrays["near_wall_rmse"].shape == (2,)
    p = boundary_layer(gcn_factory(2, 16), g, profile="poiseuille", steps=60)
    assert np.isnan(p.arrays["deltas"]).all() and p.arrays["rmse"].shape == (1,)
