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

from functools import partial

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch_geometric.nn as gnn

from im2sim.lab import (
    ModelAdapter,
    LabResult,
    equivariance_error,
    grid_graph,
    jacobian_reach,
    linear_probe,
    load_results,
    proximity_leakage,
    resolution_invariance,
    results_to_frame,
    run_suite,
    save_results,
    spectral_gain,
    tube_graph,
    u_tube_graph,
    viz,
)
from im2sim.lab.results import is_trained


@pytest.fixture(scope="module")
def results():
    tube = tube_graph(n_rings=6, n_per_ring=8, n_radial=2)
    models = {
        "gcn2": lambda i, o: ModelAdapter(gnn.GCN(i, 8, 2, out_channels=o)),
        "gcn4": lambda i, o: ModelAdapter(gnn.GCN(i, 8, 4, out_channels=o)),
    }
    exps = {
        "reach": partial(jacobian_reach, graph=tube),
        "equiv": partial(equivariance_error, graph=tube, n_trials=2),
        "probe": partial(linear_probe, graph=tube, input_fn=lambda g: g.pos[:, :1]),
        "spectral": partial(spectral_gain, graph=tube, k=6),
        "res": partial(resolution_invariance, graph=tube, fractions=(1.0, 0.5)),
        "leak": partial(proximity_leakage, graph=u_tube_graph(20)),
    }
    return run_suite(models, exps, seeds=(0, 1), untrained_channels=(1, 2), verbose=False), tube


def test_run_suite_labels_and_counts(results):
    res, _ = results
    assert len(res) == 2 * 6 * 2
    assert {r.model for r in res} == {"gcn2", "gcn4"}
    assert {r.name for r in res} >= {"reach", "leak"} and {r.seed for r in res} == {0, 1}


def test_seed_changes_initialisation(results):
    res, _ = results
    eq = [r for r in res if r.name == "equiv" and r.model == "gcn2"]
    assert eq[0].seed != eq[1].seed


def test_run_suite_skips_trained_with_adapter():
    from im2sim.lab import copy_task

    out = run_suite(
        {"a": ModelAdapter(gnn.GCN(3, 4, 1, out_channels=1))}, {"c": copy_task}, verbose=False
    )
    assert out == [] and is_trained(copy_task)


def test_frame_and_roundtrip(results, tmp_path):
    res, _ = results
    df = results_to_frame(res)
    assert {"experiment", "model", "seed", "effective_radius"} <= set(df.columns)
    save_results(res[:3], tmp_path / "r.json")
    back = load_results(tmp_path / "r.json")
    assert back[0].name == res[0].name
    assert (back[0].arrays[next(iter(back[0].arrays))] == res[0].arrays[next(iter(res[0].arrays))]).all()


def test_all_static_plots_render(results):
    res, tube = results
    by = lambda n: [r for r in res if r.name == n]
    figs = [
        viz.plot_reach_profile(by("reach")),
        viz.plot_reach_on_graph(by("reach")[0], tube),
        viz.plot_equivariance(by("equiv")),
        viz.plot_probe_r2(by("probe")[0]),
        viz.plot_spectral_gain(by("spectral")),
        viz.plot_resolution(by("res")),
        viz.plot_leakage(by("leak")),
        viz.plot_suite_summary(res),
    ]
    for f in figs:
        assert f.axes
    plt.close("all")


def test_graph_signal_2d_and_3d():
    f2 = viz.plot_graph_signal(grid_graph(5), list(range(25)))
    f3 = viz.plot_graph_signal(tube_graph(n_rings=4, n_per_ring=6, n_radial=2))
    assert f2.axes and f3.axes
    plt.close("all")


def test_trained_plots_render():
    import numpy as np

    n = 4
    step = LabResult(
        "step_sweep",
        arrays={
            "widths": np.array([0.0, 1.0]),
            "rise_true": np.array([1.0, 3.0]),
            "rise_pred": np.array([2.0, 3.5]),
            "profile_coord": np.arange(n, dtype=float),
            "profile_true_0": np.zeros(n),
            "profile_pred_0": np.zeros(n),
            "profile_true_1": np.ones(n),
            "profile_pred_1": np.ones(n),
        },
        model="m",
    )
    assert viz.plot_step_response(step).axes and viz.plot_sharpness_curve(step).axes
    copy = LabResult("copy_task", arrays={"distances": np.arange(3), "accuracy": np.ones(3)})
    assert viz.plot_accuracy_vs_distance(copy).axes
    harm = LabResult(
        "harmonic",
        arrays={"diameters": np.arange(3), "error": np.ones(3), "baseline_error": np.ones(3)},
    )
    assert viz.plot_harmonic_error(harm).axes
    bl = LabResult(
        "boundary_layer",
        arrays={
            "deltas": np.array([1.0, np.nan]),
            "rmse": np.zeros(2),
            "profile_r": np.arange(3.0),
            **{f"profile_{k}_{i}": np.zeros(3) for k in ("true", "pred") for i in range(2)},
        },
    )
    assert viz.plot_boundary_layer(bl).axes
    plt.close("all")
