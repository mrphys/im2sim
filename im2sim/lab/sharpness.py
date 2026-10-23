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

"""Experiments measuring how sharply a GNN's output can change across a graph."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from im2sim.lab.adapters import AdapterFactory, ModelAdapter
from im2sim.lab.graphs import laplacian_eigs, mean_edge_length
from im2sim.lab.results import LabResult, experiment
from im2sim.lab.training import fit, predict


def rise_width(coord: np.ndarray, values: np.ndarray, lo: float = 0.1, hi: float = 0.9) -> float:
    """Distance along ``coord`` over which ``values`` (a 0 -> 1 step) rises from ``lo`` to ``hi``.

    Values are averaged per unique coordinate and made monotone before interpolating.
    """
    uc, inv = np.unique(np.round(coord, 6), return_inverse=True)
    m = np.maximum.accumulate(np.bincount(inv, values) / np.bincount(inv))
    return float(np.interp(hi, m, uc) - np.interp(lo, m, uc))


def _normalised_coords(graph: Data) -> torch.Tensor:
    p = graph.pos
    return (p - p.mean(0)) / (p.max(0).values - p.min(0).values).max().clamp(min=1e-8)


@experiment(trained=True)
def step_sweep(
    adapter_factory: AdapterFactory,
    graph: Data,
    widths: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
    mode: str = "coords",
    axis: int = 0,
    steps: int = 400,
    lr: float = 1e-2,
    seed: int = 0,
) -> LabResult:
    """Fit a smooth-to-sharp step across the graph and measure what the model can reproduce.

    The target is ``sigmoid((p_axis - mid) / (w * h))`` where ``h`` is the mean edge length, so
    ``w`` is the step width in edge lengths (``w=0``: a hard step). The model is trained on this
    single graph, so the result measures *representational* sharpness, not generalisation.

    Args:
        adapter_factory (AdapterFactory): ``factory(in_channels, 1) -> ModelAdapter``.
        graph (Data): Graph with ``pos``.
        widths: Step widths in mean edge lengths.
        mode (str): ``"coords"``: input is the normalised position (a model must build the
            edge from geometry; exposes spectral bias). ``"identity"``: input is the target itself
            (a model must pass a sharp signal through its layers).
        axis (int): Coordinate axis normal to the step.
        steps (int): Training steps per width.
        lr (float): Learning rate.
        seed (int): Random seed.

    Returns:
        LabResult: arrays ``widths``, ``rmse``, ``rise_pred``, ``rise_true``, ``overshoot``
        and per-width profiles ``profile_true_{i}`` / ``profile_pred_{i}`` with
        ``profile_coord``; scalar ``mean_sharpness_ratio`` (predicted / true rise width, 1 is ideal
        and large means over-smoothing; measured on the sharpest non-zero width).
    """
    h = mean_edge_length(graph)
    p = graph.pos[:, axis]
    mid = 0.5 * (p.max() + p.min())
    coord = p.numpy()
    rmse, rise_p, rise_t, over = [], [], [], []
    profiles = {}
    for i, w in enumerate(widths):
        torch.manual_seed(seed)
        y = (p > mid).float() if w == 0 else torch.sigmoid((p - mid) / (w * h))
        x = _normalised_coords(graph) if mode == "coords" else y[:, None].clone()
        adapter = adapter_factory(x.size(1), 1)
        fit(adapter, graph, x, y, steps=steps, lr=lr)
        pred = predict(adapter, graph, x)[:, 0]
        rmse.append(float(((pred - y) ** 2).mean().sqrt()))
        rise_p.append(rise_width(coord, pred.numpy()))
        rise_t.append(rise_width(coord, y.numpy()))
        over.append(float(max(pred.max() - 1, 0, -pred.min())))
        profiles[f"profile_true_{i}"] = y.numpy()
        profiles[f"profile_pred_{i}"] = pred.numpy()
    nz = [i for i, w in enumerate(widths) if w > 0]
    ratio = rise_p[nz[0]] / max(rise_t[nz[0]], 1e-8) if nz else float("nan")
    return LabResult(
        name="step_sweep",
        scalars={"mean_rmse": float(np.mean(rmse)), "sharpness_ratio": float(ratio)},
        arrays={
            "widths": np.asarray(widths),
            "rmse": np.asarray(rmse),
            "rise_pred": np.asarray(rise_p),
            "rise_true": np.asarray(rise_t),
            "overshoot": np.asarray(over),
            "profile_coord": coord,
            **profiles,
        },
        seed=seed,
        meta={"mode": mode, "axis": axis, "edge_length": h},
    )


@experiment(trained=False)
def spectral_gain(
    adapter: ModelAdapter,
    graph: Data,
    k: int = 16,
    normalization: str | None = "sym",
    out_channel: int | None = None,
    seed: int = 0,
) -> LabResult:
    """Frequency response of a model: feed each low Laplacian eigenvector, measure what returns.

    Every input channel is set to the unit-norm eigenvector ``v_j`` (eigenvalue ``lambda_j``).
    For output ``y`` the gain is the mean over channels of ``|<y_c, v_j>|`` (how much of the
    signal comes back at the same frequency) and the Dirichlet ratio is
    ``(y^T L y / y^T y) / lambda_j``, the mean frequency of the output relative to the input:
    below 1 the output has shifted to lower frequencies (smoothing), above 1 to higher ones
    (sharpening or aliasing). Exactly 1 for any filter that commutes with the Laplacian, so it
    is informative for nonlinear or position-dependent models.
    Needs no training, so it works on checkpoints and on freshly initialised models alike.

    Args:
        adapter (ModelAdapter): Model under test.
        graph (Data): Graph whose Laplacian defines the frequencies.
        k (int): Number of lowest-frequency eigenvectors.
        normalization (str | None): Laplacian normalisation (see ``laplacian_eigs``).
        out_channel (int | None): Use one output channel (default all).
        seed (int): Unused.

    Returns:
        LabResult: arrays ``eigvals``, ``gain``, ``dirichlet_ratio``; scalar ``cutoff_eigval``
        (smallest eigenvalue whose gain drops below half the maximum; ``inf`` if never).
    """
    vals, vecs = laplacian_eigs(graph, k=k, normalization=normalization)
    from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

    n = graph.num_nodes
    ei, ew = get_laplacian(graph.edge_index, normalization=normalization, num_nodes=n)
    lap = to_scipy_sparse_matrix(ei, ew, n).tocsr()
    c_in = graph[adapter.input_key].size(1)
    gains, ratios = [], []
    with adapter.evaluating(), torch.no_grad():
        for j in range(len(vals)):
            v = torch.from_numpy(vecs[:, j]).float()
            y = adapter(graph, v[:, None].expand(n, c_in).contiguous())
            y = y if out_channel is None else y[:, [out_channel]]
            yn = y.numpy()
            gains.append(float(np.abs(yn.T @ vecs[:, j]).mean()))
            num = (yn * (lap @ yn)).sum() / max((yn**2).sum(), 1e-12)
            ratios.append(float(num / vals[j]) if vals[j] > 1e-8 else float("nan"))
    gains = np.asarray(gains)
    below = np.nonzero(gains < 0.5 * gains.max())[0]
    return LabResult(
        name="spectral_gain",
        scalars={"cutoff_eigval": float(vals[below[0]]) if below.size else float("inf")},
        arrays={"eigvals": vals, "gain": gains, "dirichlet_ratio": np.asarray(ratios)},
        seed=seed,
    )


@experiment(trained=True)
def boundary_layer(
    adapter_factory: AdapterFactory,
    graph: Data,
    deltas: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
    profile: str = "exp",
    inputs: str = "coords",
    steps: int = 400,
    lr: float = 1e-2,
    seed: int = 0,
) -> LabResult:
    """Fit near-wall profiles of decreasing thickness on a tube graph (CFD-style boundary layer).

    Needs a graph from :func:`im2sim.lab.graphs.tube_graph`. Targets are
    ``1 - exp(-wall_dist / (delta * radial_spacing))`` (``profile="exp"``), thickness ``delta``
    in radial node spacings, or the parabolic Poiseuille profile (``profile="poiseuille"``,
    ``deltas`` ignored). Trained on the single graph, like :func:`step_sweep`.

    Args:
        adapter_factory (AdapterFactory): ``factory(in_channels, 1) -> ModelAdapter``.
        graph (Data): Tube graph (``wall_dist``, ``radial_spacing``, ``poiseuille``).
        deltas: Boundary-layer thicknesses in radial node spacings.
        profile (str): ``"exp"`` or ``"poiseuille"``.
        inputs (str): ``"coords"`` (geometry only) or ``"wall_dist"`` (given the distance).
        steps (int): Training steps per case.
        lr (float): Learning rate.
        seed (int): Random seed.

    Returns:
        LabResult: arrays ``deltas``, ``rmse``, ``near_wall_rmse`` (outer two radial layers),
        ``profile_r`` (wall distance) and ``profile_true_{i}`` / ``profile_pred_{i}`` (means per
        wall distance).
    """
    cases = [None] if profile == "poiseuille" else list(deltas)
    wd = graph.wall_dist
    near = wd <= 1.5 * graph.radial_spacing + 1e-6
    levels, inv = np.unique(np.round(wd.numpy(), 6), return_inverse=True)
    rmse, nw, profiles = [], [], {}
    for i, d in enumerate(cases):
        torch.manual_seed(seed)
        y = graph.poiseuille if d is None else 1 - torch.exp(-wd / (d * graph.radial_spacing))
        x = _normalised_coords(graph) if inputs == "coords" else (wd / wd.max())[:, None]
        adapter = adapter_factory(x.size(1), 1)
        fit(adapter, graph, x, y, steps=steps, lr=lr)
        pred = predict(adapter, graph, x)[:, 0]
        rmse.append(float(((pred - y) ** 2).mean().sqrt()))
        nw.append(float(((pred - y)[near] ** 2).mean().sqrt()))
        cnt = np.bincount(inv)
        profiles[f"profile_true_{i}"] = np.bincount(inv, y.numpy()) / cnt
        profiles[f"profile_pred_{i}"] = np.bincount(inv, pred.numpy()) / cnt
    return LabResult(
        name="boundary_layer",
        scalars={"mean_rmse": float(np.mean(rmse)), "mean_near_wall_rmse": float(np.mean(nw))},
        arrays={
            "deltas": np.asarray([np.nan if d is None else d for d in cases]),
            "rmse": np.asarray(rmse),
            "near_wall_rmse": np.asarray(nw),
            "profile_r": levels,
            **profiles,
        },
        seed=seed,
        meta={"profile": profile, "inputs": inputs},
    )
