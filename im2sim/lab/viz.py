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

"""Plots for lab results. Every function returns a matplotlib ``Figure`` (or draws on ``ax``)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from torch_geometric.data import Data

from im2sim.lab.results import LabResult, results_to_frame


def _as_list(results) -> list[LabResult]:
    return [results] if isinstance(results, LabResult) else list(results)


def _axes(ax, figsize=(5, 3.5)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    return ax.figure, ax


def _group(results: Sequence[LabResult]) -> dict[str, list[LabResult]]:
    groups: dict[str, list[LabResult]] = {}
    for r in results:
        groups.setdefault(r.label, []).append(r)
    return groups


def _band(ax, x, runs: list[np.ndarray], label, **kw):
    """Mean line with min-max band when several seeds are given."""
    arr = np.stack([np.asarray(r, float) for r in runs])
    (line,) = ax.plot(x, arr.mean(0), label=label, **kw)
    if len(arr) > 1:
        ax.fill_between(x, arr.min(0), arr.max(0), color=line.get_color(), alpha=0.2)


def _curves(results, x_key, y_key, xlabel, ylabel, title, ax, logy=False, **kw):
    fig, ax = _axes(ax)
    for label, rs in _group(_as_list(results)).items():
        _band(ax, rs[0].arrays[x_key], [r.arrays[y_key] for r in rs], label, marker="o", **kw)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    return fig


# --------------------------------------------------------------------------------------
# graph drawing
# --------------------------------------------------------------------------------------


def plot_graph_signal(
    graph: Data,
    values=None,
    ax=None,
    cmap: str = "viridis",
    edges: bool | None = None,
    title: str | None = None,
    node_size: float = 18,
    vmin=None,
    vmax=None,
    colorbar: bool = True,
):
    """Draw a graph with nodes coloured by ``values`` (2D, or 3D if z has extent).

    Positions are projected onto their two highest-variance axes unless the graph has real 3D
    extent, in which case a 3D scatter is drawn. NaN values are drawn as faint grey nodes (used
    for "not reached"); edges are drawn by default only for graphs of up to 5000 edges.
    """
    pos = graph.pos.numpy()
    ext = pos.max(0) - pos.min(0)
    use3d = pos.shape[1] == 3 and (ext > 1e-6 * ext.max()).sum() == 3
    fig = ax.figure if ax is not None else plt.figure(figsize=(5, 4))
    if ax is None:
        ax = fig.add_subplot(111, projection="3d" if use3d else None)
    if use3d:
        cols = [0, 1, 2]
    else:
        cols = list(np.argsort(ext)[::-1][:2])
    src, dst = graph.edge_index.numpy()
    if edges is None:
        edges = src.size <= 5000
    if edges:
        if use3d:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection

            segs = np.stack([pos[src][:, cols], pos[dst][:, cols]], axis=1)
            ax.add_collection3d(Line3DCollection(segs, colors="0.8", linewidths=0.3, alpha=0.5))
        else:
            segs = np.stack([pos[src][:, cols], pos[dst][:, cols]], axis=1)
            ax.add_collection(LineCollection(segs, colors="0.8", linewidths=0.6, zorder=0))
    c = None if values is None else np.asarray(values, float)
    style = {}
    if c is not None:
        hide = ~np.isfinite(c)
        if hide.any():
            ax.scatter(*[pos[hide, i] for i in cols], s=node_size / 4, c="0.75", alpha=0.25, zorder=1)
        pos, c = pos[~hide], c[~hide]
        style = {"c": c, "cmap": cmap, "vmin": vmin, "vmax": vmax}
    sc = ax.scatter(*[pos[:, i] for i in cols], s=node_size, zorder=2, **style)
    if use3d:
        ax.set_box_aspect(ext[cols])
    else:
        ax.set_aspect("equal" if ext[cols[1]] > 0.1 * ext[cols[0]] else "auto")
        ax.autoscale()
        ax.margins(0.05)
        ax.set_xticks([])
        ax.set_yticks([])
    if values is not None and colorbar:
        fig.colorbar(sc, ax=ax, shrink=0.7)
    if title:
        ax.set_title(title)
    return fig


# --------------------------------------------------------------------------------------
# reach
# --------------------------------------------------------------------------------------


def plot_reach_profile(results, ax=None, normalise: bool = True, signed: bool = False):
    """Max sensitivity vs hop distance (log scale) from :func:`jacobian_reach`.

    With ``signed=True`` the signed gradient of the most sensitive node per hop is drawn on a symmetric-log axis
    (dotted line = zero), so cancellation between a local term and a global one (e.g. a graph norm) is visible
    as a sign change rather than a dip in ``|grad|``.
    """
    fig, ax = _axes(ax)
    key = "sens_signed" if signed else "sens_max"
    for label, rs in _group(_as_list(results)).items():
        r = rs[0]
        s = r.arrays[key]
        peak = np.abs(s).max()
        s = s / peak if normalise and peak > 0 else s
        if signed:
            ax.plot(r.arrays["hops"], s, marker="o", label=label)
        else:
            ax.semilogy(r.arrays["hops"], np.maximum(s, 1e-12), marker="o", label=label)
    thr = _as_list(results)[0].meta.get("threshold")
    if signed:
        ax.set_yscale("symlog", linthresh=thr or 1e-2)
        ax.axhline(0, ls=":", c="0.4", lw=1)
        if thr:
            for sign in (1, -1):
                ax.axhline(sign * thr, ls="--", c="0.5", lw=1)
        ax.set(xlabel="hop distance", ylabel="signed relative sensitivity", title="Message reach (signed)")
        ax.set_ylim(-2, 2)
    else:
        if thr:
            ax.axhline(thr, ls="--", c="0.5", lw=1)
        ax.set(xlabel="hop distance", ylabel="relative sensitivity", title="Message reach")
        ax.set_ylim(1e-8, 2)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    return fig


def plot_reach_on_graph(result: LabResult, graph: Data, ax=None):
    """Colour each node by its sensitivity (log10) to the probed input."""
    s = result.arrays["node_sens"]
    rel = s / max(s.max(), 1e-30)
    vals = np.where(rel > 1e-8, np.log10(np.maximum(rel, 1e-8)), np.nan)
    fig = plot_graph_signal(graph, vals, ax=ax, cmap="magma", vmin=-8, vmax=0, title=None)
    centre = result.meta.get("centre")
    target_ax = ax if ax is not None else fig.axes[0]
    if centre is not None:
        p = graph.pos[centre].numpy()
        if hasattr(target_ax, "get_zlim"):
            target_ax.scatter(*p, marker="*", s=150, c="cyan", edgecolors="k", zorder=5)
        else:
            cols = np.argsort(np.ptp(graph.pos.numpy(), axis=0))[::-1][:2]
            target_ax.scatter(*p[cols], marker="*", s=150, c="cyan", edgecolors="k", zorder=5)
    target_ax.set_title(f"log10 sensitivity of node {centre}")
    return fig


def plot_accuracy_vs_distance(results, ax=None):
    """Copy-task accuracy vs hop distance, with the 0.5 chance line."""
    fig = _curves(
        results, "distances", "accuracy", "hop distance", "accuracy", "Long-range copy", ax
    )
    fig.axes[0].axhline(0.5, ls=":", c="0.5")
    fig.axes[0].set_ylim(0.3, 1.02)
    return fig


def plot_harmonic_error(results, ax=None):
    """Relative error of the Laplace-solution task vs domain diameter (dotted: trivial baseline)."""
    fig = _curves(
        results,
        "diameters",
        "error",
        "domain diameter (hops)",
        "relative error",
        "Harmonic extension",
        ax,
    )
    r = _as_list(results)[0]
    fig.axes[0].plot(
        r.arrays["diameters"], r.arrays["baseline_error"], ls=":", c="0.4", label="boundary mean"
    )
    fig.axes[0].legend(frameon=False)
    return fig


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------


def plot_equivariance(results, ax=None):
    """Grouped bars of rotation / translation / scale error per model."""
    fig, ax = _axes(ax)
    groups = _group(_as_list(results))
    kinds = ["rotation", "translation", "scale"]
    w = 0.8 / len(groups)
    for i, (label, rs) in enumerate(groups.items()):
        vals = [np.mean([r.scalars[f"{k}_error"] for r in rs]) for k in kinds]
        ax.bar(np.arange(3) + i * w, vals, w, label=label)
    ax.set_xticks(np.arange(3) + w * (len(groups) - 1) / 2, kinds)
    ax.set(ylabel="relative output change", title="Equivariance error")
    ax.legend(frameon=False)
    return fig


def plot_probe_r2(result: LabResult, ax=None):
    """Heatmap of linear-probe R^2 (layers x geometric targets)."""
    fig, ax = _axes(ax, (1.0 + 0.9 * len(result.arrays["targets"]), 1.5 + 0.35 * len(result.arrays["layers"])))
    r2 = result.arrays["r2"]
    im = ax.imshow(np.clip(r2, -0.2, 1), vmin=-0.2, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(r2.shape[1]), result.arrays["targets"], rotation=45, ha="right")
    ax.set_yticks(range(r2.shape[0]), result.arrays["layers"])
    for i in range(r2.shape[0]):
        for j in range(r2.shape[1]):
            ax.text(j, i, f"{r2[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="w" if r2[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=ax, label="held-out R$^2$")
    ax.set_title(f"Linear probes: {result.label}")
    return fig


def plot_resolution(results, ax=None):
    """Output error vs fraction of nodes kept."""
    return _curves(
        results,
        "fractions",
        "error",
        "fraction of nodes kept",
        "relative output error",
        "Resolution invariance",
        ax,
    )


def plot_leakage(results, ax=None):
    """Per-node response along the U-tube: limb 0 (perturbed) vs limb 1 (across the gap)."""
    fig, ax = _axes(ax)
    for r in _as_list(results):
        x, limb, d = r.arrays["pos_x"], r.arrays["limb"], r.arrays["delta"]
        for k, ls in ((0, "-"), (1, "--")):
            order = np.argsort(x[limb == k])
            ax.plot(x[limb == k][order], d[limb == k][order], ls, label=f"{r.label}, limb {k}")
    ax.set(xlabel="x", ylabel="|output change|", title="Proximity leakage")
    ax.legend(frameon=False, fontsize=8)
    return fig


# --------------------------------------------------------------------------------------
# sharpness
# --------------------------------------------------------------------------------------


def plot_step_response(result: LabResult, ax=None, max_panels: int = 5):
    """Predicted vs true step profiles along the step axis, one panel per width."""
    n = min(len(result.arrays["widths"]), max_panels)
    if ax is None:
        fig, axs = plt.subplots(1, n, figsize=(2.6 * n, 2.6), sharey=True)
    else:
        fig, axs = ax.figure, np.atleast_1d(ax)
    c = result.arrays["profile_coord"]
    order = np.argsort(c)
    for i, a in enumerate(np.atleast_1d(axs)[:n]):
        a.plot(c[order], result.arrays[f"profile_true_{i}"][order], c="0.6", label="true")
        a.plot(c[order], result.arrays[f"profile_pred_{i}"][order], ".", ms=3, label="pred")
        a.set_title(f"width {result.arrays['widths'][i]:g} edges", fontsize=9)
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Step response: {result.label}", y=1.02)
    return fig


def plot_sharpness_curve(results, ax=None):
    """Predicted vs true 10-90% rise width (points above the diagonal are over-smoothed)."""
    fig, ax = _axes(ax)
    lim = 0
    for label, rs in _group(_as_list(results)).items():
        t = rs[0].arrays["rise_true"]
        _band(ax, t, [r.arrays["rise_pred"] for r in rs], label, marker="o")
        lim = max(lim, t.max(), *[r.arrays["rise_pred"].max() for r in rs])
    ax.plot([0, lim], [0, lim], ":", c="0.5", label="ideal")
    ax.set(xlabel="true rise width", ylabel="predicted rise width", title="Sharpness")
    ax.legend(frameon=False)
    return fig


def plot_spectral_gain(results, ax=None):
    """Gain and Dirichlet-energy ratio against Laplacian eigenvalue."""
    if ax is None:
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    else:
        fig, axs = ax.figure, ax
    for label, rs in _group(_as_list(results)).items():
        ev = rs[0].arrays["eigvals"]
        _band(axs[0], ev, [r.arrays["gain"] for r in rs], label, marker="o")
        _band(axs[1], ev[1:], [r.arrays["dirichlet_ratio"][1:] for r in rs], label, marker="o")
    axs[0].set(xlabel="eigenvalue (frequency)", ylabel="gain", title="Frequency response")
    axs[1].axhline(1, ls=":", c="0.5")
    axs[1].set(xlabel="eigenvalue (frequency)", ylabel="Dirichlet ratio", title="output / input frequency")
    for a in axs:
        a.grid(alpha=0.3)
    axs[0].legend(frameon=False)
    return fig


def plot_boundary_layer(result: LabResult, ax=None, max_panels: int = 4):
    """Predicted vs true wall-normal profile for each boundary-layer thickness."""
    n = min(len(result.arrays["rmse"]), max_panels)
    if ax is None:
        fig, axs = plt.subplots(1, n, figsize=(2.6 * n, 2.6), sharey=True)
    else:
        fig, axs = ax.figure, np.atleast_1d(ax)
    r = result.arrays["profile_r"]
    for i, a in enumerate(np.atleast_1d(axs)[:n]):
        a.plot(r, result.arrays[f"profile_true_{i}"], "-o", c="0.6", ms=3, label="true")
        a.plot(r, result.arrays[f"profile_pred_{i}"], "-o", ms=3, label="pred")
        d = result.arrays["deltas"][i]
        a.set_title("Poiseuille" if np.isnan(d) else f"delta = {d:g} spacings", fontsize=9)
        a.set_xlabel("wall distance")
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Boundary layer: {result.label}", y=1.02)
    return fig


# --------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------


def plot_suite_summary(results: Sequence[LabResult], ax=None):
    """Heatmap of every scalar (columns) per model (rows), each column scaled to [0, 1].

    Scalars are averaged over seeds; colours compare models within a column only.
    """
    df = results_to_frame(results)
    df = df.drop(columns=["graph", "seed"]).groupby(["model", "experiment"]).mean(numeric_only=True)
    wide = df.reset_index().pivot(index="model", columns="experiment")
    wide = wide.dropna(axis=1, how="all")
    cols = [f"{b}\n{a}" for a, b in wide.columns]
    vals = wide.to_numpy(float)
    finite = np.where(np.isfinite(vals), vals, np.nan)
    lo, hi = np.nanmin(finite, 0), np.nanmax(finite, 0)
    norm = (finite - lo) / np.where(hi - lo > 0, hi - lo, 1)
    fig, ax = _axes(ax, (1.2 + 1.1 * len(cols), 1.2 + 0.5 * len(wide)))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(wide)), wide.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.3g}", ha="center", va="center", fontsize=8,
                        color="w" if norm[i, j] > 0.6 else "k")
    ax.set_title("Suite summary (colour: relative within column)")
    return fig
