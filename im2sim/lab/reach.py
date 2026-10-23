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

"""Experiments measuring how far information travels through a GNN."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from im2sim.lab.adapters import AdapterFactory, ModelAdapter
from im2sim.lab.graphs import dirichlet_solve, grid_graph, hop_distances, path_graph
from im2sim.lab.results import LabResult, experiment
from im2sim.lab.training import fit, predict


@experiment(trained=False)
def jacobian_reach(
    adapter: ModelAdapter,
    graph: Data,
    centre: int | None = None,
    out_channel: int = 0,
    in_channel: int | None = None,
    threshold: float = 0.01,
    seed: int = 0,
) -> LabResult:
    """Sensitivity of one output node to every input node, binned by hop distance.

    Computes ``|d out[centre, out_channel] / d x|`` with autograd (no training needed).

    Args:
        adapter (ModelAdapter): Model under test (uses ``graph.<input_key>`` as input).
        graph (Data): Graph to probe.
        centre (int | None): Output node; default is the node closest to the mean position.
        out_channel (int): Output channel.
        in_channel (int | None): Restrict to one input channel (default: sum over channels).
        threshold (float): Relative sensitivity defining the *effective* radius.
        seed (int): Unused (kept for the common experiment signature).

    Returns:
        LabResult: scalars ``effective_radius`` (furthest hop whose max sensitivity is at least
        ``threshold`` x the overall max) and ``receptive_field`` (furthest hop with any
        non-negligible sensitivity), and ``global_coupling`` (max sensitivity at the farthest
        hop relative to the peak: ~0 for strictly local models, a non-zero floor when something
        couples all nodes, e.g. a normalisation layer over the whole graph, a virtual node or
        global pooling); arrays ``hops``, ``sens_max``, ``sens_sum``, ``sens_signed`` (signed gradient
        of the most sensitive node per hop; shows cancellation between local and global terms),
        ``node_sens``, ``node_hops``.
    """
    if centre is None:
        centre = int((graph.pos - graph.pos.mean(0)).norm(dim=-1).argmin())
    x = graph[adapter.input_key].detach().clone().requires_grad_(True)
    with adapter.evaluating():
        out = adapter(graph, x)[centre, out_channel]
    (grad,) = torch.autograd.grad(out, x, allow_unused=True)
    grad = torch.zeros_like(x) if grad is None else grad
    signed = grad[:, in_channel] if in_channel is not None else grad
    sens = signed.abs()
    if sens.dim() > 1:
        sens = sens.sum(1)
        signed = signed.sum(1)
    sens = sens.numpy()
    signed = signed.numpy()

    hops = hop_distances(graph, centre)
    reachable = np.isfinite(hops)
    max_hop = int(hops[reachable].max())
    sens_max = np.zeros(max_hop + 1)
    sens_sum = np.zeros(max_hop + 1)
    sens_signed = np.zeros(max_hop + 1)
    for h in range(max_hop + 1):
        mask = hops == h
        s = sens[mask]
        if s.size:
            sens_max[h], sens_sum[h] = s.max(), s.sum()
            sens_signed[h] = signed[mask][s.argmax()]  # signed gradient of the most sensitive node at this hop
    peak = sens_max.max()
    rel = sens_max / peak if peak > 0 else sens_max
    eff = np.nonzero(rel >= threshold)[0]
    rf = np.nonzero(rel > 1e-8)[0]
    return LabResult(
        name="jacobian_reach",
        scalars={
            "effective_radius": float(eff.max()) if eff.size else 0.0,
            "receptive_field": float(rf.max()) if rf.size else 0.0,
            "global_coupling": float(rel[-1]),
        },
        arrays={
            "hops": np.arange(max_hop + 1),
            "sens_max": sens_max,
            "sens_sum": sens_sum,
            "sens_signed": sens_signed,
            "node_sens": sens,
            "node_hops": hops,
        },
        meta={"centre": centre, "threshold": threshold, "out_channel": out_channel},
    )


def _accuracy_run(factory, make_graph_fn, d, steps, lr, n_train, n_test, seed):
    torch.manual_seed(seed)
    base = make_graph_fn(d)
    hops = hop_distances(base, 0)
    target = int(np.argmax(np.where(np.isfinite(hops), hops, -1)))
    n = base.num_nodes

    def batch(count):
        bits = torch.randint(0, 2, (count,)).float() * 2 - 1
        xs = torch.zeros(count, n, 3)
        xs[:, 0, 0] = bits
        xs[:, 0, 1] = 1.0
        xs[:, target, 2] = 1.0
        return bits, xs.reshape(-1, 3)

    adapter = factory(3, 1)
    bits, xs = batch(n_train)
    graphs = [base] * n_train
    ys = torch.zeros(n_train, n)
    ys[:, target] = bits
    masks = torch.zeros(n_train, n, dtype=torch.bool)
    masks[:, target] = True
    fit(
        adapter,
        graphs,
        list(xs.reshape(n_train, n, 3)),
        list(ys),
        list(masks),
        steps=steps,
        lr=lr,
    )
    bits_t, xs_t = batch(n_test)
    pred = predict(adapter, Batch.from_data_list([base.clone() for _ in range(n_test)]), xs_t)
    pred = pred.reshape(n_test, n)[:, target]
    return float((torch.sign(pred) == bits_t).float().mean()), int(hops[target])


@experiment(trained=True)
def copy_task(
    adapter_factory: AdapterFactory,
    distances=(1, 2, 4, 6, 8),
    graph_fn: Callable[[int], Data] = lambda d: path_graph(d + 1),
    steps: int = 300,
    lr: float = 1e-2,
    n_train: int = 32,
    n_test: int = 64,
    solved: float = 0.9,
    seed: int = 0,
) -> LabResult:
    """Copy a +-1 bit from node 0 to the farthest node, as a function of distance.

    Node 0 carries the bit (channel 0) and a source marker (channel 1); the target node carries
    a query marker (channel 2). A fresh model (``adapter_factory(3, 1)``) is trained per
    distance. Chance accuracy is 0.5.

    Args:
        adapter_factory (AdapterFactory): ``factory(in_channels, out_channels) -> ModelAdapter``.
        distances: Hop distances to test.
        graph_fn: ``d -> graph`` whose farthest node from node 0 is ``d`` hops away.
        steps (int): Training steps per distance.
        lr (float): Learning rate.
        n_train (int): Training samples per step (random bits).
        n_test (int): Held-out samples.
        solved (float): Accuracy counted as solved for ``max_distance_solved``.
        seed (int): Random seed.

    Returns:
        LabResult: scalar ``max_distance_solved`` (largest d reached with all smaller d also
        solved); arrays ``distances``, ``accuracy``.
    """
    acc = np.array(
        [
            _accuracy_run(adapter_factory, graph_fn, d, steps, lr, n_train, n_test, seed)[0]
            for d in distances
        ]
    )
    best = 0
    for d, a in zip(distances, acc):
        if a < solved:
            break
        best = d
    return LabResult(
        name="copy_task",
        scalars={"max_distance_solved": float(best), "mean_accuracy": float(acc.mean())},
        arrays={"distances": np.asarray(distances), "accuracy": acc},
        seed=seed,
        meta={"steps": steps, "solved": solved},
    )


def _smooth_boundary_values(pos: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random low-frequency field evaluated at ``pos`` (normalised to the unit box)."""
    p = (pos - pos.min(0).values) / (pos.max(0).values - pos.min(0).values).clamp(min=1e-8)
    w = torch.randn(p.size(1), 3, generator=gen) * 2.0
    phase = torch.rand(3, generator=gen) * 6.2832
    amp = torch.randn(3, generator=gen)
    return (amp * torch.cos(p @ w + phase)).sum(1)


@experiment(trained=True)
def harmonic_extension(
    adapter_factory: AdapterFactory,
    sizes=(5, 9, 13, 17),
    graph_fn: Callable[[int], Data] = grid_graph,
    steps: int = 400,
    lr: float = 1e-2,
    n_train: int = 16,
    n_test: int = 16,
    seed: int = 0,
) -> LabResult:
    """Learn the Laplace solution from boundary values; error vs domain size.

    Input channels are ``[is_boundary, boundary_value]``; the target is the graph-harmonic
    extension (exact sparse solve, see :func:`im2sim.lab.graphs.dirichlet_solve`). The answer
    at the centre depends on the whole boundary, so a model that cannot send information across
    the domain fails as the domain grows. Errors are relative L2 over interior nodes; the
    trivial baseline (predict the boundary mean) is reported alongside.

    Args:
        adapter_factory (AdapterFactory): ``factory(in_channels=2, out_channels=1)``.
        sizes: Argument passed to ``graph_fn`` (grid side length by default).
        graph_fn: ``size -> graph`` with a ``boundary`` mask.
        steps (int): Training steps per size.
        lr (float): Learning rate.
        n_train (int): Training graphs per size.
        n_test (int): Held-out graphs per size.
        seed (int): Random seed.

    Returns:
        LabResult: arrays ``sizes``, ``diameters`` (hops), ``error``, ``baseline_error``;
        scalar ``mean_error``.
    """
    errs, base_errs, diams = [], [], []
    for s in sizes:
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        g = graph_fn(s)
        bmask = g.boundary
        h0 = hop_distances(g, 0)
        diams.append(int(h0[np.isfinite(h0)].max()))

        def sample(count):
            xs, ys = [], []
            for _ in range(count):
                vals = _smooth_boundary_values(g.pos, gen)
                u = dirichlet_solve(g, bmask, vals)
                xs.append(torch.stack([bmask.float(), vals * bmask], dim=1))
                ys.append(u)
            return xs, ys

        xs, ys = sample(n_train)
        adapter = adapter_factory(2, 1)
        fit(adapter, [g] * n_train, xs, ys, [~bmask] * n_train, steps=steps, lr=lr)
        xt, yt = sample(n_test)
        e, b = [], []
        for x_, y_ in zip(xt, yt):
            p = predict(adapter, g, x_)[:, 0]
            inner = ~bmask
            denom = y_[inner].norm()
            e.append(float((p[inner] - y_[inner]).norm() / denom))
            mean_b = x_[bmask, 1].mean()
            b.append(float((mean_b - y_[inner]).norm() / denom))
        errs.append(np.mean(e))
        base_errs.append(np.mean(b))
    return LabResult(
        name="harmonic_extension",
        scalars={"mean_error": float(np.mean(errs))},
        arrays={
            "sizes": np.asarray(sizes),
            "diameters": np.asarray(diams),
            "error": np.asarray(errs),
            "baseline_error": np.asarray(base_errs),
        },
        seed=seed,
        meta={"steps": steps},
    )
