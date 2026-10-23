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

"""Experiments probing a GNN's understanding of geometry."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

from im2sim.lab.adapters import ModelAdapter
from im2sim.lab.graphs import hop_distances, laplacian_eigs, mean_edge_length, subsample
from im2sim.lab.results import LabResult, experiment

InputFn = Callable[[Data], torch.Tensor]


def _default_input(adapter: ModelAdapter) -> InputFn:
    return lambda g: g[adapter.input_key]


def random_rotation(dim: int, gen: torch.Generator) -> torch.Tensor:
    """Random proper rotation matrix ``[dim, dim]`` (QR of a Gaussian matrix)."""
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=gen))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def apply_coord_transform(
    adapter: ModelAdapter, graph: Data, rot: torch.Tensor, scale: float, shift: torch.Tensor
) -> Data:
    """Return ``graph`` with coordinates mapped to ``scale * pos @ rot.T + shift``."""
    return adapter.set_coords(graph, scale * graph.pos @ rot.T + shift)


# --------------------------------------------------------------------------------------
# equivariance
# --------------------------------------------------------------------------------------


@experiment(trained=False)
def equivariance_error(
    adapter: ModelAdapter,
    graph: Data,
    n_trials: int = 8,
    output_kind: str = "scalar",
    input_fn: InputFn | None = None,
    seed: int = 0,
) -> LabResult:
    """Relative output change under rotation, translation and scaling of the coordinates.

    ``input_fn`` rebuilds the node input from the *transformed* graph, so models fed raw
    coordinates (``input_fn=lambda g: g.pos``) are tested honestly; models that only use
    ``edge_index`` or relative positions inside the layers are tested with the default
    (``graph.x`` unchanged).

    Args:
        adapter (ModelAdapter): Model under test.
        graph (Data): Graph with ``pos``.
        n_trials (int): Random transforms per kind.
        output_kind (str): ``"scalar"`` (output should be unchanged) or ``"vector"`` (the
            first ``dim`` output channels are a vector: rotated with the input under
            rotation, unchanged under translation/scaling).
        input_fn (Callable | None): ``graph -> node input`` (default ``graph[input_key]``).
        seed (int): Random seed.

    Returns:
        LabResult: scalars ``rotation_error``, ``translation_error``, ``scale_error``
        (mean over trials of ``|f(Tg) - ref| / |f(g)|``) and arrays with the per-trial values.
    """
    gen = torch.Generator().manual_seed(seed)
    input_fn = input_fn or _default_input(adapter)
    dim = graph.pos.size(1)
    extent = float((graph.pos.max(0).values - graph.pos.min(0).values).max())
    eye, zero = torch.eye(dim), torch.zeros(dim)
    with adapter.evaluating(), torch.no_grad():
        base = adapter(graph, input_fn(graph))
        denom = base.norm().clamp(min=1e-12)
        errs: dict[str, list[float]] = {"rotation": [], "translation": [], "scale": []}
        for _ in range(n_trials):
            rot = random_rotation(dim, gen)
            trials = {
                "rotation": (rot, 1.0, zero),
                "translation": (eye, 1.0, (torch.rand(dim, generator=gen) - 0.5) * 4 * extent),
                "scale": (eye, float(0.5 + 1.5 * torch.rand(1, generator=gen)), zero),
            }
            for kind, (r, s, t) in trials.items():
                g = apply_coord_transform(adapter, graph, r, s, t)
                out = adapter(g, input_fn(g))
                ref = base
                if output_kind == "vector" and kind == "rotation":
                    ref = base.clone()
                    ref[:, :dim] = base[:, :dim] @ r.T
                errs[kind].append(float((out - ref).norm() / denom))
    return LabResult(
        name="equivariance_error",
        scalars={f"{k}_error": float(np.mean(v)) for k, v in errs.items()},
        arrays={f"{k}_trials": np.asarray(v) for k, v in errs.items()},
        seed=seed,
        meta={"output_kind": output_kind},
    )


# --------------------------------------------------------------------------------------
# linear probes
# --------------------------------------------------------------------------------------


def geometric_targets(graph: Data, n_eigs: int = 3) -> dict[str, torch.Tensor]:
    """Per-node geometric quantities used as linear-probe targets.

    ``coords`` (centred, unit scale), ``mean_curvature`` (norm of the umbrella-operator
    Laplacian of the positions), ``laplacian_pos`` (its vector form, ~ normal direction),
    ``landmark_hops`` (hop distance to node 0), ``eig1..`` (low Laplacian eigenvectors), plus
    ``wall_dist`` / ``normals`` when the graph provides ``wall_dist`` or faces.
    """
    pos = graph.pos
    n = graph.num_nodes
    out: dict[str, torch.Tensor] = {}
    scale = (pos - pos.mean(0)).norm(dim=-1).max().clamp(min=1e-8)
    out["coords"] = (pos - pos.mean(0)) / scale
    ei, ew = get_laplacian(graph.edge_index, normalization="rw", num_nodes=n)
    lap = to_scipy_sparse_matrix(ei, ew, n).tocsr()
    lp = torch.from_numpy(lap @ pos.numpy()).float() / mean_edge_length(graph)
    out["laplacian_pos"] = lp
    out["mean_curvature"] = lp.norm(dim=-1, keepdim=True)
    hops = hop_distances(graph, 0)
    hops = np.where(np.isfinite(hops), hops, 0)
    out["landmark_hops"] = torch.from_numpy(hops / max(hops.max(), 1)).float()[:, None]
    if n_eigs:
        _, vecs = laplacian_eigs(graph, k=n_eigs + 1)
        for i in range(1, vecs.shape[1]):
            out[f"eig{i}"] = torch.from_numpy(vecs[:, i]).float()[:, None]
    if "wall_dist" in graph:
        out["wall_dist"] = graph.wall_dist.float()[:, None]
    if "face" in graph and graph.face is not None:
        out["normals"] = T.GenerateMeshNormals()(graph.clone()).norm
    return out


def ridge_r2(
    feats: torch.Tensor, target: torch.Tensor, train: torch.Tensor, alpha: float
) -> float:
    """Held-out R^2 of a ridge regression ``feats -> target`` fit on the ``train`` mask."""
    f = feats.double()
    y = target.double()
    if y.dim() == 1:
        y = y[:, None]
    mu, sd = f[train].mean(0), f[train].std(0).clamp(min=1e-8)
    f = (f - mu) / sd
    f = torch.cat([f, torch.ones(f.size(0), 1, dtype=f.dtype)], dim=1)
    ft = f[train]
    reg = alpha * ft.size(0) * torch.eye(f.size(1), dtype=f.dtype)
    reg[-1, -1] = 0
    w = torch.linalg.solve(ft.T @ ft + reg, ft.T @ y[train])
    test = ~train
    resid = ((f[test] @ w - y[test]) ** 2).sum()
    total = ((y[test] - y[train].mean(0)) ** 2).sum().clamp(min=1e-12)
    return float(1 - resid / total)


@experiment(trained=False)
def linear_probe(
    adapter: ModelAdapter,
    graph: Data,
    targets: dict[str, torch.Tensor] | None = None,
    layers: Sequence[str] | None = None,
    alpha: float = 1e-2,
    train_frac: float = 0.7,
    input_fn: InputFn | None = None,
    seed: int = 0,
) -> LabResult:
    """How linearly decodable is each geometric quantity from each layer's embedding?

    Embeddings come from :meth:`ModelAdapter.hidden_states` (``"input"`` plus every layer
    chosen by the adapter's ``hidden_filter``). A ridge regression is fit on a random
    ``train_frac`` of the nodes and R^2 is reported on the rest, so the score is held-out
    within the graph. ``input`` is the baseline: gain over it is what the network added.

    Args:
        adapter (ModelAdapter): Model under test.
        graph (Data): Graph to embed.
        targets (dict | None): ``name -> [N] or [N, k]`` targets (default
            :func:`geometric_targets`).
        layers (Sequence[str] | None): Subset of layer names to report.
        alpha (float): Ridge strength (scaled by the number of training nodes).
        train_frac (float): Fraction of nodes used for fitting.
        input_fn (Callable | None): ``graph -> node input`` (default ``graph[input_key]``).
        seed (int): Random seed.

    Returns:
        LabResult: array ``r2`` ``[n_layers, n_targets]`` with ``layers`` and ``targets`` name
        arrays; scalar ``best_mean_r2`` (best layer's mean R^2).
    """
    gen = torch.Generator().manual_seed(seed)
    targets = targets if targets is not None else geometric_targets(graph)
    x = input_fn(graph) if input_fn else None
    states = adapter.hidden_states(graph, x)
    if layers is not None:
        states = {k: v for k, v in states.items() if k in layers}
    n = graph.num_nodes
    train = torch.zeros(n, dtype=torch.bool)
    train[torch.randperm(n, generator=gen)[: int(train_frac * n)]] = True
    r2 = np.array(
        [[ridge_r2(h, t, train, alpha) for t in targets.values()] for h in states.values()]
    )
    return LabResult(
        name="linear_probe",
        scalars={"best_mean_r2": float(r2.mean(1).max())},
        arrays={"r2": r2, "layers": np.array(list(states)), "targets": np.array(list(targets))},
        seed=seed,
        meta={"alpha": alpha},
    )


# --------------------------------------------------------------------------------------
# resolution invariance
# --------------------------------------------------------------------------------------


@experiment(trained=False)
def resolution_invariance(
    adapter: ModelAdapter,
    graph: Data,
    fractions: Sequence[float] = (1.0, 0.75, 0.5, 0.35),
    connectivity: str = "knn",
    input_fn: InputFn | None = None,
    seed: int = 0,
    **subsample_kwargs,
) -> LabResult:
    """Does the prediction at a node survive changing the sampling density around it?

    Random subsets of the nodes are kept, edges are rebuilt from their positions (k-NN with the
    source mean degree, radius, or 2D Delaunay), and the new prediction is compared with the
    full-graph prediction at the same nodes (relative L2).

    Args:
        adapter (ModelAdapter): Model under test.
        graph (Data): Reference graph.
        fractions: Fractions of nodes kept (1.0 re-builds edges only, a connectivity control).
        connectivity (str): ``"knn"``, ``"radius"`` or ``"delaunay"``.
        input_fn (Callable | None): ``graph -> node input`` (default ``graph[input_key]``).
        seed (int): Random seed.
        **subsample_kwargs: ``k`` / ``r`` forwarded to :func:`im2sim.lab.graphs.subsample`.

    Returns:
        LabResult: arrays ``fractions``, ``error``; scalar ``max_error``.
    """
    input_fn = input_fn or _default_input(adapter)
    gen = torch.Generator().manual_seed(seed)
    n = graph.num_nodes
    with adapter.evaluating(), torch.no_grad():
        ref = adapter(graph, input_fn(graph))
        errs = []
        for f in fractions:
            idx = torch.randperm(n, generator=gen)[: max(4, int(round(f * n)))].sort().values
            sub = subsample(graph, idx, connectivity, **subsample_kwargs)
            out = adapter(sub, input_fn(sub))
            errs.append(float((out - ref[idx]).norm() / ref[idx].norm().clamp(min=1e-12)))
    return LabResult(
        name="resolution_invariance",
        scalars={"max_error": float(max(errs))},
        arrays={"fractions": np.asarray(fractions), "error": np.asarray(errs)},
        seed=seed,
        meta={"connectivity": connectivity},
    )


# --------------------------------------------------------------------------------------
# proximity leakage
# --------------------------------------------------------------------------------------


@experiment(trained=False)
def proximity_leakage(
    adapter: ModelAdapter,
    graph: Data,
    amplitude: float = 1.0,
    patch_width: float | None = None,
    seed: int = 0,
) -> LabResult:
    """Does a perturbation cross the gap between two limbs that are far apart on the graph?

    Needs a graph from :func:`im2sim.lab.graphs.u_tube_graph` (``limb`` and ``mirror``
    attributes). The input is perturbed on a patch of limb 0 and the output change is read at
    the patch and at the facing nodes of limb 1.

    Args:
        adapter (ModelAdapter): Model under test.
        graph (Data): U-tube graph.
        amplitude (float): Size of the added input perturbation.
        patch_width (float | None): Half-width of the patch along x (default 2 edge lengths).
        seed (int): Unused.

    Returns:
        LabResult: scalar ``leakage_ratio`` (mean |change| on the facing nodes divided by on
        the patch) and ``hop_gap`` (graph distance patch->facing); arrays ``pos_x``, ``limb``,
        ``delta`` (per-node |change|).
    """
    pos_x = graph.pos[:, 0]
    width = patch_width if patch_width is not None else 2 * mean_edge_length(graph)
    mid = 0.5 * (pos_x.min() + pos_x.max())
    patch = (graph.limb == 0) & ((pos_x - mid).abs() <= width)
    facing = graph.mirror[patch]
    x0 = graph[adapter.input_key]
    x1 = x0.clone()
    x1[patch] += amplitude
    with adapter.evaluating(), torch.no_grad():
        delta = (adapter(graph, x1) - adapter(graph, x0)).abs().sum(1)
    on_patch = float(delta[patch].mean())
    centre = int(patch.nonzero()[len(patch.nonzero()) // 2])
    hop_gap = float(hop_distances(graph, centre)[int(graph.mirror[centre])])
    return LabResult(
        name="proximity_leakage",
        scalars={
            "leakage_ratio": float(delta[facing].mean() / max(on_patch, 1e-12)),
            "hop_gap": hop_gap,
        },
        arrays={"pos_x": pos_x.numpy(), "limb": graph.limb.numpy(), "delta": delta.numpy()},
        seed=seed,
    )


__all__ = [
    "apply_coord_transform",
    "equivariance_error",
    "geometric_targets",
    "linear_probe",
    "proximity_leakage",
    "random_rotation",
    "resolution_invariance",
    "ridge_r2",
]
