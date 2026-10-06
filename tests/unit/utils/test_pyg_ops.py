import warnings

import pytest
import torch
import torch_geometric.nn as gnn
from torch_geometric.data import Data

from im2sim.losses.feature import KnnFeatureLoss
from im2sim.losses.pointcloud import ChamferLoss
from im2sim.utils import pyg_ops

requires_compiled_knn = pytest.mark.skipif(
    not pyg_ops.HAS_COMPILED_KNN, reason="no compiled knn backend installed"
)
requires_compiled_graclus = pytest.mark.skipif(
    not pyg_ops.HAS_COMPILED_GRACLUS, reason="no compiled graclus backend installed"
)


@pytest.fixture
def force_fallback(monkeypatch):
    monkeypatch.setattr(pyg_ops, "HAS_COMPILED_KNN", False)
    monkeypatch.setattr(pyg_ops, "HAS_COMPILED_GRACLUS", False)
    monkeypatch.setattr(pyg_ops, "_warned", set())


@pytest.fixture
def no_fallback_warnings(force_fallback):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pure-PyTorch fallback.*")
        yield


def _random_batched_points(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(50, 3, generator=g)
    y = torch.rand(30, 3, generator=g)
    # Example 1 only has 2 points in x, fewer than some of the k values tested.
    batch_x = torch.repeat_interleave(torch.arange(3), torch.tensor([20, 2, 28]))
    batch_y = torch.repeat_interleave(torch.arange(3), torch.tensor([10, 10, 10]))
    return x, y, batch_x, batch_y


def _as_set(assign_index):
    return set(map(tuple, assign_index.t().tolist()))


def _reference_knn(x, y, k, batch_x=None, batch_y=None):
    """Brute force knn, one query at a time."""
    pairs = set()
    for i in range(y.size(0)):
        candidates = torch.arange(x.size(0))
        if batch_x is not None:
            candidates = candidates[batch_x == batch_y[i]]
        dist = (x[candidates] - y[i]).norm(dim=-1)
        nearest = candidates[dist.argsort()[:k]]
        pairs.update((i, int(j)) for j in nearest)
    return pairs


def _reference_knn_interpolate(feats, pos_x, pos_y, batch_x, batch_y, k):
    out = torch.zeros(pos_y.size(0), feats.size(1))
    for i in range(pos_y.size(0)):
        candidates = (batch_x == batch_y[i]).nonzero().flatten()
        sq_dist = ((pos_x[candidates] - pos_y[i]) ** 2).sum(-1)
        order = sq_dist.argsort()[:k]
        w = 1.0 / sq_dist[order].clamp(min=1e-16)
        out[i] = (feats[candidates[order]] * w[:, None]).sum(0) / w.sum()
    return out


# ---------------------------------------------------------------------------
# knn fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("k", [1, 3, 25, 100])
@pytest.mark.parametrize("max_chunk_elements", [1, 64, 2**24])
def test_knn_fallback_matches_reference(batched, k, max_chunk_elements):
    x, y, batch_x, batch_y = _random_batched_points()
    if not batched:
        batch_x = batch_y = None
    actual = pyg_ops._knn_torch(x, y, k, batch_x, batch_y, max_chunk_elements)
    assert actual.dtype == torch.long
    assert _as_set(actual) == _reference_knn(x, y, k, batch_x, batch_y)


def test_knn_fallback_returns_neighbours_sorted_by_distance(no_fallback_warnings):
    x, y, _, _ = _random_batched_points()
    row, col = pyg_ops.knn(x, y, 5)
    dist = (x[col] - y[row]).norm(dim=-1).view(-1, 5)
    assert torch.equal(row.view(-1, 5), torch.arange(y.size(0))[:, None].expand(-1, 5))
    assert (dist[:, 1:] >= dist[:, :-1]).all()


def test_knn_fallback_respects_batches(no_fallback_warnings):
    x, y, batch_x, batch_y = _random_batched_points()
    row, col = pyg_ops.knn(x, y, 3, batch_x=batch_x, batch_y=batch_y)
    assert torch.equal(batch_y[row], batch_x[col])
    # Queries in example 1 only get the 2 points available instead of 3.
    assert row.numel() == 10 * 3 + 10 * 2 + 10 * 3


def test_knn_fallback_skips_examples_missing_from_x(no_fallback_warnings):
    x = torch.rand(5, 3)
    y = torch.rand(4, 3)
    batch_x = torch.zeros(5, dtype=torch.long)
    batch_y = torch.tensor([0, 0, 1, 1])
    row, col = pyg_ops.knn(x, y, 2, batch_x=batch_x, batch_y=batch_y)
    assert set(row.tolist()) == {0, 1}


@pytest.mark.parametrize("missing", ["batch_x", "batch_y"])
def test_knn_fallback_treats_missing_batch_as_example_zero(no_fallback_warnings, missing):
    x, y, batch_x, batch_y = _random_batched_points()
    batches = {"batch_x": batch_x, "batch_y": batch_y}
    batches[missing] = None
    zeros_x = torch.zeros_like(batch_x) if missing == "batch_x" else batch_x
    zeros_y = torch.zeros_like(batch_y) if missing == "batch_y" else batch_y

    actual = pyg_ops.knn(x, y, 3, **batches)
    assert _as_set(actual) == _reference_knn(x, y, 3, zeros_x, zeros_y)


def test_knn_accepts_float_batch_vectors(no_fallback_warnings):
    x, y, batch_x, batch_y = _random_batched_points()
    row, col = pyg_ops.knn(x, y, 1, batch_x=batch_x.float(), batch_y=batch_y.float())
    assert torch.equal(batch_y[row], batch_x[col])


@pytest.mark.parametrize(
    "num_x, num_y, k",
    [(0, 4, 2), (4, 0, 2), (4, 4, 0)],
)
def test_knn_fallback_empty_result(no_fallback_warnings, num_x, num_y, k):
    out = pyg_ops.knn(torch.rand(num_x, 3), torch.rand(num_y, 3), k)
    assert out.shape == (2, 0)
    assert out.dtype == torch.long


def test_knn_fallback_finds_exact_matches(no_fallback_warnings):
    x = torch.rand(20, 3)
    perm = torch.randperm(20)
    row, col = pyg_ops.knn(x, x[perm], 1)
    assert torch.equal(col[row.argsort()], perm)


def test_knn_fallback_handles_mixed_precision(no_fallback_warnings):
    x, y, _, _ = _random_batched_points()
    actual = pyg_ops.knn(x.double(), y, 2)
    assert _as_set(actual) == _reference_knn(x.double(), y.double(), 2)


def test_knn_fallback_warns_once(force_fallback):
    x, y, _, _ = _random_batched_points()
    with pytest.warns(UserWarning, match="im2sim-install-pyg-addons") as record:
        pyg_ops.knn(x, y, 1)
        pyg_ops.knn(x, y, 1)
    assert sum("`knn`" in str(w.message) for w in record) == 1


# ---------------------------------------------------------------------------
# knn_interpolate fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 3])
def test_knn_interpolate_fallback_matches_reference(no_fallback_warnings, k):
    x, y, batch_x, batch_y = _random_batched_points()
    feats = torch.rand(x.size(0), 4)
    actual = pyg_ops.knn_interpolate(feats, x, y, batch_x, batch_y, k=k)
    expected = _reference_knn_interpolate(feats, x, y, batch_x, batch_y, k)
    torch.testing.assert_close(actual, expected)


def test_knn_interpolate_fallback_is_identity_at_source_points(no_fallback_warnings):
    pos = torch.rand(20, 3)
    feats = torch.rand(20, 2)
    torch.testing.assert_close(pyg_ops.knn_interpolate(feats, pos, pos, k=3), feats)


def test_knn_interpolate_fallback_preserves_constant_field(no_fallback_warnings):
    x, y, batch_x, batch_y = _random_batched_points()
    feats = torch.full((x.size(0), 3), 7.0)
    out = pyg_ops.knn_interpolate(feats, x, y, batch_x, batch_y, k=4)
    torch.testing.assert_close(out, torch.full((y.size(0), 3), 7.0))


def test_knn_interpolate_fallback_gradients_flow_to_features_only(no_fallback_warnings):
    x, y, _, _ = _random_batched_points()
    x.requires_grad_(True)
    feats = torch.rand(x.size(0), 2, requires_grad=True)
    pyg_ops.knn_interpolate(feats, x, y, k=3).sum().backward()
    assert feats.grad is not None and feats.grad.abs().sum() > 0
    assert x.grad is None


# ---------------------------------------------------------------------------
# graclus fallback
# ---------------------------------------------------------------------------


def _grid_edge_index(n=6):
    idx = torch.arange(n * n).view(n, n)
    right = torch.stack([idx[:, :-1].flatten(), idx[:, 1:].flatten()])
    down = torch.stack([idx[:-1].flatten(), idx[1:].flatten()])
    edge_index = torch.cat([right, down], dim=1)
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def _check_maximal_matching(cluster, edge_index):
    edges = _as_set(edge_index)
    sizes = torch.bincount(cluster, minlength=cluster.numel())
    for c in cluster.unique().tolist():
        members = (cluster == c).nonzero().flatten().tolist()
        assert c in members, "cluster id must be one of its members"
        assert len(members) <= 2
        if len(members) == 2:
            assert tuple(members) in edges
    # Greedy matching is maximal: no edge joins two unmatched nodes.
    for u, v in edges:
        if u != v:
            assert sizes[cluster[u]] == 2 or sizes[cluster[v]] == 2


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_graclus_fallback_is_maximal_matching(no_fallback_warnings, weighted, seed):
    torch.manual_seed(seed)
    edge_index = _grid_edge_index()
    weight = torch.rand(edge_index.size(1)) if weighted else None
    cluster = pyg_ops.graclus(edge_index, weight, 36)
    assert cluster.shape == (36,)
    assert cluster.dtype == torch.long
    _check_maximal_matching(cluster, edge_index)


@pytest.mark.parametrize("seed", range(5))
def test_graclus_fallback_prefers_heaviest_edge(no_fallback_warnings, seed):
    # 4-cycle where each node has exactly one heavy edge, so any visiting order gives {0,1},{2,3}.
    torch.manual_seed(seed)
    edge_index = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 0], [1, 0, 3, 2, 2, 1, 0, 3]])
    weight = torch.tensor([10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0])
    cluster = pyg_ops.graclus(edge_index, weight)
    assert cluster[0] == cluster[1]
    assert cluster[2] == cluster[3]
    assert cluster[0] != cluster[2]


def test_graclus_fallback_ignores_self_loops(no_fallback_warnings):
    edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    cluster = pyg_ops.graclus(edge_index, torch.ones(3))
    assert torch.equal(cluster, torch.arange(3))


def test_graclus_fallback_isolated_nodes_are_singletons(no_fallback_warnings):
    edge_index = torch.tensor([[0, 1], [1, 0]])
    cluster = pyg_ops.graclus(edge_index, num_nodes=4)
    assert cluster[0] == cluster[1]
    assert cluster[2] == 2
    assert cluster[3] == 3


def test_graclus_fallback_infers_num_nodes(no_fallback_warnings):
    cluster = pyg_ops.graclus(_grid_edge_index(4))
    assert cluster.shape == (16,)


def test_graclus_fallback_empty_graph(no_fallback_warnings):
    edge_index = torch.empty((2, 0), dtype=torch.long)
    assert pyg_ops.graclus(edge_index).shape == (0,)
    assert torch.equal(pyg_ops.graclus(edge_index, num_nodes=3), torch.arange(3))


def test_graclus_fallback_unsorted_edges(no_fallback_warnings):
    edge_index = _grid_edge_index()
    perm = torch.randperm(edge_index.size(1))
    weight = torch.rand(edge_index.size(1))
    cluster = pyg_ops.graclus(edge_index[:, perm], weight[perm], 36)
    _check_maximal_matching(cluster, edge_index)


def test_graclus_fallback_warns_once(force_fallback):
    with pytest.warns(UserWarning, match="`graclus`") as record:
        pyg_ops.graclus(_grid_edge_index(3))
        pyg_ops.graclus(_grid_edge_index(3))
    assert len(record) == 1


# ---------------------------------------------------------------------------
# Call sites work with the fallbacks
# ---------------------------------------------------------------------------


def test_chamfer_loss_with_fallback(no_fallback_warnings):
    coords = torch.rand(10, 3)
    batch = torch.zeros(10, dtype=torch.long)
    true_graph = Data(coords=coords, batch=batch)
    loss = ChamferLoss()
    assert loss(true_graph, Data(coords=coords.clone(), batch=batch)) == 0
    shifted = Data(coords=coords + torch.tensor([0.0, 0.0, 1e3]), batch=batch)
    assert loss(true_graph, shifted) > 0


def test_knn_feature_loss_with_fallback(no_fallback_warnings):
    coords = torch.rand(10, 3)
    true_graph = Data(coords=coords, x=torch.ones(10, 2))
    pred_graph = Data(coords=torch.rand(6, 3), x=torch.full((6, 2), 3.0))
    result = KnnFeatureLoss(mode="l1", k=2)(true_graph, pred_graph)
    torch.testing.assert_close(result, torch.tensor(2.0))


def test_mesh_ops_with_fallback(no_fallback_warnings):
    pytest.importorskip("pyvista")
    from im2sim.mesh_ops.mesh_utils import cluster_pool, rasterize

    points = torch.tensor([[0.5, 0.5, 0.5], [3.5, 3.5, 3.5]])
    dists = rasterize(points, [4, 4, 4], [1.0, 1.0, 1.0])
    assert dists.shape == (4, 4, 4)
    assert dists[0, 0, 0] == 0
    assert dists[3, 3, 3] == 0

    # Non-unit voxel sizes: centroids are at (i + 0.5) * size
    dists = rasterize(torch.tensor([[1.0, 1.0, 1.0]]), [4, 3, 2], [2.0, 2.0, 2.0])
    assert dists.shape == (4, 3, 2)
    assert dists[0, 0, 0] == 0

    n = 6
    idx = torch.arange(n * n)
    mesh = Data(
        x=torch.stack([idx // n, idx % n, torch.zeros_like(idx)], dim=-1).float(),
        edge_index=_grid_edge_index(n),
    )
    pooled = cluster_pool(mesh)
    assert n * n // 2 <= pooled.x.size(0) < n * n


# ---------------------------------------------------------------------------
# Fallbacks agree with the compiled PyG ops
# ---------------------------------------------------------------------------


@requires_compiled_knn
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("k", [1, 3, 25])
def test_knn_fallback_matches_pyg(batched, k):
    x, y, batch_x, batch_y = _random_batched_points()
    if not batched:
        batch_x = batch_y = None
    expected = gnn.knn(x, y, k, batch_x=batch_x, batch_y=batch_y)
    actual = pyg_ops._knn_torch(x, y, k, batch_x, batch_y, max_chunk_elements=64)
    assert _as_set(actual) == _as_set(expected)


@requires_compiled_knn
def test_knn_interpolate_fallback_matches_pyg(no_fallback_warnings):
    x, y, batch_x, batch_y = _random_batched_points()
    feats = torch.rand(x.size(0), 4)
    expected = gnn.knn_interpolate(feats, x, y, batch_x, batch_y, k=3)
    actual = pyg_ops.knn_interpolate(feats, x, y, batch_x, batch_y, k=3)
    torch.testing.assert_close(actual, expected)


@requires_compiled_graclus
def test_graclus_compiled_is_maximal_matching():
    edge_index = _grid_edge_index()
    cluster = pyg_ops.graclus(edge_index, torch.rand(edge_index.size(1)), 36)
    _check_maximal_matching(cluster, edge_index)
