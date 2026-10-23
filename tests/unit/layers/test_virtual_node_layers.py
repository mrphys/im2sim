import pytest
import torch
import torch_geometric as pyg

from im2sim.configs import GraphConvBlockConfig, LayerConfig
from im2sim.layers import (
    DirectionalBlock,
    DirectionalConv,
    GraphContext,
    GraphConvBlock,
    SparseOps,
    StructVirtualNode,
)
from im2sim.utils.layer_util import PYG_LAYERS

KEYS = ("wall_index", "cap_index")


def make_graph(n=12, seed=0):
    """Ring of n nodes in 3D with two structures: the even nodes (wall) and nodes 1 and 3 (cap)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.arange(n)
    return pyg.data.Data(
        x=torch.randn(n, 6, generator=g),
        coords=torch.randn(n, 3, generator=g),
        edge_index=torch.stack([torch.cat([idx, (idx + 1) % n]), torch.cat([(idx + 1) % n, idx])]),
        wall_index=idx[::2],
        cap_index=torch.tensor([1, 3]),
    )


# ---------------------------------------------------------------------------
# SparseOps
# ---------------------------------------------------------------------------


def test_sparse_ops_match_dense_reference():
    graph = make_graph()
    h = torch.randn(graph.num_nodes, 5)
    out = SparseOps(graph.coords, graph.edge_index).feats(h)

    src, dst = graph.edge_index
    expected = torch.zeros(graph.num_nodes, 5 * 5)
    for i in range(graph.num_nodes):
        nbrs = src[dst == i]
        r = graph.coords[nbrs] - graph.coords[i]
        rhat = r / r.norm(dim=-1, keepdim=True)
        diff = h[nbrs] - h[i]
        expected[i] = torch.cat(
            [h[i], h[nbrs].mean(0)] + [(rhat[:, k : k + 1] * diff).mean(0) for k in range(3)]
        )
    assert torch.allclose(out, expected, atol=1e-5)


def test_sparse_ops_translation_invariant_and_rotation_equivariant_shapes():
    graph = make_graph()
    h = torch.randn(graph.num_nodes, 4)
    a = SparseOps(graph.coords, graph.edge_index).feats(h)
    b = SparseOps(graph.coords + 10.0, graph.edge_index).feats(h)
    assert a.shape == (graph.num_nodes, 20)
    assert torch.allclose(a, b, atol=1e-4)


# ---------------------------------------------------------------------------
# GraphContext
# ---------------------------------------------------------------------------


def test_assign_later_structures_override_earlier():
    ctx = GraphContext(make_graph(), KEYS)
    assign = ctx.assign
    assert assign[0] == 0 and assign[2] == 0
    # nodes 1 and 3 are only in the cap, odd nodes elsewhere are in no structure
    assert assign[1] == 1 and assign[3] == 1 and assign[5] == -1

    graph = make_graph()
    graph.cap_index = torch.tensor([0, 1])  # node 0 is in both: the cap wins
    assert GraphContext(graph, KEYS).assign[0] == 1


def test_missing_structure_key_raises():
    with pytest.raises(ValueError, match="nope_index"):
        GraphContext(make_graph(), ("nope_index",)).assign  # noqa: B018


def test_coords_fall_back_to_pos_and_pad():
    graph = make_graph()
    graph.pos = graph.coords[:, :2]
    del graph["coords"]
    ctx = GraphContext(graph, KEYS)
    assert ctx.coords.shape == (graph.num_nodes, 3)
    assert torch.equal(ctx.coords[:, 2], torch.zeros(graph.num_nodes))


def test_pos_features_shape_and_per_graph_normalisation():
    a, b = make_graph(seed=0), make_graph(seed=1)
    b.coords = b.coords * 7 + 3  # a very different scale and offset
    batch = pyg.data.Batch.from_data_list([a, b])
    ctx = GraphContext(batch, KEYS)
    assert ctx.num_graphs == 2
    feats = ctx.pos_features(n_fourier=2)
    assert feats.shape == (batch.num_nodes, 3 + 6 * 2)
    # normalised per graph: coordinates lie in [-1, 1] for both
    assert feats[:, :3].abs().max() <= 1 + 1e-6
    single = GraphContext(a, KEYS).pos_features(n_fourier=2)
    assert torch.allclose(feats[: a.num_nodes], single, atol=1e-5)


# ---------------------------------------------------------------------------
# DirectionalConv / DirectionalBlock
# ---------------------------------------------------------------------------


def test_directional_conv_is_registered_and_matches_apply():
    assert "DirectionalConv" in PYG_LAYERS
    graph = make_graph()
    conv = DirectionalConv(6, 3)
    out = conv(graph.x, graph.edge_index, graph.coords)
    ops = SparseOps(graph.coords, graph.edge_index)
    assert out.shape == (graph.num_nodes, 3)
    assert torch.allclose(out, conv.apply(graph.x, ops))


def test_directional_conv_inside_graph_conv_block():
    graph = make_graph()
    cfg = GraphConvBlockConfig(
        depth=2,
        conv_cfg=LayerConfig(name="DirectionalConv", kwargs={}),
        norm_cfg=LayerConfig(name="LayerNorm", kwargs={"mode": "node"}),
    )
    block = GraphConvBlock(in_channels=6, out_channels=4, cfg=cfg)
    out = block(graph)
    assert out.x.shape == (graph.num_nodes, 4)
    out.x.sum().backward()
    assert all(p.grad is not None for p in block.parameters())


def test_directional_block_is_residual():
    graph = make_graph()
    block = DirectionalBlock(6)
    torch.nn.init.zeros_(block.out.weight)
    torch.nn.init.zeros_(block.out.bias)
    out = block(graph.x, SparseOps(graph.coords, graph.edge_index))
    assert torch.equal(out, graph.x)


# ---------------------------------------------------------------------------
# StructVirtualNode
# ---------------------------------------------------------------------------


def test_struct_virtual_node_shape_and_gradients():
    graph = make_graph()
    layer = StructVirtualNode(8, n_structures=2)
    h = torch.randn(graph.num_nodes, 8, requires_grad=True)
    out = layer(h, GraphContext(graph, KEYS))
    assert out.shape == h.shape
    out.sum().backward()
    assert torch.isfinite(h.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


def test_struct_virtual_node_pools_structure_means():
    graph = make_graph()
    layer = StructVirtualNode(8, n_structures=2)
    h = torch.randn(graph.num_nodes, 8)
    tokens = layer.pool(h, GraphContext(graph, KEYS))
    v = layer.pool_v(h)
    assert tokens.shape == (1, 2, 8)
    assert torch.allclose(tokens[0, 0], v[graph.wall_index].mean(0), atol=1e-6)
    assert torch.allclose(tokens[0, 1], v[graph.cap_index].mean(0), atol=1e-6)


def test_struct_virtual_node_couples_every_node():
    """Changing one node changes all nodes: the VN reaches across the graph in one layer."""
    graph = make_graph(n=40)
    layer = StructVirtualNode(8, n_structures=2).eval()
    ctx = GraphContext(graph, KEYS)
    h = torch.randn(graph.num_nodes, 8)
    h2 = h.clone()
    h2[0] += 1.0  # node 0 is in the wall structure
    delta = (layer(h2, ctx) - layer(h, ctx)).abs().sum(1)
    assert (delta[1:] > 0).all()


def test_struct_virtual_node_empty_structure_is_zero_token():
    graph = make_graph()
    graph.cap_index = torch.zeros(0, dtype=torch.long)
    layer = StructVirtualNode(8, n_structures=2)
    tokens = layer.pool(torch.randn(graph.num_nodes, 8), GraphContext(graph, KEYS))
    assert torch.equal(tokens[0, 1], torch.zeros(8))


def test_struct_virtual_node_batch_matches_single_graphs():
    graphs = [make_graph(seed=0), make_graph(n=9, seed=1)]
    layer = StructVirtualNode(8, n_structures=2).eval()
    hs = [torch.randn(g.num_nodes, 8) for g in graphs]
    batch = pyg.data.Batch.from_data_list(graphs)
    out = layer(torch.cat(hs), GraphContext(batch, KEYS))
    expected = torch.cat([layer(h, GraphContext(g, KEYS)) for h, g in zip(hs, graphs)])
    assert torch.allclose(out, expected, atol=1e-5)


def test_struct_virtual_node_checks_structure_count():
    layer = StructVirtualNode(8, n_structures=3)
    with pytest.raises(ValueError, match="3 structures"):
        layer(torch.randn(12, 8), GraphContext(make_graph(), KEYS))
