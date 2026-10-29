import pytest
import torch
import torch_geometric as pyg

from im2sim.configs import LayerConfig, StructVNConfig
from im2sim.lab import ModelAdapter, jacobian_reach, tube_graph
from im2sim.models import Im2SimBase, StructVN, StructVNDecoder

KEYS = ["wall_index", "cap_1_index", "cap_2_index"]


def small_cfg(**kwargs):
    return StructVNConfig(
        hidden_channels=8,
        n_blocks=2,
        n_local=1,
        n_fourier=2,
        **kwargs,
    )


def mesh(seed=0, in_channels=5):
    """Tube mesh with a wall and two end caps as structures."""
    g = tube_graph(n_rings=6, n_per_ring=8, n_radial=2)
    gen = torch.Generator().manual_seed(seed)
    z = g.pos[:, 2]
    g.wall_index = g.is_wall.nonzero().squeeze(1)
    g.cap_1_index = (g.is_cap & (z == z.min())).nonzero().squeeze(1)
    g.cap_2_index = (g.is_cap & (z == z.max())).nonzero().squeeze(1)
    g.coords = g.pos
    g.x = torch.randn(g.num_nodes, in_channels, generator=gen)
    return g


# ---------------------------------------------------------------------------
# StructVN
# ---------------------------------------------------------------------------


def test_forward_shape_and_input_untouched():
    graph = mesh()
    x = graph.x.clone()
    model = StructVN(5, 3, small_cfg(), KEYS)
    out = model(graph)
    assert out.x.shape == (graph.num_nodes, 3)
    assert torch.equal(graph.x, x)
    assert torch.equal(out.edge_index, graph.edge_index)


def test_batch_matches_single_graphs():
    graphs = [mesh(0), mesh(1)]
    model = StructVN(5, 3, small_cfg(), KEYS).eval()
    with torch.no_grad():
        out = model(pyg.data.Batch.from_data_list(graphs)).x
        expected = torch.cat([model(g).x for g in graphs])
    assert torch.allclose(out, expected, atol=1e-5)


def test_translation_invariant():
    graph = mesh()
    shifted = graph.clone()
    shifted.coords = shifted.coords + 5.0
    model = StructVN(5, 3, small_cfg(), KEYS).eval()
    with torch.no_grad():
        assert torch.allclose(model(graph).x, model(shifted).x, atol=1e-4)


@pytest.mark.parametrize("n_structures", [1, 2])
def test_structure_keys_set_the_number_of_virtual_nodes(n_structures):
    model = StructVN(5, 3, small_cfg(), KEYS[:n_structures])
    assert all(vn.n_structures == n_structures for vn in model.far)
    assert model(mesh()).x.shape[1] == 3


def test_checkpointing_does_not_change_outputs_or_gradients():
    graph = mesh()
    a = StructVN(5, 3, small_cfg(checkpointing=True), KEYS).train()
    b = StructVN(5, 3, small_cfg(checkpointing=False), KEYS).train()
    b.load_state_dict(a.state_dict())
    for model in (a, b):
        model(graph).x.pow(2).mean().backward()
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-6)


def test_state_dict_roundtrip():
    graph = mesh()
    a, b = StructVN(5, 3, small_cfg(), KEYS).eval(), StructVN(5, 3, small_cfg(), KEYS).eval()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        assert torch.equal(a(graph).x, b(graph).x)


GAT = LayerConfig(name="GATConv", kwargs={"heads": 2, "concat": False})


@pytest.mark.parametrize(
    "conv_cfg",
    [
        LayerConfig(name="DirectionalConv", kwargs={}),
        LayerConfig(name="GCNConv", kwargs={}),
        LayerConfig(name="SAGEConv", kwargs={}),
        GAT,
    ],
    ids=lambda cfg: cfg.name,
)
@pytest.mark.parametrize("checkpointing", [False, True])
def test_conv_cfg_selects_the_local_convolution(conv_cfg, checkpointing):
    graph = mesh()
    model = StructVN(5, 3, small_cfg(conv_cfg=conv_cfg, checkpointing=checkpointing), KEYS).train()
    assert all(type(b.mix).__name__ == conv_cfg.name for local in model.local for b in local)
    out = model(graph)
    assert out.x.shape == (graph.num_nodes, 3)
    out.x.pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_default_conv_is_gat():
    model = StructVN(5, 3, small_cfg(), KEYS)
    assert small_cfg().conv_cfg.name == "GATConv"
    assert not model.uses_sparse_ops


def test_directional_conv_keeps_the_state_dict_keys():
    model = StructVN(5, 3, small_cfg(conv_cfg=LayerConfig(name="DirectionalConv", kwargs={})), KEYS)
    assert model.uses_sparse_ops
    assert "local.0.0.mix.weight" in model.state_dict()
    assert "local.0.0.norm.weight" in model.state_dict()


def test_unknown_conv_raises():
    with pytest.raises(ValueError, match="not found"):
        StructVN(5, 3, small_cfg(conv_cfg=LayerConfig(name="NotAConv", kwargs={})), KEYS)


def test_conv_cfg_in_decoder_and_config_roundtrip(tmp_path):
    cfg = small_cfg(conv_cfg=GAT, protocol="predict", pred_feature_key="cfd")
    cfg.save(tmp_path / "cfg.json")
    assert StructVNConfig.load(tmp_path / "cfg.json") == cfg
    decoder = StructVNDecoder(5, 3, cfg, KEYS)
    assert decoder(mesh()).cfd.shape[1] == 3


def test_config_save_and_load(tmp_path):
    cfg = small_cfg(protocol="update", pred_feature_key="cfd", pred_feature_channels=[0, 1])
    cfg.save(tmp_path / "cfg.json")
    assert StructVNConfig.load(tmp_path / "cfg.json") == cfg


def test_lab_sees_global_coupling_through_the_virtual_nodes():
    """The lab probe of untrained models: every node is coupled to every other node."""
    graph = mesh(in_channels=1)
    adapter = ModelAdapter(StructVN(1, 1, small_cfg(), KEYS))
    assert adapter.kind == "pyg_data"
    result = jacobian_reach(adapter, graph, centre=20)
    assert result.scalars["global_coupling"] > 0
    assert result.scalars["receptive_field"] == result.arrays["hops"].max()


# ---------------------------------------------------------------------------
# StructVNDecoder
# ---------------------------------------------------------------------------


def test_decoder_predicts_into_cfd():
    graph = mesh()
    decoder = StructVNDecoder(5, 4, small_cfg(pred_feature_key="cfd"), KEYS)
    out = decoder(graph)
    assert out.cfd.shape == (graph.num_nodes, 4)
    assert out.x.shape == graph.x.shape
    assert "cfd" not in graph


def test_decoder_with_projected_features_and_update_protocol():
    graph = mesh()
    projected = torch.randn(graph.num_nodes, 3)
    cfg = small_cfg(protocol="update", pred_feature_key="cfd")
    decoder = StructVNDecoder(5 + 3, 4, cfg, KEYS, graph_channels=5)
    graph.cfd = torch.ones(graph.num_nodes, 4)
    out = decoder(graph, projected)
    assert out.x.shape == graph.x.shape
    assert not torch.allclose(out.cfd, graph.cfd)


def test_decoder_include_ids_only_writes_selected_nodes():
    graph = mesh()
    graph.cfd = torch.zeros(graph.num_nodes, 4)
    cfg = small_cfg(pred_feature_key="cfd", include_ids=["cap_1_index"])
    out = StructVNDecoder(5, 4, cfg, KEYS)(graph)
    mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    mask[graph.cap_1_index] = True
    assert (out.cfd[mask].abs().sum(1) > 0).all()
    assert torch.equal(out.cfd[~mask], torch.zeros_like(out.cfd[~mask]))


def test_decoder_in_im2sim_base():
    """The decoder is a drop-in `graph_decoder`: the image encoder and projection are stubs."""

    class Encoder(torch.nn.Module):
        in_channels = 1

        def forward(self, image):
            return image

    class Projection(torch.nn.Module):
        def forward(self, image_features, graph):
            return image_features.mean() * torch.ones(graph.num_nodes, 3)

    graph = mesh()
    model = Im2SimBase(
        image_shape=(4, 4, 4),
        image_encoder=Encoder(),
        graph_decoder=StructVNDecoder(
            5 + 3, 4, small_cfg(pred_feature_key="cfd"), KEYS, graph_channels=5
        ),
        projections=Projection(),
    )
    image = torch.randn(1, 1, 4, 4, 4, requires_grad=True)
    out = model(image, graph)
    assert out.cfd.shape == (graph.num_nodes, 4)
    out.cfd.sum().backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0


@pytest.mark.parametrize("keys", [[], "wall_index"])
def test_structure_keys_must_be_a_non_empty_sequence_of_names(keys):
    with pytest.raises(ValueError, match="structure_keys"):
        StructVN(5, 3, small_cfg(), keys)


def test_structure_keys_are_required():
    with pytest.raises(TypeError):
        StructVN(5, 3, small_cfg())
    with pytest.raises(TypeError):
        StructVNDecoder(5, 3, small_cfg())
