import pytest
import torch
import torch_geometric as pyg

from im2sim.configs import HalfUNetConfig, ReverseHalfUNetConfig, UNetConfig
from im2sim.layers import MaskRasterizer
from im2sim.models import HalfUNet, Im2SimBase


@pytest.mark.parametrize("config_cls", [HalfUNetConfig, UNetConfig, ReverseHalfUNetConfig])
def test_segmentation_presets_do_not_normalise_the_output_logits(config_cls):
    for mode in ("single_class_segmentation_mode", "multiclass_segmentation_mode"):
        cfg = getattr(config_cls(), mode)()
        assert cfg.out_block_cfg.norm_cfg.name is None


def test_single_class_segmentation_can_fit_an_empty_mask():
    torch.manual_seed(0)
    model = HalfUNet(1, 1, 2, HalfUNetConfig().single_class_segmentation_mode())
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.rand(2, 1, 32, 32)

    for _ in range(150):
        optimiser.zero_grad()
        y = model(x)
        loss = torch.nn.functional.binary_cross_entropy(y, torch.zeros_like(y))
        loss.backward()
        optimiser.step()

    assert loss.item() < 0.05


class _LearnableProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, image_features, graph):
        return image_features.mean(dim=(2, 3, 4))[graph.batch] * self.scale


class _Encoder(torch.nn.Module):
    in_channels = 2

    def forward(self, x):
        return [x, x]


class _Decoder(torch.nn.Module):
    def forward(self, in_graph, projected_features):
        return in_graph


def test_projection_list_is_registered_as_submodules():
    model = Im2SimBase(
        (4, 4, 4), _Encoder(), _Decoder(), [_LearnableProjection(), _LearnableProjection()]
    )

    assert len(list(model.parameters())) == 2
    assert len(model.state_dict()) == 2


def _batched_graph(n_graphs):
    graphs = [
        pyg.data.Data(x=torch.rand(5, 1), coords=torch.rand(5, 3) * 7) for _ in range(n_graphs)
    ]
    return pyg.data.Batch.from_data_list(graphs)


class _OneChannelProjection(torch.nn.Module):
    def forward(self, image_features, graph):
        return torch.zeros(graph.coords.shape[0], 1)


class _PassThroughEncoder(torch.nn.Module):
    in_channels = 2

    def forward(self, x):
        return x


@pytest.mark.parametrize("n_graphs", [1, 2, 3])
def test_blank_image_matches_the_graph_batch_size(n_graphs):
    model = Im2SimBase(
        (8, 8, 8),
        _PassThroughEncoder(),
        _Decoder(),
        _OneChannelProjection(),
        rasterizer=MaskRasterizer(),
    )

    out = model(None, _batched_graph(n_graphs))
    assert out.num_graphs == n_graphs


def test_exclude_ids_selects_the_remaining_nodes_as_long_tensor():
    from im2sim.models.gnn_wrappers import _get_selected_nodes

    graph = pyg.data.Data(x=torch.rand(6, 1), excl=torch.tensor([1, 4]))
    ids = _get_selected_nodes(graph, exclude_ids=["excl"])
    assert ids.tolist() == [0, 2, 3, 5]

    everything = pyg.data.Data(x=torch.rand(3, 1), excl=torch.arange(3))
    ids = _get_selected_nodes(everything, exclude_ids=["excl"])
    assert ids.dtype == torch.long and ids.numel() == 0
    assert everything.x[ids].shape == (0, 1)


def test_exclude_ids_scales_to_large_graphs():
    import time

    from im2sim.models.gnn_wrappers import _get_selected_nodes

    n = 200_000
    graph = pyg.data.Data(x=torch.rand(n, 1), excl=torch.arange(0, n, 10))
    start = time.perf_counter()
    ids = _get_selected_nodes(graph, exclude_ids=["excl"])
    assert ids.numel() == n - n // 10
    assert time.perf_counter() - start < 2.0


def test_id_keys_without_index_in_the_name_warn_for_batches():
    from im2sim.models.gnn_wrappers import _get_selected_nodes

    graphs = [pyg.data.Data(x=torch.rand(4, 1), inlet_ids=torch.tensor([0, 1])) for _ in range(2)]
    batch = pyg.data.Batch.from_data_list(graphs)
    with pytest.warns(UserWarning, match="index"):
        _get_selected_nodes(batch, include_ids=["inlet_ids"])
