import pytest
import torch
from torch_geometric.data import Data

from im2sim.layers.rasterization import FeatureRasterizer, MaskRasterizer, rasterise_feats


@pytest.fixture
def graph():
    coords = torch.tensor([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [6.0, 2.0, 3.0]])
    x = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    return Data(coords=coords, x=x, batch=torch.zeros(3, dtype=torch.long))


def test_rasterise_feats_places_features_at_points(graph):
    img = rasterise_feats(graph.coords, graph.x, (8, 8, 8))

    assert img.shape == (8, 8, 8, 2)
    torch.testing.assert_close(img[1, 1, 1], graph.x[0])
    torch.testing.assert_close(img[5, 5, 5], graph.x[1])
    # Voxels far from all points are left empty
    assert torch.all(img[7, 7, 0] == 0)


def test_feature_rasterizer_replaces_last_channels(graph):
    image = torch.full((1, 3, 8, 8, 8), -1.0)

    out = FeatureRasterizer()(graph, image)

    assert out.shape == (1, 3, 8, 8, 8)
    assert torch.all(out[:, 0] == -1.0)
    torch.testing.assert_close(out[0, 1:, 5, 5, 5], graph.x[1])


def test_feature_rasterizer_channel_selection(graph):
    image = torch.zeros(1, 2, 8, 8, 8)

    out = FeatureRasterizer(feature_channels=[1])(graph, image)

    assert out.shape == (1, 2, 8, 8, 8)
    assert out[0, 1, 1, 1, 1] == 10.0


def test_mask_rasterizer_replaces_last_channel(graph):
    image = torch.full((1, 2, 8, 8, 8), -1.0)

    out = MaskRasterizer()(graph, image)

    assert out.shape == (1, 2, 8, 8, 8)
    assert torch.all(out[:, 0] == -1.0)
    assert out[0, 1, 1, 1, 1] == 1.0
    assert out[0, 1, 7, 7, 0] == 0.0
