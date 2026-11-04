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


# -----------------------------
# Regression tests
# -----------------------------


def test_dilate_mask_keeps_singleton_spatial_dims():
    from im2sim.layers.rasterization import dilate_mask, pointcloud_to_mask

    mask = pointcloud_to_mask(torch.rand(5, 3) * 3, (4, 4, 1), [1, 1, 1])
    assert dilate_mask(mask).shape == (4, 4, 1)


def test_rasterizers_accept_graph_without_batch():
    import torch_geometric as pyg

    from im2sim.layers import FeatureRasterizer, MaskRasterizer

    g = pyg.data.Data(x=torch.rand(5, 1), coords=torch.rand(5, 3) * 7)
    image = torch.zeros(1, 2, 8, 8, 8)
    assert MaskRasterizer()(g, image).shape == image.shape
    assert FeatureRasterizer(feature_channels=[0])(g, image).shape == image.shape


def test_mask_rasterizer_keeps_empty_graphs_in_the_batch():
    import torch_geometric as pyg

    from im2sim.layers import MaskRasterizer

    g = pyg.data.Data(
        x=torch.rand(3, 1),
        coords=torch.rand(3, 3) * 7,
        batch=torch.zeros(3, dtype=torch.long),
    )
    image = torch.zeros(2, 2, 8, 8, 8)  # the second graph has no nodes
    out = MaskRasterizer()(g, image)
    assert out.shape == image.shape
    assert out[1, -1].sum() == 0


def test_feature_rasterizer_respects_voxel_sizes():
    import torch_geometric as pyg

    from im2sim.layers import FeatureRasterizer

    coords = torch.tensor([[4.0, 4.0, 4.0]])
    g = pyg.data.Data(x=torch.ones(1, 1), coords=coords, batch=torch.zeros(1, dtype=torch.long))
    image = torch.zeros(1, 1, 8, 8, 8)
    unit = FeatureRasterizer(voxel_sizes=(1.0, 1.0, 1.0))(g, image)
    coarse = FeatureRasterizer(voxel_sizes=(2.0, 2.0, 2.0))(g, image)
    # a point at 4.0 is voxel (4,4,4) with unit voxels and voxel (2,2,2) with 2.0 voxels
    assert unit[0, 0, 4, 4, 4] == 1 and coarse[0, 0, 2, 2, 2] == 1
    assert coarse[0, 0, 4, 4, 4] == 0


def test_projection_raises_clearly_if_batch_exceeds_image_batch():
    import pytest
    import torch_geometric as pyg

    from im2sim.layers import TrilinearProjection

    g = pyg.data.Data(coords=torch.rand(4, 3) * 3, batch=torch.tensor([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="batch size"):
        TrilinearProjection(4)(torch.rand(1, 2, 4, 4, 4), g)


def test_projection_accepts_graph_without_batch():
    import torch_geometric as pyg

    from im2sim.layers import TrilinearProjection

    g = pyg.data.Data(coords=torch.rand(4, 3) * 3)
    assert TrilinearProjection(4)(torch.rand(1, 2, 4, 4, 4), g).shape == (4, 2)
