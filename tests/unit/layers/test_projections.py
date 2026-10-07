import pytest
import torch
from torch_geometric.data import Data

from im2sim.layers.projections import TrilinearProjection


def _graph(coords):
    return Data(coords=coords, batch=torch.zeros(coords.shape[0], dtype=torch.long))


def test_integer_coords_return_voxel_values():
    image = torch.randn(1, 3, 4, 4, 4)
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [3.0, 3.0, 3.0], [2.0, 0.0, 1.0]])

    out = TrilinearProjection(image_dim=4)(image, _graph(coords))

    expected = torch.stack([image[0, :, int(x), int(y), int(z)] for x, y, z in coords])
    assert torch.allclose(out, expected)


@pytest.mark.parametrize("coord", [[0.5, 1.25, 2.75], [1.0, 1.5, 2.0], [2.9, 0.1, 3.0]])
def test_matches_linear_function(coord):
    # Trilinear interpolation of a linear field is exact.
    grid = torch.arange(4, dtype=torch.float32)
    gx, gy, gz = torch.meshgrid(grid, grid, grid, indexing="ij")
    image = (1.0 + 2.0 * gx - 3.0 * gy + 0.5 * gz)[None, None]
    x, y, z = coord

    out = TrilinearProjection(image_dim=4)(image, _graph(torch.tensor([coord])))

    assert torch.allclose(out, torch.tensor([[1.0 + 2.0 * x - 3.0 * y + 0.5 * z]]))


def test_coords_are_scaled_to_feature_map():
    image = torch.randn(1, 2, 4, 4, 4)
    coords = torch.tensor([[2.0, 4.0, 6.0]])

    out = TrilinearProjection(image_dim=8)(image, _graph(coords))

    assert torch.allclose(out, image[0, :, 1, 2, 3][None])


@pytest.mark.parametrize(
    "batch", [[0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1, 1]]
)
def test_batched_graphs_match_single_graphs(batch):
    image = torch.randn(2, 3, 4, 4, 4)
    batch = torch.tensor(batch)
    coords = torch.rand(len(batch), 3) * 3
    proj = TrilinearProjection(image_dim=4)

    out = proj(image, Data(coords=coords, batch=batch))

    assert out.shape == (len(batch), 3)
    for i in range(2):
        mask = batch == i
        assert torch.allclose(out[mask], proj(image[i : i + 1], _graph(coords[mask])))
