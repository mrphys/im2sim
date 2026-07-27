import torch

from im2sim.data.mesh_utils import *


def test_make_padded_batch_shapes():
    x = torch.tensor([[1], [2], [3], [4], [5]])
    batch = torch.tensor([0, 0, 0, 1, 2])

    padded_x, mask = make_padded_batch(x, batch)

    assert padded_x.shape == (3, 3, 1)
    assert mask.shape == (3, 3)


def test_make_padded_batch_mask_correct():
    x = torch.randn(5, 2)
    batch = torch.tensor([0, 0, 0, 1, 2])

    _, mask = make_padded_batch(x, batch)

    expected = torch.tensor([
        [True, True, True],
        [True, False, False],
        [True, False, False],
    ])

    assert torch.equal(mask, expected)

def test_compute_edge_lengths_simple():
    points = torch.tensor([
        [0.0, 0.0],
        [3.0, 4.0],
    ])
    edges = torch.tensor([[0], [1]])

    distances = compute_edge_lengths(points, edges)

    assert torch.allclose(distances, torch.tensor([5.0]))


def test_hard_threshold():
    y = torch.tensor([0.5, 1.0, 1.5])

    out = hard_threshold(y, threshold=1.0)

    expected = torch.tensor([1.0, 0.0, 0.0])
    assert torch.equal(out, expected)


def test_soft_threshold_range():
    y = torch.tensor([0.0, 1.5, 3.0])
    out = soft_threshold(y)

    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_soft_threshold_behavior():
    y_low = torch.tensor([0.0])
    y_high = torch.tensor([10.0])

    assert soft_threshold(y_low) > soft_threshold(y_high)


def test_set_attrs():
    data = Data()
    attrs = {
        "x": torch.tensor([[1.0]]),
        "edge_index": torch.tensor([[0], [0]])
    }

    set_attrs(data, attrs)

    assert torch.equal(data.x, attrs["x"])
    assert torch.equal(data.edge_index, attrs["edge_index"])