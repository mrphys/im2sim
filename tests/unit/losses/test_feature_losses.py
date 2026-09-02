
import pytest
import torch
from torch_geometric.data import Data

from im2sim.losses.feature import KnnFeatureLoss


def make_graph(coords, x, batch=None):
    graph = Data(
        coords=torch.tensor(coords, dtype=torch.float32),
        x=torch.tensor(x, dtype=torch.float32),
    )

    if batch is not None:
        graph.batch = torch.tensor(batch, dtype=torch.long)

    return graph


# ---------------------------------------------------------------------------
# Initialisation / validation
# ---------------------------------------------------------------------------


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Mode must be either 'l1' or 'l2'"):
        KnnFeatureLoss(mode="invalid")


@pytest.mark.parametrize("mode", ["l1", "l2"])
def test_valid_modes(mode):
    loss = KnnFeatureLoss(mode=mode)

    assert loss.mode == mode


def test_constructor_arguments_are_stored():
    loss = KnnFeatureLoss(
        mode="l2",
        k=5,
        feature_key="features",
        feature_channels=[1, 3],
    )

    assert loss.mode == "l2"
    assert loss.k == 5
    assert loss.feature_key == "features"
    assert loss.feature_channels == [1, 3]


# ---------------------------------------------------------------------------
# Required graph attributes
# ---------------------------------------------------------------------------


def test_missing_true_coords_raises():
    true_graph = Data(x=torch.randn(3, 2))
    pred_graph = Data(
        x=torch.randn(4, 2),
        coords=torch.randn(4, 3),
    )

    loss = KnnFeatureLoss()

    with pytest.raises(
        ValueError,
        match="Both true_graph and pred_graph must have 'coords' attribute",
    ):
        loss(true_graph, pred_graph)


def test_missing_pred_coords_raises():
    true_graph = Data(
        x=torch.randn(3, 2),
        coords=torch.randn(3, 3),
    )
    pred_graph = Data(x=torch.randn(4, 2))

    loss = KnnFeatureLoss()

    with pytest.raises(
        ValueError,
        match="Both true_graph and pred_graph must have 'coords' attribute",
    ):
        loss(true_graph, pred_graph)


def test_missing_true_feature_raises():
    true_graph = Data(coords=torch.randn(3, 3))
    pred_graph = Data(
        x=torch.randn(4, 2),
        coords=torch.randn(4, 3),
    )

    print(true_graph)
    print(pred_graph)
    loss = KnnFeatureLoss()

    with pytest.raises(
        ValueError,
        match=f"Both true_graph and pred_graph must have non-None 'x' attribute."
    ):
        loss(true_graph, pred_graph)


def test_missing_pred_feature_raises():
    true_graph = Data(
        x=torch.randn(3, 2),
        coords=torch.randn(3, 3),
    )
    pred_graph = Data(coords=torch.randn(4, 3))

    loss = KnnFeatureLoss()

    with pytest.raises(
        ValueError,
        match=f"Both true_graph and pred_graph must have non-None 'x' attribute.",
    ):
        loss(true_graph, pred_graph)


# ---------------------------------------------------------------------------
# Single graph interpolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("l1", 1.0),
        ("l2", 1.0),
    ],
)
def test_loss_for_constant_offset(mode, expected):
    """
    Every true feature is 1 and every predicted feature is 2.

    KNN interpolation of a constant field should remain constant at 1,
    giving an error of exactly 1.
    """
    true_graph = make_graph(
        coords=[
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        x=[
            [1.0],
            [1.0],
            [1.0],
        ],
    )

    pred_graph = make_graph(
        coords=[
            [0.25, 0.0],
            [1.25, 0.0],
        ],
        x=[
            [2.0],
            [2.0],
        ],
    )

    loss = KnnFeatureLoss(mode=mode, k=2)

    result = loss(true_graph, pred_graph)

    assert torch.isclose(result, torch.tensor(expected))


def test_identical_constant_features_have_zero_loss():
    true_graph = make_graph(
        coords=[
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        x=[
            [3.0],
            [3.0],
            [3.0],
        ],
    )

    pred_graph = make_graph(
        coords=[
            [0.2, 0.0],
            [0.8, 0.0],
            [1.7, 0.0],
        ],
        x=[
            [3.0],
            [3.0],
            [3.0],
        ],
    )

    loss = KnnFeatureLoss(mode="l2", k=2)

    result = loss(true_graph, pred_graph)

    assert torch.isclose(result, torch.tensor(0.0))


# ---------------------------------------------------------------------------
# L1 vs L2
# ---------------------------------------------------------------------------


def test_l1_and_l2_losses_are_different():
    true_graph = make_graph(
        coords=[
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        x=[
            [0.0],
            [0.0],
            [0.0],
        ],
    )

    pred_graph = make_graph(
        coords=[
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        x=[
            [2.0],
            [2.0],
            [2.0],
        ],
    )

    l1 = KnnFeatureLoss(mode="l1", k=1)(true_graph, pred_graph)
    l2 = KnnFeatureLoss(mode="l2", k=1)(true_graph, pred_graph)

    assert torch.isclose(l1, torch.tensor(2.0))
    assert torch.isclose(l2, torch.tensor(4.0))


# ---------------------------------------------------------------------------
# Feature channels
# ---------------------------------------------------------------------------


def test_feature_channels_selects_requested_channels():
    true_graph = make_graph(
        coords=[
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        x=[
            [1.0, 10.0],
            [1.0, 10.0],
            [1.0, 10.0],
        ],
    )

    pred_graph = make_graph(
        coords=[
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        x=[
            [1.0, 20.0],
            [1.0, 20.0],
            [1.0, 20.0],
        ],
    )

    loss = KnnFeatureLoss(
        mode="l1",
        k=1,
        feature_channels=[0],
    )

    result = loss(true_graph, pred_graph)

    # Channel 0 is identical.
    assert torch.isclose(result, torch.tensor(0.0))


def test_feature_channels_can_select_multiple_channels():
    true_graph = make_graph(
        coords=[
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        x=[
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ],
    )

    pred_graph = make_graph(
        coords=[
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        x=[
            [2.0, 4.0, 6.0],
            [2.0, 4.0, 6.0],
        ],
    )

    loss = KnnFeatureLoss(
        mode="l1",
        k=1,
        feature_channels=[0, 2],
    )

    result = loss(true_graph, pred_graph)

    # Errors are |1 - 2| and |3 - 6| => mean = 2.
    assert torch.isclose(result, torch.tensor(2.0))


# ---------------------------------------------------------------------------
# Custom feature key
# ---------------------------------------------------------------------------


def test_custom_feature_key():
    true_graph = Data(
        coords=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        ),
        features=torch.tensor(
            [
                [1.0],
                [1.0],
            ]
        ),
    )

    pred_graph = Data(
        coords=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        ),
        features=torch.tensor(
            [
                [3.0],
                [3.0],
            ]
        ),
    )

    loss = KnnFeatureLoss(
        mode="l1",
        k=1,
        feature_key="features",
    )

    result = loss(true_graph, pred_graph)

    assert torch.isclose(result, torch.tensor(2.0))


# ---------------------------------------------------------------------------
# Batch handling
# ---------------------------------------------------------------------------


def test_unbatched_graphs_are_supported():
    true_graph = make_graph(
        coords=[
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        x=[
            [1.0],
            [1.0],
    ])

    pred_graph = make_graph(
        coords=[
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        x=[
            [2.0],
            [2.0],
        ],
    )

    loss = KnnFeatureLoss(mode="l1", k=1)

    result = loss(true_graph, pred_graph)

    assert torch.isclose(result, torch.tensor(1.0))


def test_batched_graphs_do_not_interpolate_between_graphs():
    """
    Verify that KNN interpolation respects graph boundaries.

    The two graphs are spatially very far apart. If batch information were
    ignored, points from the other graph could become nearest neighbours.
    """
    true_graph = make_graph(
        coords=[
            [0.0, 0.0],
            [1.0, 0.0],
            [100.0, 0.0],
            [101.0, 0.0],
        ],
        x=[
            [1.0],
            [1.0],
            [10.0],
            [10.0],
        ],
        batch=[0, 0, 1, 1],
    )

    pred_graph = make_graph(
        coords=[
            [0.5, 0.0],
            [100.5, 0.0],
        ],
        x=[
            [1.0],
            [10.0],
        ],
        batch=[0, 1],
    )

    loss = KnnFeatureLoss(mode="l1", k=2)

    result = loss(true_graph, pred_graph)

    assert torch.isclose(result, torch.tensor(0.0))


# ---------------------------------------------------------------------------
# Gradient behaviour
# ---------------------------------------------------------------------------


def test_loss_has_gradient_with_respect_to_prediction():
    true_graph = make_graph(
        coords=[
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        x=[
            [1.0],
            [2.0],
            [3.0],
        ],
    )

    pred_graph = make_graph(
        coords=[
            [0.5, 0.0],
            [1.5, 0.0],
        ],
        x=[
            [2.0],
            [2.0],
        ],
    )

    pred_graph.x.requires_grad_(True)

    loss = KnnFeatureLoss(mode="l1", k=2)

    result = loss(true_graph, pred_graph)
    result.backward()

    assert pred_graph.x.grad is not None
    assert torch.any(pred_graph.x.grad != 0)

