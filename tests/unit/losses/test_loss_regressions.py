import pytest
import torch
import torch_geometric as pyg

from im2sim.losses import AspectRatioLoss, FaceNormalLoss, FocalTverskyLoss, TverskyLoss
from im2sim.losses.pointcloud import _compute_batch_chamfer


@pytest.mark.parametrize("loss_cls", [FocalTverskyLoss, TverskyLoss])
def test_tversky_losses_have_finite_gradients_for_a_perfect_prediction(loss_cls):
    y_true = torch.zeros(1, 2, 4, 4)
    y_true[:, 0] = 1
    y_pred = y_true.clone().requires_grad_(True)

    loss = loss_cls()(y_true, y_pred)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(y_pred.grad).all()


def test_tversky_loss_value_unchanged_for_imperfect_prediction():
    y_true = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
    y_pred = torch.tensor([[[0.8, 0.6, 0.2, 0.1]]])
    loss = TverskyLoss(alpha=0.3, beta=0.7, epsilon=1e-5)(y_true, y_pred)

    tp, fp, fn = 1.4, 0.3, 0.6
    expected = 1 - (tp + 1e-5) / (tp + 0.3 * fp + 0.7 * fn + 1e-5)
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_chamfer_skips_batch_elements_without_a_counterpart():
    y1 = torch.rand(6, 3)
    b1 = torch.tensor([0, 0, 0, 1, 1, 1])
    y2 = torch.rand(3, 3)
    b2 = torch.tensor([0, 0, 0])

    loss = _compute_batch_chamfer(y1, y2, b1, b2)

    # only batch element 0 can be matched, so the result equals chamfer on that element alone
    expected = _compute_batch_chamfer(y1[:3], y2)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_aspect_ratio_of_a_collapsed_tetrahedron_is_finite():
    graph = pyg.data.Data(coords=torch.zeros(4, 3), cells=torch.tensor([[0], [1], [2], [3]]))
    assert torch.isfinite(AspectRatioLoss("cells", supervised=False)(None, graph))


def test_face_normal_consistency_ignores_single_face_graphs():
    coords = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    graph = pyg.data.Data(
        coords=coords,
        face=torch.tensor([[0], [1], [2]]),
        batch=torch.zeros(3, dtype=torch.long),
    )
    loss = FaceNormalLoss("face", supervised=False)(None, graph)

    # the std of a single normal is undefined (NaN), such graphs must not contribute
    assert loss.item() == 0.0
