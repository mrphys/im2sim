import numpy as np
import numpy as np
import pytest
import torch

from im2sim.losses.confusion_losses import (
    DiceLoss,
    FocalTverskyLoss,
    IoULoss,
    TverskyLoss,
)


LOSS_CLASSES = {
    "FocalTverskyLoss": FocalTverskyLoss,
    "TverskyLoss": TverskyLoss,
    "DiceLoss": DiceLoss,
    "IoULoss": IoULoss,
}


@pytest.mark.parametrize("loss_name", LOSS_CLASSES)
@pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 3, 4, 5),
        (4, 2, 8, 8),
        (2, 4, 3, 5, 5),
    ],
)
def test_loss_matches_reference(
    loss_name,
    average,
    shape,
):
    rng = np.random.default_rng(42)

    y_true = rng.random(shape).astype(np.float32)
    y_pred = rng.random(shape).astype(np.float32)

    # Reference implementation
    cm = _compute_confusion_matrix(
        y_true,
        y_pred,
        average,
    )

    args = {
        "alpha": 0.3,
        "beta": 0.7,
        "gamma": 0.75,
        "epsilon": 1e-5,
    }

    value = _compute_loss(
        loss_name,
        cm,
        args,
    )

    class_weights = None

    expected = _compute_average(
        value,
        cm,
        average,
        class_weights,
    )

    expected = np.mean(expected)

    # PyTorch implementation
    loss_fn = LOSS_CLASSES[loss_name](
        average=average,
        epsilon=args["epsilon"],
    )

    y_true_tensor = torch.from_numpy(y_true)
    y_pred_tensor = torch.from_numpy(y_pred)

    actual = loss_fn(
        y_true_tensor,
        y_pred_tensor,
    )

    np.testing.assert_allclose(
        actual.detach().numpy(),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def _compute_confusion_matrix(y_true, y_pred, average):

    if average == 'micro':
        axis = tuple(range(1, y_true.ndim))
    else:
        axis = tuple(range(2, y_true.ndim))

    tp = np.sum(y_true * y_pred, axis=axis)
    tn = np.sum((1 - y_true) * (1 - y_pred), axis=axis)
    fp = np.sum((1 - y_true) * y_pred, axis=axis)
    fn = np.sum(y_true * (1 - y_pred), axis=axis)

    return tp, tn, fp, fn


def _compute_loss(name, cm, args):  # pylint: disable=missing-function-docstring
    tp, _, fp, fn = cm
    eps = args['epsilon']
    if name == 'FocalTverskyLoss':
        return (1.0 - ((tp + eps) / \
        (tp + args['alpha'] * fp + args['beta'] * fn + eps))) ** args['gamma']
    if name == 'TverskyLoss':
        return 1.0 - ((tp + eps) / \
        (tp + args['alpha'] * fp + args['beta'] * fn + eps))
    if name == 'DiceLoss':
        return 1.0 - ((tp + eps) / (tp + 0.5 * fp + 0.5 * fn + eps))
    if name == 'IoULoss':
        return 1.0 - ((tp + eps) / (tp + 1.0 * fp + 1.0 * fn + eps))
    raise ValueError(f"Invalid loss name: {name}")


def _compute_average(value, cm, average, class_weights):  # pylint: disable=missing-function-docstring
    tp, _, _, fn = cm
    if average == 'micro':
        return value
    if average == 'macro':
        return np.mean(value, axis=1)
    if average == 'weighted':
        if class_weights is None:
            class_weights = (tp + fn) / np.sum(tp + fn, axis=1, keepdims=True)
        return np.sum(value * class_weights, axis=1)
    
    raise ValueError(f"Invalid average mode: {average}")




    