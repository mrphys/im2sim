from dataclasses import dataclass

import torch
from torch import Tensor, nn

from im2sim.src.utils import api_util


@dataclass
class ConfusionMatrix:
    true_positives: Tensor
    true_negatives: Tensor
    false_positives: Tensor
    false_negatives: Tensor


class ConfusionLoss(nn.Module):
    """Base class for losses derived from a soft confusion matrix.

    Args:
        average: How to average over classes. One of `"micro"`,
            `"macro"`, or `"weighted"`.
        class_weights: Optional weights used for weighted averaging.
        reduction: Reduction applied to the batch. One of `"none"`,
            `"mean"`, or `"sum"`.

    Notes:
        Inputs are expected to follow the standard PyTorch channel-first
        convention:

        `[batch, channels, *spatial]`

        The confusion matrix is calculated independently for each batch
        element and channel, with all spatial dimensions reduced.
    """

    def __init__(
        self,
        average: str = "macro",
        class_weights: Tensor | None = None,
        reduction: str = "mean",
    ):
        """"""
        super().__init__()

        if average not in {"micro", "macro", "weighted"}:
            raise ValueError(
                f"Invalid average: {average!r}. Expected 'micro', 'macro', or 'weighted'."
            )

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"Invalid reduction: {reduction!r}. Expected 'none', 'mean', or 'sum'."
            )

        self.average = average
        self.reduction = reduction

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.as_tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(
        self,
        y_true: Tensor,
        y_pred: Tensor,
    ) -> Tensor:
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true and y_pred must have the same shape, "
                f"got {y_true.shape} and {y_pred.shape}."
            )

        y_true = y_true.to(dtype=y_pred.dtype)

        # PyTorch convention:
        #
        #     [B, C, *spatial]
        #
        # Therefore all dimensions after channel are spatial dimensions.
        spatial_dims = tuple(range(2, y_pred.ndim))

        confusion_matrix = ConfusionMatrix(
            true_positives=(y_pred * y_true).sum(dim=spatial_dims),
            true_negatives=((1 - y_pred) * (1 - y_true)).sum(dim=spatial_dims),
            false_positives=(y_pred * (1 - y_true)).sum(dim=spatial_dims),
            false_negatives=((1 - y_pred) * y_true).sum(dim=spatial_dims),
        )

        # Micro averaging combines all classes before calculating
        # the loss.
        if self.average == "micro":
            confusion_matrix = ConfusionMatrix(
                true_positives=confusion_matrix.true_positives.sum(
                    dim=1,
                    keepdim=True,
                ),
                true_negatives=confusion_matrix.true_negatives.sum(
                    dim=1,
                    keepdim=True,
                ),
                false_positives=confusion_matrix.false_positives.sum(
                    dim=1,
                    keepdim=True,
                ),
                false_negatives=confusion_matrix.false_negatives.sum(
                    dim=1,
                    keepdim=True,
                ),
            )

        loss = self._forward(confusion_matrix)
        loss = self._average(loss, confusion_matrix)

        return self._reduce(loss)

    def _forward(self, confusion_matrix: ConfusionMatrix) -> Tensor:
        """Compute the per-class loss.

        Returns:
            Tensor with shape `[B, C]`.
        """
        raise NotImplementedError

    def _average(
        self,
        class_values: Tensor,
        confusion_matrix: ConfusionMatrix,
    ) -> Tensor:
        if self.average == "micro":
            # Micro averaging has already combined the classes.
            return class_values[:, 0]

        if self.average == "macro":
            return class_values.mean(dim=1)

        if self.average == "weighted":
            if self.class_weights is not None:
                class_weights = self.class_weights.to(
                    device=class_values.device,
                    dtype=class_values.dtype,
                )

                # Allow either:
                #   [C]
                # or
                #   [B, C]
                if class_weights.ndim == 1:
                    class_weights = class_weights.unsqueeze(0)

            else:
                true_instances = confusion_matrix.true_positives + confusion_matrix.false_negatives

                class_weights = torch.nan_to_num(
                    true_instances / true_instances.sum(dim=1, keepdim=True),
                    nan=0.0,
                )

            return (class_values * class_weights).sum(dim=1)

        raise ValueError(f"Unknown average mode: {self.average}")

    def _reduce(self, loss: Tensor) -> Tensor:
        if self.reduction == "none":
            return loss

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        raise ValueError(f"Unknown reduction: {self.reduction}")


@api_util.export("losses.FocalTverskyLoss")
class FocalTverskyLoss(ConfusionLoss):
    r"""Focal Tversky loss.

    The focal Tversky loss [1,2] is:

    .. math::
        L =
        \left(
            1 -
            \frac{\mathrm{TP} + \epsilon}
            {\mathrm{TP} + \alpha \mathrm{FP} + \beta \mathrm{FN} + \epsilon}
        \right)^\gamma

    Args:
        alpha (float):
            Weight given to false positives. Default is `0.3`.
            Increasing alpha will penalize false positives more heavily, which can be useful in cases where false positives are more detrimental than false negatives.

        beta (float):
            Weight given to false negatives. Default is `0.7`.
            Increasing beta will penalize false negatives more heavily, which can be useful in cases where false negatives are more detrimental than false positives.

        gamma (float):
            Focusing parameter. Default is `0.75`.
            Increasing gamma will focus the loss more on hard-to-classify examples, which can be useful in cases of class imbalance.

        epsilon (float):
            Smoothing factor. Default is `1e-5`.

        average (str):
            Class averaging strategy. Default is `"macro"`. One of `"micro"`, `"macro"`, or `"weighted"`.
            `"micro"`: Calculate metrics globally by counting the total true positives, false negatives and false positives.
            `"macro"`: Calculate metrics for each class, and find their unweighted mean. This does not take class imbalance into account.
            `"weighted"`: Calculate metrics for each class, and find their average weighted by support (the number of true instances for each class).

        class_weights (torch.Tensor):
            Optional class weights. Default is None. If provided, should be a 1D tensor of shape `[C]` or a 2D tensor of shape `[B, C]`.

        reduction (str):
            Batch reduction strategy. Default is "mean". One of `"none"`, `"mean"`, or `"sum"`.


    Note:

        Inputs `y_true` and `y_pred` are expected to have shape `[Batch, Channels, *Spatial]`,
        with channel `i` containing labels/predictions for class `i`. `y_true[:, i, ...]`
        is 1 if the element represented by `y_true[...]` is a member of class `i` and
        0 otherwise. `y_pred[:,i,...]` is the predicted probability, in the range
        `[0.0, 1.0]`, that the element represented by `y_pred[...]` is a member of
        class `i`.

        The loss is computed for each batch element `y_true[i, ...]` and `y_pred[i, ...]`,
        and then reduced over this dimension as specified by argument `reduction`.

        This loss works for binary, multiclass and multilabel classification and/or
        segmentation. In multiclass/multilabel problems, the different classes are
        combined according to the `average` and `class_weights` arguments.

    References:

        .. [1] Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017, September).
            Tversky loss function for image segmentation using 3D fully convolutional
            deep networks. In International workshop on machine learning in medical
            imaging (pp. 379-387). Springer, Cham.

        .. [2] Abraham, N., & Khan, N. M. (2019, April). A novel focal tversky loss
            function with improved attention u-net for lesion segmentation. In 2019
            IEEE 16th international symposium on biomedical imaging (ISBI 2019)
            (pp. 683-687). IEEE.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        epsilon: float = 1e-5,
        average: str = "macro",
        class_weights: Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__(
            average=average,
            class_weights=class_weights,
            reduction=reduction,
        )

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def _forward(
        self,
        confusion_matrix: ConfusionMatrix,
    ) -> Tensor:
        """ " """
        true_positives = confusion_matrix.true_positives
        false_positives = confusion_matrix.false_positives
        false_negatives = confusion_matrix.false_negatives

        numerator = true_positives

        denominator = true_positives + self.alpha * false_positives + self.beta * false_negatives

        tversky_index = (numerator + self.epsilon) / (denominator + self.epsilon)

        return (1.0 - tversky_index).pow(self.gamma)


@api_util.export("losses.TverskyLoss")
class TverskyLoss(FocalTverskyLoss):
    r"""Tversky loss.

    The Tversky loss is:

    .. math::
        L =
        \left(
            1 -
            \frac{\mathrm{TP} + \epsilon}
            {\mathrm{TP} + \alpha \mathrm{FP} + \beta \mathrm{FN} + \epsilon}
        \right)

    Args:
        alpha (float):
            Weight given to false positives. Default is `0.3`.
            Increasing alpha will penalize false positives more heavily, which can be useful in cases where false positives are more detrimental than false negatives.

        beta (float):
            Weight given to false negatives. Default is `0.7`.
            Increasing beta will penalize false negatives more heavily, which can be useful in cases where false negatives are more detrimental than false positives.

        epsilon (float):
            Smoothing factor. Default is `1e-5`.

        average (str):
            Class averaging strategy. Default is `"macro"`. One of `"micro"`, `"macro"`, or `"weighted"`.
            `"micro"`: Calculate metrics globally by counting the total true positives, false negatives and false positives.
            `"macro"`: Calculate metrics for each class, and find their unweighted mean. This does not take class imbalance into account.
            `"weighted"`: Calculate metrics for each class, and find their average weighted by support (the number of true instances for each class).

        class_weights (torch.Tensor):
            Optional class weights. Default is None. If provided, should be a 1D tensor of shape `[C]` or a 2D tensor of shape `[B, C]`.

        reduction (str):
            Batch reduction strategy. Default is "mean". One of `"none"`, `"mean"`, or `"sum"`.



    Note:
        Inputs `y_true` and `y_pred` are expected to have shape `[Batch, Channels, *Spatial]`,
        with channel `i` containing labels/predictions for class `i`. `y_true[:, i, ...]`
        is 1 if the element represented by `y_true[...]` is a member of class `i` and
        0 otherwise. `y_pred[:,i,...]` is the predicted probability, in the range
        `[0.0, 1.0]`, that the element represented by `y_pred[...]` is a member of
        class `i`.

        The loss is computed for each batch element `y_true[i, ...]` and `y_pred[i, ...]`,
        and then reduced over this dimension as specified by argument `reduction`.

        This loss works for binary, multiclass and multilabel classification and/or
        segmentation. In multiclass/multilabel problems, the different classes are
        combined according to the `average` and `class_weights` arguments.

    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        epsilon: float = 1e-5,
        average: str = "macro",
        class_weights: Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__(
            alpha=alpha,
            beta=beta,
            gamma=1.0,
            epsilon=epsilon,
            average=average,
            class_weights=class_weights,
            reduction=reduction,
        )


@api_util.export("losses.DiceLoss")
class DiceLoss(TverskyLoss):
    r"""Dice loss, also known as F1 loss.

    The Dice loss is:

    .. math::
        L =
        \left(
            1 -
            \frac{\mathrm{TP} + \epsilon}
            {\mathrm{TP} + 0.5 * \mathrm{FP} + 0.5 * \mathrm{FN} + \epsilon}
        \right)

    Args:

        epsilon (float):
            Smoothing factor. Default is `1e-5`.

        average (str):
            Class averaging strategy. Default is `"macro"`. One of `"micro"`, `"macro"`, or `"weighted"`.
            `"micro"`: Calculate metrics globally by counting the total true positives, false negatives and false positives.
            `"macro"`: Calculate metrics for each class, and find their unweighted mean. This does not take class imbalance into account.
            `"weighted"`: Calculate metrics for each class, and find their average weighted by support (the number of true instances for each class).

        class_weights (torch.Tensor):
            Optional class weights. Default is None. If provided, should be a 1D tensor of shape `[C]` or a 2D tensor of shape `[B, C]`.

        reduction (str):
            Batch reduction strategy. Default is "mean". One of `"none"`, `"mean"`, or `"sum"`.

    Note:
        Inputs `y_true` and `y_pred` are expected to have shape `[Batch, Channels, *Spatial]`,
        with channel `i` containing labels/predictions for class `i`. `y_true[:, i, ...]`
        is 1 if the element represented by `y_true[...]` is a member of class `i` and
        0 otherwise. `y_pred[:,i,...]` is the predicted probability, in the range
        `[0.0, 1.0]`, that the element represented by `y_pred[...]` is a member of
        class `i`.

        The loss is computed for each batch element `y_true[i, ...]` and `y_pred[i, ...]`,
        and then reduced over this dimension as specified by argument `reduction`.

        This loss works for binary, multiclass and multilabel classification and/or
        segmentation. In multiclass/multilabel problems, the different classes are
        combined according to the `average` and `class_weights` arguments.
    """

    def __init__(
        self,
        epsilon: float = 1e-5,
        average: str = "macro",
        class_weights: Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__(
            alpha=0.5,
            beta=0.5,
            epsilon=epsilon,
            average=average,
            class_weights=class_weights,
            reduction=reduction,
        )


@api_util.export("losses.IoULoss")
class IoULoss(TverskyLoss):
    r"""Intersection over Union (IoU) or Jaccard loss.

    The IoU loss is:

    .. math::
        L =
        \left(
            1 -
            \frac{\mathrm{TP} + \epsilon}
            {\mathrm{TP} + \mathrm{FP} + \mathrm{FN} + \epsilon}
        \right)

    Args:
        epsilon (float):
            Smoothing factor. Default is `1e-5`.

        average (str):
            Class averaging strategy. Default is `"macro"`. One of `"micro"`, `"macro"`, or `"weighted"`.
            `"micro"`: Calculate metrics globally by counting the total true positives, false negatives and false positives.
            `"macro"`: Calculate metrics for each class, and find their unweighted mean. This does not take class imbalance into account.
            `"weighted"`: Calculate metrics for each class, and find their average weighted by support (the number of true instances for each class).

        class_weights (torch.Tensor):
            Optional class weights. Default is None. If provided, should be a 1D tensor of shape `[C]` or a 2D tensor of shape `[B, C]`.

        reduction (str):
            Batch reduction strategy. Default is "mean". One of `"none"`, `"mean"`, or `"sum"`.

    Note:
        Inputs `y_true` and `y_pred` are expected to have shape `[Batch, Channels, *Spatial]`,
        with channel `i` containing labels/predictions for class `i`. `y_true[:, i, ...]`
        is 1 if the element represented by `y_true[...]` is a member of class `i` and
        0 otherwise. `y_pred[:,i,...]` is the predicted probability, in the range
        `[0.0, 1.0]`, that the element represented by `y_pred[...]` is a member of
        class `i`.

        The loss is computed for each batch element `y_true[i, ...]` and `y_pred[i, ...]`,
        and then reduced over this dimension as specified by argument `reduction`.

        This loss works for binary, multiclass and multilabel classification and/or
        segmentation. In multiclass/multilabel problems, the different classes are
        combined according to the `average` and `class_weights` arguments.
    """

    def __init__(
        self,
        epsilon: float = 1e-5,
        average: str = "macro",
        class_weights: Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__(
            alpha=1.0,
            beta=1.0,
            epsilon=epsilon,
            average=average,
            class_weights=class_weights,
            reduction=reduction,
        )
