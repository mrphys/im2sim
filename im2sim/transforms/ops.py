import torch

from im2sim.data.core import FittableOperation, InvertibleOperation, Operation, register_op


# ------------------------------------------------------------------------------------
# OP FUNCTION LIBRARY
# ------------------------------------------------------------------------------------
eps = 1e-8


def normtorange(x, max=None, min=None, a=0, b=1):
    """
    Normalizes the input tensor `x` to a specified range [a, b].

    Args:
        x (torch.Tensor) : Input tensor to be normalized.
        max (float, optional): Maximum value for normalization. If None, uses the maximum of `x`.
        min (float, optional): Minimum value for normalization. If None, uses the minimum of `x`.
        a (float, optional): Lower bound of the target range. Default is 0.
        b (float, optional): Upper bound of the target range. Default is 1.
    """
    if min is None:
        min = x.min()
    if max is None:
        max = x.max()
    return a + ((x - min) * (b - a)) / (max - min + eps)


def inv_normtorange(x, max=None, min=None, a=0, b=1):
    """
    Inverse normalization of the input tensor `x` from a specified range [a, b] back to its original range.

    Args:
        x (torch.Tensor) : Input tensor to be inverse normalized.
        max (float, optional): Maximum value for inverse normalization. If None, uses the maximum of `x`.
        min (float, optional): Minimum value for inverse normalization. If None, uses the minimum of `x`.
        a (float, optional): Lower bound of the original range. Default is 0.
        b (float, optional): Upper bound of the original range. Default is 1.
    """
    if min is None:
        min = x.min()
    if max is None:
        max = x.max()
    return min + ((x - a) * (max - min)) / (b - a)


def normalise(x, max=None, min=None):
    """
    Normalizes the input tensor `x` to the range [0, 1].

    Args:
        x (torch.Tensor) : Input tensor to be normalized.
        max (float, optional): Maximum value for normalization. If None, uses the maximum of `x`.
        min (float, optional): Minimum value for normalization. If None, uses the minimum of `x`.
    """
    return normtorange(x, max, min)


def inv_normalise(x, max, min):
    """
    Inverse normalization of the input tensor `x` from the range [0, 1] back to its original range.

    Args:
        x (torch.Tensor) : Input tensor to be inverse normalized.
        max (float): Maximum value for inverse normalization.
        min (float): Minimum value for inverse normalization.
    """
    return inv_normtorange(x, max, min)


def standardise(x, mean=None, std=None):
    """
    Standardizes the input tensor `x` to have zero mean and unit variance.

    Args:
        x (torch.Tensor) : Input tensor to be standardized.
        mean (float, optional): Mean value for standardization. If None, uses the mean of `x`.
        std (float, optional): Standard deviation for standardization. If None, uses the standard deviation of `x`.
    """
    if mean is None:
        mean = x.mean()
    if std is None:
        std = x.std()
    return (x - mean) / (std + eps)


def inv_standardise(x, mean=None, std=None):
    """
    Inverse standardization of the input tensor `x` from zero mean and unit variance back to its original distribution.

    Args:
        x (torch.Tensor) : Input tensor to be inverse standardized.
        mean (float, optional): Mean value for inverse standardization. If None, uses the mean of `x`.
        std (float, optional): Standard deviation for inverse standardization. If None, uses the standard deviation of `x`.
    """
    return x * std + mean


# ------------------------------------------------------------------------------------
# SIMPLE OPERATIONS LIBRARY
# ------------------------------------------------------------------------------------


@register_op
class NormOp(Operation):
    """
    Normalizes the input tensor to the range [0, 1].
    """
    def forward(self, x):
        return normalise(x)


@register_op
class RangeNormOp(Operation):
    """
    Normalizes the input tensor to a specified range [a, b].

    Args:
        a (float): Lower bound of the target range.
        b (float): Upper bound of the target range.
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self, x):
        return normtorange(x, a=self.a, b=self.b)


@register_op
class ZScoreOp(Operation):
    """
    Standardizes the input tensor to have zero mean and unit variance.
    """
    def forward(self, x):
        return standardise(x)


# ------------------------------------------------------------------------------------
# INVERTIBLE OPERATIONS LIBRARY
# ------------------------------------------------------------------------------------


@register_op
class PowerScaleOp(InvertibleOperation):
    """
    Scales the input tensor by raising it to a specified power.

    Args:
        exp (float): The exponent to which the input tensor will be raised.
        preserve_sign (bool, optional): If True, preserves the sign of the input tensor. Default is True.
        eps (float, optional): A small value added to the absolute value of the input tensor to avoid numerical issues. Default is 1e-8.
    """
    def __init__(self, exp, preserve_sign=True, eps=1e-8):
        if exp == 0:
            raise ValueError("exp must not be 0")
        self.exp = exp
        self.preserve_sign = preserve_sign
        self.eps = eps

    def forward(self, x):
        if self.preserve_sign:
            return torch.sign(x) * torch.pow(torch.abs(x) + self.eps, self.exp)
        else:
            return torch.pow(x, self.exp)

    def inverse(self, x):
        if self.preserve_sign:
            return torch.sign(x) * torch.pow(torch.abs(x) + self.eps, 1 / self.exp)
        else:
            return torch.pow(x, 1 / self.exp)


# ------------------------------------------------------------------------------------
# FITTABLE OPERATIONS LIBRARY
# ------------------------------------------------------------------------------------


@register_op
class FitNormOp(FittableOperation):
    """
    Normalizes the input tensor to the range [0, 1] based on the fitted maximum and minimum values.

    Args:
        max (float, optional): Maximum value for normalization. If None, will be fitted from the data.
        min (float, optional): Minimum value for normalization. If None, will be fitted from the data.
    """
    def __init__(self):
        self.max = torch.Tensor([-torch.inf])
        self.min = torch.Tensor([torch.inf])

    def forward(self, x):
        return normalise(x, self.max, self.min)

    def inverse(self, x):
        return inv_normalise(x, self.max, self.min)

    def fit_step(self, x):
        self.max = torch.maximum(self.max, x.max())
        self.min = torch.minimum(self.min, x.min())

    def complete_fit(self):
        pass


@register_op
class FitRangeNormOp(FittableOperation):
    """
    Normalizes the input tensor to a specified range [a, b] based on the fitted maximum and minimum values.

    Args:
        a (float): Lower bound of the target range.
        b (float): Upper bound of the target range.
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.max = torch.Tensor([-torch.inf])
        self.min = torch.Tensor([torch.inf])

    def forward(self, x):
        return normtorange(x, self.max, self.min, self.a, self.b)

    def inverse(self, x):
        return inv_normtorange(x, self.max, self.min, self.a, self.b)

    def fit_step(self, x):
        self.max = torch.maximum(self.max, x.max())
        self.min = torch.minimum(self.min, x.min())

    def complete_fit(self):
        pass


@register_op
class FitZScoreOp(FittableOperation):
    """
    Standardizes the input tensor to have zero mean and unit variance based on the fitted mean and standard deviation.
    """
    def __init__(self):
        self.sum = 0
        self.sq_sum = 0
        self.numel = 0

    def forward(self, data):
        return standardise(data, self.mean, self.std)

    def inverse(self, data):
        return inv_standardise(data, self.mean, self.std)

    def fit_step(self, data):
        self.sum += data.sum()
        self.sq_sum += (data**2).sum()
        self.numel += data.numel()

    def complete_fit(self):
        self.mean = self.sum / self.numel
        self.std = torch.sqrt(self.sq_sum / self.numel - self.mean**2)
