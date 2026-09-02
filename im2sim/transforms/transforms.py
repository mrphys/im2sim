from im2sim.data.core import Operation, Transform
from im2sim.transforms.ops import (
    FitNormOp,
    FitRangeNormOp,
    FitZScoreOp,
    NormOp,
    PowerScaleOp,
    RangeNormOp,
    ZScoreOp,
)


def transform_from_fn(fn,
                    keys: list[str], 
                    attr: str = None, 
                    channels: list[int] = None, 
                    per_channel: bool = False, 
                    channel_dim: int = -1, 
                    name: str = None
)-> Transform:

    class FnOp(Operation):
        def forward(self, x):
            return fn(x)

    return Transform(
        op=FnOp(),
        keys=keys,
        attr=attr,
        channels=channels,
        per_channel=per_channel,
        channel_dim=channel_dim,
        name=name,
    )


# ------------------------------------------------------------------------------------
# SIMPLE TRANSFORM FACTORIES
# ------------------------------------------------------------------------------------


def Norm(keys: list[str], 
        attr: str = None, 
        channels: list[int] = None, 
        per_channel: bool = False, 
        channel_dim: int = -1, 
        name: str = None) -> Transform:
    """
    Factory function to create a normalization transform using `im2sim.data.ops.NormOp`.

    Args:
        keys (list): List of keys to which the normalization will be applied.
        attr (str, optional): Object attribute to transform. For example in a `pyg.data.Data` object this could be `'x'`. Defaults to None.
        channels (list, optional): List of channels to which the normalization will be applied. Defaults to None.
        per_channel (bool, optional): Whether to apply normalization per channel. Defaults to False.
        channel_dim (int, optional): Dimension of the channel axis. Defaults to -1.
        name (str, optional): Name of the transform. Defaults to None.

    Returns:
        Transform: An instance of `im2sim.data.Transform` configured with the normalization operation.
    """
    return Transform(
        op=NormOp(),
        keys=keys,
        attr=attr,
        channels=channels,
        per_channel=per_channel,
        channel_dim=channel_dim,
        name=name,
    )


def RangeNorm(llim: float,
            hlim: float,
            keys: list[str], 
            attr: str = None, 
            channels: list[int] = None, 
            per_channel: bool = False, 
            channel_dim: int = -1, 
            name: str = None
) -> Transform:
    """
    Factory function to create a range normalization transform using `im2sim.data.ops.RangeNormOp`.

    Args:
        llim (float): Lower limit of the target range for normalization.
        hlim (float): Upper limit of the target range for normalization.
        keys (list): List of keys to which the normalization will be applied.
        attr (str, optional): Object attribute to transform. For example in a `pyg.data.Data` object this could be `'x'`. Defaults to None.
        channels (list, optional): List of channels to which the normalization will be applied. Defaults to None.
        per_channel (bool, optional): Whether to apply normalization per channel. Defaults to False.
        channel_dim (int, optional): Dimension of the channel axis. Defaults to -1.
        name (str, optional): Name of the transform. Defaults to None.

    Returns:
        Transform: An instance of `im2sim.data.Transform` configured with the RangeNorm operation.
    """
    return Transform(
        op=RangeNormOp(a=llim, b=hlim),
        keys=keys,
        attr=attr,
        channels=channels,
        per_channel=per_channel,
        channel_dim=channel_dim,
        name=name,
    )


def ZScore(keys: list[str], 
            attr: str = None, 
            channels: list[int] = None, 
            per_channel: bool = False, 
            channel_dim: int = -1, 
            name: str = None) -> Transform:
    """
    Factory function to create a z-score normalization transform using `im2sim.data.ops.ZScoreOp`.

    Args:
        keys (list): List of keys to which the z-score normalization will be applied.
        attr (str, optional): Object attribute to transform. For example in a `pyg.data.Data` object this could be `'x'`. Defaults to None.
        channels (list, optional): List of channels to which the z-score normalization will be applied. Defaults to None.
        per_channel (bool, optional): Whether to apply z-score normalization per channel. Defaults to False.
        channel_dim (int, optional): Dimension of the channel axis. Defaults to -1.
        name (str, optional): Name of the transform. Defaults to None.

    Returns:
        Transform: An instance of `im2sim.data.Transform` configured with the ZScore operation.
    """
    return Transform(
        op=ZScoreOp(),
        keys=keys,
        attr=attr,
        channels=channels,
        per_channel=per_channel,
        channel_dim=channel_dim,
        name=name,
    )


# ------------------------------------------------------------------------------------
# INVERTIBLE TRANSFORM FACTORIES
# ------------------------------------------------------------------------------------


def PowerScaling(
    exp,
    preserve_sign,
    keys: list[str], 
    attr: str = None, 
    channels: list[int] = None, 
    per_channel: bool = False, 
    channel_dim: int = -1, 
    name: str = None) -> Transform:
    """
    Factory function to create a power scaling transform using `im2sim.data.ops.PowerScaleOp`.

    Args:
        exp (float): The exponent to which the input tensor will be raised.
        preserve_sign (bool): If True, preserves the sign of the input tensor.
        keys (list): List of keys to which the power scaling will be applied.
        attr (str, optional): Object attribute to transform. For example in a `pyg.data.Data` object this could be `'x'`. Defaults to None.
        channels (list, optional): List of channels to which the power scaling will be applied. Defaults to None.
        per_channel (bool, optional): Whether to apply power scaling per channel. Defaults to False.
        channel_dim (int, optional): Dimension of the channel axis. Defaults to -1.
        name (str, optional): Name of the transform. Defaults to None.

    Returns:
        Transform: An instance of `im2sim.data.Transform` configured with the PowerScale operation.
    """
    return Transform(
        op=PowerScaleOp(exp=exp, preserve_sign=preserve_sign),
        keys=keys,
        attr=attr,
        channels=channels,
        per_channel=per_channel,
        channel_dim=channel_dim,
        name=name,
    )


# ------------------------------------------------------------------------------------
# FITTABLE TRANSFORM FACTORIES
# ------------------------------------------------------------------------------------


def FitNorm(keys: list[str], 
            attr: str = None, 
            channels: list[int] = None, 
            per_channel: bool = False, 
            channel_dim: int = -1, 
            name: str = None) -> Transform:
    """
    Factory function to create a fit normalization transform using `im2sim.data.ops.FitNormOp`.

    Args:
        keys (list): List of keys to which the fit normalization will be applied.
        attr (str, optional): Object attribute to transform. For example in a `pyg.data.Data` object this could be `'x'`. Defaults to None.
        channels (list, optional): List of channels to which the fit normalization will be applied. Defaults to None.
        per_channel (bool, optional): Whether to apply fit normalization per channel. Defaults to False.
        channel_dim (int, optional): Dimension of the channel axis. Defaults to -1.
        name (str, optional): Name of the transform. Defaults to None.

    Returns:
        Transform: An instance of `im2sim.data.Transform` configured with the fittable normalization operation.
    """
    return Transform(
        op=FitNormOp(),
        keys=keys,
        attr=attr,
        channels=channels,
        per_channel=per_channel,
        channel_dim=channel_dim,
        name=name,
    )


def FitRangeNorm(
    llim,
    hlim,
    keys: list[str], 
    attr: str = None, 
    channels: list[int] = None, 
    per_channel: bool = False, 
    channel_dim: int = -1, 
    name: str = None) -> Transform:
    """
    Factory function to create a fit range normalization transform using `im2sim.data.ops.FitRangeNormOp`.
    
    Args: 
        llim (float): Lower limit of the target range for normalization.
        hlim (float): Upper limit of the target range for normalization.
        keys (list): List of keys to which the fit range normalization will be applied.
        attr (str, optional): Object attribute to transform. For example in a `pyg.data.Data` object this could be `'x'`. Defaults to None.
        channels (list, optional): List of channels to which the fit range normalization will be applied. Defaults to None.
        per_channel (bool, optional): Whether to apply fit range normalization per channel. Defaults to False.
        channel_dim (int, optional): Dimension of the channel axis. Defaults to -1.
        name (str, optional): Name of the transform. Defaults to None.

    Returns:
        Transform: An instance of `im2sim.data.Transform` configured with the fittable range normalization operation.
    """
    return Transform(
        op=FitRangeNormOp(a=llim, b=hlim),
        keys=keys,
        attr=attr,
        channels=channels,
        per_channel=per_channel,
        channel_dim=channel_dim,
        name=name,
    )


def FitZScore(keys: list[str], 
            attr: str = None, 
            channels: list[int] = None, 
            per_channel: bool = False, 
            channel_dim: int = -1, 
            name: str = None) -> Transform:
    """
    Factory function to create a fit z-score normalization transform using `im2sim.data.ops.FitZScoreOp`.

    Args:
        keys (list): List of keys to which the fit z-score normalization will be applied.
        attr (str, optional): Object attribute to transform. For example in a `pyg.data.Data` object this could be `'x'`. Defaults to None.
        channels (list, optional): List of channels to which the fit z-score normalization will be applied. Defaults to None.
        per_channel (bool, optional): Whether to apply fit z-score normalization per channel. Defaults to False.
        channel_dim (int, optional): Dimension of the channel axis. Defaults to -1.
        name (str, optional): Name of the transform. Defaults to None.

    Returns:
        Transform: An instance of `im2sim.data.Transform` configured with the fittable z-score normalization operation.
    """
    return Transform(
        op=FitZScoreOp(),
        keys=keys,
        attr=attr,
        channels=channels,
        per_channel=per_channel,
        channel_dim=channel_dim,
        name=name,
    )
