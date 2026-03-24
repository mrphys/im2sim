from .ops import *
from .core import *



def transform_from_fn(fn, keys,attr=None,channels=None,per_channel=False,channel_dim=-1,name=None):

    class FnOp(Operation):

        def forward(self, x):
            return fn(x)

    return Transform(op=FnOp(), keys=keys,attr=attr,channels=channels,per_channel=per_channel,channel_dim=channel_dim,name=name)


# ------------------------------------------------------------------------------------
# SIMPLE TRANSFORM FACTORIES
# ------------------------------------------------------------------------------------

    
def Norm(keys,attr=None,channels=None,per_channel=False,channel_dim=-1,name=None):
    return Transform(op=NormOp(), keys=keys,attr=attr,channels=channels,per_channel=per_channel,channel_dim=channel_dim,name=name)

def RangeNorm(llim, hlim, keys,attr=None,channels=None,per_channel=False,channel_dim=-1,name=None):
    return Transform(op=RangeNormOp(a=llim, b=hlim), keys=keys,attr=attr,channels=channels,per_channel=per_channel,channel_dim=channel_dim,name=name)

def ZScore(keys,attr=None,channels=None,per_channel=False,channel_dim=-1,name=None):
    return Transform(op=ZScoreOp(), keys=keys,attr=attr,channels=channels,per_channel=per_channel,channel_dim=channel_dim,name=name)
    
# ------------------------------------------------------------------------------------
# INVERTIBLE TRANSFORM FACTORIES
# ------------------------------------------------------------------------------------

def PowerScaling(exp,preserve_sign, keys,attr=None,channels=None,per_channel=False,channel_dim=-1,name=None):
    return Transform(op=PowerScaleOp(exp=exp,preserve_sign=preserve_sign), keys=keys,attr=attr,channels=channels,per_channel=per_channel,channel_dim=channel_dim,name=name)

# ------------------------------------------------------------------------------------
# FITTABLE TRANSFORM FACTORIES
# ------------------------------------------------------------------------------------


def FitNorm(keys,attr=None,channels=None,per_channel=False,channel_dim=-1,name=None):
    return Transform(op=FitNormOp(), keys=keys,attr=attr,channels=channels,per_channel=per_channel,channel_dim=channel_dim,name=name)

def FitRangeNorm(llim, hlim, keys,attr=None,channels=None,per_channel=False,channel_dim=-1,name=None):
    return Transform(op=FitRangeNormOp(a=llim, b=hlim), keys=keys,attr=attr,channels=channels,per_channel=per_channel,channel_dim=channel_dim,name=name)

def FitZScore(keys,attr=None,channels=None,per_channel=False,channel_dim=-1,name=None):
    return Transform(op=FitZScoreOp(), keys=keys,attr=attr,channels=channels,per_channel=per_channel,channel_dim=channel_dim,name=name)



