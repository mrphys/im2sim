from .core import *


# ------------------------------------------------------------------------------------
# OP FUNCTION LIBRARY
# ------------------------------------------------------------------------------------
eps = 1e-8

def normtorange(x, max=None, min=None, a=0, b=1):
    if min==None:
        min = x.min()
    if max==None:
        max = x.max()
    return a + ((x-min)*(b-a))/(max-min+eps)

def inv_normtorange(x, max=None, min=None, a=0, b=1):
    if min==None:
        min = x.min()
    if max==None:
        max = x.max()
    return min + ((x-a)*(max-min))/(b-a)


def normalise(x, max=None, min=None):
    return normtorange(x, max, min)

def inv_normalise(x, max, min):
    return inv_normtorange(x, max, min)

def standardise(x, mean=None, std=None):
    if mean==None:
        mean=x.mean()
    if std==None:
        std=x.std()
    return (x-mean)/(std+eps)

def inv_standardise(x, mean=None, std=None):
    return x*std + mean





# ------------------------------------------------------------------------------------
# SIMPLE OPERATIONS LIBRARY
# ------------------------------------------------------------------------------------

@register_op
class NormOp(Operation):

    def forward(self, x):
        return normalise(x)

@register_op
class RangeNormOp(Operation):

    def __init__(self, a, b):
        self.a = a
        self.b = b
    def forward(self, x):
        return normtorange(x, a=self.a, b=self.b)
    
@register_op
class ZScoreOp(Operation):

    def forward(self, x):
        return standardise(x)
    
# ------------------------------------------------------------------------------------
# INVERTIBLE OPERATIONS LIBRARY
# ------------------------------------------------------------------------------------

@register_op
class PowerScaleOp(InvertibleOperation):

    def __init__(self, exp, preserve_sign=True):
        self.exp = exp

    def forward(self, x):
        return torch.sign(x) * torch.pow(torch.abs(x), self.exp)

    def inverse(self, x):
        return torch.sign(x) * torch.pow(torch.abs(x), 1/self.exp)


# ------------------------------------------------------------------------------------
# FITTABLE OPERATIONS LIBRARY
# ------------------------------------------------------------------------------------


@register_op
class FitNormOp(FittableOperation):

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
        self.sq_sum += (data ** 2).sum()
        self.numel += data.numel()

    
    def complete_fit(self):
        self.mean = self.sum / self.numel
        self.std = torch.sqrt(self.sq_sum / self.numel - self.mean ** 2)

 