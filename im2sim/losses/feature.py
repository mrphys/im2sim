
import logging

import torch

logger = logging.getLogger(__name__)

def mse(gr1, gr2):
    return torch.mean((gr1.x[...,3:] - gr2.x[...,3:])**2)

    
