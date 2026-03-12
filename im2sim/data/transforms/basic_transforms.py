from abc import ABC, abstractmethod
import torch
import copy
from torch_geometric.data import Data
from .transform_fn import *


class BaseTransform(ABC):

    def __init__(self, key, attr=None, channels="all", channel_dim=-1, name=None):
        self.key = key
        self.attr = attr
        self.channel_dim = channel_dim

        if channels == "all":
            self.call_fn = self.forward

        elif channels == "each":
            self.call_fn = self._per_channel

        elif isinstance(channels, list):
            self.channels = channels
            self.call_fn = self._select_channels

        elif isinstance(channels, int):
            self.channels = [channels]
            self.call_fn = self._select_channels

        else:
            raise TypeError(
                "channels must be 'all', 'each', int or List[int]"
            )
        
        self.name = f'{name}_{key}' if name is not None else f"{self.__class__.__name__}(key={self.key})"

    @abstractmethod
    def forward(self, x):
        pass

    def _per_channel(self, x):
        x = torch.moveaxis(x, self.channel_dim, 0)
        out = torch.stack([self.forward(c) for c in x])
        return torch.moveaxis(out, 0, self.channel_dim)

    def _select_channels(self, x):
        x = torch.moveaxis(x, self.channel_dim, 0)
        for i in self.channels:
            x[i] = self.forward(x[i])
        return torch.moveaxis(x, 0, self.channel_dim)

    def __call__(self, data):
        data = copy.copy(data)
        x = data[self.key]

        if isinstance(x, Data):
            attr = self.attr if self.attr is not None else "x"
            x = copy.copy(x)

            if attr == 'PyG0000':
                x = self.call_fn(x)

            else:
                tensor = getattr(x, attr)
                tensor = self.call_fn(tensor)
                setattr(x, attr, tensor)

        elif isinstance(x, torch.Tensor):
            x = self.call_fn(x)

        else:
            raise TypeError(
                f"{self.key} must be Tensor or PyG Data"
            )

        data[self.key] = x
        return data

    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name
    
class Normalise(BaseTransform):
    def forward(self, data):
        return normalise(data)

class Standardise(BaseTransform):
    def forward(self, data):
        return standardise(data)


    
    
def transform_from_fn(fn, key, attr=None, channels="all", channel_dim=-1, name=None):

    class FnTransform(BaseTransform):

        def forward(self, x):
            return fn(x)

    return FnTransform(key=key, attr=attr, channels=channels, channel_dim=channel_dim, name=name)

def pyg_transform(fn, key, channels="all", channel_dim=-1, name=None):
    return transform_from_fn(fn=fn, key=key, attr='PyG0000', channels=channels, channel_dim=channel_dim, name=name)

