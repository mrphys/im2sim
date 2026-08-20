import inspect
import re
from collections.abc import Callable
from copy import copy
from typing import Any

import torch
import torch_geometric.nn as gnn


def make_registry(lib: Any, regex: Callable):

    registry = NormalizedDict(
        {name: getattr(lib, name) for name in dir(lib) if re.search(regex, name)}
    )

    def register(cls=None, *, name=None):
        def decorator(obj):
            registry[name or obj.__name__] = obj
            return obj

        return decorator(cls) if cls else decorator

    return registry, register


def normalize_key(key: str) -> str:
    return key.replace(" ", "").replace("_", "").lower()


class NormalizedDict:
    def __init__(self, data: dict):
        self._data = {}
        for key, val in data.items():
            self.__setitem__(key, val)

    def __setitem__(self, key, value):
        self._data[normalize_key(key)] = value

    def __getitem__(self, key):
        return self._data[normalize_key(key)]

    def __str__(self):
        return self._data.__str__()


activation_pattern = re.compile(
    r"(ReLU|ELU|LeakyReLU|PReLU|RReLU|GELU|SiLU|"
    r"Sigmoid|Tanh|Softmax|Softplus|SELU|CELU|"
    r"Threshold|Hardtanh|Hardswish|Mish)"
)
ACTIVATIONS, register_activation = make_registry(torch.nn, activation_pattern)


layer_pattern = r"(Conv|Pool|Norm|Upsample|PixelShuffle|Dropout)"


IMAGE_LAYERS, register_image_layer = make_registry(torch.nn, layer_pattern)
GRAPH_LAYERS, register_graph_layer = make_registry(gnn, layer_pattern)


def register_with_ranks(base_name, ranks=(1, 2, 3)):
    def decorator(cls):
        original_init = cls.__init__
        original_signature = inspect.signature(original_init)

        parameters = [
            p for name, p in original_signature.parameters.items() if name not in ("self", "rank")
        ]

        signature = original_signature.replace(parameters=parameters)

        for r in ranks:
            name = f"{base_name}{r}d"

            def __init__(self, *args, _rank=r, _init=original_init, **kwargs):
                _init(self, *args, rank=_rank, **kwargs)

            __init__.__signature__ = signature

            layer_cls = type(
                name,
                (cls,),
                {"__init__": __init__},
            )
            layer_cls.__signature__ = signature

            register_image_layer(name=name)(layer_cls)

        return cls

    return decorator


def get_image_layer(name: str, rank: int) -> torch.nn.Module:
    """
    Get a PyTorch layer by name, with optional arguments.
    """

    if name is None:
        return torch.nn.Identity

    rank_name = f"{name}{rank}d"

    try:
        return IMAGE_LAYERS[rank_name]
    except KeyError:
        pass

    try:
        return IMAGE_LAYERS[name]
    except KeyError:
        ValueError(f"Layer {name} with rank {rank} not found in PyTorch layers registry")


def get_activation(name: str | None) -> torch.nn.Module:
    if name is None:
        return torch.nn.Identity()

    activation = ACTIVATIONS[name]

    if activation is torch.nn.Softmax:
        return torch.nn.Softmax(dim=1)

    return activation()


class PyGParameterError(TypeError):
    pass


class PyGWrapperError(TypeError):
    pass


def _match_attrs_to_signature(graph, module):
    sig = inspect.signature(module.forward)

    attrs = []

    for name, _param in sig.parameters.items():
        if name == "self":
            continue

        if hasattr(graph, name):
            value = getattr(graph, name)

            # Ignore methods/functions
            if callable(value):
                continue

            attrs.append(name)

    return attrs


# This will not be useable for pooling layers as they have multiple return Tensors. If requiered we will have to add this functionality.
class PyG_Wrapper(torch.nn.Module):
    """
    A wrapper for PyG modules to make the forward method accept and return a PyG Data object instead of separated attributes
    Needs to be initialised with a pre-initialised PyG Module.
    """

    def __init__(self, pyg_module: torch.nn.Module):
        super().__init__()
        self.pyg_module = pyg_module

    def forward(self, graph):
        attrs = _match_attrs_to_signature(graph, self.pyg_module)
        out = self.pyg_module(**{attr: getattr(graph, attr) for attr in attrs})
        if not isinstance(out, torch.Tensor):
            raise PyGWrapperError(
                f"PyG layers that have multiple outputs like {self.pyg_module.__class__.__name__} are not currently supported. Consider changing PyG layer or writing a custom wrapper"
            )

        out_graph = copy(graph)
        out_graph.x = out
        return out_graph


def get_graph_layer(
    name: str, args: list[Any] = None, kwargs: dict[str, Any] = None
) -> PyG_Wrapper:

    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    module = GRAPH_LAYERS[name](*args, **kwargs)
    return PyG_Wrapper(module)


def standardize_spatial_factors(factors, rank):
    """
    Convert a sequence of spatial factors into a standardized list of tuples.
    """
    standardized = []

    for f in factors:
        if isinstance(f, int):
            standardized.append(tuple([f] * rank))
        elif isinstance(f, (tuple, list)):
            standardized.append(tuple(f))
        else:
            raise TypeError(f"Each factor must be an int, tuple, or list, got {type(f).__name__}")

    return standardized


def apply_residual_connection(*inputs, connection_type: str = "add"):
    if len(inputs) == 0:
        raise ValueError("At least one input tensor is required")

    if connection_type.lower().strip() == "add":
        return torch.stack(inputs, dim=0).sum(dim=0)

    elif connection_type.lower().strip() == "concat":
        return torch.cat(inputs, dim=1)

    elif connection_type.lower().strip() == "multiply":
        result = inputs[0]
        for x in inputs[1:]:
            result = result * x
        return result

    elif connection_type.lower().strip() == "average":
        return torch.stack(inputs, dim=0).mean(dim=0)

    else:
        raise ValueError(f"Unsupported residual connection type: {connection_type}")


def call_with_supported_kwargs(fn, kwargs):
    sig = inspect.signature(fn)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**supported)
