from torch import nn
import torch.nn.functional as F


def get_image_layer(name, rank):
  """Get an N-D layer object.

  Args:
    name: A `str`. The name of the requested layer.
    rank: An `int`. The rank of the requested layer.

  Returns:
    A `torch.nn.Module` object.

  Raises:
    ValueError: If the requested layer is unknown.
  """
  try:
    return _IMAGE_LAYERS[(name, rank)]
  except KeyError as err:
    raise ValueError(
        f"Could not find a layer with name '{name}' and rank {rank}.") from err
  
  
def get_graph_layer(name):
  """Get an graph layer object.

  Args:
    name: A `str`. The name of the requested layer.

  Returns:
    A `torch.nn.Module` object.

  Raises:
    ValueError: If the requested layer is unknown.
  """
  try:
    return _GRAPH_LAYERS[name] 
  except KeyError as err:
    raise ValueError(
        f"Could not find an activation with name '{name}'") from err
  
def get_default_kwargs(name):
  """Get an graph layer object.

  Args:
    name: A `str`. The name of the requested layer.

  Returns:
    A `torch.nn.Module` object.

  Raises:
    ValueError: If the requested layer is unknown.
  """
  try:
    return _LAYER_KWARGS[name] 
  except KeyError as err:
    raise ValueError(
        f"Could not find an activation with name '{name}'") from err
  
  
def get_activation(name):
  """Get an activation object.

  Args:
    name: A `str`. The name of the requested layer.

  Returns:
    A `torch.nn.Module` object.

  Raises:
    ValueError: If the requested activation is unknown.
  """
  try:
    return _ACTIVATIONS[name]
  except KeyError as err:
    raise ValueError(
        f"Could not find an activation with name '{name}'") from err


_IMAGE_LAYERS = {
    ('AveragePooling', 1): nn.AvgPool1d,
    ('AveragePooling', 2): nn.AvgPool2d,
    ('AveragePooling', 3): nn.AvgPool3d,
    ('Conv', 1): nn.Conv1d,
    ('Conv', 2): nn.Conv2d,
    ('Conv', 3): nn.Conv3d,
    ('ConvTranspose', 1): nn.ConvTranspose1d,
    ('ConvTranspose', 2): nn.ConvTranspose2d,
    ('ConvTranspose', 3): nn.ConvTranspose3d,
    ('MaxPool', 1): nn.MaxPool1d,
    ('MaxPool', 2): nn.MaxPool2d,
    ('MaxPool', 3): nn.MaxPool3d,
    ('Dropout', 1): nn.Dropout1d,
    ('Dropout', 2): nn.Dropout2d,
    ('Dropout', 3): nn.Dropout3d,
    ('ZeroPadding', 1): nn.ZeroPad1d,
    ('ZeroPadding', 2): nn.ZeroPad2d,
    ('ZeroPadding', 3): nn.ZeroPad3d,
    ('BatchNorm', 1): nn.BatchNorm1d,
    ('BatchNorm', 2): nn.BatchNorm2d,
    ('BatchNorm', 3): nn.BatchNorm3d,
    ('InstanceNorm', 1): nn.InstanceNorm1d,
    ('InstanceNorm', 2): nn.InstanceNorm2d,
    ('InstanceNorm', 3): nn.InstanceNorm3d
}

# _GRAPH_LAYERS = {
#   'ChebConv': gnn.ChebConv,
#   'GraphConv': gnn.GraphConv,
#   'GCNConv': gnn.GCNConv,
#   'GATConv': gnn.GATConv,
#   'InstanceNorm': gnn.InstanceNorm,
#   'BatchNorm': gnn.BatchNorm,
#   'GraphNorm': gnn.GraphNorm
# }

# _LAYER_KWARGS = {
#   'ChebConv': {'K':3},
#   'GraphConv': {},
#   'GCNConv': {},
#   'GATConv': {}
# }


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
    "softmax": nn.Softmax
}


# def init_weights(m):
#     if isinstance(m, gnn.ChebConv):
#         for lin in m.lins:
#             nn.init.kaiming_normal_(lin.weight, nonlinearity='leaky_relu')
#             lin.weight.data *= 0.1
#             if lin.bias is not None:
#                 nn.init.zeros_(lin.bias)


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
            raise TypeError(
                f"Each factor must be an int, tuple, or list, got {type(f).__name__}"
            )

    return standardized