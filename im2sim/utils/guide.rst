``utils`` holds the internals that other modules are built on. It exports nothing at package
level, so import from its submodules. Most users only need it to register custom layers.

Layer registries (``im2sim.utils.layer_util``)
----------------------------------------------

Every ``LayerConfig`` name in :doc:`configs` is resolved through one of these registries. The
lookup order is shown in :doc:`layers`.

.. list-table::
    :header-rows: 1
    :widths: 22 42 36

    * - Registry
      - Filled with
      - Look up / extend with
    * - ``IMAGE_LAYERS``
      - ``torch.nn`` layers whose names contain ``Conv``, ``Pool``, ``Norm``, ``Upsample``,
        ``PixelShuffle`` or ``Dropout``, plus the im2sim image layers
      - ``get_image_layer(name, rank)``, ``register_image_layer``
    * - ``GRAPH_LAYERS``
      - im2sim layers that take and return a graph (``GraphDropout``, ``EdgeDropout``, ...)
      - ``get_graph_layer(name, args, kwargs)``, ``register_graph_layer``
    * - ``PYG_LAYERS``
      - ``torch_geometric.nn`` layers whose names contain ``Conv``, ``Pool``, ``Norm`` or
        ``Dropout``, plus a few im2sim layers written in the PyG style
      - looked up by ``get_graph_layer``, ``register_pyg_layer``
    * - ``ACTIVATIONS``
      - ``torch.nn`` activations (``ReLU``, ``GELU``, ``Softmax``, ...)
      - ``get_activation(name)``, ``register_activation``

``get_image_layer`` returns a **class** and leaves construction to the caller.
``get_graph_layer`` returns a constructed **module** that maps a graph to a graph. Names are
matched without regard to case, spaces or underscores.

The ``register_*`` functions are decorators. They take an optional ``name`` and default to the
class name:

.. code-block:: python

    import torch
    from im2sim.utils.layer_util import register_activation

    @register_activation(name="Swish")
    class Swish(torch.nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

To register an image layer under ``"<Name>1d"``, ``"<Name>2d"`` and ``"<Name>3d"`` at once, give
its ``__init__`` a ``rank`` argument and decorate the class with ``register_with_ranks("<Name>")``.
The im2sim depthwise, ghost and attention layers are registered this way.

PyG operations without compiled extras (``im2sim.utils.pyg_ops``)
-----------------------------------------------------------------

``knn``, ``knn_interpolate`` and ``graclus`` are drop-in replacements for the PyTorch Geometric
functions of the same names. PyG runs these through optional compiled packages (``pyg-lib`` or
``torch_cluster``) that are not available for every PyTorch version. The wrappers use the
compiled backend when PyG can find one and otherwise fall back to a pure-PyTorch version with a
one-time warning. The losses, ``FeatureRasterizer`` and ``mesh_ops`` use these wrappers, so ``im2sim`` runs either way.
The compiled backends are faster; install them with the ``im2sim-install-pyg-addons`` command.
