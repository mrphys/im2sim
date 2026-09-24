im2sim.layers
=============

.. automodule:: im2sim.layers

Classes
-------

.. autosummary::
    :toctree: layers
    :template: layers/class.rst
    :nosignatures:

    ChannelDropout
    ConditionedSqueezeExcite
    DefaultGraphNorm
    DepthwiseConv
    DepthwiseSeparableConv
    EdgeDropout
    EfficientChannelAttn
    FeatureRasterizer
    GhostConv
    GraphActivation
    GraphConvBlock
    GraphDropout
    GraphECA
    GraphSE
    ImageConvBlock
    MaskRasterizer
    NodeDropout
    SqueezeExcite
    TrilinearProjection

Functions
---------

.. autosummary::
    :toctree: layers
    :template: layers/function.rst
    :nosignatures:

    

Guide
=====

`layers` provides a collection of building blocks for constructing and customising
deep learning models in `im2sim`.

The module supports standard `torch` layers, `PyTorch Geometric (PyG)` layers,
and custom layers designed specifically for image and graph processing. These
layers can be composed into larger building blocks and complete model
architectures.

The `layers` module also provides utilities for:

* dynamically accessing image and graph layers;
* wrapping PyG modules for use with graph data;
* constructing configurable image and graph blocks; and
* transferring features between image and graph domains.

Layer hierarchy
----------------

The layers in this module are designed to be composed from general-purpose
components into increasingly specialised building blocks.

At the lowest level, individual layers can be accessed dynamically using
:func:`get_image_layer` and :func:`get_graph_layer`:

.. code-block:: python


    from im2sim.layers import get_image_layer, get_graph_layer

    conv_layer = get_image_layer(
        name="Conv",
        kwargs={
            "in_channels": 16,
            "out_channels": 32,
            "kernel_size": 3,
            "padding": "same",
        },
    )

    graph_layer = get_graph_layer(
        name="GATConv",
        kwargs={
            "in_channels": 16,
            "out_channels": 32,
        },
    )


The exact arguments required by a layer depend on the registered layer type.
For PyG layers, `get_graph_layer` handles the construction and wrapping
required for use with graph data.

Higher-level blocks provide configurable compositions of these layers. For
example, an image convolutional block can be created using
:class:`ImageConvBlock` and :class:`ImageConvBlockConfig`:

.. code-block:: python

    from im2sim.configs import ImageConvBlockConfig
    from im2sim.layers import ImageConvBlock

    cfg = ImageConvBlockConfig(
        depth=3,
        activation="ReLU",
    )

    block = ImageConvBlock(
        in_channels=16,
        out_channels=32,
        rank=3,
        cfg=cfg,
)


The configuration object controls the structure of the block, while
`in_channels`, `out_channels`, and `rank` define the input/output
dimensions and spatial dimensionality.

Graph-based blocks are configured similarly using
:class:`GraphConvBlock` and
:class:`GraphConvBlockConfig`:

.. code-block:: python


    from im2sim.configs import GraphConvBlockConfig
    from im2sim.layers import GraphConvBlock

    cfg = GraphConvBlockConfig(
        depth=3,
        activation="ReLU",
    )

    graph_block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )


Individual layers can also be customised through the configuration object:

.. code-block:: python


    from im2sim.configs import GraphConvBlockConfig, LayerConfig
    from im2sim.layers import GraphConvBlock

    cfg = GraphConvBlockConfig(
        depth=2,
        activation="ReLU",
        conv=LayerConfig(
            name="GATConv",
            kwargs={"heads": 1},
        ),
    )

    graph_block = GraphConvBlock(
        in_channels=16,
        out_channels=32,
        cfg=cfg,
    )


Torch and PyG Module Integration
---------------------------------

The `layers` module integrates standard `torch` modules and
`PyG` modules through layer registries. Image layers can be accessed through
`get_image_layer` and graph layers through `get_graph_layer`:

.. code-block:: python


    from im2sim.layers import get_image_layer, get_graph_layer

    conv = get_image_layer(
        name="Conv",
        kwargs={
            "in_channels": 16,
            "out_channels": 32,
            "kernel_size": 3,
            "padding": "same",
        },
    )

    gcn = get_graph_layer(
        name="GCNConv",
        kwargs={
            "in_channels": 16,
            "out_channels": 32,
        },
    )


PyG layers are wrapped internally so that they can operate within the graph
layer interface used by `im2sim`.

Custom Layers
--------------

In addition to standard `torch` and `PyG` modules, the `layers` module
provides custom layers for image and graph processing.

Examples of image-processing layers include:

* :class:`DepthwiseConv`
* :class:`DepthwiseSeparableConv`
* :class:`SqueezeExcite`
* :class:`EfficientChannelAttn`

Graph-processing functionality includes components such as:

* :class:`GraphConvBlock`
* :class:`GraphDropout`
* :class:`GraphECA`
* :class:`GraphSE`

These components can be used independently or composed into larger
architectures.

Creating Custom Layers
----------------------

New layers can be registered with the appropriate layer registry. A custom
graph layer can, for example, be implemented as a `torch.nn.Module`:

.. code-block:: python

    import torch

    from im2sim.layers import register_graph_layer

    @register_graph_layer(name="CustomGraphLayer")
    class CustomGraphLayer(torch.nn.Module):

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.linear = torch.nn.Linear(
                in_channels,
                out_channels,
            )

        def forward(self, graph):
            graph.x = self.linear(graph.x)
            return graph


Once registered, the layer can be constructed dynamically:

.. code-block:: python

    from im2sim.layers import get_graph_layer

    layer = get_graph_layer(
        name="CustomGraphLayer",
        kwargs={
            "in_channels": 16,
            "out_channels": 32,
        },
    )


Rasterizing and Projecting Features
-----------------------------------

The `layers` module includes utilities for transferring features between
image and graph domains.

For example, :class:`TrilinearProjection` can project image features onto graph
nodes, while :class:`MaskRasterizer` can rasterize graph geometry or features
back into an image representation:

.. code-block:: python


    from im2sim.layers import MaskRasterizer, TrilinearProjection

    projection = TrilinearProjection(
        image_size=128,
    )

    rasterizer = MaskRasterizer()



Interaction with `configs`
--------------------------

The `layers` module is designed to work closely with the `configs` module.
Configuration objects allow layer and block architectures to be defined
separately from their construction.

For example, an image convolutional block can be configured using
:class:`ImageConvBlockConfig`:

.. code-block:: python

    from im2sim.configs import ImageConvBlockConfig
    from im2sim.layers import ImageConvBlock

    cfg = ImageConvBlockConfig(
        depth=3,
        activation="ReLU",
    )

    block = ImageConvBlock(
        in_channels=1,
        out_channels=16,
        rank=3,
        cfg=cfg,
    )


Individual layer properties can also be specified using `LayerConfig`:

.. code-block:: python

    from im2sim.configs import ImageConvBlockConfig, LayerConfig
    from im2sim.layers import ImageConvBlock

    cfg = ImageConvBlockConfig(
        depth=3,
        activation="ReLU",
        conv=LayerConfig(
            name="Conv",
            kwargs={
                "kernel_size": 5,
                "padding": "same",
            },
        ),
    )

    block = ImageConvBlock(
        in_channels=1,
        out_channels=16,
        rank=3,
        cfg=cfg,
    )


Configuration objects can also be modified and reused when constructing
different models, making it possible to maintain consistent architectures
across experiments.

Summary
-------

The `layers` module provides a flexible system for constructing deep learning
models from reusable image and graph components. Standard `torch` and `PyG`
layers can be accessed dynamically, while higher-level blocks such as
:class:`ImageConvBlock` and :class:`GraphConvBlock` provide configurable
architectural building blocks.

The integration with `configs` separates architectural configuration from
layer construction, making layers and blocks straightforward to customise,
reuse, and compose into larger image-to-graph models.


