im2sim.models
=============

.. automodule:: im2sim.models

Classes
-------

.. autosummary::
    :toctree: models
    :template: models/class.rst
    :nosignatures:

    HalfUNet
    Im2SimBase
    Im2SimGen2
    ReverseHalfUNet
    SimpleGraphDecoder
    UNet

Functions
---------

.. autosummary::
    :toctree: models
    :template: models/function.rst
    :nosignatures:

    

Guide
=====

``models`` provides complete neural network architectures for building
machine-learning models in ``im2sim``.

Models are composed from the reusable building blocks in
``im2sim.layers`` and configured using objects from ``im2sim.configs``. This
separation allows architectures to be customised without directly
constructing the underlying PyTorch modules.

The models module supports both conventional image-processing architectures
and models that couple image and graph representations. The latter provide
the basis for using ``im2sim`` as a framework for learning image-to-geometry
and image-to-physics mappings.

Image model construction
-------------------

Models in ``im2sim`` are constructed by combining a model class with a
configuration object.

For example:

.. code-block:: python

    from im2sim.configs import UNetConfig
    from im2sim.models import UNet

    cfg = UNetConfig(
        filters=[32, 64, 128],
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        rank=3,
        cfg=cfg,
    )


The model is the blueprint for the assembly of the model while the 
configuration object defines the specific parameters and components of the architecture.

This separation means that the same model type can be reused with
different configurations.

Graph decoder protocols
------------------------

The graph decoder provides a common interface for applying graph neural
networks to an input graph while controlling how the predicted values are
written back to the graph.

:class:`SimpleGraphDecoder` combines:

* a sequence of :class:`GraphConvBlock` instances;
* optional image features projected onto graph nodes;
* a prediction feature stored on the graph; and
* a configurable graph protocol controlling how the prediction is applied.

The decoder supports two protocols:

.. code-block:: text


    ┌───────────────────┐
    │   Input graph     │
    │                   │
    │ x + image features│
    └─────────┬─────────┘
              │
              ▼
      GraphConvBlock(s)
              │
              ▼
      Graph prediction
              │
        ┌─────┴─────┐
        │           │
        ▼           ▼
    update      predict
        │           │
        ▼           ▼
    add to       replace
    existing      existing
    value         value
        │           │
        └─────┬─────┘
              ▼
         Output graph



`update` protocol
------------------

The `update` protocol treats the graph decoder output as an increment to an
existing graph feature.

The update is:

.. code-block:: text

    graph.feature ← graph.feature + decoder(graph)

For example, suppose a graph already contains a three-channel prediction
feature:

.. code-block:: python

    cfg = SimpleGraphDecoderConfig(
        protocol="update",
        pred_feature_key="prediction",
        block_cfg=GraphConvBlockConfig(),
    )

    decoder = SimpleGraphDecoder(
        in_channels=32,
        out_channels=3,
        cfg=cfg,
    )


The decoder produces a three-channel update which is added to
`graph.prediction`.

This makes the protocol useful for iterative refinement. For example, a
model can start from an initial geometry and repeatedly predict corrections:

.. code-block:: text


    Initial graph
        │
        ▼
    decoder update
        │
        ▼
    graph + Δgraph
        │
        ▼
    decoder update
        │
        ▼
    graph + Δgraph
        │
        ▼
    refined graph


This is particularly useful when a model is intended to learn a deformation
rather than directly predict the final value.

`predict` protocol
-------------------

The `predict` protocol treats the graph decoder output as the new value of a
graph feature.

The prediction is:

.. code-block:: text


graph.feature ← decoder(graph)


For example:

.. code-block:: python


    cfg = SimpleGraphDecoderConfig(
        protocol="predict",
        pred_feature_key="pressure",
        block_cfg=GraphConvBlockConfig(),
    )

    decoder = SimpleGraphDecoder(
        in_channels=32,
        out_channels=1,
        cfg=cfg,
    )


The decoder output replaces the selected channels of
`graph.pressure`.

This protocol is therefore appropriate when the graph contains the input
geometry and features required to make a prediction, but the target feature
itself should be predicted directly.

Selecting the predicted feature
--------------------------------

The feature being updated or predicted is selected using
`pred_feature_key`.

The default is `"x"`, meaning that the graph's main node feature tensor is
the prediction target.

For example:

.. code-block:: python

    cfg = SimpleGraphDecoderConfig(
        protocol="predict",
        pred_feature_key="pressure",
        block_cfg=GraphConvBlockConfig(),
    )

    decoder = SimpleGraphDecoder(
        in_channels=32,
        out_channels=1,
        cfg=cfg,
    )


When the specified graph feature does not already exist, it is initialised to
zeros with the required number of output channels.

Predicting selected channels
----------------------------

Specific channels of the prediction feature can be selected using
`pred_feature_channels`.

For example:

.. code-block:: python


    cfg = SimpleGraphDecoderConfig(
        protocol="update",
        pred_feature_key="velocity",
        pred_feature_channels=[0, 2],
        block_cfg=GraphConvBlockConfig(),
    )

    decoder = SimpleGraphDecoder(
        in_channels=32,
        out_channels=2,
        cfg=cfg,
    )


In this case, the decoder operates only on channels `0` and `2` of
`graph.velocity`.

This allows different model components to operate on different parts of a
larger graph feature without replacing the entire feature tensor.

Using node selection
---------------------

The decoder can also restrict its operation to a subset of graph nodes.

Nodes can be selected using ``include_ids``:

.. code-block:: python

    cfg = SimpleGraphDecoderConfig(
        protocol="update",
        pred_feature_key="displacement",
        include_ids=["inlet_nodes", "outlet_nodes"],
        block_cfg=GraphConvBlockConfig(),
    )


Only nodes associated with the specified graph attributes are updated.

Alternatively, nodes can be excluded using ``exclude_ids``:

.. code-block:: python

    cfg = SimpleGraphDecoderConfig(
        protocol="update",
        pred_feature_key="displacement",
        exclude_ids=["fixed_nodes"],
        block_cfg=GraphConvBlockConfig(),
    )


All nodes except those associated with ``fixed_nodes`` are then updated.

If both are provided, ``include_ids`` takes precedence.

The graph therefore acts as both the input representation and a convenient
mechanism for specifying which subsets of the graph are trainable or
predictable.

Existing prediction features as inputs
--------------------------------------

When ``pred_feature_key`` is not ``"x"``, the existing prediction feature is
also made available to the graph network.

For example:

.. code-block:: python

    cfg = SimpleGraphDecoderConfig(
        protocol="update",
        pred_feature_key="displacement",
        pred_feature_channels=[0, 1, 2],
        block_cfg=GraphConvBlockConfig(),
    )


If ``graph.displacement`` already contains the current displacement, those
selected channels are concatenated with ``graph.x`` before the graph network
is applied.

This is particularly useful for iterative updates because the decoder can use
the current state when predicting the next correction:

.. code-block:: text

    graph features
        +
    current prediction
        │
        ▼
    GNN
        │
        ▼
    update
        │
        ▼
    updated prediction



Image-to-graph models
----------------------

A central purpose of ``im2sim`` is to connect image-based information with
graph-based representations.

An image-to-graph model can contain several stages:

.. code-block:: text

    Image
    │
    ▼
    Image encoder
    │
    ▼
    Image features
    │
    ▼
    Image → graph projection
    │
    ▼
    Graph processing
    │
    ▼
    Graph prediction
    │
    ▼
    Updated geometry / physical quantities


The image encoder extracts spatial features from the input image. A
projection transfers those features onto graph nodes, allowing graph
processing layers to combine image-derived information with existing graph
features and connectivity.

The resulting graph representation can then be used to predict geometry,
graph features, or other quantities associated with the underlying system.


``Im2SimGen2``
---------------

:class:`Im2SimGen2` provides a complete image-to-graph architecture built from
these components.

It combines an image encoder, image-to-graph projection, graph processing,
and optional rasterisation into a single model:

.. code-block:: text

                      ┌─────────────────┐
                      │   Input image   │
                      └────────┬────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │ Image encoder   │
                      └────────┬────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │   Projection    │
                      └────────┬────────┘
                               │
                               ▼
    ┌──────────────┐  ┌─────────────────┐
    │ Input graph  ├─►│  Graph decoder  │
    └──────────────┘  └────────┬────────┘
                               │
                               ▼
                          Output graph


A complete ``Im2SimGen2`` model can be constructed by specifying the image
and graph channel dimensions together with the encoder, graph decoder,
projection, and rasterisation components:

.. code-block:: python

    from im2sim.configs import (
        GraphConvBlockConfig,
        HalfUNetConfig,
        SimpleGraphDecoderConfig,
    )
    from im2sim.layers import MaskRasterizer, TrilinearProjection
    from im2sim.models import Im2SimGen2

    encoder_cfg = HalfUNetConfig(
        n_levels=3,
        hidden_channels=64,
    )

    decoder_cfg = SimpleGraphDecoderConfig(
        block_cfg=GraphConvBlockConfig(),
    )

    model = Im2SimGen2(
        image_channels=3,
        projection_channels=64,
        graph_channels=32,
        out_channels=32,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        projection=TrilinearProjection(128),
        rasterizer=MaskRasterizer(),
        n_iters=3,
    )


Rasterisation and image feedback
---------------------------------

The optional rasteriser provides the connection back from the graph domain to
the image domain.

This is particularly useful for iterative architectures, where an updated
graph can be rasterised into an image representation and used during a later
iteration.

For example:

.. code-block:: python

    rasterizer = MaskRasterizer()

    model = Im2SimGen2(
        image_channels=3,
        projection_channels=64,
        graph_channels=32,
        out_channels=32,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        projection=TrilinearProjection(128),
        rasterizer=rasterizer,
        n_iters=3,
    )


The projection and rasterisation modules therefore define the interface
between the dense image representation and the graph representation.

Iterative graph updates
------------------------

A defining feature of `Im2SimGen2` is its ability to refine the graph
iteratively.

Rather than predicting the complete graph state in a single step, the model
can repeatedly update the graph representation:

.. code-block:: text

    Image features
        │
        ▼
    Graph update
        │
        ▼
    Updated geometry
        │
        ▼
    Rasterisation
        │
        ▼
    Next iteration


The number of refinement steps is controlled using ``n_iters``:

.. code-block:: python

    model = Im2SimGen2(
        image_channels=3,
        projection_channels=64,
        graph_channels=32,
        out_channels=32,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        projection=TrilinearProjection(128),
        rasterizer=MaskRasterizer(),
        n_iters=3,
    )


A model can therefore learn a sequence of geometry updates rather than a
single direct mapping.


Intermediate graph states
--------------------------

Intermediate graph states can optionally be returned during the iterative
forward pass:

.. code-block:: python

    model = Im2SimGen2(
        image_channels=3,
        projection_channels=64,
        graph_channels=32,
        out_channels=32,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        projection=TrilinearProjection(128),
        rasterizer=MaskRasterizer(),
        n_iters=3,
        return_intermediate_graphs=True,
    )


This can be useful when intermediate geometries are required for visualisation
or when losses are applied at multiple stages of the refinement process.

Using graph features
--------------------

The graph representation can contain both geometric and non-geometric
features.

For example, a PyTorch Geometric graph can contain node features, coordinates,
and connectivity:

.. code-block:: python

    from torch_geometric.data import Data

    graph = Data(
        x=node_features,
        pos=node_coordinates,
        edge_index=edge_index,
    )


This allows image information to be combined with an existing graph
representation rather than requiring the image to be converted entirely into
a graph before processing.

The resulting model can therefore use:

* image-derived features;
* node coordinates;
* existing node features; and
* graph connectivity.

Image-to-graph prediction
--------------------------

The simplest use of `Im2SimGen2` is to combine an image with an existing
graph and predict an updated graph.

Conceptually:

.. code-block:: text

    Medical image + input graph
                │
                ▼
        Im2SimGen2
                │
                ▼
        predicted graph


The input graph may represent a template, approximate geometry, or other
structured representation that can be refined using information from the
image.




Building custom image-to-graph models with ``Im2SimBase``
----------------------------------------------------------

While :class:`Im2SimGen2` provides a complete image-to-graph architecture,
:class:`Im2SimBase` provides a more flexible foundation for building custom
models.

Rather than prescribing a particular encoder, decoder, or graph-processing
architecture, ``Im2SimBase`` allows the individual components of an
image-to-graph model to be assembled independently.

Conceptually, the base model provides the structure:

.. code-block:: text

    Image
    │
    ▼
    Image encoder
    │
    ▼
    Projection
    │
    ▼
    Graph processing
    │
    ▼
    Graph output


Optional components can then be added around this structure, such as graph
rasterisation or iterative updates.

This makes ``Im2SimBase`` useful when the standard ``Im2SimGen2`` architecture
does not provide the required level of control.

For example, the individual components can be constructed independently:

.. code-block:: python

    from im2sim.configs import (
        GraphConvBlockConfig,
        HalfUNetConfig,
        SimpleGraphDecoderConfig,
    )
    from im2sim.layers import MaskRasterizer, TrilinearProjection

    encoder_cfg = HalfUNetConfig(
        n_levels=3,
        hidden_channels=64,
    )

    decoder_cfg = SimpleGraphDecoderConfig(
        block_cfg=GraphConvBlockConfig(
            depth=2,
        ),
    )

    projection = TrilinearProjection(128)

    rasterizer = MaskRasterizer()


These components can then be combined using ``Im2SimBase`` to define a
custom image-to-graph architecture.

The important distinction is that ``Im2SimGen2`` provides a predefined
architecture, whereas ``Im2SimBase`` is intended as an architectural
framework. The latter is therefore the appropriate starting point when a
model requires custom:

* image encoders;
* graph processing modules;
* image-to-graph projections;
* graph-to-image rasterisers;
* iterative update schemes; or
* combinations of these components.

This also makes it possible to reuse the same image encoder or graph
processing module across different model architectures without rewriting the
overall image-to-graph framework.


Saving and loading models
--------------------------

Models are standard PyTorch modules and can therefore use the normal PyTorch
``state_dict`` interface:

.. code-block:: python

    import torch

    torch.save(model.state_dict(), "model.pth")

    model.load_state_dict(
        torch.load("model.pth")
    )


Model configurations should be saved alongside the learned parameters so
that the architecture can be reproduced:

.. code-block:: python

    cfg.save("model_config.yaml")


A model can then be reconstructed from the saved configuration before loading
its parameters.

Summary
-------

The ``models`` module provides configurable neural network architectures for
``im2sim``.

The models range from reusable image-processing networks to image-to-graph
architectures that combine dense image features with structured graph
representations.

The main image-to-graph workflow is:

.. code-block:: text

    Image
    │
    ▼
    Image encoder
    │
    ▼
    Image features
    │
    ▼
    Image ↔ graph interface
    │
    ▼
    Graph processing
    │
    ▼
    Graph prediction


:class:`Im2SimGen2` provides a complete implementation of this workflow with
iterative graph refinement, projection, and optional rasterisation.

:class:`Im2SimBase` provides a more general framework for constructing custom
image-to-graph architectures from independently selected components.

Together, these models provide both a ready-to-use architecture and a
flexible foundation for developing new image-based digital twin models.


