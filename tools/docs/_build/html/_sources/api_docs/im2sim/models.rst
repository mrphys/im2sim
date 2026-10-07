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

``models`` contains complete networks built from the blocks in :doc:`layers` and configured with
the classes in :doc:`configs`. There are two families:

* **Image models**: :class:`UNet`, :class:`HalfUNet` and :class:`ReverseHalfUNet`. These map an
  image to an image and can also serve as image encoders.
* **Image-to-graph models**: :class:`SimpleGraphDecoder`, :class:`Im2SimBase` and
  :class:`Im2SimGen2`. These combine an image with a graph, such as a template mesh, and predict
  new graph features or node positions.

Building an image model
-----------------------

Every model takes its input and output channels and its spatial ``rank`` as arguments, and
everything else from its config. The same config can build a 2D or a 3D model:

.. code-block:: python

    from im2sim.configs import UNetConfig
    from im2sim.models import UNet

    cfg = UNetConfig(filters=[32, 64, 128]).single_class_segmentation_mode()

    model = UNet(in_channels=1, out_channels=1, rank=3, cfg=cfg)

:class:`HalfUNet` fuses its levels by addition instead of using a full decoder, which makes it a
light image encoder. ``Im2SimGen2`` uses it for that role. :class:`ReverseHalfUNet` keeps the
decoders and drops the encoders, and supports deep supervision.

The graph decoder
-----------------

:class:`SimpleGraphDecoder` is the graph half of every image-to-graph model. It takes a graph,
optionally with image features already sampled at its nodes, runs a stack of
:class:`~im2sim.layers.GraphConvBlock` layers, and writes the result into one feature of the
graph.

.. figure:: models/diagrams/graph_decoder.svg
   :alt: graph.x, projected image features and, optionally, the current prediction feature are
         concatenated, passed through n_blocks GraphConvBlocks and a single-convolution output
         block, and the output is written into graph.<pred_feature_key> with += or =.

.. code-block:: python

    from im2sim.configs import GraphConvBlockConfig, SimpleGraphDecoderConfig
    from im2sim.models import SimpleGraphDecoder

    cfg = SimpleGraphDecoderConfig(
        n_blocks=2,
        block_cfg=GraphConvBlockConfig(depth=2),
        protocol="update",
        pred_feature_key="coords",
    )

    # graph.x has 32 channels and the projected image features have 64.
    decoder = SimpleGraphDecoder(
        in_channels=32 + 64, out_channels=3, cfg=cfg, graph_channels=32
    )

    graph = decoder(graph, projected_features)

The configuration answers three questions.

Which feature is written? (``pred_feature_key``)
    The graph attribute to write. The default is ``"x"``. Any other key, such as ``"coords"`` or
    ``"pressure"``, is created as zeros if it is missing. Its current value is also appended to
    the decoder input, so the decoder can see the state it is changing. The extra
    ``out_channels`` input channels are added automatically, so ``in_channels`` is only
    ``graph.x`` plus the projected features. Since the decoder does not write ``graph.x``, a small
    MLP maps ``graph.x`` plus the projected features back to ``graph_channels`` channels, so
    ``graph.x`` keeps its width from one iteration to the next. Pass ``graph_channels`` (the
    width of ``graph.x``) whenever projected features are used.

    With ``"x"``, the output is written into the first channels of ``graph.x`` and the projected
    features are then dropped, so the written channels must lie within the original
    ``graph.x``.

How is it written? (``protocol``)
    ``"update"`` adds the output to the existing value (``feature += output``), so the decoder
    learns a correction or a displacement. This suits iterative refinement and mesh deformation.
    ``"predict"`` overwrites the value (``feature = output``), so the decoder predicts the
    quantity directly, for example a pressure field on a fixed mesh.

Where is it written? (``pred_feature_channels``, ``include_ids``, ``exclude_ids``)
    Restricts the write to some channels and some nodes. ``include_ids`` and ``exclude_ids`` name
    graph attributes that hold node indices, such as the ``inlet_index`` attributes created by
    :doc:`mesh_ops`.

.. figure:: models/diagrams/node_channel_selection.svg
   :alt: An 8-node by 3-channel velocity feature in which only rows for inlet and outlet nodes
         and columns 0 and 2 are written from a 2-channel decoder output.

.. code-block:: python

    # Update only the x and z velocity of the inlet and outlet nodes.
    cfg = SimpleGraphDecoderConfig(
        protocol="update",
        pred_feature_key="velocity",
        pred_feature_channels=[0, 2],
        include_ids=["inlet_index", "outlet_index"],
    )
    decoder = SimpleGraphDecoder(
        in_channels=32 + 64, out_channels=2, cfg=cfg, graph_channels=32
    )

``out_channels`` must equal the number of channels written: ``len(pred_feature_channels)``, or the
width of the feature when no channels are selected. The selection limits only the write.
Message passing still runs over every node, so nodes that are not written still pass
information to the nodes that are.

Image-to-graph models
---------------------

:class:`Im2SimBase` connects four components:

* an **image encoder**, ``image -> image_features``;
* one or more **projections**, ``(image_features, graph) -> [N, C]`` features per node;
* a **graph decoder**, ``(graph, projected_features) -> graph``;
* an optional **rasterizer**, ``(graph, image) -> image``, which writes the current graph back
  into the image.

Calling the model runs the loop below ``n_iters`` times and returns the final graph, or a list
with the graph after every iteration when ``return_intermediate_graphs=True``.

.. figure:: models/diagrams/im2sim_forward.svg
   :alt: The image is encoded, the features are sampled at graph.coords by the projection, and
         the graph decoder updates the graph. The updated graph feeds the next iteration. If a
         rasterizer is given, it writes the graph into the image, which is encoded again.

Two details of the loop matter in practice:

* The projection samples features at ``graph.coords``. A decoder with
  ``pred_feature_key="coords"`` and the ``"update"`` protocol moves the nodes, so each iteration
  samples the image at the new node positions.
* Without a rasterizer, the image is encoded once. With one, the rasterizer overwrites the last
  channel(s) of the image with the current graph and the encoder runs again on every iteration.
  Reserve a channel for it, for example ``image_channels=2`` for one intensity channel plus one
  mask channel.

:class:`Im2SimGen2` is ``Im2SimBase`` with a :class:`HalfUNet` encoder and a
:class:`SimpleGraphDecoder`, built from their configs:

.. code-block:: python

    from im2sim.configs import (
        GraphConvBlockConfig,
        HalfUNetConfig,
        SimpleGraphDecoderConfig,
    )
    from im2sim.layers import MaskRasterizer, TrilinearProjection
    from im2sim.models import Im2SimGen2

    model = Im2SimGen2(
        image_shape=(128, 128, 128),
        image_channels=2,        # intensity + the mask written by the rasterizer
        projection_channels=64,  # encoder output channels sampled at each node
        graph_channels=32,       # channels of graph.x
        out_channels=3,          # channels written by the decoder
        encoder_cfg=HalfUNetConfig(n_levels=3, hidden_channels=64),
        decoder_cfg=SimpleGraphDecoderConfig(
            block_cfg=GraphConvBlockConfig(depth=2),
            pred_feature_key="coords",
        ),
        projection=TrilinearProjection(image_dim=128),
        rasterizer=MaskRasterizer(),
        n_iters=3,
    )

    out_graph = model(image, graph)

The graph must have ``coords``, and ``batch`` for the projection and rasterizer. Graphs batched
by the ``DataLoader`` in :doc:`data` already have ``batch``. For a single graph, set
``graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)``.

``image_input`` may be ``None`` when a rasterizer is given. The model then starts from an empty
image of shape ``image_shape`` and rasterizes the input graph into it.

Custom architectures
~~~~~~~~~~~~~~~~~~~~

Use ``Im2SimBase`` directly to swap in your own components:

.. code-block:: python

    from im2sim.models import HalfUNet, Im2SimBase, SimpleGraphDecoder

    encoder = HalfUNet(
        in_channels=2, out_channels=64, rank=3, cfg=HalfUNetConfig(n_levels=3)
    )
    decoder = SimpleGraphDecoder(
        in_channels=32 + 64,
        out_channels=3,
        cfg=SimpleGraphDecoderConfig(pred_feature_key="coords"),
        graph_channels=32,  # channels of graph.x, restored after each iteration
    )

    model = Im2SimBase(
        image_shape=(128, 128, 128),
        image_encoder=encoder,
        graph_decoder=decoder,
        projections=TrilinearProjection(image_dim=128),
        rasterizer=MaskRasterizer(),
        n_iters=3,
    )

``Im2SimBase`` checks the argument **names** of each component's ``forward`` when it is built, and
raises a ``ValueError`` if they differ:

.. list-table::
    :header-rows: 1

    * - Component
      - Required ``forward`` signature
    * - graph decoder
      - ``forward(self, in_graph, projected_features)``
    * - projection
      - ``forward(self, image_features, graph)``
    * - rasterizer
      - ``forward(self, graph, image_input)``

To sample several encoder levels, pass a list of projections. The encoder must then return a list
of feature maps with one entry per projection, and the decoder receives a list of projected
features. ``SimpleGraphDecoder`` expects a single tensor, so this needs a custom decoder.

Saving and loading
------------------

Models are ordinary ``torch.nn.Module`` objects. Save the weights with ``state_dict`` and the
architecture with the config's ``save()``:

.. code-block:: python

    torch.save(model.state_dict(), "unet.pt")
    cfg.save("unet.json")

    cfg = UNetConfig.load("unet.json")
    model = UNet(in_channels=1, out_channels=1, rank=3, cfg=cfg)
    model.load_state_dict(torch.load("unet.pt"))

For ``Im2SimGen2``, save ``encoder_cfg`` and ``decoder_cfg.block_cfg`` this way.
``SimpleGraphDecoderConfig`` has no ``save()`` (see :doc:`configs`), so store its remaining fields,
and the constructor arguments, alongside them.

