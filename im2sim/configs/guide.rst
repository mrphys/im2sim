``configs`` holds the dataclasses that describe ``im2sim`` models and their parts. Models and
blocks are built from a config object instead of a long list of constructor arguments, so an
architecture can be changed, copied, saved and reloaded without touching any PyTorch modules.

Configs also provide:

* ``mod()`` for making a modified copy;
* presets, such as ``add_se()`` or ``segmentation_mode()``, for common changes;
* ``save()`` / ``load()`` and ``to_dict()`` / ``from_dict()`` for serialisation.

How configs nest
----------------

Configs are nested. A model config holds one block config per level, and each block config holds
:class:`LayerConfig` objects. A ``LayerConfig`` names a single layer and the keyword arguments
passed to it when the layer is built.

.. figure:: configs/diagrams/config_hierarchy.svg
   :alt: A UNetConfig containing LayerConfigs for pooling and upsampling and a list of
         ImageConvBlockConfigs, each of which contains LayerConfigs for its convolution,
         normalisation, dropout and attention. Next to it, the same tree as JSON.

.. code-block:: python

    from im2sim.configs import ImageConvBlockConfig, LayerConfig, UNetConfig

    block_cfg = ImageConvBlockConfig(
        depth=2,
        activation="ReLU",
        conv_cfg=LayerConfig(name="Conv", kwargs={"kernel_size": 3, "padding": "same"}),
        norm_cfg=LayerConfig(name="InstanceNorm", kwargs={"affine": True}),
    )

    cfg = UNetConfig(
        filters=[32, 64, 128, 256],
        pool_cfg=LayerConfig(name="MaxPool", kwargs={"kernel_size": 2}),
        block_cfg=block_cfg,
    )

``block_cfg`` is a template: when ``encoder_block_cfg`` and ``decoder_block_cfg`` are not given,
``UNetConfig`` gives every level its own copy of it. To configure levels individually, pass a list
with one ``ImageConvBlockConfig`` per level instead.

Configs carry no spatial rank. Layer names are resolved when the model is built (see
:doc:`layers`), so ``"Conv"`` becomes ``Conv2d`` or ``Conv3d`` depending on the ``rank`` passed to
the model, and the same config can build a 2D or a 3D network.

Modifying configs
-----------------

``mod()`` returns a new config with some fields replaced:

.. code-block:: python

    block_cfg = ImageConvBlockConfig(depth=4, activation="LeakyReLU")
    small_block_cfg = block_cfg.mod(depth=2, activation="ReLU")

Presets are methods that apply a common change and return the config, so they can be chained:

.. code-block:: python

    cfg = (
        UNetConfig(filters=[32, 32, 32])
        .to_depthwise_separable()
        .add_input_residual()
    )

There are presets for task set-up (``reconstruction_mode()``, ``single_class_segmentation_mode()``,
``multiclass_segmentation_mode()``), convolution types (``to_depthwise_separable()``,
``to_ghost_depthwise()``), attention (``add_se()``, ``add_eca()``), dilation and residual
connections. The API pages for each config class list them all. Model-level presets such as
``UNetConfig.add_se()`` apply the change to every encoder and decoder block.

.. important::

    Presets change the config **in place** and return the same object. ``mod()`` makes a new
    top-level object, but the nested configs inside it are shared with the original. Applying a
    preset to a ``mod()`` copy can therefore change the original too. Use ``copy.deepcopy`` when
    the original must stay untouched:

    .. code-block:: python

        import copy

        base_cfg = UNetConfig(filters=[32, 64, 128])
        se_cfg = copy.deepcopy(base_cfg).add_se()   # base_cfg is unchanged

Residual connections
--------------------

``ImageConvBlockConfig`` and ``GraphConvBlockConfig`` describe residual connections with two
fields. ``residual_connections`` is a dict ``{target: [sources]}``, and ``residual_type`` sets how
the tensors are combined: ``"add"``, ``"concat"``, ``"multiply"`` or ``"average"``.

Each index refers to a point between layers. ``0`` is the block input and ``k`` is the output of
layer ``k``. Each source tensor is merged into the target point before the next layer runs.

.. figure:: configs/diagrams/residual_connections.svg
   :alt: Four copies of a depth-4 block showing where add_input_residual, add_conv1_residual,
         add_input_concat_residual and a custom two-source concat connection attach.

The three residual presets set both fields for you:

.. code-block:: python

    ImageConvBlockConfig(depth=4).add_input_residual()         # {4: [0]}, "add"
    ImageConvBlockConfig(depth=4).add_conv1_residual()         # {4: [1]}, "add"
    ImageConvBlockConfig(depth=4).add_input_concat_residual()  # {3: [0]}, "concat"

Connections can also be written directly:

.. code-block:: python

    cfg = ImageConvBlockConfig(
        depth=4,
        residual_connections={3: [0, 1]},
        residual_type="concat",
    )

``"add"``, ``"multiply"`` and ``"average"`` need the merged tensors to have the same number of
channels. ``"concat"`` increases the channel count, which is why ``add_input_concat_residual()``
targets the input of the last layer rather than its output: the last layer maps the wider tensor
back to ``out_channels``. Graph blocks default to ``residual_type="average"``, image blocks to
``"add"``.

Saving and loading
------------------

``save()`` writes a config, including all nested configs, to a JSON file. Each nested object is
stored with its class name, so ``load()`` rebuilds the full tree:

.. code-block:: python

    cfg = UNetConfig(filters=[32, 64, 128, 256])
    cfg.save("unet.json")

    loaded_cfg = UNetConfig.load("unet.json")

``to_dict()`` and ``from_dict()`` do the same without touching the file system, which is useful
for storing a config inside an experiment log or a checkpoint:

.. code-block:: python

    config_dict = cfg.to_dict()
    cfg = UNetConfig.from_dict(config_dict)

Save the config next to the model's ``state_dict`` so that the architecture can be rebuilt before
the weights are loaded.

.. note::

    :class:`SimpleGraphDecoderConfig` is a plain dataclass. It does not inherit from the config
    base class, so it has no ``mod()``, ``save()`` or ``load()``. Use ``dataclasses.replace`` and
    ``dataclasses.asdict`` instead. Its ``block_cfg`` is a ``GraphConvBlockConfig`` and supports
    everything described above.
