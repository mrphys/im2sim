im2sim.configs
==============

.. automodule:: im2sim.configs

Classes
-------

.. autosummary::
    :toctree: configs
    :template: configs/class.rst
    :nosignatures:

    GraphConvBlockConfig
    HalfUNetConfig
    ImageConvBlockConfig
    LayerConfig
    ReverseHalfUNetConfig
    SimpleGraphDecoderConfig
    UNetConfig

Functions
---------

.. autosummary::
    :toctree: configs
    :template: configs/function.rst
    :nosignatures:

    

Guide
=====


``configs`` provides a structured configuration system for defining and customising
the components of ``im2sim`` models.

Configurations are implemented as dataclasses and can be composed hierarchically.
For example, a model configuration can contain configurations for its convolutional
blocks, which in turn contain configurations for individual layers. This makes it
possible to customise a model at different levels of detail without having to
construct the underlying PyTorch modules manually.

The configuration system also provides utilities for:

* modifying existing configurations;
* applying common configuration presets;
* recursively serialising configurations; and
* saving and loading configurations from files.

Configuration hierarchy
-----------------------

The configuration classes are designed to be composed from general-purpose
configurations into increasingly specialised components.

At the lowest level, :class:`LayerConfig` describes an individual layer or module:

.. code-block:: python

    from im2sim.configs import LayerConfig

    conv_cfg = LayerConfig(
        name="Conv",
        kwargs={
            "kernel_size": 3,
            "padding": "same",
        },
    )


A ``LayerConfig`` specifies the name of the layer and the keyword arguments
that should be passed to it when the layer is constructed.

Higher-level configurations can then use ``LayerConfig`` objects to describe
their constituent layers. For example, an :class:`ImageConvBlockConfig` can
specify the convolution, normalisation, dropout, and attention layers within
a convolutional block:

.. code-block:: python


    from im2sim.configs import ImageConvBlockConfig, LayerConfig

    block_cfg = ImageConvBlockConfig(
        depth=4,
        activation="LeakyReLU",
        conv_cfg=LayerConfig(
            name="Conv",
            kwargs={"kernel_size": 5, "padding": "same"},
        ),
        norm_cfg=LayerConfig(
            name="BatchNorm",
            kwargs={"affine": True},
        ),
        dropout_cfg=LayerConfig(
            name="Dropout",
            kwargs={"p": 0.5},
        ),
    )


This hierarchical structure allows configurations to be reused across
different parts of a model.

Configuring model architectures
-------------------------------

Complete model configurations can be built by combining these lower-level
configurations. For example, a :class:`UNetConfig` controls the architecture
of a U-Net while allowing the convolutional blocks to be customised:

.. code-block:: python

    from im2sim.configs import ImageConvBlockConfig, LayerConfig, UNetConfig

    block_cfg = ImageConvBlockConfig(
        depth=4,
        activation="LeakyReLU",
        out_activation="sigmoid",
        conv_cfg=LayerConfig(
            name="Conv",
            kwargs={"kernel_size": 5, "padding": "same"},
        ),
        norm_cfg=LayerConfig(
            name="BatchNorm",
            kwargs={"affine": True},
        ),
        dropout_cfg=LayerConfig(
            name="Dropout",
            kwargs={"p": 0.5},
        ),
        attn_cfg=LayerConfig(
            name="SqueezeExcite",
            kwargs={},
        ),
    )

    cfg = UNetConfig(
        filters=[32, 64, 128, 256],
        pool_cfg=LayerConfig(
            name="MaxPool",
            kwargs={"kernel_size": 2},
        ),
        upsample_cfg=LayerConfig(
            name="Upsample",
            kwargs={
                "scale_factor": 2,
                "mode": "bilinear",
            },
        ),
        block_cfg=block_cfg,
    )

Modifying configurations
------------------------

Configurations can be modified after they have been created. The
:meth:`Config.mod` method creates a modified copy of a configuration while
leaving the original configuration unchanged.

For example:

.. code-block:: python

    block_cfg = ImageConvBlockConfig(
        depth=4,
        activation="LeakyReLU",
    )

    mini_block_cfg = block_cfg.mod(
        depth=2,
        activation="ReLU",
    )


This is particularly useful when several configurations share a common
starting point.

The same approach can be used with model configurations:

.. code-block:: python

    cfg = UNetConfig(
        filters=[32, 64, 128, 256],
    )

    cfg = cfg.mod(
        encoder_block_cfg=ImageConvBlockConfig(depth=2),
        decoder_block_cfg=ImageConvBlockConfig(depth=4),
    )


Configuration presets
---------------------

Many configurations provide methods for applying common architectural
modifications without manually changing individual fields.

For example, convolutional blocks can be configured for reconstruction or
segmentation tasks:

.. code-block:: python

    cfg = ImageConvBlockConfig(
        depth=3,
        activation="ReLU",
    )

    cfg = cfg.reconstruction_mode()


For a U-Net, task-specific presets can be applied to the complete
configuration:

.. code-block:: python

    cfg = UNetConfig(
        filters=[32, 64, 128, 256],
    )

    cfg = cfg.single_class_segmentation_mode()


For multi-class segmentation, the corresponding preset can be used:

.. code-block:: python

    cfg = UNetConfig(
        filters=[32, 64, 128, 256],
    )

    cfg = cfg.multiclass_segmentation_mode()


Other presets can modify the architecture itself. For example, depthwise
separable convolutions and residual connections can be enabled together:

.. code-block:: python

    cfg = UNetConfig(
        filters=[32, 32, 32],
    )

    cfg = cfg.to_depthwise_separable()
    cfg = cfg.add_input_residual()


These methods can also be chained:

.. code-block:: python

    cfg = (
        UNetConfig(filters=[32, 32, 32])
        .to_depthwise_separable()
        .add_input_residual()
    )


Available presets include modifications to convolution types, residual
connections, dilation, attention mechanisms, and task-specific
configurations. Refer to the individual configuration classes for the full
set of available transformations.

Working with residual connections
---------------------------------

``ImageConvBlockConfig``  and ``GraphConvBlockConfig`` provides utilities for configuring residual
connections within convolutional blocks.

For example, an input-to-output residual connection can be added with:

.. code-block:: python

    cfg = ImageConvBlockConfig(depth=3)
    cfg = cfg.add_input_residual()


Residual connections can also use concatenation rather than addition:

.. code-block:: python

    cfg = ImageConvBlockConfig(depth=3)
    cfg = cfg.add_input_concat_residual()


Alternatively, a residual connection can be added from the first convolution
to the final convolution:

.. code-block:: python

    cfg = ImageConvBlockConfig(depth=3)
    cfg = cfg.add_conv1_residual()


Residual connections can also be specified directly using
`residual_connections`:

.. code-block:: python

    cfg = ImageConvBlockConfig(
        depth=4,
        residual_connections={3: [0, 1]},
        residual_type="concat",
    )


The keys identify target layers and the values identify the source layers.
Layer `0` represents the input to the block.
Layer `1` represents the input to the first convolution, and so on. 

To add a residual connection from the input to the output of the final convolution, the following configuration can be used:

.. code-block:: python

    cfg = ImageConvBlockConfig(
        depth=4,
        residual_connections={4: [0]},
        residual_type="add",
    )


Applying modifications across a model
--------------------------------------

Because model configurations contain nested configuration objects, a
modification can be propagated to multiple components.

For example, encoder and decoder blocks can be created from a common base
configuration and modified independently:

.. code-block:: python

    block_cfg = ImageConvBlockConfig(
        depth=3,
        activation="ReLU",
        out_activation="sigmoid",
    )

    encoder_block_cfg = [
        block_cfg.mod(depth=2)
        for _ in range(4)
    ]

    decoder_block_cfg = [
        block_cfg.mod(depth=4)
        for _ in range(4)
    ]

    cfg = UNetConfig(
        filters=[32, 64, 128, 256],
        encoder_block_cfg=encoder_block_cfg,
        decoder_block_cfg=decoder_block_cfg,
    )


This approach is useful when a model requires consistent configuration across
multiple levels while retaining control over individual blocks.

Saving and loading Configurations
---------------------------------

Configurations can be serialised and saved for later use. Nested
configurations, lists, dictionaries, and enumerated values are handled
recursively by the configuration system.

For example, a model configuration can be saved and subsequently restored:

.. code-block:: python

    cfg = UNetConfig(
        filters=[32, 64, 128, 256],
    )

    cfg.save("my_config.yaml")

    loaded_cfg = UNetConfig.load("my_config.yaml")


This makes configurations convenient for experiment management and for
reproducing model architectures across different runs.

The configuration system can also convert configurations to dictionaries:

.. code-block:: python

    config_dict = cfg.to_dict()


and reconstruct them from dictionaries:

.. code-block:: python

    cfg = UNetConfig.from_dict(config_dict)


Summary
-------

The ``configs`` module provides a composable way to define model architectures
and their components. Rather than specifying model parameters directly when
constructing each model, configurations can be built hierarchically and then
modified or transformed using reusable presets.

The general workflow is:

.. code-block:: text

    LayerConfig
        │
        ▼
    ImageConvBlockConfig
        │
        ▼
    Model configuration (e.g. UNetConfig)
        │
        ├── modify with mod()
        ├── apply architectural presets
        └── save/load configuration


This allows the same configuration objects to be reused across experiments
while keeping model architecture definitions explicit and reproducible.

