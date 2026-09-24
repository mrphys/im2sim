`transforms` provides tensor transformations for preprocessing, normalising, scaling, and inverting data used by `im2sim` datasets and training pipelines.

The transforms are designed around a common :class:`~im2sim.data.Transform` interface. A transform wraps an operation and specifies **which data entries, attributes, and channels** the operation should act on. 
This allows the same operation to be applied to tensors, PyG data objects, individual channels, or multiple dataset fields.


Overview
--------

The available transforms fall into three categories:

.. list-table::
:header-rows: 1
:widths: 20 30 30 20


* - Category
  - Purpose
  - Dataset statistics
  - Invertible
* - Simple normalisation
  - Normalise individual inputs
  - Computed from each input
  - No
* - Invertible scaling
  - Change the distribution of values while retaining recoverability
  - Not required
  - Yes
* - Fittable normalisation
  - Apply consistent normalisation across a dataset
  - Learned during ``fit()``
  - Yes


Transform application
---------------------

Transforms are configured with a set of ``keys`` identifying the entries in the data dictionary to which they should be applied.

For example, a tensor stored under ``"image"`` can be normalised with:

.. code-block:: python

    from im2sim.transforms import Norm

    transform = Norm(keys=["image"])


The transform can also operate on an attribute of an object. This is particularly useful for PyTorch Geometric data:

.. code-block:: python

    transform = Norm(
        keys=["graph"],
        attr="x",
    )


Here, the operation is applied to ``graph.x`` rather than replacing the entire ``graph`` object. The ``attr="all"`` option can instead be used when an operation should receive the complete object.

Transforms can additionally be restricted to particular channels:

.. code-block:: python

    transform = Norm(
        keys=["graph"],
        attr="x",
        channels=[0, 1, 2],
    )


By default, selected channels share the same operation. Setting ``per_channel=True`` creates an independent operation for each channel. This distinction is particularly important for fittable transforms, where `per_channel=True` results in separate statistics being fitted for each channel.



Invertible transforms
---------------------

Invertible transforms provide both a forward transformation and an ``inverse()`` operation. 
They are useful when data must be transformed for model training but converted back to its original representation for interpretation, visualisation, or physical evaluation.


Fittable transforms
-------------------

Fittable transforms are intended for datasets where normalisation statistics should be determined from a collection of training samples rather than independently for every sample.

The general workflow is:

.. code-block:: python

    transform = FitZScore(keys=["image"])

    pipeline = Pipeline([
        transform,
    ])

    pipeline.fit(dataset)

    dataset.transforms = pipeline

During fitting, the transform accumulates statistics over the dataset. 
Once fitting has completed, the resulting statistics are reused for subsequent samples.
A fittable transform cannot be applied before it has been fitted.


.. important::

    Fit transforms should be fitted using the **training dataset only**. 
    Fitting using validation or test samples would allow information from those samples to influence the preprocessing statistics.



Per-channel transforms
----------------------

All transform factories support:

``channels``
    Selects which channels are transformed.

``per_channel``
    Creates an independent operation for each selected channel.

``channel_dim``
    Specifies the dimension containing the channels.

For example:

.. code-block:: python

    transform = FitZScore(
        keys=["graph"],
        attr="x",
        channels=[0, 1, 2],
        per_channel=True,
        channel_dim=-1,
    )

With ``per_channel=True``, each channel obtains its own fitted mean and standard deviation rather than sharing statistics across channels.

This is particularly useful when different channels represent different physical quantities or have substantially different numerical scales.


Custom transforms
-----------------

``transform_from_fn`` provides a lightweight way to turn an arbitrary Python function into an ``im2sim`` transform.

For example:

.. code-block:: python

    from im2sim.transforms import transform_from_fn

    transform = transform_from_fn(
        fn=lambda x: x * 1000,
        keys=["length"],
    )

The resulting transform uses the supplied function as its operation while retaining the normal ``Transform`` interface, including key, attribute, and channel selection.

This is useful for simple one-off preprocessing operations that do not justify defining a new ``Operation`` class.

For reusable or more complex operations, defining an ``Operation`` subclass is preferable because operations can then participate in the operation registry and support additional functionality such as inversion or fitting.


Pipelines
---------

Transforms are normally combined using ``Pipeline``.

.. code-block:: python

    from im2sim.data import Pipeline
    from im2sim.transforms import FitZScore, PowerScaling

    pipeline = Pipeline([
        PowerScaling(
            exp=0.5,
            preserve_sign=True,
            keys=["pressure"],
        ),
        FitZScore(
            keys=["pressure"],
        ),
    ])

    pipeline.fit(train_dataset)

    sample = pipeline(sample)

Transforms are applied sequentially in the order in which they are defined. When ``Pipeline.inverse()`` is used, invertible transforms are applied in reverse order. Non-invertible transforms are skipped.

The order of transforms is therefore significant. For example, if a power transformation is intended to reduce skew before standardisation, the power transformation should precede ``FitZScore``.


Choosing a transform
--------------------

The appropriate transform depends primarily on whether the statistics should be determined **per sample** or **from the training dataset**, and whether the transformation needs to be reversible.

.. list-table::
    :header-rows: 1
    :widths: 25 35 40

    * - Transform
      - Use when
      - Main consideration
    * - ``Norm``
      - Each sample should independently occupy ``[0, 1]``
      - Absolute sample-to-sample scale is discarded
    * - ``RangeNorm``
      - Each sample should independently occupy a specified range
      - Statistics are calculated per input
    * - ``ZScore``
      - Each sample should be centred and scaled independently
      - Mean and standard deviation are calculated per input
    * - ``FitNorm``
      - A consistent ``[0, 1]`` scale is required across a dataset
      - Must be fitted on representative training data
    * - ``FitRangeNorm``
      - A consistent arbitrary range is required across a dataset
      - Must be fitted on representative training data
    * - ``FitZScore``
      - A consistent standardised representation is required across a dataset
      - Must be fitted on representative training data
    * - ``PowerScaling``
      - A skewed or large dynamic range should be compressed
      - Transformation can be inverted


Training considerations
-----------------------

For machine-learning applications, the distinction between **sample-wise** and **dataset-wise** normalisation is important.

Sample-wise transforms such as ``Norm`` and ``ZScore`` can remove information about the absolute magnitude of a variable. For example, normalising each CFD sample independently can make two samples with very different pressure ranges appear numerically similar.

Fittable transforms instead preserve the relative scale between samples because the same statistics are used throughout the dataset.

A typical supervised-learning pipeline is therefore:

.. code-block:: python

    pipeline = Pipeline([
        FitZScore(
            keys=["image"],
        ),
        FitZScore(
            keys=["graph"],
            attr="x",
            per_channel=True,
        ),
    ])

    pipeline.fit(train_dataset)

    train_sample = pipeline(train_sample)
    val_sample = pipeline(val_sample)
    test_sample = pipeline(test_sample)

The fitting stage should be performed once using the training data, after which the fitted state should be reused for validation, testing, inference, and deployment.


Saving fitted transforms
------------------------

Transforms and pipelines expose configuration and state information that can be serialised. This allows fitted preprocessing to be saved and restored alongside a trained model.

.. code-block:: python

    from im2sim.data import save_pipeline, load_pipeline

    save_pipeline(pipeline, "pipeline.pt")

    pipeline = load_pipeline("pipeline.pt")

The configuration records the transform type and its arguments, while the state contains fitted operation parameters such as dataset statistics.


Design pattern
--------------

The transform system separates **what an operation does** from **where it is applied**.

An ``Operation`` defines the numerical transformation:

.. code-block:: text

    Operation
        |
        +-- NormOp
        +-- RangeNormOp
        +-- ZScoreOp
        +-- PowerScaleOp
        +-- FitNormOp
        +-- FitRangeNormOp
        +-- FitZScoreOp

A ``Transform`` wraps the operation and handles data selection:

.. code-block:: text

    Transform
        |
        +-- keys
        +-- attr
        +-- channels
        +-- per_channel
        +-- channel_dim
        |
        +-- Operation

This separation means that the same operation can be reused for different dataset fields without duplicating its implementation.
