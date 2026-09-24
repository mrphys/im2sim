im2sim.data
===========

.. automodule:: im2sim.data

Classes
-------

.. autosummary::
    :toctree: data
    :template: data/class.rst
    :nosignatures:

    Dataset
    FittableOperation
    InvertibleOperation
    Operation
    Pipeline
    Transform

Functions
---------

.. autosummary::
    :toctree: data
    :template: data/function.rst
    :nosignatures:

    DataLoader
    collate
    register_op

Guide
=====

``data`` provides the data loading and preprocessing framework used by
``im2sim`` models.

The module is designed around a flexible pipeline in which each sample is
represented as a dictionary of tensors and/or PyTorch Geometric data
objects. Data can then be processed using composable operations and
transforms before being passed to a model.

The main components are:

* ``Dataset`` for defining how individual cases are loaded;
* ``DataLoader`` for batching samples during training;
* ``Operation`` for implementing data-processing operations;
* ``Transform`` for applying operations to selected parts of a sample; and
* ``Pipeline`` for composing multiple transforms into a preprocessing
  workflow.

The individual transforms provided by ``im2sim`` are described separately
in the transforms guide.

Data pipeline
------------

The data module separates loading, preprocessing, and batching:

.. code-block:: text


    Case files
        │
        ▼
    Dataset
        │
        │ load_fn
        ▼
    Sample dictionary
        │
        ├── image
        ├── mask
        ├── graph
        └── other tensors / data
        │
        ▼
    Transform / Pipeline
        │
        ▼
    Processed sample
        │
        ▼
    DataLoader
        │
        ▼
    Batched model input


This separation allows the same dataset definition to be used with
different preprocessing pipelines.

A dataset is responsible for loading a case, while transforms determine
how the loaded data is processed.

Datasets
---------

``Dataset`` provides a lightweight template for creating datasets from a
collection of cases.

A dataset is defined by:

* a list of case identifiers;
* a ``load_fn`` that loads the data associated with a case; and
* an optional set of transforms.

For example:

.. code-block:: python

    from im2sim.data import Dataset

    cases = [
        "case1",
        "case2",
        "case3",
    ]

    def load(case):
        return {
            "image": load_image(case),
            "graph": load_graph(case),
        }

    dataset = Dataset(
        load_fn=load,
        cases=cases,
    )


The ``load_fn`` receives a case identifier and returns a dictionary
containing the data for that case.

The dataset itself does not impose a particular medical-imaging data
format. This allows a dataset to contain whatever combination of images,
masks, meshes, graphs, or other tensors is required by the model.

For example, a sample might contain:

.. code-block:: python

    {
        "image": image,
        "segmentation": segmentation,
        "mesh": mesh,
    }


or:

.. code-block:: python

    {
        "image": image,
        "template": template,
        "out_graph": graph,
    }


This dictionary-based representation allows transforms to operate on
individual components without requiring a specialised dataset class for
every model.



Transforms
---------

``Transform`` provides the interface between an operation and a sample
dictionary.

A transform specifies:

* which sample keys it operates on;
* optionally, which attribute of an object should be modified;
* optionally, which channels should be processed; and
* the operation that should be applied.

For example, an operation can be restricted to a particular sample key:

.. code-block:: python


    transform = Transform(
        op=my_operation,
        keys="image",
    )


Transforms can also operate on multiple keys. This is useful when several
parts of a sample must undergo the same spatial transformation.

The transform layer therefore separates what an operation does from
where it is applied.

The available operations and their specific behaviour are described in
the transforms guide :doc:`transforms`.

Pipelines
--------

``Pipeline`` combines multiple transforms into an ordered preprocessing
sequence.

For example:

.. code-block:: python

    pipeline = Pipeline([
        image_transform,
        spatial_transform,
        normalisation_transform,
    ])

    dataset = Dataset(
        load_fn=load,
        cases=cases,
        transforms=pipeline,
    )


Transforms are applied in the order in which they appear in the
pipeline.

This ordering is important when operations depend on the output of
previous operations. For example, spatial preprocessing may need to occur
before a normalisation operation.

A pipeline can also determine which transforms apply to a particular
sample based on the keys present in that sample. This allows a single
pipeline to contain transforms for different data components without
requiring every sample to contain every possible key.

Invertible preprocessing
-----------------------

Some operations are invertible.

An ``InvertibleOperation`` provides both a forward operation and an
``inverse`` operation. A ``Transform`` wrapping such an operation can
therefore be reversed:

.. code-block:: python

    transformed = transform.forward(sample)

    original = transform.inverse(transformed)


When several invertible transforms are used in a ``Pipeline``, the inverse
pipeline is applied in reverse order.

For example:

.. code-block:: text


    Original data
        │
        ▼
    Transform A
        │
        ▼
    Transform B
        │
        ▼
    Model input

    Model output
        │
        ▼
    inverse B
        │
        ▼
    inverse A
        │
        ▼
    Original space


This is particularly useful for spatial preprocessing where model outputs
need to be mapped back into the original image or physical coordinate
system.

Only operations that provide an inverse are reversed by the pipeline.

Fittable operations
-------------------

Some preprocessing operations need to determine parameters from the
dataset before they can be applied.

These are represented by ``FittableOperation``.

A fittable operation has three stages:

.. code-block:: text

    Dataset
    │
    ▼
    fit_step(...)
    │
    │ repeated over batches
    ▼
    complete_fit()
    │
    ▼
    Fitted operation
    │
    ▼
    Transform data


For example, a normalisation operation may need to estimate statistics
from the training dataset before it can transform individual samples.

A ``Pipeline`` handles this fitting process automatically:

.. code-block:: python

    pipeline = Pipeline([
        transform_a,
        normalisation_transform,
        transform_c,
    ])

    pipeline.fit(dataset)


Transforms earlier in the pipeline are applied before a fittable transform
is fitted. This means that fitting can operate on the same representation
that will subsequently be passed to the model.

Training considerations
~~~~~~~~~~~~~~~~~~~~~~~

Fittable preprocessing should generally be fitted using the training
dataset only.

The fitted state can then be reused for validation and test data. This
prevents information from the validation or test sets from influencing
preprocessing parameters.

A fittable transform also cannot be used for forward or inverse processing
until it has been fitted.

Channel-specific processing
---------------------------

Transforms can optionally operate on selected channels rather than an
entire tensor.

This is useful when different channels represent different physical
quantities and should not undergo identical preprocessing.

For example, a transform can be configured to operate only on selected
channels:

.. code-block:: python

    transform = Transform(
        op=my_operation,
        keys="image",
        channels=[0, 2],
    )


Transforms can also create separate operation instances for individual
channels using ``per_channel=True``.

This distinction is important for multimodal data where channels may have
different units, distributions, or physical meanings.

Working with image and graph data
--------------------------------

The data module is designed to support hybrid image/graph models.

A single sample can contain both dense image tensors and graph objects:

.. code-block:: python

    sample = {
        "image": image,
        "template": template,
        "graph": graph,
    }


This allows a model to combine image-derived information with geometric
or graph-based representations without requiring separate data-loading
systems.

For example, an image-to-simulation model might load:

.. code-block:: text

    Medical image
        │
        ├──────────────┐
        │              │
        ▼              ▼
    Image          Template mesh
        │              │
        │              ▼
        │          Graph data
        │              │
        └───────┬──────┘
                ▼
            Model input


The same sample can therefore carry all of the inputs and targets required
by a hybrid model.

Batching
-------

``DataLoader`` provides a wrapper around PyTorch's standard
``DataLoader`` with an ``im2sim``-specific collate function.

Tensor values are batched using standard PyTorch stacking in dim 0, while PyTorch
Geometric ``Data`` objects are combined into a `PyG Batch <https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html?highlight=batching>`_

For example, a dataset returning:

.. code-block:: python

    {
        "image": image,
        "graph": graph,
    }


produces batches containing:

.. code-block:: python

    {
        "image": batched_images,
        "graph": batched_graph,
    }


This means image and graph data can be loaded together using the same
``DataLoade`r`.

Standard PyTorch data-loading arguments such as ``batch_size``,
``shuffle``, ``num_workers``, and ``pin_memory`` can be passed directly:

.. code-block:: python

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


Serialisation and reproducibility
--------------------------------

Operations and pipelines expose configuration and state information so
that preprocessing can be reproduced.

A pipeline can be represented by its configuration:

.. code-block:: python

    config = pipeline.config()


and its fitted state can be stored separately:

.. code-block:: python

    state = pipeline.state_dict()


The complete pipeline can also be saved and subsequently restored:

.. code-block:: python

    from im2sim.data.core import save_pipeline, load_pipeline

    save_pipeline(
        pipeline,
        "pipeline.pt",
    )

    pipeline = load_pipeline("pipeline.pt")


This is particularly useful for fitted preprocessing, where reproducing
the transformation requires both the definition of the operations and
their fitted state.

Designing a data pipeline
------------------------

A typical ``im2sim`` training pipeline separates four stages:

.. code-block:: text


    1. Load
    │
    ▼
    Dataset
    │
    ▼
    2. Transform
    │
    ├── spatial processing
    ├── intensity processing
    ├── graph processing
    └── other preprocessing
    │
    ▼
    3. Batch
    │
    ▼
    DataLoader
    │
    ▼
    4. Train
    │
    ▼
    Model


The dataset should generally be responsible for finding and loading
data, rather than implementing preprocessing logic.

Preprocessing should instead be expressed through transforms and
pipelines. This keeps the dataset reusable and makes preprocessing
explicit and reproducible.


Training and validation pipelines
---------------------------------

Training, validation, and test datasets can use different pipelines while
sharing the same underlying dataset definition.

For example:

.. code-block:: python

    train_dataset = Dataset(
        load_fn=load,
        cases=train_cases,
        transforms=train_pipeline,
    )

    val_dataset = Dataset(
        load_fn=load,
        cases=val_cases,
        transforms=val_pipeline,
    )


The training pipeline can contain stochastic augmentation, while
validation and test pipelines can use deterministic preprocessing.

Fitted preprocessing parameters should be obtained from the training data
and then reused for validation and test data rather than fitted
independently.

Summary
-------

The ``data`` module provides a common framework for loading and
preprocessing the heterogeneous data used by ``im2sim`` models.

The general workflow is:

.. code-block:: text


    Case identifiers
        │
        ▼
    Dataset
        │
        │ load_fn
        ▼
    Sample dictionary
        │
        ▼
    Pipeline
        │
        ├── Operations
        ├── Invertible operations
        └── Fittable operations
        │
        ▼
    DataLoader
        │
        ▼
    Batched model input


The key design principle is that data loading and data processing are
separate.

``Dataset`` defines how a case is loaded, ``Transform`` determines where
an operation is applied, ``Pipeline`` defines the preprocessing sequence,
and ``DataLoader`` handles batching.

This provides a common data interface for models that operate on images,
graphs, meshes, or combinations of these representations while keeping
the preprocessing pipeline explicit and reusable.


