``transforms`` provides ready-made normalisation and scaling transforms for the pipelines described
in :doc:`data`. Each function in this module builds a :class:`~im2sim.data.Transform` around an
operation from :mod:`im2sim.transforms`, so they all share the same arguments for choosing what
to change.

Choosing what a transform changes
---------------------------------

.. figure:: transforms/diagrams/transform_targeting.svg
   :alt: keys selects the "graph" entry of the sample, attr selects graph.x, and channels selects
         columns 0 to 2. With per_channel=False one operation covers all three columns; with
         per_channel=True each column gets its own operation and statistics.

.. code-block:: python

    from im2sim.transforms import FitZScore

    transform = FitZScore(
        keys=["graph"],      # entry of the sample dict
        attr="x",            # attribute of the graph
        channels=[0, 1, 2],  # columns of graph.x; channel 3 is left alone
        per_channel=True,    # separate mean and std for each of the three
    )

``channel_dim`` (default ``-1``) says which axis holds the channels. That suits graph features
``[N, C]``. For channel-first images, count from the end: ``-4`` for ``[C, D, H, W]`` or ``-3``
for ``[C, H, W]``. A positive index such as ``0`` fails when the transform is fitted, because
fitting sees the batched ``[1, C, D, H, W]`` tensor.

Use ``per_channel=True`` when channels hold different quantities, such as pressure and velocity,
so that each gets its own statistics. With ``per_channel=False``, the selected channels are
treated as one pool of values.

Available transforms
--------------------

.. list-table::
    :header-rows: 1
    :widths: 22 38 20 20

    * - Transform
      - Result
      - Statistics from
      - Invertible
    * - ``Norm``
      - values in ``[0, 1]``
      - each sample
      - no
    * - ``RangeNorm(llim, hlim)``
      - values in ``[llim, hlim]``
      - each sample
      - no
    * - ``ZScore``
      - mean 0, std 1
      - each sample
      - no
    * - ``FitNorm``
      - ``[0, 1]`` over the training set
      - training set
      - yes
    * - ``FitRangeNorm(llim, hlim)``
      - ``[llim, hlim]`` over the training set
      - training set
      - yes
    * - ``FitZScore``
      - mean 0, std 1 over the training set
      - training set
      - yes
    * - ``PowerScaling(exp, preserve_sign)``
      - ``sign(x) · |x|^exp``, which compresses a wide range
      - none
      - yes

Per-sample or fitted?
~~~~~~~~~~~~~~~~~~~~~

The main decision is where the statistics come from. Per-sample transforms (``Norm``,
``RangeNorm``, ``ZScore``) rescale every sample on its own, so differences in magnitude
**between** samples disappear. Fitted transforms (``Fit*``) learn one set of statistics from the
training set and apply it to every sample, so those differences survive.

.. figure:: transforms/diagrams/sample_vs_dataset.svg
   :alt: Two samples with maxima 20 and 100. After Norm both span 0 to 1. After FitNorm fitted
         with maximum 100 they span 0 to 0.2 and 0 to 1.

For physical fields such as CFD pressure, where the absolute level matters, use a fitted
transform. Per-sample transforms suit inputs whose absolute scale is arbitrary, such as MR image
intensities. Per-sample transforms are also not invertible, because the statistics they used are
not kept.

Fitted transforms must be fitted before use, through ``Pipeline.fit`` on the training set (see
:doc:`data`). Using one before fitting raises a ``RuntimeError``.

Ordering
~~~~~~~~

Transforms run in the order they are listed, and ``Pipeline.inverse`` undoes them in reverse. To
reduce skew before standardising, put ``PowerScaling`` first. ``FitZScore`` is then fitted on the
already-scaled values:

.. code-block:: python

    from im2sim.data import Pipeline
    from im2sim.transforms import FitZScore, PowerScaling

    pipeline = Pipeline([
        PowerScaling(exp=0.5, preserve_sign=True, keys=["graph"], attr="pressure"),
        FitZScore(keys=["graph"], attr="pressure"),
    ])
    pipeline.fit(train_dataset)

    sample = pipeline(sample)                # sqrt-scale, then z-score
    original = pipeline.inverse(sample)      # undo z-score, then undo sqrt-scale

``PowerScaling`` adds a small ``eps`` to ``|x|``, so the round trip is close to, but not exactly,
the original values.

One-off transforms
------------------

``transform_from_fn`` wraps any function as a transform with the same ``keys`` / ``attr`` /
``channels`` arguments:

.. code-block:: python

    from im2sim.transforms import transform_from_fn

    to_mm = transform_from_fn(fn=lambda x: x * 1000, keys=["graph"], attr="coords")

Such transforms cannot be inverted or fitted, and a pipeline that contains one cannot be reloaded
with ``load_pipeline``. For anything reusable, write an ``Operation`` subclass and register it with
``@register_op``, as shown in :doc:`data`.
