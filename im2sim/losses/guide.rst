``losses`` provides differentiable objectives for training models across
image, graph, point-cloud, and mesh prediction tasks.

The losses are grouped according to the type of prediction being trained.
Each group makes different assumptions about the representation of the
prediction and target, and therefore has different requirements when used
during training.

The main loss categories are:

* confusion losses for image segmentation;
* SSIM loss for image reconstruction;
* feature losses for comparing graph-based fields;
* point-cloud losses for comparing predicted and target geometry; and
* mesh losses for controlling the quality and validity of predicted
  meshes.

Losses can also be combined to provide multiple training objectives. This
is particularly useful for mesh-based models, where geometric accuracy
and mesh quality may need to be optimised simultaneously.

Loss categories

---------------

The different loss categories are intended for different types of
prediction:

.. code-block:: text

    Prediction
        │
        ├── Image segmentation
        │      └── Confusion losses
        │
        ├── Image reconstruction
        │      └── SSIM loss
        │
        ├── Graph / field prediction
        │      └── KNN feature loss
        │
        ├── Point-cloud geometry
        │      └── Chamfer loss
        │
        └── Mesh deformation / generation
            ├── Chamfer loss
            ├── Edge-length deviation
            ├── Aspect ratio
            ├── Face normals
            └── Inversion


Image segmentation

-----------------

The confusion losses are intended for segmentation models where the
prediction and target are spatial tensors containing class probabilities
and labels.

They include:

.. code-block:: text

    ConfusionLoss
        ├── DiceLoss
        ├── TverskyLoss
        ├── FocalTverskyLoss
        └── IoULoss


These losses are based on soft versions of the confusion matrix rather
than treating every voxel or pixel independently. This makes them useful
for segmentation problems where the overlap between the predicted and
target regions is more important than the absolute number of correctly
classified background pixels.

They are particularly useful for highly imbalanced segmentation problems,
where the foreground may occupy only a small fraction of the image.

The losses support binary, multiclass, and multilabel segmentation. The
prediction should therefore be supplied as a channel-first tensor:

.. code-block:: text

    [batch, channels, *spatial_dimensions]


The class reduction strategy can be selected using ``"micro"``, ``"macro"``
or ``"weighted"`` averaging.

For example:

.. code-block:: python

    from im2sim.losses import DiceLoss

    loss_fn = DiceLoss()

    loss = loss_fn(
        target,
        prediction,
    )


Training considerations
~~~~~~~~~~~~~~~~~~~~~~~~

Confusion losses optimise overlap, rather than voxel-wise accuracy.
They should therefore generally be interpreted alongside a suitable
segmentation metric such as Dice or IoU.

The choice of class reduction is important for multiclass problems.
``"macro"`` gives each class equal importance, whereas ``"micro"`` is
dominated by the total number of pixels or voxels.

For strongly imbalanced problems, class weighting or a Tversky-style loss
can be used to change the relative importance of false positives and false
negatives.

These losses also require predictions that represent probabilities or
soft class assignments rather than discrete class labels.


Image reconstruction
-------------------

``SSIMLoss`` is intended for image-to-image reconstruction tasks where
preserving local image structure is important.

Unlike a point-wise loss such as L1 or L2, SSIM compares local image
statistics and therefore considers luminance, contrast, and structural
similarity.

It is useful for problems such as:

* image reconstruction;
* image enhancement;
* learned image restoration; and
* undersampled-image reconstruction.

For example:

.. code-block:: python

    from im2sim.losses import SSIMLoss

    loss_fn = SSIMLoss()

    loss = loss_fn(
        prediction,
        target,
    )


The loss operates on channel-first tensors. The spatial dimensionality is
specified using ``rank``:

.. code-block:: text

    rank=2    [B, C, H, W]
    rank=3    [B, C, D, H, W]


It can also be used to apply a 2D SSIM calculation independently across
slices of a higher-dimensional image.

Training considerations
~~~~~~~~~~~~~~~~~~~~~~~

SSIM is sensitive to the dynamic range of the input images. ``max_val``
should therefore correspond to the range in which the images are
represented.

The Gaussian filtering parameters also affect the spatial scale over
which structural similarity is measured.

SSIM is a structural loss rather than a direct pixel-wise error. It can
therefore preserve image structure while allowing small local intensity
differences that would be penalised more strongly by L1 or L2 losses.

For reconstruction problems, SSIM can also be combined with a point-wise
loss when both structural similarity and intensity accuracy are required.

Graph and field prediction
---------------------------

``KnnFeatureLoss`` is intended for predicting features defined on graphs
when the nodes of the predicted and target graphs are not in direct
correspondence.

This is particularly relevant to mesh-based models where the geometry
changes during prediction. The predicted mesh may contain the same
physical domain as the target mesh while having different node positions
or connectivity.

Instead of comparing feature ``i`` on one graph with feature ``i`` on
the other graph, the loss uses the spatial coordinates of the graphs to
interpolate target features onto the predicted nodes using
k-nearest-neighbour interpolation.

Conceptually:

.. code-block:: text

    target graph
        │
        │ KNN interpolation
        ▼
    predicted coordinates
        │
        ▼
    feature comparison


For example:

.. code-block:: python

    from im2sim.losses import KnnFeatureLoss

    loss_fn = KnnFeatureLoss(
        mode="l1",
        k=3,
    )

    loss = loss_fn(
        target_graph,
        prediction_graph,
    )


The graphs must therefore provide coordinates and graph features. The graph coordinates must be stored in the ``coords`` attribute. 
The feature tensor defaults to ``x`` but another graph attribute can be selected using ``feature_key``.

Training considerations
~~~~~~~~~~~~~~~~~~~~~~~~

The main advantage of this loss is that node correspondence is not
required.

This makes it suitable for models where graph geometry is being predicted
or deformed at the same time as graph features.

The value of ``k`` controls the neighbourhood used to interpolate the
target features. A small value makes the interpolation more local, while
larger values provide a broader neighbourhood.

The loss can use either L1 or L2 feature differences.

It is important that the coordinates of the two graphs use the same
physical coordinate system. Otherwise, the spatial interpolation does not
represent a meaningful correspondence.


Point cloud geometry
--------------------

``ChamferLoss`` is intended for comparing the geometry of two point
clouds or graph-based meshes when the individual points are not directly
corresponding.

It measures the nearest-neighbour distance between the two sets of
coordinates in both directions:

This makes it suitable for mesh deformation and geometry prediction where
the number or arrangement of nodes can change.

For example:

.. code-block:: python

    from im2sim.losses import ChamferLoss

    loss_fn = ChamferLoss()

    loss = loss_fn(
        target_graph,
        prediction_graph,
    )


The graphs require coordinate information and batching information.

A subset of points can optionally be selected using ``id_key``. This can
be useful when only a particular part of the geometry should contribute
to the loss.

Training considerations
~~~~~~~~~~~~~~~~~~~~~~~

Chamfer loss measures geometric proximity, rather than mesh quality.

A prediction can therefore achieve a low Chamfer loss while still having
poor element quality, distorted triangles or tetrahedra, or inverted
elements.

For mesh deformation models, Chamfer loss is consequently often most
useful as one component of a larger objective.

It is also insensitive to point-to-point correspondence, which makes it
appropriate when the predicted and target meshes have different node
locations.

Mesh quality and validity
-------------------------

The mesh losses are intended for models that predict or deform meshes.

Unlike ``ChamferLoss``, which measures where the mesh is located, these
losses constrain properties of the mesh itself:

.. code-block:: text

    Mesh prediction
        │
        ├── geometric position
        │      └── ChamferLoss
        │
        ├── edge distribution
        │      └── EdgeLengthDeviationLoss
        │
        ├── element shape
        │      └── AspectRatioLoss
        │
        ├── surface smoothness
        │      └── FaceNormalLoss
        │
        └── element validity
                └── InversionLoss


These losses are particularly useful as regularisation terms when a model
is free to move mesh nodes and therefore has no guarantee that the
resulting mesh remains well-conditioned.


Supervised and unsupervised mesh losses
---------------------------------------

Several mesh losses support both supervised and unsupervised operation.

In supervised mode, the predicted mesh is compared with a target mesh. The
loss can therefore penalise degradation relative to the target.

In unsupervised mode, the loss is calculated from the predicted mesh
alone. This allows mesh quality constraints to be used even when a target
mesh is unavailable.

For example:

.. code-block:: python


    from im2sim.losses import EdgeLengthDeviationLoss

    loss_fn = EdgeLengthDeviationLoss(
        supervised=False,
    )

    loss = loss_fn(
        None,
        prediction_graph,
    )


This distinction is useful when separating the objectives of a mesh
prediction model:

.. code-block:: text

    supervised
        │
        ├── match target geometry
        └── preserve target mesh properties

    unsupervised
        │
        └── prevent invalid / poor-quality geometry


Mesh loss requirements
----------------------

The different mesh losses require different graph attributes.

.. list-table::
    :header-rows: 1

    * - Loss
      - Required information
      - Main purpose
    * - ``ChamferLoss``
      - ``coords``, ``batch``
      - Geometric position
    * - ``EdgeLengthDeviationLoss``
      - ``coords``, ``edge_index``
      - Edge-length regularity
    * - ``AspectRatioLoss``
      - ``coords``, tetrahedral cell connectivity
      - Element shape
    * - ``FaceNormalLoss``
      - ``coords``, triangular face connectivity
      - Surface geometry
    * - ``InversionLoss``
      - ``coords``, tetrahedral cell connectivity
      - Element validity


The names of the connectivity attributes are configurable using the
corresponding ``edge_key``, ``cell_key`` or ``face_key`` arguments.

Combining losses during training
--------------------------------

The loss categories are generally complementary rather than alternatives.

For example, a mesh deformation model may need to satisfy three different
objectives:

.. code-block:: text

        Target geometry
            │
            ▼
        ChamferLoss

        Mesh regularity
            │
            ▼
        EdgeLengthDeviationLoss
            +
        AspectRatioLoss

        Mesh validity
            │
            ▼
        InversionLoss


These objectives can be combined using weighting factors:

.. code-block:: python

    geometry_loss = chamfer(
        target_graph,
        prediction_graph,
    )

    quality_loss = edge_loss(
        None,
        prediction_graph,
    )

    validity_loss = inversion_loss(
        None,
        prediction_graph,
    )

    loss = (
        geometry_loss
        + 0.1 * quality_loss
        + 0.1 * validity_loss
    )


The relative weighting determines the trade-off between matching the
target and enforcing geometric constraints.

In practice, this means that a mesh loss should not generally be selected
in isolation. A geometry loss can encourage the predicted mesh to reach
the target while a quality or validity loss prevents the deformation from
producing undesirable elements.

Loss selection by problem
-------------------------

The following provides a general guide for selecting a loss category:

.. list-table::
    :header-rows: 1

    * - Problem
      - Suitable losses
      - Main consideration
    * - Binary / multiclass segmentation
      - Dice, Tversky, Focal Tversky, IoU
      - Class imbalance and false-positive / false-negative weighting
    * - Image reconstruction
      - SSIM
      - Preserve local image structure and dynamic range
    * - Graph field prediction
      - KNN feature loss
      - No direct node correspondence required
    * - Point-cloud / mesh geometry
      - Chamfer
      - Measures position, not mesh quality
    * - Mesh deformation
      - Chamfer + mesh losses
      - Geometry and mesh quality must be balanced
    * - Mesh validity
      - Inversion
      - Penalises invalid tetrahedral elements


Summary
--------

The ``losses`` module groups training objectives according to the
representation being predicted.

The general workflow is:

.. code-block:: text

    Image
    │
    ├── segmentation ──► confusion losses
    │
    └── reconstruction ──► SSIMLoss

    Graph
    │
    └── field prediction ──► KnnFeatureLoss

    Geometry
    │
    ├── point / mesh position ──► ChamferLoss
    │
    └── mesh quality
            ├── edge lengths
            ├── aspect ratio
            ├── face normals
            └── element validity

The key distinction is between losses that measure agreement with a
target and losses that impose constraints on the prediction itself.

For image and feature prediction, the losses primarily measure agreement
with a target. For mesh-based models, geometry losses such as Chamfer can
be combined with mesh-quality and validity losses to constrain the
predicted geometry during training.

When combining losses, the relative weighting of each objective becomes
part of the model's training configuration and should be selected
according to the desired balance between prediction accuracy and
geometric regularity.

