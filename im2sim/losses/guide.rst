``losses`` provides training objectives for images, graph features, point clouds and meshes. They
fall into two kinds:

* **agreement losses** compare a prediction with a target: the confusion losses, ``SSIMLoss``,
  ``KnnFeatureLoss`` and ``ChamferLoss``;
* **quality losses** constrain the predicted mesh itself, with or without a target:
  ``EdgeLengthDeviationLoss``, ``AspectRatioLoss``, ``FaceNormalLoss`` and ``InversionLoss``.

.. list-table:: Choosing a loss
    :header-rows: 1
    :widths: 26 34 40

    * - Prediction
      - Loss
      - Main consideration
    * - Segmentation
      - ``DiceLoss``, ``IoULoss``, ``TverskyLoss``, ``FocalTverskyLoss``
      - class imbalance; weighting of false positives against false negatives
    * - Image reconstruction
      - ``SSIMLoss``
      - local structure; ``max_val`` must match the data range
    * - Field on a graph
      - ``KnnFeatureLoss``
      - no node correspondence needed; shared coordinate system required
    * - Node positions
      - ``ChamferLoss``
      - measures position only, not mesh quality
    * - Mesh deformation
      - ``ChamferLoss`` + quality losses
      - balance accuracy against element quality

.. important::

    Argument order differs between losses. ``SSIMLoss`` is called as ``loss(prediction, target)``.
    Every other loss is called as ``loss(target, prediction)``. For the graph losses, the target is
    ignored (pass ``None``) when the loss is unsupervised.

Segmentation
------------

The confusion losses build a soft confusion matrix from probabilities, per batch element and
channel, and turn it into an overlap score. They optimise overlap rather than per-voxel accuracy,
so a small foreground counts as much as a large background.

.. code-block:: python

    from im2sim.losses import DiceLoss, TverskyLoss

    loss_fn = DiceLoss(average="macro")
    loss = loss_fn(target, prediction)  # both [B, C, *spatial], prediction as probabilities

* Inputs are channel-first, ``[B, C, *spatial]``, with one channel per class, and the target and
  prediction must have the same shape. Predictions must be probabilities, for example after
  ``sigmoid`` or ``softmax``, not class labels.
* ``average`` sets how classes are combined. ``"macro"`` gives each class equal weight.
  ``"micro"`` pools all voxels, so large classes dominate. ``"weighted"`` uses ``class_weights``
  if they are given, and otherwise weights each class by its share of the target voxels.
* ``DiceLoss`` and ``IoULoss`` weight false positives and false negatives equally. The Tversky
  losses let you weight them differently, and ``FocalTverskyLoss`` also emphasises hard examples.

Image reconstruction
--------------------

``SSIMLoss`` returns ``1 - SSIM``. SSIM compares local means, variances and covariances under a
Gaussian window, so it rewards preserved structure rather than exact intensities.

.. code-block:: python

    from im2sim.losses import SSIMLoss

    loss_fn = SSIMLoss(max_val=1.0, rank=3)
    loss = loss_fn(prediction, target)  # [B, C, D, H, W]

* ``max_val`` must match the range of the data: ``1.0`` for images in ``[0, 1]``.
* ``filter_size`` and ``filter_sigma`` set the spatial scale of the comparison.
* ``rank=2`` on a 3D tensor computes 2D SSIM slice by slice.
* SSIM tolerates small intensity offsets. Add an L1 term when absolute intensities matter.

Fields on graphs
----------------

``KnnFeatureLoss`` compares node features on two graphs whose nodes do not correspond, for
example a predicted pressure field on a deformed mesh against a simulation on the true mesh.

.. figure:: losses/diagrams/knn_feature_loss.svg
   :alt: Target and predicted nodes at different positions. For one predicted node, the three
         nearest target nodes are interpolated with inverse squared distance weights and
         compared with the predicted value.

.. code-block:: python

    from im2sim.losses import KnnFeatureLoss

    loss_fn = KnnFeatureLoss(mode="l1", k=3, feature_key="pressure")
    loss = loss_fn(target_graph, prediction_graph)

Both graphs need ``coords`` and the ``feature_key`` attribute, in the same coordinate system.
``feature_channels`` restricts the comparison to some channels. Small ``k`` keeps the
interpolation local, and larger ``k`` smooths it. With batched graphs, neighbours are searched
only within the same sample.

Point positions
---------------

``ChamferLoss`` measures how close two point sets are without matching points one-to-one. The
two sets may have different sizes.

.. figure:: losses/diagrams/chamfer.svg
   :alt: Left: each target point with an arrow to its nearest predicted point. Right: each
         predicted point with an arrow to its nearest target point. The loss is the sum of the two
         mean arrow lengths.

.. code-block:: python

    from im2sim.losses import ChamferLoss

    loss_fn = ChamferLoss()                       # all nodes
    wall_fn = ChamferLoss(id_key="wall_index")    # only the wall nodes on both sides
    loss = loss_fn(target_graph, prediction_graph)

Both directions are needed. The target-to-prediction term alone is low when every target point
has a nearby prediction, even if extra predicted points are scattered elsewhere. The other term
alone is low when predictions sit on the target but leave parts of it uncovered.

Chamfer says nothing about the elements between the nodes. A mesh can match the target surface
closely and still contain slivers or inverted tetrahedra. For mesh models, pair it with quality
losses.

Mesh quality
------------

.. figure:: losses/diagrams/mesh_quality.svg
   :alt: Four panels with a lower-loss and a higher-loss example each: an even versus uneven
         triangle strip, an equilateral versus sliver triangle, a flat cap with parallel normals
         versus a warped cap, and a correctly oriented versus inverted element.

.. list-table::
    :header-rows: 1
    :widths: 26 30 44

    * - Loss
      - Needs on the graph
      - Measures
    * - ``EdgeLengthDeviationLoss``
      - ``coords``, ``edge_index``
      - std / mean of all edge lengths
    * - ``AspectRatioLoss(cell_key)``
      - ``coords``, tetrahedra ``[4, M]``
      - mean over tetrahedra of longest edge / mean edge (1 for a regular tetrahedron)
    * - ``FaceNormalLoss(face_key)``
      - ``coords``, ``batch``, triangles ``[3, M]``
      - spread of the unit face normals; zero for a flat set of faces
    * - ``InversionLoss(cell_key, min_vol)``
      - ``coords``, tetrahedra ``[4, M]``
      - mean of ``max(0, min_vol - V)`` over signed tetrahedron volumes ``V``

``cell_key`` and ``face_key`` name the graph attribute that holds the connectivity, such as the
``*_cell_index`` attributes created by :doc:`mesh_ops`.

``FaceNormalLoss`` pulls faces towards a **common plane**, not towards a smooth surface. Apply it
to faces that should be flat, such as inlet and outlet caps, not to a whole curved vessel wall.

Supervised and unsupervised
~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``supervised=False``, the first three losses ignore the target and return the measure for
the prediction alone. Pass ``None`` as the target. With ``supervised=True``, the default:

* ``EdgeLengthDeviationLoss`` and ``AspectRatioLoss`` return
  ``relu(measure(pred) - measure(target))²``. The prediction is penalised only for being **worse**
  than the target, never for being better.
* ``FaceNormalLoss`` adds the distance between the mean face normals of the two meshes to the
  prediction's own spread.

``InversionLoss`` is always unsupervised, and ``ChamferLoss`` is always supervised.

Combining losses
----------------

For a model that deforms a template mesh, a typical objective has a term for where the mesh is
and terms for what it looks like:

.. code-block:: python

    from im2sim.losses import ChamferLoss, EdgeLengthDeviationLoss, InversionLoss

    chamfer = ChamferLoss()
    edges = EdgeLengthDeviationLoss(supervised=True)
    inversion = InversionLoss(cell_key="vol_cell_index")

    loss = (
        chamfer(target_graph, prediction_graph)
        + 0.1 * edges(target_graph, prediction_graph)
        + 0.1 * inversion(None, prediction_graph)
    )

The weights trade accuracy against element quality and are part of the training configuration.
Too little regularisation leaves inverted or degenerate elements that break later simulation.
Too much stops the mesh from reaching the target.
