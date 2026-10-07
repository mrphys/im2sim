im2sim.plot
===========

.. automodule:: im2sim.plot

Classes
-------

.. autosummary::
    :toctree: plot
    :template: plot/class.rst
    :nosignatures:

    PointCloudPlot

Functions
---------

.. autosummary::
    :toctree: plot
    :template: plot/function.rst
    :nosignatures:

    

Guide
=====

``plot`` contains :class:`PointCloudPlot`, a grid of 3D scatter plots for comparing point clouds,
such as predicted and target node positions, or the graph after each iteration of an
``Im2SimGen2`` model.

Comparing point clouds
----------------------

Pass one ``[N, 3]`` array per subplot, filled row by row, and optionally one value per point to
colour by:

.. code-block:: python

    from im2sim.plot import PointCloudPlot

    plot = PointCloudPlot(
        nrows=1,
        ncols=2,
        point_sets=[target.coords.numpy(), prediction.coords.detach().numpy()],
        color_sets=[target.pressure[:, 0].numpy(), prediction.pressure[:, 0].detach().numpy()],
        titles=["target", "prediction"],
        norm_mode="all",
    )
    plot.save_image("comparison.png")

* ``norm_mode`` chooses which subplots share a colour scale: ``"all"``, ``"row"``, ``"col"`` or
  ``"none"`` (each subplot scaled on its own). Use ``"all"`` when colours should be comparable
  across subplots.
* The axis limits of every subplot are taken from the **last** point cloud in ``point_sets``, so
  put the reference geometry last if the clouds differ in extent.
* ``elev`` and ``azim`` set the viewing angle. ``titles`` takes one title per subplot or a single
  string for the whole figure.

Animating a sequence
--------------------

``animate`` draws one frame per entry of ``point_sequence_sets``. Each entry is a list with one
point cloud per subplot. This is a quick way to watch an iterative model refine a mesh:

.. code-block:: python

    graphs = model(image, graph)  # model built with return_intermediate_graphs=True
    frames = [[g.coords.detach().numpy()] for g in graphs]

    plot = PointCloudPlot(nrows=1, ncols=1, point_sets=frames[-1])
    plot.animate(frames, filename="refinement.gif", fps=2)

Files ending in ``.gif`` are written with Pillow; other extensions need ``ffmpeg``. Both
``save_image`` and ``animate`` close the figure, so create a new ``PointCloudPlot`` for each
output.

