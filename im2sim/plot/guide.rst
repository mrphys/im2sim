``plot`` draws 3D results with pyvista, so it needs the ``mesh`` extra
(``pip install -e ".[mesh]"``). :class:`PointCloudPlot` is a quick grid of point clouds for
comparing, for example, predicted and target node positions, or the graph after each iteration
of an ``Im2SimGen2`` model. :class:`Figure` is the general version: it combines meshes, graphs,
streamlines, point clouds and line charts in one image or animation (see `Figures`_ below).

Rendering happens off screen. Before the first render, ``plot`` checks in a subprocess that this
works, and raises a ``RuntimeError`` if not: without a working OpenGL setup VTK would otherwise
crash the Python process. In a container with no display, install OSMesa
(``apt-get install libosmesa6 libgl1-mesa-dri``), and under conda start Python with
``LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`` so Mesa can load. Set
``IM2SIM_SKIP_RENDER_CHECK=1`` to skip the check.

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
* ``elev`` and ``azim`` set the viewing angle, as in matplotlib with ``y`` vertical. ``titles``
  takes one title per subplot or a single string for the whole figure.
* ``figsize`` is in inches; the image is ``figsize * dpi`` pixels (``save_image(..., dpi=200)``).
* Subplots that share a colour scale share one colour bar.

Animating a sequence
--------------------

``animate`` draws one frame per entry of ``point_sequence_sets``. Each entry is a list with one
point cloud per subplot. This is a quick way to watch an iterative model refine a mesh:

.. code-block:: python

    graphs = model(image, graph)  # model built with return_intermediate_graphs=True
    frames = [[g.coords.detach().numpy()] for g in graphs]

    plot = PointCloudPlot(nrows=1, ncols=1, point_sets=frames[-1])
    plot.animate(frames, filename="refinement.gif", fps=2)

``.gif`` files need ``imageio``; other extensions are written as movies and need
``imageio-ffmpeg``. ``draw_frame(point_sets, color_sets)`` replaces the data drawn by the next
``save_image``.

Plotting meshes with edges
--------------------------

:class:`MeshPlot` is the pyvista counterpart of :class:`PointCloudPlot`. It takes the same grid,
``titles``, ``cmap`` and ``norm_mode`` arguments, but draws each subplot as a mesh so its edges
are visible. It needs the ``mesh`` extra (``pip install -e ".[mesh]"``).

Each entry of ``meshes`` is either a pyvista dataset or a ``(points, edge_index)`` tuple, so a
PyTorch Geometric graph can be drawn with its edges directly:

.. code-block:: python

    import pyvista as pv
    from im2sim.plot import MeshPlot

    mesh = pv.read("case-0350.vtu")
    wall = mesh.extract_cells(mesh["CellEntityIds"] == 1).extract_surface()

    plot = MeshPlot(
        nrows=1,
        ncols=2,
        meshes=[wall, (graph.coords, graph.edge_index)],
        scalars=["pressure", graph.pressure[:, 0]],
        titles=["wall", "graph"],
        norm_mode="all",
    )
    plot.save_image("mesh_comparison.png")

* ``scalars`` takes, per subplot, an array with one value per point or per cell, the name of an
  array on the mesh, or ``None`` for a uniform colour. Vector values are drawn as their magnitude.
* ``clim`` fixes the colour limits, overriding ``norm_mode``. Use it for animations so colours
  mean the same thing in every frame.
* The volume of a tetrahedral mesh renders as its outer surface. Use ``style="wireframe"`` to see
  every edge, or clip it first (``mesh.clip(normal="z", crinkle=True)``) to see the cells inside.
* With ``link_views=True`` (the default) all subplots share one camera, fitted to every mesh.
  ``camera_position``, ``elev`` and ``azim`` set the view.
* Rendering is off screen. In a container with no GPU, install ``libosmesa6`` and, under conda,
  start Python with ``LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`` so Mesa can load.

``animate`` takes one entry per frame, each a list with one item per subplot: an ``[N, 3]``
array of new point positions, a new mesh, or ``None`` to keep the geometry. Pass ``None`` for
the whole entry to only change the colours:

.. code-block:: python

    plot = MeshPlot(1, 1, [wall], scalars=[pressures[0]], clim=(p_min, p_max))
    plot.animate([None] * len(pressures), [[p] for p in pressures], filename="cycle.gif")

``.gif`` output needs ``imageio``; other extensions are written as movies with
``imageio-ffmpeg``. Like :class:`PointCloudPlot`, both ``save_image`` and ``animate`` close the
plot.

Figures
-------

:class:`Figure` is the general, configurable way to plot. It separates four things:

* **Data**: each :class:`Panel` takes a sequence of frames. Anything can be given: a pyvista mesh,
  a PyTorch Geometric graph (``pos`` or ``coords``, ``edge_index``, ``face`` and any per-node
  attributes, which become fields), a ``(points, edge_index)`` tuple, an ``[N, 3]`` array, a list
  of these (one per frame), or a :class:`MeshSequence`. A panel with one frame stays still while
  the others animate.
* **Layers**: what to draw from each frame, stacked in order: :class:`Surface`,
  :class:`Wireframe`, :class:`Points`, :class:`Graph` and :class:`Streamlines`. Every layer can
  ``select`` cells (e.g. ``{"CellEntityIds": 1}``), ``clip`` a volume open and colour by a field
  name or a function. Vector fields are coloured by magnitude. A panel can instead hold
  :class:`Trace` layers: a line chart of one value per frame (a field's mean, a function of the
  mesh, or an array), with a marker that moves with the animation.
* **Colour scales**: layers refer to a :class:`ColorScale` by name. Layers with the same name
  share one range and one colour bar, across panels. ``clim`` is ``"frame"`` (rescale every
  frame), ``"global"`` (one range over all frames) or a fixed ``(vmin, vmax)``.
* **Output**: ``save`` picks the format from the extension, ``to_array`` returns an RGB array
  (e.g. for logging), and the figure stays open until ``close``.

.. code-block:: python

    from im2sim.plot import (
        ColorScale, Figure, Graph, Panel, Streamlines, Surface, Trace, VTUSequence,
        vector_field,
    )

    cfd = VTUSequence(
        "Mr_Test_2001-{:04d}.vtu",
        steps=range(301, 401),  # loaded lazily, one file per frame
        fields={"U": vector_field("x-velocity", "y-velocity", "z-velocity")},
    )

    with Figure(
        panels=[
            Panel(cfd, [
                Surface(select={"CellEntityIds": 1}, opacity=0.12, show_edges=False),
                Streamlines("U", seeds={"CellEntityIds": 4}, select={"CellEntityIds": 0},
                            color="U", scale="speed"),
            ], title="CFD"),
            Panel([prediction_graph], [Graph("pressure")], title="prediction"),
            Panel(cfd, [Trace("U", label="mean speed")], title="waveform"),
        ],
        scales={"speed": ColorScale("inferno", clim="global")},
        frame_label="auto",  # the timestep, from the VTUSequence
    ) as fig:
        fig.save("cycle.gif", fps=12)
        fig.save("peak.png", frame=18)
        image = fig.to_array(frame=18)

Field names are stripped of surrounding whitespace when loaded. A new kind of layer subclasses
:class:`Layer` and overrides ``geometry(mesh, source)``, which returns the mesh to draw.

Tests that render are skipped unless ``IM2SIM_RENDER_TESTS=1`` is set, since they need a working
OpenGL setup.
