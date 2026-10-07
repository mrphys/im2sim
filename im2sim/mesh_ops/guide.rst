``mesh_ops`` converts PyVista meshes into PyTorch Geometric graphs and provides small mesh
utilities used by the losses and models.

.. note::

    ``mesh_ops`` needs the optional ``mesh`` dependencies (see the installation instructions on
    :doc:`/index`). It is not imported by ``import im2sim``, so import it explicitly:

    .. code-block:: python

        import im2sim.mesh_ops as M     # works

        import im2sim
        im2sim.mesh_ops.get_edges_tet   # AttributeError unless im2sim.mesh_ops was imported

From a PyVista mesh to a graph
------------------------------

A simulation mesh stored as ``.vtu`` carries node positions, cells, a ``CellEntityIds`` array
that labels each cell with the structure it belongs to, and point data such as pressure. Each
``mesh_ops`` function turns one of these into graph attributes:

.. figure:: mesh_ops/diagrams/mesh_to_pyg.svg
   :alt: points become coords, tetrahedral cells become edge_index through get_edges_tet,
         CellEntityIds become per-structure node and cell indices, and point data becomes a
         node feature tensor.

.. code-block:: python

    import pyvista as pv
    import torch
    from torch_geometric.data import Data

    import im2sim.mesh_ops as M

    mesh = pv.read("example_mesh.vtu")

    data = Data()
    data.coords = torch.from_numpy(mesh.points).float()
    data.edge_index = M.get_edges_tet(mesh)

    # Map CellEntityIds to structure names.
    names = {1: "wall", 2: "inlet", 3: "outlet"}

    # Node indices per structure: data.wall_index, data.inlet_index, data.outlet_index
    M.set_attrs(data, M.get_structure_ids(mesh, names))

    # Cell connectivity per structure, [3, M] for triangles:
    # data.wall_cell_index, data.inlet_cell_index, data.outlet_cell_index
    M.set_attrs(data, M.get_structure_cells(mesh, names))

    # Node features [N, F], one column per scalar point-data array.
    data.cfd = M.get_node_features(mesh, ["pressure", "velocity_x", "velocity_y", "velocity_z"])

Points to watch:

* ``get_edges_tet`` takes its edges from the cells with ``CellEntityIds == 0``, which is
  where tetrahedral meshers usually put the volume. For surface meshes without entity ids, such
  as ``.vtk`` surfaces, use ``get_edges_surf``.
* ``get_structure_cells`` raises an error if an id in ``names`` does not occur in the mesh.
* ``get_node_features`` expects one **scalar** array per name. Split vector fields into
  components first.

The attributes these functions produce are the inputs used elsewhere in ``im2sim``:

* ``coords`` is read by ``TrilinearProjection``, ``MaskRasterizer`` and every graph loss.
* ``*_index`` attributes can be passed as ``include_ids`` / ``exclude_ids`` to the graph decoder
  (see :doc:`models`), or as ``id_key`` to ``ChamferLoss``.
* ``*_cell_index`` attributes can be passed as ``cell_key`` / ``face_key`` to the mesh quality
  losses (see :doc:`losses`).

Other utilities
---------------

``get_structure_edges``
    Edge indices per structure, like ``get_structure_ids`` but returning ``*_edge_index``.
``compute_edge_lengths``
    Euclidean length of each edge in an ``edge_index``.
``make_padded_batch``
    Converts concatenated batched nodes into a padded ``[B, max_nodes, C]`` tensor and a mask.
``cluster_pool``
    Coarsens a mesh graph with Graclus clustering, weighted by inverse edge length.
``rasterize``
    Distance from every voxel centre of a grid to the nearest point of a point cloud.
``hard_threshold``, ``soft_threshold``
    Turn such a distance map into a mask: a step, or a differentiable sigmoid.
