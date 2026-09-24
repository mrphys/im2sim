im2sim.mesh_ops
===============

.. automodule:: im2sim.mesh_ops

Classes
-------

.. autosummary::
    :toctree: mesh_ops
    :template: mesh_ops/class.rst
    :nosignatures:

    

Functions
---------

.. autosummary::
    :toctree: mesh_ops
    :template: mesh_ops/function.rst
    :nosignatures:

    cluster_pool
    compute_edge_lengths
    get_edges
    get_edges_surf
    get_edges_tet
    get_node_features
    get_structure_cells
    get_structure_edges
    get_structure_ids
    hard_threshold
    make_padded_batch
    rasterize
    set_attrs
    soft_threshold

Guide
=====

``mesh_ops`` provides utilities for manipulating and processing 3D meshes.

.. note::
    ``mesh_ops`` is an optional module within ``im2sim`` and requires the
    additional mesh dependencies to be installed.

    See the installation instructions on the :doc:`../index` for more
    information.

    ``mesh_ops`` is not imported automatically by ``im2sim``. It must be
    imported explicitly before its functions can be used.

    .. code-block:: python

        # This works
        import im2sim.mesh_ops as M

        pv_mesh = pv.read("example_mesh.vtu")
        edge_index = M.get_edges_tet(pv_mesh)

        # This does not work
        import im2sim

        pv_mesh = pv.read("example_mesh.vtu")
        edge_index = im2sim.mesh_ops.get_edges_tet(pv_mesh)


Mesh properties and representations
-----------------------------------

The module is designed to work with PyVista meshes and provides utilities for
converting mesh representations between PyVista and PyTorch Geometric (PyG).

For example, a tetrahedral mesh stored in a `.vtu` file can be converted to
a simple PyG representation as follows:

.. code-block:: python

    import pyvista as pv
    from torch_geometric.data import Data
    import im2sim.mesh_ops as M

    # Load the PyVista mesh from a .vtu file
    pv_mesh = pv.read("example_mesh.vtu")

    coords = torch.tensor(pv_mesh.points, dtype=torch.float32)

    edge_index = M.get_edges_tet(pv_mesh)

    # Convert the PyVista mesh to a PyTorch Geometric representation
    pyg_mesh = Data(coords=coords, edge_index=edge_index)


Surface meshes can be converted in a similar way using
:func:`get_edges_surf`:

.. code-block:: python


    import pyvista as pv
    from torch_geometric.data import Data
    import im2sim.mesh_ops as M

    # Load the PyVista surface mesh from a .vtk file
    pv_mesh = pv.read("example_mesh.vtk")

    coords = torch.tensor(pv_mesh.points, dtype=torch.float32)

    edge_index = M.get_edges_surf(pv_mesh)

    # Convert the PyVista mesh to a PyTorch Geometric representation
    pyg_mesh = Data(coords=coords, edge_index=edge_index)


Working with mesh structures
----------------------------

``mesh_ops`` also provides utilities for working with labelled mesh
substructures, such as vessel walls, inlets, and outlets. These can be used
to identify regions of interest and provide structural information to
downstream machine-learning models.

For example, structure identifiers can be extracted from a PyVista mesh and
stored as attributes on a PyG ``Data`` object:

.. code-block:: python

    import pyvista as pv
    from torch_geometric.data import Data
    import im2sim.mesh_ops as M

    # Load the PyVista mesh
    mesh = pv.read("example_mesh.vtu")

    # Create a PyTorch Geometric Data object to hold the mesh representation
    data = Data()

    # Set the coordinates and edge index of the mesh
    data.coords = torch.from_numpy(mesh.points)
    data.edge_index = M.get_edges_tet(mesh)

    # Map PyVista CellEntityIds to structure names
    structure_ids = {
            1: "wall",
            2: "inlet",
            3: "outlet",
        }

    # Get the structure node indices and set them as attributes
    # on the PyG Data object. The resulting attributes are
    # 'wall_index', 'inlet_index', and 'outlet_index'.
    structure_ids = M.get_structure_ids(mesh, structure_ids)
    M.set_attrs(data, structure_ids)

    # Get the structure cell indices and set them as attributes
    # on the PyG Data object. The resulting attributes are
    # 'wall_cell_index', 'inlet_cell_index', and 'outlet_cell_index'.
    cell_ids = M.get_structure_cells(mesh, structure_ids)
    M.set_attrs(data, cell_ids)

    # Define the feature names to extract. These should correspond to
    # column names in the VTU file's point_data.
    feature_names = ["pressure", "velocity"]

    # Create a 'cfd' attribute to hold the node features.
    # The resulting tensor has shape (num_nodes, num_features).
    data.cfd = M.get_node_features(mesh, feature_names)


In addition to structure extraction, ``mesh_ops`` supports loading nodal data
from VTU files and assigning attributes recursively to PyG ``Data`` objects.
This is useful for constructing graph representations that combine mesh
geometry, anatomical structures, boundary conditions, material properties,
and simulation outputs.

These utilities can therefore be used to prepare PyVista meshes for use as
inputs or targets in machine-learning workflows based on PyTorch Geometric.


