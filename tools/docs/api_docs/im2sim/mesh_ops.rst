im2sim.mesh_ops
===============

`mesh_ops` is a set of utilities that allow for the manipulation of 3D meshes. 
It includes functions for computing various properties of meshes such as normals, curvature, and surface area. 

It is specifically compatible with Pyvista pointgrid objects and can be used to convert mesh representations between PyVista and Pytorch Geometric formats.

For example if you have a PyVista mesh saved as a .vtu file, you can follow the worksflow below to convert it to a simple Pytorch Geometric mesh representation:

.. code-block:: python

    import pyvista as pv
    from torch_geometric.data import Data
    import im2sim.mesh_ops as M

    # Load the PyVista mesh from a .vtu file
    pv_mesh = pv.read("example_mesh.vtu")

    coords = torch.tensor(pv_mesh.points, dtype=torch.float32)

    edge_index = M.get_edges_tet(pv_mesh)

    # Convert the PyVista mesh to a Pytorch Geometric mesh representation
    pyg_mesh = Data(coords=coords, edge_index=edge_index)






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
im2sim.mesh_ops
===============

`mesh_ops` is a set of utilities that allow for the manipulation of 3D meshes. 
It includes functions for computing various properties of meshes such as normals, curvature, and surface area. 

It is specifically compatible with Pyvista pointgrid objects and can be used to convert mesh representations between PyVista and Pytorch Geometric formats.

For example if you have a PyVista mesh saved as a .vtu file, you can follow the worksflow below to convert it to a simple Pytorch Geometric mesh representation:

.. code-block:: python

    import pyvista as pv
    from torch_geometric.data import Data
    import im2sim.mesh_ops as M

    # Load the PyVista mesh from a .vtu file
    pv_mesh = pv.read("example_mesh.vtu")

    coords = torch.tensor(pv_mesh.points, dtype=torch.float32)

    edge_index = M.get_edges_tet(pv_mesh)

    # Convert the PyVista mesh to a Pytorch Geometric mesh representation
    pyg_mesh = Data(coords=coords, edge_index=edge_index)






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
