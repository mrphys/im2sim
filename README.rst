IM2SIM
======

`For documentation, tutorials and examples, please visit`

https://mrphys.github.io/im2sim/


.. start-intro

`im2sim` is a library designed to simplify the development of ML-accelerated
digital twins based on medical images.

This includes two main components:

1. Deep Learning (DL) frameworks based on
   `PyTorch <https://pytorch.org>`_ and
   `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_.

2. Mesh processing frameworks based on
   `VTK <https://vtk.org>`_ and
   `PyVista <https://docs.pyvista.org/index.html>`_.


`Features`

* Image and Mesh Data Processing
* DL models for medical imaging applications
* DL models for digital twin applications
* Hybrid DL models for simulation outputs directly from images
* Building blocks for custom DL models
* Visualisation utilities

`Installation`

To install the base package for imaging tasks, run:

.. code-block:: bash

   pip install im2sim

To install with optional mesh dependencies, run:

.. code-block:: bash

   pip install "im2sim[mesh]"

To install the compiled PyTorch Geometric extensions used for fast GNN ops
(run after PyTorch is installed for your CUDA version), run:

.. code-block:: bash

   pip install im2sim
   im2sim-install-pyg-addons

To install the latest development version from GitHub, run:

.. code-block:: bash

   pip install git+https://github.com/mrphys/im2sim.git

.. end-intro
