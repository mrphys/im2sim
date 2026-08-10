im2sim
======

``im2sim`` is a library designed to simplify the development of ML-accelerated
digital twins based on medical images.

This includes two main components:

1. Deep Learning (DL) frameworks based on
   `PyTorch <https://pytorch.org>`_ and
   `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_

2. Mesh processing frameworks based on
   `VTK <https://vtk.org>`_ and
   `PyVista <https://docs.pyvista.org/index.html>`_

Features
--------

* Image and Mesh Data Processing
* DL models for medical imaging applications
* DL models for digital twin applications
* Hybrid DL models for simulation outputs directly from images
* Building blocks for custom DL models
* Visualisation utilities

Installation
------------

1. Install the CUDA dependencies:

.. code-block:: bash

   pip install torch-scatter torch-cluster \
      -f https://data.pydata.org/whl/torch-2.3.1+cu121.html

2. Install the repository:

.. code-block:: bash

   pip install git+https://github.com/mrphys/im2sim.git