# ==============================================================================
# Copyright 2026 University College London.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Lab: controlled experiments on what a GNN can do.

Experiments are grouped by the question they answer:

* :mod:`~im2sim.lab.reach`: how far do messages travel?
* :mod:`~im2sim.lab.geometry`: how well does the model understand geometry?
* :mod:`~im2sim.lab.sharpness`: how sharply can the signal change?

All experiments take a :class:`ModelAdapter` (or, for trained ones, a factory building one),
so im2sim models and plain PyG models are tested identically, on toy graphs from
:mod:`~im2sim.lab.graphs` or on real graphs via :func:`~im2sim.lab.graphs.from_mesh_graph`.
Results are :class:`LabResult` objects that :mod:`~im2sim.lab.viz` knows how to plot.
"""

from im2sim.lab import geometry, graphs, reach, sharpness, viz
from im2sim.lab.adapters import AdapterFactory, ModelAdapter, dummy_image
from im2sim.lab.geometry import (
    equivariance_error,
    geometric_targets,
    linear_probe,
    proximity_leakage,
    resolution_invariance,
)
from im2sim.lab.graphs import (
    barbell_graph,
    from_mesh_graph,
    grid_graph,
    path_graph,
    resample,
    ring_graph,
    tree_graph,
    tube_graph,
    u_tube_graph,
)
from im2sim.lab.reach import copy_task, harmonic_extension, jacobian_reach
from im2sim.lab.results import LabResult, load_results, results_to_frame, save_results
from im2sim.lab.runner import run_suite
from im2sim.lab.sharpness import boundary_layer, spectral_gain, step_sweep

__all__ = [
    "AdapterFactory",
    "LabResult",
    "ModelAdapter",
    "barbell_graph",
    "boundary_layer",
    "copy_task",
    "dummy_image",
    "equivariance_error",
    "from_mesh_graph",
    "geometric_targets",
    "geometry",
    "graphs",
    "grid_graph",
    "harmonic_extension",
    "jacobian_reach",
    "linear_probe",
    "load_results",
    "path_graph",
    "proximity_leakage",
    "reach",
    "resample",
    "resolution_invariance",
    "results_to_frame",
    "ring_graph",
    "run_suite",
    "save_results",
    "sharpness",
    "spectral_gain",
    "step_sweep",
    "tree_graph",
    "tube_graph",
    "u_tube_graph",
    "viz",
]
