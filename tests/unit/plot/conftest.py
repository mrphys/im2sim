import os

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

# Rendering needs a working OpenGL setup (a display, EGL or OSMesa), which CI may not have.
render = pytest.mark.skipif(
    os.environ.get("IM2SIM_RENDER_TESTS") != "1",
    reason="set IM2SIM_RENDER_TESTS=1 to run tests that render",
)


def make_box(speed=1.0, pressure_offset=0.0):
    """A tetrahedral box: x < 2 is wall (id 1), the rest volume (id 0), flow along +x."""
    box = pv.ImageData(dimensions=(6, 4, 4)).triangulate()
    centers = box.cell_centers().points
    box.cell_data["CellEntityIds"] = np.where(centers[:, 0] < 2, 1, 0)
    box.point_data["U"] = np.tile([speed, 0.0, 0.0], (box.n_points, 1))
    box.point_data["pressure"] = box.points[:, 0] + pressure_offset
    return box


@pytest.fixture
def box():
    return make_box()
