import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from im2sim.plot.layers import (  # noqa: E402
    COLOR,
    Graph,
    Layer,
    Points,
    Streamlines,
    Surface,
    color_array,
    select_cells,
)
from im2sim.plot.scales import ColorScale, value_range  # noqa: E402


def test_select_cells_by_field(box):
    wall = select_cells(box, {"CellEntityIds": 1})
    assert 0 < wall.n_cells < box.n_cells
    assert set(np.unique(wall["CellEntityIds"])) == {1}


def test_select_cells_by_mask(box):
    assert select_cells(box, lambda m: np.arange(m.n_cells) < 3).n_cells == 3


def test_build_colours_by_field_without_touching_source(box):
    built = Surface("pressure").build(box)

    np.testing.assert_allclose(color_array(built), box["pressure"])
    assert built.active_scalars_name == COLOR
    assert COLOR not in box.point_data


def test_vector_fields_are_coloured_by_magnitude(box):
    box["U"] = np.tile([3.0, 4.0, 0.0], (box.n_points, 1))
    np.testing.assert_allclose(color_array(Surface("U").build(box)), 5.0)


def test_build_applies_select_clip_and_surface(box):
    built = Surface(select={"CellEntityIds": 0}, clip={"normal": "x"}, surface=True).build(box)
    assert isinstance(built, pv.PolyData)
    assert built.n_points < box.n_points


def test_scale_name_defaults():
    assert Layer().scale_name is None
    assert Layer("pressure").scale_name == "pressure"
    assert Layer(lambda m: m.points[:, 0]).scale_name == "color"
    assert Layer("U", scale="speed").scale_name == "speed"


def test_wrong_number_of_colour_values_is_an_error(box):
    with pytest.raises(ValueError, match="colour values"):
        Layer(lambda m: np.zeros(3)).build(box)


def test_graph_draws_every_cell_edge(box):
    built = Graph().build(box)
    assert built.n_lines == built.n_cells > 0


def test_graph_keeps_existing_graph_lines():
    graph = pv.PolyData(np.random.rand(3, 3), lines=[2, 0, 1, 2, 1, 2])
    assert Graph().build(graph).n_lines == 2


def test_points_layer_passes_point_options():
    kwargs = Points(point_size=3).mesh_kwargs()
    assert kwargs["style"] == "points" and kwargs["point_size"] == 3


def test_streamlines_follow_the_flow(box):
    seeds = np.array([[0.0, y, z] for y in (1.0, 2.0) for z in (1.0, 2.0)])
    layer = Streamlines(
        "U", seeds=seeds, color="U", tube_radius=0, integration={"max_step_length": 0.2}
    )

    built = layer.build(box)

    assert built.n_lines > 0
    assert built.bounds[1] > 4.5  # traced along +x to (nearly) the far side, at x = 5
    yz = built.points[:, 1:]
    assert yz.min() > 0.99 and yz.max() < 2.01  # straight along x, no drift in y or z
    np.testing.assert_allclose(color_array(built), 1.0)


def test_streamlines_reuse_their_seeds_and_subsample(box):
    layer = Streamlines("U", seeds={"CellEntityIds": 1}, n_seeds=5)
    layer.build(box)
    seeds = layer._seed_points.copy()

    layer.build(box)

    assert len(seeds) == 5
    np.testing.assert_array_equal(layer._seed_points, seeds)


def test_streamlines_without_flow_are_empty(box):
    box["U"] = np.zeros((box.n_points, 3))
    built = Streamlines("U", seeds=np.array([[1.0, 1.0, 1.0]]), color="U").build(box)
    assert built.n_points == 0
    assert color_array(built).size == 0


def test_color_scale_validates_clim():
    assert ColorScale(clim=(0, 1)).clim == (0.0, 1.0)
    with pytest.raises(ValueError):
        ColorScale(clim="sometimes")
    with pytest.raises(ValueError):
        ColorScale(clim=(0, 1, 2))


def test_value_range_ignores_missing_and_non_finite():
    assert value_range([None, np.array([]), np.array([1.0, np.nan, 3.0])]) == (1.0, 3.0)
    assert value_range([None]) is None


def test_trace_of_a_field_is_its_mean_per_frame(box):
    from im2sim.plot.data import MeshSequence
    from im2sim.plot.layers import Trace

    seq = MeshSequence([box, box], labels=[300, 302])
    x, y = Trace("U").values(seq)  # vectors use their magnitude
    np.testing.assert_allclose(x, [300, 302])
    np.testing.assert_allclose(y, [1.0, 1.0])
    assert Trace("pressure").y_label == "pressure"


def test_trace_of_a_function_or_array(box):
    from im2sim.plot.data import MeshSequence
    from im2sim.plot.layers import Trace

    seq = MeshSequence([box] * 3, labels=["a", "b", "c"])
    x, y = Trace(lambda m: m["pressure"].max()).values(seq)
    np.testing.assert_allclose(x, [0, 1, 2])  # labels aren't numbers, so frame index
    np.testing.assert_allclose(y, [5.0, 5.0, 5.0])
    _, y = Trace([1, 2, 3], x=[0.1, 0.2, 0.3]).values(seq)
    np.testing.assert_allclose(y, [1, 2, 3])
    with pytest.raises(ValueError, match="2 values for 3 frames"):
        Trace([1, 2]).values(seq)
