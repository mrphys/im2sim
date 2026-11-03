import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from im2sim.plot.mesh import MeshPlot, compute_norm_ranges, graph_to_mesh  # noqa: E402


def test_graph_to_mesh_dedupes_undirected_edges_and_self_loops():
    points = np.random.rand(4, 3)
    edge_index = np.array([[0, 1, 1, 2, 3], [1, 0, 2, 1, 3]])  # 0-1 and 1-2 both ways, 3-3 loop

    mesh = graph_to_mesh(points, edge_index)

    assert mesh.n_points == 4
    assert mesh.n_lines == 2
    lines = mesh.lines.reshape(-1, 3)[:, 1:]
    assert {tuple(e) for e in lines} == {(0, 1), (1, 2)}


def test_graph_to_mesh_accepts_torch_and_faces():
    torch = pytest.importorskip("torch")
    points = torch.rand(3, 3)

    mesh = graph_to_mesh(points, faces=torch.tensor([[0, 1, 2]]))

    assert mesh.n_faces_strict == 1


def test_graph_to_mesh_points_only_has_vertices():
    mesh = graph_to_mesh(np.random.rand(5, 3))
    assert mesh.n_verts == 5


@pytest.mark.parametrize(
    "norm_mode, expected",
    [
        ("none", [(0, 1), (2, 3), (4, 5), (6, 7)]),
        ("all", [(0, 7)] * 4),
        ("row", [(0, 3), (0, 3), (4, 7), (4, 7)]),
        ("col", [(0, 5), (2, 7), (0, 5), (2, 7)]),
    ],
)
def test_compute_norm_ranges(norm_mode, expected):
    colors = [np.array([2 * i, 2 * i + 1]) for i in range(4)]
    assert compute_norm_ranges(colors, 2, 2, norm_mode) == expected


def test_compute_norm_ranges_skips_uncoloured_subplots():
    ranges = compute_norm_ranges([np.array([1.0, 2.0]), None], 1, 2, "all")
    assert ranges == [(1.0, 2.0), None]


def test_compute_norm_ranges_rejects_unknown_mode():
    with pytest.raises(ValueError):
        compute_norm_ranges([np.zeros(2)], 1, 1, "bad")


def test_meshplot_rejects_wrong_number_of_meshes():
    with pytest.raises(ValueError, match="expected 2 meshes"):
        MeshPlot(1, 2, [pv.Sphere()])


def test_meshplot_rejects_scalars_of_wrong_length():
    with pytest.raises(ValueError, match="scalars have 3 values"):
        MeshPlot(1, 1, [pv.Sphere()], scalars=[np.zeros(3)])
