import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from im2sim.plot.data import (  # noqa: E402
    MeshSequence,
    VTUSequence,
    as_sequence,
    clean_field_names,
    to_pyvista,
    vector_field,
)


def test_to_pyvista_returns_pyvista_meshes_unchanged(box):
    assert to_pyvista(box) is box


def test_to_pyvista_points_array():
    mesh = to_pyvista(np.random.rand(7, 3))
    assert mesh.n_points == 7
    assert mesh.n_verts == 7


def test_to_pyvista_tuple_is_a_graph():
    mesh = to_pyvista((np.random.rand(3, 3), np.array([[0, 1], [1, 2]])))
    assert mesh.n_lines == 2


def test_to_pyvista_pyg_graph_carries_node_fields():
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric.data")
    graph = pyg.Data(
        pos=torch.rand(4, 3),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        pressure=torch.rand(4, 1),
        velocity=torch.rand(4, 3),
        y=torch.rand(2),  # not per node, so skipped
    )

    mesh = to_pyvista(graph)

    assert mesh.n_points == 4
    assert mesh.n_lines == 3
    assert mesh["pressure"].shape == (4,)
    assert mesh["velocity"].shape == (4, 3)
    assert "y" not in mesh.point_data


def test_to_pyvista_pyg_graph_needs_positions():
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric.data")
    with pytest.raises(ValueError, match="pos"):
        to_pyvista(pyg.Data(x=torch.rand(4, 1), edge_index=torch.tensor([[0], [1]])))


def test_to_pyvista_rejects_unknown_input():
    with pytest.raises(TypeError):
        to_pyvista("mesh.vtu")


def test_clean_field_names_strips_padding(box):
    box.point_data["   speed "] = np.zeros(box.n_points)
    clean_field_names(box)
    assert "speed" in box.point_data
    assert "   speed " not in box.point_data


def test_vector_field_stacks_components(box):
    box["vx"], box["vy"] = box.points[:, 0], box.points[:, 1]
    assert vector_field("vx", "vy")(box).shape == (box.n_points, 2)


def test_mesh_sequence_prepares_meshes_without_touching_inputs(box):
    seq = MeshSequence([box], fields={"p2": lambda m: 2 * m["pressure"]})

    mesh = seq[0]

    np.testing.assert_allclose(mesh["p2"], 2 * box["pressure"])
    assert "p2" not in box.point_data
    assert seq[0] is mesh  # cached


def test_mesh_sequence_loader_is_lazy():
    calls = []

    def loader(i):
        calls.append(i)
        return np.full((2, 3), float(i))

    seq = MeshSequence(loader=loader, length=5)

    assert len(seq) == 5 and calls == []
    assert seq[-1].points[0, 0] == 4.0
    assert calls == [4]
    with pytest.raises(IndexError):
        seq[5]


def test_mesh_sequence_needs_exactly_one_source():
    with pytest.raises(ValueError):
        MeshSequence()
    with pytest.raises(ValueError):
        MeshSequence([1], loader=lambda i: i, length=1)


def test_vtu_sequence_reads_numbered_files(tmp_path, box):
    for step in (3, 4):
        box.point_data["  padded"] = np.full(box.n_points, float(step))
        box.save(tmp_path / f"case-{step:04d}.vtu")

    seq = VTUSequence(tmp_path / "case-{:04d}.vtu", steps=[3, 4])

    assert seq.labels == ["3", "4"]
    assert seq[1]["padded"][0] == 4.0


def test_as_sequence(box):
    assert len(as_sequence(box)) == 1
    assert len(as_sequence([box, box])) == 2
    seq = MeshSequence([box])
    assert as_sequence(seq) is seq
