import pytest
import torch
from torch_geometric.data import Data

from im2sim.losses.mesh import (
    MeshLoss,
    EdgeLengthDeviationLoss,
    AspectRatioLoss,
    FaceNormalLoss,
    InversionLoss,
    edge_length_deviation_loss,
    aspect_ratio_loss,
    face_norm_consistency,
    face_norm_loss,
    inversion_loss,
    _edge_length_deviation,
    _aspect_ratio,
    _face_norm,
    tet_det,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tetra_graph():
    """Single regular tetrahedron."""
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 3**0.5 / 2, 0.0],
            [0.5, 3**0.5 / 6, (2.0 / 3.0) ** 0.5],
        ],
        dtype=torch.float32,
    )

    # All six unique edges.
    edge_index = torch.tensor(
        [
            [0, 0, 0, 1, 1, 2],
            [1, 2, 3, 2, 3, 3],
        ],
        dtype=torch.long,
    )

    cells = torch.tensor(
        [[0], [1], [2], [3]],
        dtype=torch.long,
    )

    return Data(
        coords=coords,
        edge_index=edge_index,
        cells=cells,
        x=coords.clone(),
    )


@pytest.fixture
def two_tetra_graph():
    """
    Two disconnected tetrahedra with the same geometry.

    Used for testing batched face-normal consistency.
    """
    coords = torch.tensor(
        [
            # tetrahedron 1
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],

            # tetrahedron 2
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    # Three faces per tetrahedron, deliberately all lying in the same
    # orientation so their normals are identical.
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [4, 5, 6],
            [4, 5, 7],
            [4, 6, 7],
        ],
        dtype=torch.long,
    ).T

    batch = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1],
        dtype=torch.long,
    )

    return Data(
        coords=coords,
        faces=faces,
        batch=batch,
    )


# ---------------------------------------------------------------------------
# MeshLoss
# ---------------------------------------------------------------------------

def test_mesh_loss_is_abstract():
    with pytest.raises(TypeError):
        MeshLoss()


def test_mesh_loss_requires_coords():
    class DummyLoss(MeshLoss):
        def __init__(self):
            super().__init__(required_attrs=[])

        def _compute_loss(self, true_graph, pred_graph):
            return torch.tensor(1.0)

    true_graph = Data()
    pred_graph = Data()

    loss = DummyLoss()

    with pytest.raises(ValueError, match="coords"):
        loss(true_graph, pred_graph)


def test_mesh_loss_supervised_requires_attributes():
    class DummyLoss(MeshLoss):
        def __init__(self):
            super().__init__(required_attrs=["foo"])

        def _compute_loss(self, true_graph, pred_graph):
            return torch.tensor(1.0)

    true_graph = Data(coords=torch.zeros(1, 3))
    pred_graph = Data(coords=torch.zeros(1, 3))

    with pytest.raises(ValueError, match="foo"):
        DummyLoss()(true_graph, pred_graph)


def test_mesh_loss_unsupervised_only_requires_prediction_attributes():
    class DummyLoss(MeshLoss):
        def __init__(self):
            super().__init__(required_attrs=["foo"], supervised=False)

        def _compute_loss(self, true_graph, pred_graph):
            return torch.tensor(1.0)

    true_graph = Data()
    pred_graph = Data(
        coords=torch.zeros(1, 3),
        foo=torch.tensor([1]),
    )

    loss = DummyLoss()(true_graph, pred_graph)

    # This test currently fails because MeshLoss.forward does not return
    # self._compute_loss(...).
    assert torch.equal(loss, torch.tensor(1.0))


# ---------------------------------------------------------------------------
# Edge-length deviation
# ---------------------------------------------------------------------------

def test_edge_length_deviation_is_zero_for_equal_edge_lengths():
    # Equilateral triangle.
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 3**0.5 / 2, 0.0],
        ]
    )

    edges = torch.tensor(
        [
            [0, 0, 1],
            [1, 2, 2],
        ]
    )

    deviation = _edge_length_deviation(points, edges)

    assert torch.isclose(deviation, torch.tensor(0.0), atol=1e-6)


def test_edge_length_deviation_is_positive_for_unequal_edges():
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )

    edges = torch.tensor(
        [
            [0, 0, 1],
            [1, 2, 2],
        ]
    )

    deviation = _edge_length_deviation(points, edges)

    assert deviation > 0


def test_edge_length_deviation_loss_is_zero_when_deviation_does_not_increase(
    tetra_graph,
):
    # Same graph => same deviation.
    loss = edge_length_deviation_loss(tetra_graph, tetra_graph)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_edge_length_deviation_loss_only_penalizes_increased_deviation(
    tetra_graph,
):
    true_graph = tetra_graph

    # Make one edge much longer, increasing deviation.
    pred_graph = tetra_graph.clone()
    pred_graph.coords = tetra_graph.coords.clone()
    pred_graph.coords[3] *= 5.0

    loss = edge_length_deviation_loss(true_graph, pred_graph)

    assert loss > 0


def test_edge_length_deviation_loss_does_not_penalize_reduced_deviation():
    # Construct two sets of edge lengths where predicted variation is smaller.
    true_coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )

    pred_coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    edges = torch.tensor(
        [
            [0, 0, 1],
            [1, 2, 2],
        ]
    )

    true_graph = Data(coords=true_coords, edge_index=edges)
    pred_graph = Data(coords=pred_coords, edge_index=edges)

    loss = edge_length_deviation_loss(true_graph, pred_graph)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_edge_length_deviation_loss_module_supervised(tetra_graph):
    loss_fn = EdgeLengthDeviationLoss(supervised=True)

    # This exposes the missing return in MeshLoss.forward.
    loss = loss_fn(tetra_graph, tetra_graph)

    assert loss is not None
    assert torch.isclose(loss, torch.tensor(0.0))


def test_edge_length_deviation_loss_module_unsupervised(tetra_graph):
    loss_fn = EdgeLengthDeviationLoss(supervised=False)

    loss = loss_fn(None, tetra_graph)

    assert loss is not None
    assert torch.isclose(loss, _edge_length_deviation(
        tetra_graph.coords,
        tetra_graph.edge_index,
    ))


def test_edge_length_loss_requires_edge_index(tetra_graph):
    true_graph = Data(coords=tetra_graph.coords)

    loss_fn = EdgeLengthDeviationLoss()

    with pytest.raises(ValueError, match="edge_index"):
        loss_fn(true_graph, true_graph)


# ---------------------------------------------------------------------------
# Aspect ratio
# ---------------------------------------------------------------------------

def test_aspect_ratio_is_one_for_regular_tetrahedron(tetra_graph):
    aspect_ratio = _aspect_ratio(
        tetra_graph.coords,
        tetra_graph.cells,
    )

    assert torch.isclose(
        aspect_ratio,
        torch.tensor(1.0),
        atol=1e-5,
    )


def test_aspect_ratio_is_greater_than_one_for_distorted_tetrahedron(
    tetra_graph,
):
    coords = tetra_graph.coords.clone()
    coords[3] *= 3.0

    aspect_ratio = _aspect_ratio(
        coords,
        tetra_graph.cells,
    )

    assert aspect_ratio > 1.0


def test_aspect_ratio_loss_is_zero_for_identical_meshes(tetra_graph):
    loss = aspect_ratio_loss(
        tetra_graph,
        tetra_graph,
        "cells",
    )

    assert torch.isclose(loss, torch.tensor(0.0))


def test_aspect_ratio_loss_penalizes_increased_aspect_ratio(tetra_graph):
    pred_graph = tetra_graph.clone()
    pred_graph.coords = tetra_graph.coords.clone()
    pred_graph.coords[3] *= 3.0

    loss = aspect_ratio_loss(
        tetra_graph,
        pred_graph,
        "cells",
    )

    assert loss > 0


def test_aspect_ratio_loss_does_not_penalize_reduced_aspect_ratio():
    # Ground truth is distorted; prediction is regular.
    coords_true = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )

    coords_pred = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 3**0.5 / 2, 0.0],
            [0.5, 3**0.5 / 6, (2.0 / 3.0) ** 0.5],
        ]
    )

    cells = torch.tensor([[0], [1], [2], [3]])

    true_graph = Data(coords=coords_true, cells=cells)
    pred_graph = Data(coords=coords_pred, cells=cells)

    loss = aspect_ratio_loss(true_graph, pred_graph, "cells")

    assert torch.isclose(loss, torch.tensor(0.0))


def test_aspect_ratio_loss_module_supervised(tetra_graph):
    loss_fn = AspectRatioLoss("cells", supervised=True)

    loss = loss_fn(tetra_graph, tetra_graph)

    assert loss is not None
    assert torch.isclose(loss, torch.tensor(0.0))


def test_aspect_ratio_loss_module_unsupervised(tetra_graph):
    loss_fn = AspectRatioLoss("cells", supervised=False)

    loss = loss_fn(None, tetra_graph)

    expected = _aspect_ratio(
        tetra_graph.coords,
        tetra_graph.cells,
    )

    # Unsupervised AspectRatioLoss returns the aspect ratio directly.
    assert loss is not None
    assert torch.isclose(loss, expected)


# ---------------------------------------------------------------------------
# Face normals
# ---------------------------------------------------------------------------

def test_face_normal_is_unit_vector():
    face = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    normal = _face_norm(face)

    assert torch.allclose(
        torch.linalg.norm(normal, dim=-1),
        torch.ones(1),
    )


def test_face_normal_has_expected_direction():
    face = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    normal = _face_norm(face)

    expected = torch.tensor([0.0, 0.0, 1.0])

    assert torch.allclose(normal, expected)


def test_face_normal_flips_when_vertex_order_is_reversed():
    face1 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    face2 = face1[[0, 2, 1]]

    normal1 = _face_norm(face1)
    normal2 = _face_norm(face2)

    assert torch.allclose(normal1, -normal2)


def test_face_normal_consistency_is_zero_for_identical_normals():
    # Three copies of the same face.
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    faces = torch.tensor(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
        ]
    )

    graph = Data(
        coords=coords,
        faces=faces,
        batch=torch.zeros(3, dtype=torch.long),
    )

    loss = face_norm_consistency(graph, "faces")

    assert torch.isclose(loss, torch.tensor(0.0))


def test_face_normal_consistency_is_positive_for_different_normals():
    coords = torch.tensor(
        [
            # xy plane
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],

            # xz plane
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    faces = torch.tensor(
        [
            [0, 3],
            [1, 4],
            [2, 5],
        ]
    )

    graph = Data(
        coords=coords,
        faces=faces,
        batch=torch.zeros(6, dtype=torch.long),
    )

    loss = face_norm_consistency(graph, "faces")

    assert loss > 0


def test_face_norm_loss_is_consistency_for_identical_graphs(two_tetra_graph):
    loss = face_norm_loss(
        two_tetra_graph,
        two_tetra_graph,
        "faces",
    )
    consistency = face_norm_consistency(two_tetra_graph, "faces")

    assert torch.isclose(loss, consistency)


def test_face_norm_loss_penalizes_different_normal_direction(two_tetra_graph):
    true_graph = two_tetra_graph

    pred_graph = two_tetra_graph.clone()
    pred_graph.coords = two_tetra_graph.coords.clone()

    # Reflect the second/third coordinates to alter face orientation.
    pred_graph.coords[:, 1] *= -1

    loss = face_norm_loss(
        true_graph,
        pred_graph,
        "faces",
    )

    assert loss > 0


def test_face_normal_loss_module_supervised(two_tetra_graph):
    loss_fn = FaceNormalLoss(
        face_key="faces",
        supervised=True,
    )

    loss = loss_fn(two_tetra_graph, two_tetra_graph)
    consistency = face_norm_consistency(two_tetra_graph, "faces")

    print(loss)
    assert loss is not None
    assert torch.isclose(loss, consistency)


def test_face_normal_loss_module_unsupervised(two_tetra_graph):
    loss_fn = FaceNormalLoss(
        face_key="faces",
        supervised=False,
    )

    loss = loss_fn(None, two_tetra_graph)

    expected = face_norm_consistency(
        two_tetra_graph,
        "faces",
    )

    assert loss is not None
    assert torch.isclose(loss, expected)


def test_face_normal_loss_requires_batch(two_tetra_graph):
    graph = two_tetra_graph.clone()
    del graph.batch

    loss_fn = FaceNormalLoss("faces")

    with pytest.raises(ValueError, match="batch"):
        loss_fn(graph, graph)


# ---------------------------------------------------------------------------
# Tetrahedron determinant / inversion
# ---------------------------------------------------------------------------

def test_tet_det_for_unit_tetrahedron():
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    cells = torch.tensor([[0], [1], [2], [3]])

    det = tet_det(coords, cells)

    # Signed 6 * volume.
    assert torch.allclose(det, torch.tensor([1.0]))


def test_tet_det_changes_sign_when_vertices_are_swapped():
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    cells1 = torch.tensor([[0], [1], [2], [3]])
    cells2 = torch.tensor([[0], [2], [1], [3]])

    det1 = tet_det(coords, cells1)
    det2 = tet_det(coords, cells2)

    assert torch.allclose(det1, -det2)


def test_inversion_loss_is_zero_for_valid_tetrahedron():
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    cells = torch.tensor([[0], [1], [2], [3]])

    loss = inversion_loss(
        coords,
        cells,
        min_vol=1e-3,
    )

    assert torch.isclose(loss, torch.tensor(0.0))


def test_inversion_loss_is_positive_for_inverted_tetrahedron():
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Swap two vertices => negative signed volume.
    cells = torch.tensor([[0], [2], [1], [3]])

    loss = inversion_loss(
        coords,
        cells,
        min_vol=1e-3,
    )

    assert loss > 0


def test_inversion_loss_penalizes_tetrahedron_below_min_volume():
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.001],
        ]
    )

    cells = torch.tensor([[0], [1], [2], [3]])

    # Actual volume = 1/6000 ≈ 0.0001667.
    min_vol = 1e-3

    loss = inversion_loss(
        coords,
        cells,
        min_vol=min_vol,
    )

    expected = min_vol - (1.0 / 6000.0)

    assert torch.allclose(
        loss,
        torch.tensor(expected),
        atol=1e-6,
    )


def test_inversion_loss_respects_min_volume():
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    cells = torch.tensor([[0], [1], [2], [3]])

    loss = inversion_loss(
        coords,
        cells,
        min_vol=1.0,
    )

    # Unit tetrahedron volume is 1/6.
    expected = 1.0 - 1.0 / 6.0

    assert torch.allclose(
        loss,
        torch.tensor(expected),
        atol=1e-6,
    )


def test_inversion_loss_module_uses_prediction_only(tetra_graph):
    loss_fn = InversionLoss(
        cell_key="cells",
        min_vol=1e-3,
    )

    # Ground truth deliberately has no useful geometry.
    true_graph = Data()

    loss = loss_fn(true_graph, tetra_graph)

    expected = inversion_loss(
        tetra_graph.coords,
        tetra_graph.cells,
        min_vol=1e-3,
    )

    assert loss is not None
    assert torch.allclose(loss, expected)


def test_inversion_loss_requires_cell_attribute(tetra_graph):
    graph = Data(coords=tetra_graph.coords)

    loss_fn = InversionLoss("cells")

    with pytest.raises(ValueError, match="cells"):
        loss_fn(None, graph)

