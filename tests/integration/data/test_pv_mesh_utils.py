import hypothesis.strategies as st
import numpy as np
import pyvista as pv
from hypothesis import given
from torch_geometric.data import Data

from im2sim.data.mesh_utils import *


def build_tetra_mesh():
    points = np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [0, 1, 0],  # 2
            [0, 0, 1],  # 3
            [1, 1, 1],  # 4 (second tetra apex)
        ]
    ).astype(np.float32)

    # Two tetrahedra
    cells = np.hstack(
        [
            [4, 0, 1, 2, 3],  # tetra 1
            [4, 1, 2, 3, 4],  # tetra 2
        ]
    )

    celltypes = np.array([pv.CellType.TETRA, pv.CellType.TETRA])
    grid = pv.UnstructuredGrid(cells, celltypes, points)

    grid["CellEntityIds"] = np.array([0] * 2)  # Assign the same entity ID to both tetrahedra
    grid["vtkOriginalPointIds"] = np.arange(len(points))
    return grid


def build_triangle_mesh():
    # 4 points forming a square
    points = np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
        ]
    ).astype(np.float32)

    # Two triangle faces: (0,1,2) and (0,2,3)
    faces = np.hstack(
        [
            [3, 0, 1, 2],
            [3, 0, 2, 3],
        ]
    )
    grid = pv.PolyData(points, faces)
    grid["CellEntityIds"] = np.array([0] * 2)  # Assign the same entity ID to both tetrahedra
    grid["vtkOriginalPointIds"] = np.arange(len(points))
    return grid


def make_square_mesh():
    # Four nodes in a square
    x = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )

    # Undirected edges
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1, 2, 3],
            [1, 0, 3, 2, 3, 2, 1, 0],
        ]
    )

    return Data(x=x, edge_index=edge_index)


def to_set(elems):
    """
    Convert edge/cell array to set of sorted tuples.
    Works for edges/cells
    - (N, m)
    where N is number of edges/cells and m is nodes per edge/cell

    """
    return {tuple(sorted(e)) for e in elems}


def test_get_edges_surf():
    mesh = build_triangle_mesh()

    edges = get_edges_surf(mesh).numpy().T
    print(edges.shape)
    edge_set = to_set(edges)

    expected = np.array(
        [[0, 1], [1, 2], [0, 2], [2, 3], [0, 3], [1, 0], [2, 1], [2, 0], [3, 2], [3, 0]]
    )

    expected = to_set(expected)

    assert edge_set == expected
    assert len(edge_set) == 5


def test_get_edges_tet():
    mesh = build_tetra_mesh()

    edges = get_edges_tet(mesh).numpy().T
    edge_set = to_set(edges)

    expected = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
            [1, 4],
            [2, 4],
            [3, 4],
            [1, 0],
            [2, 0],
            [3, 0],
            [2, 1],
            [3, 1],
            [3, 2],
            [4, 1],
            [4, 2],
            [4, 3],
        ]
    )

    expected = to_set(expected)

    assert edge_set == expected
    assert len(edge_set) == 9


def test_get_structure_cells():
    tet_mesh = build_tetra_mesh()

    tet_cells = get_structure_cells(tet_mesh, {0: "vol"})["vol_cell_index"].numpy().T

    tet_cell_set = to_set(tet_cells)

    tet_expected = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])

    tet_expected = to_set(tet_expected)

    assert tet_cell_set == tet_expected
    assert len(tet_cell_set) == 2


def test_get_structure_ids():
    tet_mesh = build_tetra_mesh()

    tet_ids = set(get_structure_ids(tet_mesh, {0: "vol"})["vol_index"].numpy().tolist())

    tet_expected = {0, 1, 2, 3, 4}

    assert tet_ids == tet_expected
    assert len(tet_ids) == 5


def test_cluster_pool_reduces_nodes():
    mesh = make_square_mesh()

    pooled = cluster_pool(mesh)

    assert isinstance(pooled, Data)
    assert pooled.x.shape[1] == mesh.x.shape[1]
    assert pooled.x.shape[0] <= mesh.x.shape[0]
    assert pooled.edge_index.shape[0] == 2


def test_cluster_pool_preserves_feature_range():
    mesh = make_square_mesh()

    pooled = cluster_pool(mesh)

    # Averaging cannot create values outside the original range
    assert torch.all(pooled.x >= mesh.x.min())
    assert torch.all(pooled.x <= mesh.x.max())


def test_cluster_pool_handles_zero_length_edges():
    mesh = Data(
        x=torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]]),
    )

    pooled = cluster_pool(mesh)

    assert torch.isfinite(pooled.x).all()


def test_rasterize_output_shape():
    points = torch.tensor([[0.0, 0.0, 0.0]])

    result = rasterize(points, im_shape=[4, 4, 4], vox_sizes=[1.0, 1.0, 1.0])

    assert result.shape == (4, 4, 4)


def test_rasterize_single_point_minimum():

    points = torch.tensor([[0.5, 0.5, 0.5]])

    result = rasterize(points, im_shape=[4, 4, 4], vox_sizes=[1.0, 1.0, 1.0])

    assert result[0, 0, 0] == 0


def test_rasterize_two_points():

    points = torch.tensor([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]])

    result = rasterize(points, im_shape=[4, 4, 4], vox_sizes=[1.0, 1.0, 1.0])

    assert result[0, 0, 0] == result[-1, -1, -1]
    assert result[0, 0, 0] < result[1, 1, 1]


@given(st.integers(min_value=1, max_value=10))
def test_rasterize_never_returns_negative(n):

    points = torch.rand(n, 3)

    result = rasterize(points, [8, 8, 8], [1.0, 1.0, 1.0])

    assert torch.all(result >= 0)
