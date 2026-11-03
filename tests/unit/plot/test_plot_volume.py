import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

from im2sim.plot import Mask, MeshMask, VolumePlot, rasterise_mesh, rasterise_nodes  # noqa: E402
from im2sim.plot.volume import _boundary  # noqa: E402


@pytest.fixture
def image():
    return np.random.default_rng(0).random((12, 10, 6))


@pytest.fixture
def cube_mask():
    mask = np.zeros((12, 10, 6), dtype=bool)
    mask[3:8, 2:7, 1:4] = True
    return mask


def test_boundary_is_ring():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    ring = _boundary(mask)
    assert ring.sum() == 8
    assert not ring[2, 2]


def test_render_shape_and_overlay(image, cube_mask):
    plain = VolumePlot(image, dpi=50, figsize=(2, 2))
    masked = VolumePlot(image, [Mask(cube_mask, color="red")], dpi=50, figsize=(2, 2))
    a, b = plain.render(2), masked.render(2)
    assert a.shape == b.shape == (100, 100, 3)
    assert a.dtype == np.uint8
    assert not np.array_equal(a, b)  # slice 2 intersects the cube
    assert np.array_equal(plain.render(5), masked.render(5))  # slice 5 does not


def test_only_masked_skips_empty_slices(image, cube_mask):
    plot = VolumePlot(image, [cube_mask], only_masked=True, dpi=40, figsize=(2, 2))
    assert len(plot.frames()) == 3  # slices 1, 2, 3


def test_axis_and_channel_selection(image, cube_mask):
    plot = VolumePlot(image[None, None] * np.ones((1, 2, 1, 1, 1)), [cube_mask], axis=0, channel=1)
    assert plot.image.shape == image.shape
    assert len(plot.slices) == 12


def test_mask_shape_mismatch(image):
    plot = VolumePlot(image, [Mask(np.ones((3, 3, 3)))])
    with pytest.raises(ValueError, match="shape"):
        plot.frames()


def test_label_map(image):
    labels = np.zeros(image.shape, dtype=int)
    labels[:3] = 1
    labels[3:6] = 2
    voxels = Mask(labels, labels=[2]).voxels(image.shape, (1, 1, 1), (0, 0, 0))
    assert voxels[3:6].all() and not voxels[:3].any()


def test_save_gif(tmp_path, image, cube_mask):
    from PIL import Image

    out = tmp_path / "sweep.gif"
    VolumePlot(image, [Mask(cube_mask, name="cube")], dpi=40, figsize=(2, 2)).save_gif(out, fps=5)
    with Image.open(out) as gif:
        assert gif.n_frames == image.shape[2]


def test_bad_slice_and_image(image):
    with pytest.raises(ValueError, match="out of range"):
        VolumePlot(image, slices=[99])
    with pytest.raises(ValueError, match="3D"):
        VolumePlot(image[0])


# ---- mesh rasterisation (needs pyvista) ----------------------------------------------------


def test_rasterise_sphere_fills_inside():
    pv = pytest.importorskip("pyvista")
    sphere = pv.Sphere(radius=5.0, center=(8, 8, 8), theta_resolution=40, phi_resolution=40)
    mask = rasterise_mesh(sphere, (17, 17, 17))
    assert mask.shape == (17, 17, 17) and mask.dtype == bool
    assert mask[8, 8, 8] and not mask[0, 0, 0]
    expected = 4 / 3 * np.pi * 5**3
    assert abs(mask.sum() - expected) / expected < 0.1
    # indexed as [x, y, z]: a sphere offset along x must shift the mask along axis 0
    shifted = pv.Sphere(radius=3.0, center=(4, 8, 8))
    com = np.argwhere(rasterise_mesh(shifted, (17, 17, 17))).mean(axis=0)
    assert com[0] < 6 and abs(com[1] - 8) < 1 and abs(com[2] - 8) < 1


def test_rasterise_shell_is_hollow():
    pv = pytest.importorskip("pyvista")
    sphere = pv.Sphere(radius=5.0, center=(8, 8, 8))
    shell = rasterise_mesh(sphere, (17, 17, 17), fill=False)
    assert shell.any() and not shell[8, 8, 8]


def test_rasterise_respects_spacing_and_origin():
    pv = pytest.importorskip("pyvista")
    sphere = pv.Sphere(radius=5.0, center=(10, 10, 10))
    mask = rasterise_mesh(sphere, (11, 11, 11), spacing=(2, 2, 2), origin=(0, 0, 0))
    assert mask[5, 5, 5] and not mask[0, 0, 0]


def test_mesh_mask_caches_and_draws(image):
    pv = pytest.importorskip("pyvista")
    sphere = pv.Sphere(radius=3.0, center=(6, 5, 3))
    mm = MeshMask(sphere, method="exact", color="cyan", name="mesh")
    first = mm.voxels(image.shape, (1, 1, 1), (0, 0, 0))
    assert mm.voxels(image.shape, (1, 1, 1), (0, 0, 0)) is first
    plot = VolumePlot(image, [sphere], dpi=40, figsize=(2, 2))  # bare mesh is wrapped
    assert isinstance(plot.masks[0], MeshMask)
    assert plot.render(3).shape == (80, 80, 3)


def _ball(pv, radius, center):
    """Nodes filling a ball, like a volume mesh (a surface alone leaves the middle empty)."""
    points = np.random.default_rng(0).uniform(-1, 1, (4000, 3))
    points = points[np.linalg.norm(points, axis=1) <= 1] * radius + center
    return pv.PolyData(points)


def test_rasterise_nodes_matches_layer_convention():
    pv = pytest.importorskip("pyvista")
    pytest.importorskip("torch_geometric")
    ball = _ball(pv, 4.0, (8.5, 8.5, 8.5))
    mask = rasterise_nodes(ball, (17, 17, 17))
    assert mask.shape == (17, 17, 17) and mask.dtype == bool
    assert mask[8, 8, 8] and not mask[0, 0, 0]
    assert abs(np.argwhere(mask).mean(axis=0) - 8.5).max() < 1


def test_mesh_mask_default_uses_nodes_and_rejects_bad_method(image):
    pv = pytest.importorskip("pyvista")
    pytest.importorskip("torch_geometric")
    ball = _ball(pv, 3.0, (6.5, 5.5, 3.5))
    assert MeshMask(ball).voxels(image.shape, (1, 1, 1), (0, 0, 0)).any()
    with pytest.raises(ValueError, match="method"):
        MeshMask(ball, method="fast")
