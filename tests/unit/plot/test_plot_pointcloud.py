import numpy as np
import pytest
from conftest import render

pytest.importorskip("pyvista")

from im2sim.plot import PointCloudPlot  # noqa: E402
from im2sim.plot.layers import color_array  # noqa: E402


def clouds(n, size=20):
    rng = np.random.default_rng(0)
    return [rng.random((size, 3)) + i for i in range(n)]


def test_checks_its_arguments():
    with pytest.raises(ValueError, match="norm_mode"):
        PointCloudPlot(1, 1, clouds(1), norm_mode="bad")
    with pytest.raises(ValueError, match="expected 2 point sets"):
        PointCloudPlot(1, 2, clouds(1))
    with pytest.raises(ValueError, match="titles"):
        PointCloudPlot(1, 2, clouds(2), titles=["a"])


@pytest.mark.parametrize(
    "norm_mode, expected",
    [
        ("none", ["plot 0", "plot 1", "plot 2", "plot 3"]),
        ("all", ["values"] * 4),
        ("row", ["row 0", "row 0", "row 1", "row 1"]),
        ("col", ["col 0", "col 1", "col 0", "col 1"]),
    ],
)
def test_norm_mode_picks_shared_scales(norm_mode, expected):
    pc = PointCloudPlot(2, 2, clouds(4), norm_mode=norm_mode)
    assert [pc._scale(i) for i in range(4)] == expected


def test_figure_uses_colours_given_as_columns():
    points = clouds(2)
    colors = [p[:, :1] for p in points]  # [N, 1], as the training scripts pass them
    fig = PointCloudPlot(1, 2, points, colors, titles="both")._figure([points], [colors], dpi=50)

    np.testing.assert_allclose(color_array(fig.built(1, 0, 0)), points[1][:, 0])
    assert fig.window_size == (300, 150)
    assert fig.title == "both"


def test_camera_follows_matplotlib_angles():
    pc = PointCloudPlot(1, 1, [np.array([[-1.0, -1, -1], [1, 1, 1]])], elev=0, azim=-90)
    position, focal, up = pc.camera
    np.testing.assert_allclose(np.subtract(position, focal), [0, 0, 1], atol=1e-12)
    assert up == (0.0, 1.0, 0.0)


def test_draw_frame_replaces_the_data():
    pc = PointCloudPlot(1, 1, clouds(1))
    new = clouds(1, size=5)
    pc.draw_frame(point_sets=new)
    assert pc.point_sets[0] is new[0]


@render
def test_save_image_and_animate(tmp_path):
    points = clouds(2)
    pc = PointCloudPlot(1, 2, points, [p[:, 0] for p in points], figsize=(2, 1), norm_mode="all")
    pc.save_image(tmp_path / "a.png", dpi=50)
    pc.animate([points, [p + 1 for p in points]], filename=str(tmp_path / "a.gif"), fps=2)
    assert (tmp_path / "a.png").stat().st_size > 0
    assert (tmp_path / "a.gif").stat().st_size > 0
