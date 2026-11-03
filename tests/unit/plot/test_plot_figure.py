import numpy as np
import pytest
from conftest import make_box, render

pv = pytest.importorskip("pyvista")

from im2sim.plot import ColorScale, Figure, Panel, Streamlines, Surface, Trace  # noqa: E402
from im2sim.plot import figure as figure_module  # noqa: E402


def frames(n):
    return [make_box(speed=i, pressure_offset=10 * i) for i in range(n)]


def test_static_panels_repeat_in_every_frame(box):
    fig = Figure([Panel(frames(3), [Surface()]), Panel(box, [Surface()])])
    assert fig.n_frames == 3
    assert fig.built(1, 0, 2) is fig.built(1, 0, 0)


def test_panels_with_different_frame_counts_are_an_error():
    with pytest.raises(ValueError, match="different numbers of frames"):
        Figure([Panel(frames(2), [Surface()]), Panel(frames(3), [Surface()])])


def test_panels_must_fit_the_grid(box):
    with pytest.raises(ValueError, match="do not fit"):
        Figure([Panel(box), Panel(box), Panel(box)], grid=(1, 2))


def test_missing_scales_get_defaults(box):
    fig = Figure([Panel(box, [Surface("pressure"), Surface()])])
    assert set(fig.scales) == {"pressure"}


def test_frame_limits_are_shared_across_panels():
    seq = frames(2)
    fig = Figure([Panel(seq, [Surface("pressure")]), Panel(seq, [Surface("pressure")])])
    assert fig.color_limits(1)["pressure"] == (10.0, 15.0)


def test_global_limits_cover_every_frame():
    fig = Figure(
        [Panel(frames(3), [Surface("pressure")])],
        scales={"pressure": ColorScale(clim="global")},
    )
    assert fig.color_limits(0)["pressure"] == (0.0, 25.0)


def test_fixed_limits(box):
    fig = Figure([Panel(box, [Surface("pressure")])], scales={"pressure": ColorScale(clim=(1, 2))})
    assert fig.color_limits(0)["pressure"] == (1.0, 2.0)


def test_layers_with_one_scale_share_limits(box):
    fig = Figure(
        [Panel(box, [Surface("pressure", scale="s"), Surface(lambda m: -m["pressure"], scale="s")])]
    )
    assert fig.color_limits(0)["s"] == (-5.0, 5.0)


def test_auto_frame_label_uses_sequence_labels(box):
    fig = Figure([Panel(frames(2), [Surface()])], frame_label="auto")
    assert fig.frame_label(1) == "1"


def test_save_checks_frame_arguments(box, tmp_path):
    fig = Figure([Panel(box, [Surface()])])
    with pytest.raises(ValueError, match="frame"):
        fig.save(tmp_path / "a.png", frames=[0])
    with pytest.raises(ValueError, match="frames"):
        fig.save(tmp_path / "a.gif", frame=0)


def test_panels_cannot_mix_traces_and_meshes(box):
    with pytest.raises(ValueError, match="mixes chart layers"):
        Figure([Panel(box, [Surface(), Trace("pressure")])])


def test_trace_panels_have_no_colour_scale(box):
    fig = Figure([Panel(frames(2), [Trace("pressure")]), Panel(box, [Surface("pressure")])])
    assert set(fig.scales) == {"pressure"}
    assert [p for p, _, _ in fig._layers()] == [1]


def test_check_rendering_raises_instead_of_crashing(monkeypatch):
    class Crashed:
        returncode = -11

    monkeypatch.setattr(figure_module, "_rendering_works", False)
    monkeypatch.delenv("IM2SIM_SKIP_RENDER_CHECK", raising=False)
    monkeypatch.setattr(figure_module.subprocess, "run", lambda *a, **k: Crashed())
    with pytest.raises(RuntimeError, match="cannot render off screen"):
        figure_module.check_rendering()

    monkeypatch.setenv("IM2SIM_SKIP_RENDER_CHECK", "1")
    figure_module.check_rendering()  # skipped


@render
def test_to_array_and_save(tmp_path):
    seq = frames(3)
    with Figure(
        [
            Panel(seq, [Surface("pressure")], title="pressure"),
            Panel(seq, [Streamlines("U", seeds={"CellEntityIds": 1}, color="U")]),
            Panel(seq, [Trace("pressure", label="mean"), Trace("U")], title="trace"),
        ],
        window_size=(300, 100),
        frame_label="auto",
    ) as fig:
        image = fig.to_array(1)
        png = fig.save(tmp_path / "frame.png", frame=2)
        gif = fig.save(tmp_path / "all.gif", fps=5)

    assert image.shape == (100, 300, 3) and image.dtype == np.uint8
    assert png.stat().st_size > 0
    assert gif.stat().st_size > 0


@render
def test_renders_a_frame_with_no_streamlines(box):
    still = make_box(speed=0.0)
    fig = Figure([Panel([still, box], [Streamlines("U", seeds={"CellEntityIds": 1}, color="U")])])
    assert fig.to_array(0).ndim == 3
    assert fig.to_array(1).ndim == 3
    fig.close()
