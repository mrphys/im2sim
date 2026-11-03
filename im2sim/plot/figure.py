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

import copy
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from im2sim.plot.data import _import_pyvista, as_sequence
from im2sim.plot.layers import COLOR, TRACE_COLORS, color_array
from im2sim.plot.scales import ColorScale, value_range

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

_RENDER_CHECK = (
    "import pyvista as pv; p = pv.Plotter(off_screen=True); p.add_mesh(pv.Sphere()); "
    "p.screenshot(None, return_img=True)"
)
_rendering_works = False


def check_rendering():
    """
    Raise a RuntimeError if pyvista cannot render off screen in this environment.

    Without a working OpenGL setup VTK can crash the whole Python process rather than raise, so
    the check runs once, in a subprocess. Set ``IM2SIM_SKIP_RENDER_CHECK=1`` to skip it.
    """
    global _rendering_works
    if _rendering_works or os.environ.get("IM2SIM_SKIP_RENDER_CHECK") == "1":
        return
    result = subprocess.run(
        [sys.executable, "-c", _RENDER_CHECK], capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pyvista cannot render off screen here (test render exited with code "
            f"{result.returncode}). In a container without a display, install OSMesa "
            "(`apt-get install libosmesa6 libgl1-mesa-dri`); under conda, also start Python "
            "with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`. See the plot guide."
        )
    _rendering_works = True


def _is_chart(layer):
    return getattr(layer, "is_chart", False)


@dataclass
class Panel:
    """
    One panel of a :class:`Figure`: some data and the layers to draw from it.

    Args:
        data: Anything :func:`~im2sim.plot.data.as_sequence` accepts: a mesh, a PyG graph, a
            list with one per frame, or a :class:`~im2sim.plot.data.MeshSequence`. A single
            frame is shown unchanged in every frame of an animation.
        layers (list): Drawn in order, so later layers go on top. Either mesh layers
            (:class:`~im2sim.plot.layers.Surface`, ...) or chart layers
            (:class:`~im2sim.plot.layers.Trace`), not both.
        title (str, optional): Title shown at the top of the panel.
    """

    data: Any
    layers: list = field(default_factory=list)
    title: str | None = None


class Figure:
    """
    A grid of panels that renders to an image, an array or an animation.

    Every panel's data is a sequence of frames. Panels with one frame stay still while the
    others animate; all others must have the same number of frames.

    Args:
        panels (list of Panel): Filled into the grid row by row.
        grid (tuple, optional): ``(nrows, ncols)``. Default is one row.
        scales (dict, optional): ``{name: ColorScale}``. Layers refer to scales by name; any
            name not given here gets ``ColorScale()``.
        camera (str or list, optional): pyvista camera position, e.g. "xy", "iso" or a
            ``[position, focal_point, view_up]`` list. Default is "xy".
        elev (float, optional): Camera elevation in degrees, applied after ``camera``.
        azim (float, optional): Camera azimuth in degrees, applied after ``camera``.
        link_views (bool, optional): Share one camera between all panels, fitted to every mesh in
            the first frame. Default is True.
        bounds (sequence of 6 floats, optional): Fit the camera to these
            ``(xmin, xmax, ymin, ymax, zmin, zmax)`` instead of to the meshes.
        window_size (tuple, optional): Image size in pixels. Default is 500 per panel.
        background (str, optional): Default is "white".
        frame_label (callable or str, optional): Text shown in the bottom-left corner: a function
            ``fn(frame) -> str``, or "auto" for the labels of the first animated panel's data
            (e.g. the timesteps of a :class:`~im2sim.plot.data.VTUSequence`). Default is None.
        title (str, optional): Title for the whole figure, in the top-left corner.
        cache (bool, optional): Keep each frame's built layers in memory, so ``"global"`` colour
            scales and repeated saves don't rebuild them. Default is True.
    """

    def __init__(
        self,
        panels,
        grid=None,
        scales=None,
        camera="xy",
        elev=0.0,
        azim=0.0,
        link_views=True,
        bounds=None,
        window_size=None,
        background="white",
        frame_label=None,
        title=None,
        cache=True,
    ):
        self.panels = list(panels)
        if not self.panels:
            raise ValueError("a figure needs at least one panel")
        self.grid = (1, len(self.panels)) if grid is None else tuple(grid)
        if self.grid[0] * self.grid[1] < len(self.panels):
            raise ValueError(f"{len(self.panels)} panels do not fit in a {self.grid} grid")

        for i, panel in enumerate(self.panels):
            charts = {_is_chart(layer) for layer in panel.layers}
            if len(charts) > 1:
                raise ValueError(f"panel {i} mixes chart layers (Trace) with mesh layers")

        self.sequences = [as_sequence(p.data) for p in self.panels]
        lengths = {len(s) for s in self.sequences} - {1}
        if len(lengths) > 1:
            raise ValueError(
                f"panels have different numbers of frames: {sorted(lengths)}. "
                "Every panel needs the same number, or a single frame."
            )
        self.n_frames = lengths.pop() if lengths else 1

        self.scales = dict(scales or {})
        for panel in self.panels:
            for layer in panel.layers:
                if layer.scale_name is not None:
                    self.scales.setdefault(layer.scale_name, ColorScale())

        if frame_label == "auto":
            animated = [s for s in self.sequences if len(s) == self.n_frames]
            labels = animated[0].labels
            frame_label = labels.__getitem__
        self.frame_label = frame_label

        self.camera = camera
        self.elev = elev
        self.azim = azim
        self.link_views = link_views
        self.bounds = bounds
        self.title = title
        self.window_size = window_size or (500 * self.grid[1], 500 * self.grid[0])
        self.background = background
        self.cache = cache

        self._built = {}
        self._global_ranges = {}
        self._plotter = None

    # ---------------------------------------------------------
    # BUILDING AND COLOUR LIMITS
    # ---------------------------------------------------------

    def _data_index(self, p, frame):
        if not 0 <= frame < self.n_frames:
            raise IndexError(f"frame {frame} out of range for {self.n_frames} frames")
        return 0 if len(self.sequences[p]) == 1 else frame

    def built(self, p, layer_index, frame):
        """The mesh layer ``layer_index`` of panel ``p`` draws in ``frame``."""
        key = (p, layer_index, self._data_index(p, frame))
        if key in self._built:
            return self._built[key]
        mesh = self.sequences[p][key[2]]
        out = self.panels[p].layers[layer_index].build(mesh)
        if self.cache:
            self._built[key] = out
        return out

    def _layers(self):
        """``(panel, layer index, layer)`` for every mesh layer."""
        for p, panel in enumerate(self.panels):
            for li, layer in enumerate(panel.layers):
                if not _is_chart(layer):
                    yield p, li, layer

    def _is_chart_panel(self, p):
        return any(_is_chart(layer) for layer in self.panels[p].layers)

    def color_limits(self, frame):
        """``{scale name: (vmin, vmax) or None}`` for ``frame``."""
        limits = {}
        for name, scale in self.scales.items():
            if not isinstance(scale.clim, str):
                limits[name] = scale.clim
                continue
            members = [(p, li) for p, li, layer in self._layers() if layer.scale_name == name]
            if scale.clim == "global":
                if name not in self._global_ranges:
                    self._global_ranges[name] = value_range(
                        color_array(self.built(p, li, f))
                        for f in range(self.n_frames)
                        for p, li in members
                    )
                limits[name] = self._global_ranges[name]
            else:
                limits[name] = value_range(
                    color_array(self.built(p, li, frame)) for p, li in members
                )
        return limits

    # ---------------------------------------------------------
    # RENDERING
    # ---------------------------------------------------------

    def _setup(self):
        pv = _import_pyvista()
        check_rendering()
        theme = copy.deepcopy(pv.global_theme)
        theme.allow_empty_mesh = True  # e.g. no streamlines in a frame with no flow
        plotter = pv.Plotter(
            shape=self.grid,
            off_screen=True,
            window_size=self.window_size,
            border=False,
            theme=theme,
        )
        plotter.set_background(self.background)

        limits = self.color_limits(0)
        bars_shown = set()
        bars_per_panel = {}
        self._actors = {}

        for p, li, layer in self._layers():
            plotter.subplot(*divmod(p, self.grid[1]))
            mesh = self.built(p, li, 0)
            kwargs = layer.mesh_kwargs()
            name = layer.scale_name
            if name is None:
                actor = plotter.add_mesh(mesh, color=layer.solid_color, **kwargs)
            else:
                scale = self.scales[name]
                show_bar = scale.show_bar and name not in bars_shown
                bar_args = {}
                if show_bar:
                    # Bars are matched by title, so each scale's bar gets its own.
                    k = bars_per_panel.get(p, 0)
                    bar_args = {
                        "title": (name if scale.title is None else scale.title)
                        + " " * len(bars_shown),
                        "color": "black",
                        "vertical": True,
                        "position_x": 0.85 - 0.13 * k,
                        "width": 0.05,
                        "fmt": scale.fmt,
                        "title_font_size": 14,
                        "label_font_size": 12,
                    }
                    bars_per_panel[p] = k + 1
                    bars_shown.add(name)
                actor = plotter.add_mesh(
                    mesh,
                    scalars=COLOR,
                    cmap=scale.cmap,
                    clim=limits[name] or (0.0, 1.0),
                    show_scalar_bar=show_bar,
                    scalar_bar_args=bar_args,
                    **kwargs,
                )
            self._actors[(p, li)] = actor

        self._charts = {}
        for p, panel in enumerate(self.panels):
            plotter.subplot(*divmod(p, self.grid[1]))
            if self._is_chart_panel(p):
                self._charts[p] = self._add_chart(plotter, p)
            elif panel.title:
                plotter.add_text(panel.title, position="upper_edge", font_size=10, color="black")

        if self.title:
            plotter.subplot(0, 0)
            plotter.add_text(self.title, position="upper_left", font_size=12, color="black")

        self._label = None
        if self.frame_label is not None:
            plotter.subplot(0, 0)
            self._label = plotter.add_text("", position="lower_left", font_size=10, color="black")

        self._plotter = plotter
        self._set_camera()

    def _add_chart(self, plotter, p):
        pv = _import_pyvista()
        panel, sequence = self.panels[p], self.sequences[p]
        traces = panel.layers
        chart = pv.Chart2D(
            x_label=next((t.x_label for t in traces if t.x_label), ""),
            y_label=next((t.y_label for t in traces if t.y_label), ""),
        )
        chart.background_color = self.background
        if panel.title:
            chart.title = panel.title

        series = [t.values(sequence) for t in traces]
        for i, (trace, (x, y)) in enumerate(zip(traces, series, strict=True)):
            color = trace.color or TRACE_COLORS[i % len(TRACE_COLORS)]
            chart.line(x, y, color=color, width=trace.width, label=trace.label or "")
        chart.legend_visible = any(t.label for t in traces)

        x_lo, x_hi = value_range(x for x, _ in series) or (0.0, 1.0)
        y_lo, y_hi = value_range(y for _, y in series) or (0.0, 1.0)
        pad = 0.05 * (y_hi - y_lo or 1.0)
        y_lo, y_hi = y_lo - pad, y_hi + pad
        chart.x_range = [x_lo, x_hi if x_hi > x_lo else x_lo + 1.0]
        chart.y_range = [y_lo, y_hi]

        # The current frame: a dashed vertical line, and a dot on each trace.
        cursor = chart.line([x_lo, x_lo], [y_lo, y_hi], color="black", width=1.0, style="--")
        dots = [chart.scatter([x[0]], [y[0]], color="black", size=8) for x, y in series]
        plotter.add_chart(chart)
        return series, cursor, dots, (y_lo, y_hi)

    def _update_chart(self, p, frame):
        series, cursor, dots, (y_lo, y_hi) = self._charts[p]
        i = self._data_index(p, frame)
        for (x, y), dot in zip(series, dots, strict=True):
            dot.update([x[i]], [y[i]])
        x_now = series[0][0][i]
        cursor.update([x_now, x_now], [y_lo, y_hi])

    def _set_camera(self):
        plotter = self._plotter
        if self.link_views:
            plotter.link_views()
            panels = [0]
        else:
            panels = range(len(self.panels))

        for p in panels:
            meshes = [
                self.built(q, li, 0)
                for q, li, _ in self._layers()
                if (self.link_views or q == p) and self.built(q, li, 0).n_points
            ]
            plotter.subplot(*divmod(p, self.grid[1]))
            plotter.camera_position = self.camera
            if self.elev:
                plotter.camera.elevation = self.elev
            if self.azim:
                plotter.camera.azimuth = self.azim
            if self.bounds is not None:
                plotter.reset_camera(bounds=list(self.bounds))
            elif meshes:
                b = np.array([m.bounds for m in meshes])
                bounds = [
                    b[:, 0].min(), b[:, 1].max(),
                    b[:, 2].min(), b[:, 3].max(),
                    b[:, 4].min(), b[:, 5].max(),
                ]  # fmt: skip
                plotter.reset_camera(bounds=bounds)

    def render(self, frame=0):
        """Draw ``frame``. Called by ``save`` and ``to_array``; rarely needed directly."""
        if self._plotter is None:
            self._setup()
        limits = self.color_limits(frame)

        for p, li, layer in self._layers():
            mesh = self.built(p, li, frame)
            mapper = self._actors[(p, li)].mapper
            mapper.SetInputData(mesh)
            if layer.scale_name is not None:
                if COLOR in mesh.cell_data and COLOR not in mesh.point_data:
                    mapper.SetScalarModeToUseCellData()
                else:
                    mapper.SetScalarModeToUsePointData()
                if limits[layer.scale_name] is not None:
                    mapper.scalar_range = limits[layer.scale_name]

        for p in self._charts:
            self._update_chart(p, frame)

        if self._label is not None:
            self._label.SetText(0, str(self.frame_label(frame)))  # 0 = lower-left corner
        self._plotter.render()

    def to_array(self, frame=0):
        """Render ``frame`` and return it as an ``[H, W, 3]`` uint8 array."""
        self.render(frame)
        return self._plotter.screenshot(None, return_img=True)

    def save(self, path, frame=None, frames=None, fps=10):
        """
        Write an image or an animation, chosen by the file extension.

        Args:
            path (str or Path): ``.png``/``.jpg``/... for one frame; ``.gif`` (needs ``imageio``)
                or a movie format such as ``.mp4`` (needs ``imageio-ffmpeg``) for several.
            frame (int, optional): The frame for an image. Default is 0.
            frames (iterable of int, optional): The frames for an animation. Default is all.
            fps (int, optional): Frames per second of an animation. Default is 10.

        Returns:
            Path: ``path``.
        """
        path = Path(path)
        if path.suffix.lower() in _IMAGE_EXTENSIONS:
            if frames is not None:
                raise ValueError(f"{path.suffix} holds one image; use `frame`, not `frames`")
            self.render(0 if frame is None else frame)
            self._plotter.screenshot(str(path))
            return path

        if frame is not None:
            raise ValueError(f"{path.suffix} is an animation; use `frames`, not `frame`")
        frames = range(self.n_frames) if frames is None else frames
        if self._plotter is None:
            self._setup()
        if path.suffix.lower() == ".gif":
            self._plotter.open_gif(str(path), fps=fps)
        else:
            self._plotter.open_movie(str(path), framerate=fps)
        try:
            for f in frames:
                self.render(f)
                self._plotter.write_frame()
        finally:
            self._plotter.mwriter.close()
            self._plotter.mwriter = None
        return path

    def close(self):
        """Free the renderer. The figure can still be rendered again afterwards."""
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
