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

import numpy as np

from im2sim.plot.data import _import_pyvista, _to_numpy
from im2sim.plot.figure import Figure, Panel
from im2sim.plot.layers import Points
from im2sim.plot.scales import ColorScale

_NORM_MODES = ("all", "row", "col", "none")


class PointCloudPlot:
    """
    A grid of 3D point clouds, e.g. predicted and target node positions.

    A shortcut for a `im2sim.plot.Figure` of
    `im2sim.plot.Points` layers, rendered with pyvista. For anything more (edges,
    surfaces, streamlines, traces) use `Figure` directly.

    Args:
        nrows (int): Number of rows in the grid.
        ncols (int): Number of columns in the grid.
        point_sets (list of np.ndarray): One `[N, 3]` point cloud per subplot, filled row by row.
            Extra columns are ignored.
        color_sets (list of np.ndarray, optional): One value per point for each point cloud
            (`[N]` or `[N, 1]`). If None, all points are one colour.
        figsize (tuple, optional): Size in inches, as in matplotlib; the image is
            `figsize * dpi` pixels. Default is 3 inches per subplot.
        cmap (str, optional): Colormap. Default is `"Blues_r"`.
        norm_mode (str, optional): Which subplots share a colour scale (and colour bar): `"all"`,
            `"row"`, `"col"` or `"none"`. Default is `"none"`.
        bound_mode (str, optional): Unused. The camera is always fitted to the last point cloud
            in `point_sets`. Default is `"all"`.
        titles (str or list of str, optional): A single figure title, or one title per subplot.
        elev (float, optional): Camera elevation in degrees, with y as the vertical axis.
            Default is `20`.
        azim (float, optional): Camera azimuth in degrees, as in matplotlib. Default is `90`.
        point_size (float, optional): Point size in pixels at `dpi=100`. Default is `5`.
    """

    def __init__(
        self,
        nrows,
        ncols,
        point_sets,
        color_sets=None,
        figsize=None,
        cmap="Blues_r",
        norm_mode="none",
        bound_mode="all",
        titles=None,
        elev=20,
        azim=90,
        point_size=5.0,
    ):
        n_plots = nrows * ncols
        if norm_mode not in _NORM_MODES:
            raise ValueError("norm_mode must be 'all', 'row', 'col', or 'none'")
        if len(point_sets) != n_plots:
            raise ValueError(f"expected {n_plots} point sets for a {nrows}x{ncols} grid")
        if color_sets is not None and len(color_sets) != n_plots:
            raise ValueError("color_sets must have one entry per subplot")

        self.title = None
        if isinstance(titles, str):
            self.title, titles = titles, None
        if titles is not None and len(titles) != n_plots:
            raise ValueError("titles must match number of subplots")

        self.nrows = nrows
        self.ncols = ncols
        self.cmap = cmap
        self.norm_mode = norm_mode
        self.titles = titles
        self.elev = elev
        self.azim = azim
        self.point_size = point_size
        self.figsize = (ncols * 3, nrows * 3) if figsize is None else figsize
        self.point_sets = list(point_sets)
        self.color_sets = None if color_sets is None else list(color_sets)

        # Like the matplotlib version, every subplot is framed on the last point cloud.
        last = np.asarray(_to_numpy(self.point_sets[-1]), dtype=float)[:, :3]
        lo, hi = last.min(axis=0), last.max(axis=0)
        self.bounds = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]
        self.camera = self._camera(lo, hi)

    def _camera(self, lo, hi):
        # matplotlib's view direction for (elev, azim) with vertical_axis="y".
        e, a = np.radians(self.elev), np.radians(self.azim)
        direction = np.array([np.cos(e) * np.cos(a), np.sin(e), -np.cos(e) * np.sin(a)])
        center = (lo + hi) / 2
        return [tuple(center + direction), tuple(center), (0.0, 1.0, 0.0)]

    def _scale(self, i):
        r, c = divmod(i, self.ncols)
        return {"all": "values", "row": f"row {r}", "col": f"col {c}", "none": f"plot {i}"}[
            self.norm_mode
        ]

    @staticmethod
    def _cloud(points, colors):
        pv = _import_pyvista()
        points = np.asarray(_to_numpy(points), dtype=float)[:, :3]
        cloud = pv.PolyData(points)
        if colors is not None:
            cloud["values"] = np.asarray(_to_numpy(colors), dtype=float).reshape(len(points), -1)[
                :, 0
            ]
        return cloud

    def _figure(self, point_frames, color_frames, dpi):
        """A Figure with one frame per entry of `point_frames` (each a list per subplot)."""
        colored = color_frames is not None
        panels = []
        for i in range(self.nrows * self.ncols):
            clouds = [
                self._cloud(points[i], color_frames[f][i] if colored else None)
                for f, points in enumerate(point_frames)
            ]
            if colored:
                layer = Points("values", scale=self._scale(i), point_size=self._px(dpi))
            else:
                layer = Points(solid_color="steelblue", point_size=self._px(dpi))
            panels.append(Panel(clouds, [layer], None if self.titles is None else self.titles[i]))

        scales = {self._scale(i): ColorScale(self.cmap, title="") for i in range(len(panels))}
        return Figure(
            panels,
            grid=(self.nrows, self.ncols),
            scales=scales,
            camera=self.camera,
            bounds=self.bounds,
            window_size=(int(self.figsize[0] * dpi), int(self.figsize[1] * dpi)),
            title=self.title,
        )

    def _px(self, dpi):
        return self.point_size * dpi / 100

    # ---------------------------------------------------------
    # DRAW ONE FRAME
    # ---------------------------------------------------------

    def draw_frame(self, point_sets=None, color_sets=None):
        """Replace the point clouds and/or colours drawn by the next `save_image`."""
        if point_sets is not None:
            self.point_sets = list(point_sets)
        if color_sets is not None:
            self.color_sets = list(color_sets)

    # ---------------------------------------------------------
    # SAVE IMAGE
    # ---------------------------------------------------------

    def save_image(self, filename, dpi=200):
        colors = None if self.color_sets is None else [self.color_sets]
        with self._figure([self.point_sets], colors, dpi) as fig:
            fig.save(filename)

    # ---------------------------------------------------------
    # ANIMATE
    # ---------------------------------------------------------

    def animate(
        self,
        point_sequence_sets,
        color_sequence_sets=None,
        filename="animation.gif",
        fps=15,
        dpi=100,
    ):
        """
        Write one frame per entry of `point_sequence_sets`, each a list with one point cloud
        per subplot. `.gif` needs `imageio`; other extensions need `imageio-ffmpeg`.
        Colour scales are recomputed every frame.
        """
        if color_sequence_sets is None and self.color_sets is not None:
            color_sequence_sets = [self.color_sets] * len(point_sequence_sets)
        with self._figure(point_sequence_sets, color_sequence_sets, dpi) as fig:
            fig.save(filename, fps=fps)
