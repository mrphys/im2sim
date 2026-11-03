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

from im2sim.plot.data import _import_pyvista, _to_numpy, graph_to_mesh

_SCALARS = "_im2sim_scalars"


def compute_norm_ranges(color_sets, nrows, ncols, norm_mode):
    """
    Shared colour limits for a grid of subplots, filled row by row.

    `norm_mode` is `"all"`, `"row"`, `"col"` or `"none"`. Entries of `color_sets` that
    are `None` are ignored, and `"none"` returns each subplot's own range.
    """
    if norm_mode not in ("all", "row", "col", "none"):
        raise ValueError("norm_mode must be 'all', 'row', 'col', or 'none'")

    def value_range(indices):
        vals = [color_sets[i] for i in indices if color_sets[i] is not None]
        if not vals:
            return None
        vals = np.concatenate([np.ravel(v) for v in vals])
        return float(vals.min()), float(vals.max())

    n = len(color_sets)
    ranges = []
    for idx in range(n):
        if color_sets[idx] is None:
            ranges.append(None)
            continue
        r, c = divmod(idx, ncols)
        if norm_mode == "none":
            group = [idx]
        elif norm_mode == "all":
            group = range(n)
        elif norm_mode == "row":
            group = [r * ncols + j for j in range(ncols) if r * ncols + j < n]
        else:
            group = [i * ncols + c for i in range(nrows) if i * ncols + c < n]
        ranges.append(value_range(group))
    return ranges


class MeshPlot:
    """
    A grid of pyvista renders for comparing meshes, with edges drawn.

    The pyvista counterpart of :class:`PointCloudPlot`: the same grid, colouring and
    `norm_mode` options, but each subplot is a mesh so its edges (graph connectivity, surface
    triangles or tetrahedra) are visible.

    Args:
        nrows (int): Number of rows in the grid.
        ncols (int): Number of columns in the grid.
        meshes (list): One entry per subplot, filled row by row. Each entry is a pyvista dataset
            (e.g. from `pv.read`), or a `(points, edge_index)` tuple which is converted with
            :func:`graph_to_mesh`.
        scalars (list, optional): One entry per subplot: an array with one value per point (or
            per cell), the name of an array on the mesh, or `None` for a uniform colour.
        cmap (str, optional): Colormap. Default is `"Blues_r"`, matching :class:`PointCloudPlot`.
        norm_mode (str, optional): Which subplots share a colour scale: `"all"`, `"row"`, `"col"` or
            `"none"`. Default is `"none"`.
        clim (tuple or list of tuple, optional): Fixed `(vmin, vmax)` colour limits, either one
            pair for every subplot or one per subplot. Overrides `norm_mode` and stays fixed
            across animation frames. Default is `None`.
        titles (str or list of str, optional): A single figure title, or one title per subplot.
        show_edges (bool, optional): Draw cell edges. Default is `True`.
        edge_color (str, optional): Colour of the edges. Default is `"black"`.
        line_width (float, optional): Width of the edges in pixels. Default is `1`.
        style (str, optional): pyvista render style: "surface", "wireframe" or "points".
            Use "wireframe" to see every edge of a volume mesh rather than only its outer
            surface. Default is `"surface"`.
        opacity (float, optional): Mesh opacity. Default is `1`.
        color (str, optional): Mesh colour for subplots with no scalars. Default is `"lightsteelblue"`.
        scalar_bar_title (str, optional): Label for the colour bars. Default is `""`.
        window_size (tuple, optional): Size of the image in pixels. Default is
            `(ncols * 400, nrows * 400)`.
        camera_position (str or list, optional): pyvista camera position, e.g. "xy", "iso" or a
            `[position, focal_point, view_up]` list. Default is `"xy"`.
        elev (float, optional): Camera elevation in degrees, applied after `camera_position`.
            Default is `0`.
        azim (float, optional): Camera azimuth in degrees, applied after `camera_position`.
            Default is `0`.
        link_views (bool, optional): Share one camera between all subplots, fitted to all of the
            meshes. Default is `True`.
        background (str, optional): Background colour. Default is `"white"`.
    """

    def __init__(
        self,
        nrows,
        ncols,
        meshes,
        scalars=None,
        cmap="Blues_r",
        norm_mode="none",
        clim=None,
        titles=None,
        show_edges=True,
        edge_color="black",
        line_width=1.0,
        style="surface",
        opacity=1.0,
        color="lightsteelblue",
        scalar_bar_title="",
        window_size=None,
        camera_position="xy",
        elev=0.0,
        azim=0.0,
        link_views=True,
        background="white",
    ):
        pv = _import_pyvista()

        n_plots = nrows * ncols
        if len(meshes) != n_plots:
            raise ValueError(
                f"expected {n_plots} meshes for a {nrows}x{ncols} grid, got {len(meshes)}"
            )
        if scalars is None:
            scalars = [None] * n_plots
        elif len(scalars) != n_plots:
            raise ValueError("scalars must have one entry per subplot")

        suptitle = None
        if isinstance(titles, str):
            suptitle, titles = titles, None
        if titles is not None and len(titles) != n_plots:
            raise ValueError("titles must match number of subplots")

        if window_size is None:
            window_size = (ncols * 400, nrows * 400)

        self.nrows = nrows
        self.ncols = ncols
        self.norm_mode = norm_mode
        if clim is not None and np.ndim(clim) == 1:
            clim = [clim] * n_plots
        if clim is not None and len(clim) != n_plots:
            raise ValueError("clim must be one (vmin, vmax) pair or one pair per subplot")
        self.clim = clim
        self.link_views = link_views
        self.camera_position = camera_position
        self.elev = elev
        self.azim = azim

        # Copies, so that frames drawn by `animate` do not modify the caller's meshes.
        self.meshes = [self._as_mesh(m).copy() for m in meshes]
        values = [self._resolve_scalars(m, s) for m, s in zip(self.meshes, scalars, strict=True)]
        ranges = self._color_ranges(values)

        self.plotter = pv.Plotter(
            shape=(nrows, ncols), off_screen=True, window_size=window_size, border=False
        )
        self.plotter.set_background(background)
        self.actors = []

        for i, (mesh, vals, clim) in enumerate(zip(self.meshes, values, ranges, strict=True)):
            self.plotter.subplot(*divmod(i, ncols))
            kwargs = dict(
                show_edges=show_edges,
                edge_color=edge_color,
                line_width=line_width,
                style=style,
                opacity=opacity,
            )
            if _is_line_mesh(pv, mesh):
                # A graph has no faces to put edges on, so the lines themselves are the edges.
                kwargs.update(show_edges=False, line_width=max(line_width, 1.0))
            if vals is None:
                actor = self.plotter.add_mesh(mesh, color=color, **kwargs)
            else:
                self._set_scalars(mesh, vals)
                # Scalar bars are keyed by title, so give each subplot its own.
                bar_title = scalar_bar_title + " " * i
                actor = self.plotter.add_mesh(
                    mesh,
                    scalars=_SCALARS,
                    cmap=cmap,
                    clim=clim,
                    scalar_bar_args={
                        "title": bar_title,
                        "color": "black",
                        "vertical": True,
                        "position_x": 0.85,
                        "width": 0.05,
                        "fmt": "%.3g",
                    },
                    **kwargs,
                )
            if titles is not None and titles[i] is not None:
                self.plotter.add_text(titles[i], position="upper_edge", font_size=10, color="black")
            self.actors.append(actor)

        if suptitle is not None:
            self.plotter.subplot(0, 0)
            self.plotter.add_text(suptitle, position="upper_left", font_size=12, color="black")

        if link_views:
            self.plotter.link_views()
        self._reset_camera()

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _as_mesh(mesh):
        if isinstance(mesh, tuple):
            return graph_to_mesh(*mesh)
        return mesh

    @staticmethod
    def _resolve_scalars(mesh, scalars):
        if scalars is None:
            return None
        if isinstance(scalars, str):
            return np.asarray(mesh[scalars])
        scalars = _to_numpy(scalars)
        if scalars.ndim > 1:
            scalars = scalars.reshape(len(scalars), -1)
            scalars = scalars[:, 0] if scalars.shape[1] == 1 else np.linalg.norm(scalars, axis=1)
        if len(scalars) not in (mesh.n_points, mesh.n_cells):
            raise ValueError(
                f"scalars have {len(scalars)} values but the mesh has {mesh.n_points} points "
                f"and {mesh.n_cells} cells"
            )
        return scalars

    def _color_ranges(self, values):
        if self.clim is not None:
            return [
                tuple(c) if v is not None else None for c, v in zip(self.clim, values, strict=True)
            ]
        return compute_norm_ranges(values, self.nrows, self.ncols, self.norm_mode)

    @staticmethod
    def _set_scalars(mesh, vals):
        if len(vals) == mesh.n_points:
            mesh.point_data[_SCALARS] = vals
            preference = "point"
        else:
            mesh.cell_data[_SCALARS] = vals
            preference = "cell"
        # The renderer colours by the active array, and `copy_from` in `draw_frame` replaces
        # it with whatever the new mesh had active.
        mesh.set_active_scalars(_SCALARS, preference=preference)

    def _reset_camera(self):
        if self.link_views:
            # One shared camera: set it once, fitted to the bounds of every mesh.
            b = np.array([m.bounds for m in self.meshes])
            bounds = [
                b[:, 0].min(),
                b[:, 1].max(),
                b[:, 2].min(),
                b[:, 3].max(),
                b[:, 4].min(),
                b[:, 5].max(),
            ]
            subplots = [(0, None if not self.meshes else bounds)]
        else:
            subplots = [(i, None) for i in range(len(self.meshes))]

        for i, bounds in subplots:
            self.plotter.subplot(*divmod(i, self.ncols))
            self.plotter.camera_position = self.camera_position
            if self.elev:
                self.plotter.camera.elevation = self.elev
            if self.azim:
                self.plotter.camera.azimuth = self.azim
            self.plotter.reset_camera(bounds=bounds)

    # ---------------------------------------------------------
    # DRAW ONE FRAME
    # ---------------------------------------------------------

    def draw_frame(self, meshes=None, scalars=None):
        """
        Update the subplots in place.

        Args:
            meshes (list, optional): One entry per subplot: an `[N, 3]` array of new point
                positions (same connectivity), a new pyvista dataset, or `None` to keep the
                current geometry.
            scalars (list, optional): New colour values per subplot, as in `__init__`.
                Subplots that had no scalars stay uniformly coloured.
        """
        n_plots = len(self.meshes)
        meshes = [None] * n_plots if meshes is None else meshes
        scalars = [None] * n_plots if scalars is None else scalars

        for mesh, new in zip(self.meshes, meshes, strict=True):
            if new is None:
                continue
            if isinstance(new, tuple) or hasattr(new, "n_points"):
                mesh.copy_from(self._as_mesh(new))
            else:
                mesh.points = _to_numpy(new)

        values = []
        for mesh, actor, new in zip(self.meshes, self.actors, scalars, strict=True):
            vals = self._resolve_scalars(mesh, new)
            if vals is not None and actor.mapper.scalar_visibility:
                self._set_scalars(mesh, vals)
            else:
                vals = None
            values.append(vals)

        ranges = self._color_ranges(values)
        for actor, clim in zip(self.actors, ranges, strict=True):
            if clim is not None:
                actor.mapper.scalar_range = clim

        self.plotter.render()

    # ---------------------------------------------------------
    # SAVE IMAGE
    # ---------------------------------------------------------

    def save_image(self, filename):
        self.plotter.screenshot(filename)
        self.plotter.close()

    # ---------------------------------------------------------
    # ANIMATE
    # ---------------------------------------------------------

    def animate(
        self, mesh_sequence_sets, scalar_sequence_sets=None, filename="animation.gif", fps=15
    ):
        """
        Write one frame per entry of `mesh_sequence_sets`.

        Each entry is a list with one item per subplot, in any form accepted by
        :meth:`draw_frame`. Pass `None` for the whole entry to keep the geometry fixed and only
        change the scalars. Files ending in `.gif` need `imageio`; other extensions are
        written as movies and need `imageio-ffmpeg`.
        """
        if scalar_sequence_sets is None:
            scalar_sequence_sets = [None] * len(mesh_sequence_sets)
        if len(scalar_sequence_sets) != len(mesh_sequence_sets):
            raise ValueError(
                "mesh_sequence_sets and scalar_sequence_sets must have the same length"
            )

        if filename.endswith(".gif"):
            self.plotter.open_gif(filename, fps=fps)
        else:
            self.plotter.open_movie(filename, framerate=fps)

        for meshes, scalars in zip(mesh_sequence_sets, scalar_sequence_sets, strict=True):
            self.draw_frame(meshes=meshes, scalars=scalars)
            self.plotter.write_frame()

        self.plotter.close()


def _is_line_mesh(pv, mesh):
    return isinstance(mesh, pv.PolyData) and mesh.n_lines > 0 and mesh.n_lines == mesh.n_cells
