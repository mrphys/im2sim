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

"""
Layers: what to draw from a frame's mesh, and how.

A layer turns the panel's mesh for a frame into the geometry it draws (`build`), and says how
to draw it (`mesh_kwargs`). Several layers can be stacked in one panel. To add a new kind of
layer, subclass `im2sim.plot.Layer` and override `geometry`.
"""

import numpy as np

from im2sim.plot.data import _import_pyvista, _to_numpy

#: Name of the array a built mesh is coloured by.
COLOR = "_im2sim_color"


def select_cells(mesh, select):
    """
    Extract the cells matching `select`.

    `select` is a dict of `{cell field: value or list of values}`, all of which must match
    (e.g. `{"CellEntityIds": 1}`), or a function `fn(mesh)` returning a boolean mask over
    cells.
    """
    if callable(select):
        mask = np.asarray(select(mesh), dtype=bool)
    else:
        mask = np.ones(mesh.n_cells, dtype=bool)
        for field, value in select.items():
            mask &= np.isin(np.asarray(mesh.cell_data[field]), np.atleast_1d(value))
    return mesh.extract_cells(np.flatnonzero(mask))


def _extract_surface(mesh):
    try:
        return mesh.extract_surface(algorithm="dataset_surface")
    except TypeError:  # pyvista before the `algorithm` option
        return mesh.extract_surface()


def color_array(mesh):
    """The array `mesh` is coloured by, or None if it has none."""
    if COLOR in mesh.point_data:
        return np.asarray(mesh.point_data[COLOR])
    if COLOR in mesh.cell_data:
        return np.asarray(mesh.cell_data[COLOR])
    return None


class Layer:
    """
    Base class for everything drawn in a panel.

    Args:
        color (str or callable, optional): What to colour by: the name of a field, or a function
            `fn(built_mesh) -> array` with one value per point or cell. Vector fields are
            coloured by their magnitude. None draws a single `solid_color`.
        scale (str, optional): Name of the `im2sim.plot.ColorScale` to use. Layers with the same
            scale name share colour limits and one colour bar. Default is the `color` field
            name, or `"color"` when `color` is a function.
        select (dict or callable, optional): Keep only matching cells, before anything else.
            See `im2sim.plot.layers.select_cells`.
        clip (dict, optional): Arguments for `pyvista`'s `clip`, applied after `select`,
            e.g. `{"normal": "z", "crinkle": True}` to cut a volume open.
        surface (bool, optional): Reduce a volume to its outer surface. Default is `False`.
        opacity (float, optional): Default is `1`.
        solid_color (str, optional): Colour when `color` is None. Default is `"lightsteelblue"`.
        show_edges (bool, optional): Draw cell edges. Default depends on the layer.
        edge_color (str, optional): Default is `"black"`.
        line_width (float, optional): Width of edges and lines in pixels. Default is `1`.
        **kwargs: Passed on to `pyvista.Plotter.add_mesh`.
    """

    style = "surface"
    default_show_edges = False

    def __init__(
        self,
        color=None,
        *,
        scale=None,
        select=None,
        clip=None,
        surface=False,
        opacity=1.0,
        solid_color="lightsteelblue",
        show_edges=None,
        edge_color="black",
        line_width=1.0,
        **kwargs,
    ):
        self.color = color
        self.scale = scale
        self.select = select
        self.clip = clip
        self.surface = surface
        self.opacity = opacity
        self.solid_color = solid_color
        self.show_edges = self.default_show_edges if show_edges is None else show_edges
        self.edge_color = edge_color
        self.line_width = line_width
        self.kwargs = kwargs

    @property
    def scale_name(self):
        """The colour scale this layer uses, or None if it is a single colour."""
        if self.color is None:
            return None
        if self.scale is not None:
            return self.scale
        return self.color if isinstance(self.color, str) else "color"

    def build(self, mesh):
        """Return the mesh to draw for one frame, with its colour values as the active array."""
        out = mesh
        if self.select is not None:
            out = select_cells(out, self.select)
        if self.clip is not None:
            out = out.clip(**self.clip)
        if self.surface:
            out = _extract_surface(out)
        out = self.geometry(out, mesh)
        if out is mesh:
            out = mesh.copy(deep=False)  # so the colour array isn't added to the source mesh

        values = self.color_values(out)
        if values is not None:
            if len(values) == out.n_points:
                out.point_data[COLOR] = values
                out.set_active_scalars(COLOR, preference="point")
            elif len(values) == out.n_cells:
                out.cell_data[COLOR] = values
                out.set_active_scalars(COLOR, preference="cell")
            else:
                raise ValueError(
                    f"{type(self).__name__}: {len(values)} colour values for a mesh with "
                    f"{out.n_points} points and {out.n_cells} cells"
                )
        return out

    def geometry(self, mesh, source):
        """
        Derive the geometry to draw. Override in subclasses.

        Args:
            mesh: The frame's mesh after `select`, `clip` and `surface`.
            source: The frame's full mesh, before any of those.
        """
        return mesh

    def color_values(self, built):
        if self.color is None:
            return None
        if built.n_points == 0:
            return np.empty(0)
        values = built[self.color] if isinstance(self.color, str) else self.color(built)
        values = _to_numpy(values).astype(float)
        if values.ndim > 1:
            values = values.reshape(len(values), -1)
            values = values[:, 0] if values.shape[1] == 1 else np.linalg.norm(values, axis=1)
        return values

    def mesh_kwargs(self):
        """Keyword arguments for `pyvista.Plotter.add_mesh`."""
        return dict(
            style=self.style,
            opacity=self.opacity,
            show_edges=self.show_edges,
            edge_color=self.edge_color,
            line_width=self.line_width,
            **self.kwargs,
        )


class Surface(Layer):
    """Filled cells with their edges drawn. A volume shows as its outer surface."""

    default_show_edges = True


class Wireframe(Layer):
    """Every cell edge as a line, including the inside of a volume."""

    style = "wireframe"

    def __init__(self, color=None, *, solid_color="slategray", **kwargs):
        super().__init__(color, solid_color=solid_color, **kwargs)


class Points(Layer):
    """The mesh's points, like `im2sim.plot.PointCloudPlot`. `point_size` is in pixels."""

    style = "points"

    def __init__(self, color=None, *, point_size=6.0, render_points_as_spheres=True, **kwargs):
        super().__init__(
            color,
            point_size=point_size,
            render_points_as_spheres=render_points_as_spheres,
            **kwargs,
        )


class Graph(Layer):
    """
    Graph edges as lines.

    A graph input (e.g. a PyG `Data` with `edge_index`) is drawn as is; any other mesh is
    drawn as one line per cell edge, which is the graph `im2sim.mesh_ops.get_edges_tet` builds.
    """

    def geometry(self, mesh, source):
        pv = _import_pyvista()
        if isinstance(mesh, pv.PolyData) and mesh.n_lines and mesh.n_lines == mesh.n_cells:
            return mesh
        return mesh.extract_all_edges()


class Streamlines(Layer):
    """
    Streamlines of a vector field, traced through the layer's mesh.

    Use `select` to pick the volume to trace through (e.g. `{"CellEntityIds": 0}`).
    The seeds are chosen from the first frame drawn and reused for every frame, so lines change
    smoothly over an animation. Use a separate `Streamlines` per panel.

    Args:
        vectors (str): Name of the `[N, 3]` vector field, e.g. from
            `im2sim.plot.vector_field`.
        seeds: Where to start the lines: a dict selecting cells of the full mesh (e.g. the inlet,
            `{"CellEntityIds": 4}`), a `[K, 3]` array, a pyvista mesh, or a function
            `fn(full_mesh) -> [K, 3]`.
        n_seeds (int, optional): Use a random subset of at most this many seeds. Default is `300`.
        direction (str, optional): "forward", "backward" or "both". "both" works whichever way
            the flow goes through the seeds. Default is `"both"`.
        tube_radius (float, optional): Draw tubes of this radius. Default is `0`.2% of the
            mesh's diagonal. Use 0 for plain lines.
        max_steps (int, optional): Integration steps per line. Default is `4000`.
        random_state (int, optional): Seed for picking the seed subset. Default is `0`.
        integration (dict, optional): More options for pyvista's `streamlines_from_source`.
            Lines stop at their last step inside the mesh, so they can end up to one step short
            of an outlet; `{"max_step_length": 0.2}` (in cell lengths) brings them closer.
        **kwargs: As for `im2sim.plot.Layer`.
    """

    def __init__(
        self,
        vectors,
        seeds,
        color=None,
        *,
        n_seeds=300,
        direction="both",
        tube_radius=None,
        max_steps=4000,
        random_state=0,
        integration=None,
        **kwargs,
    ):
        super().__init__(color, **kwargs)
        self.integration = dict(integration or {})
        self.vectors = vectors
        self.seeds = seeds
        self.n_seeds = n_seeds
        self.direction = direction
        self.tube_radius = tube_radius
        self.max_steps = max_steps
        self.random_state = random_state
        self._seed_points = None

    def _choose_seeds(self, mesh, source):
        pv = _import_pyvista()
        if isinstance(self.seeds, dict):
            points = select_cells(source, self.seeds).points
        elif isinstance(self.seeds, pv.DataSet):
            points = self.seeds.points
        elif callable(self.seeds):
            points = _to_numpy(self.seeds(source))
        else:
            points = _to_numpy(self.seeds)
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(points) > self.n_seeds:
            rng = np.random.default_rng(self.random_state)
            points = points[rng.choice(len(points), self.n_seeds, replace=False)]
        # Nudge the seeds towards the middle, so none start exactly on the boundary.
        return points + 0.02 * (np.asarray(mesh.center) - points)

    def geometry(self, mesh, source):
        pv = _import_pyvista()
        if self._seed_points is None:
            self._seed_points = self._choose_seeds(mesh, source)
        lines = mesh.streamlines_from_source(
            pv.PolyData(self._seed_points),
            vectors=self.vectors,
            integration_direction=self.direction,
            max_steps=self.max_steps,
            max_length=10 * mesh.length,
            **self.integration,
        )
        if lines.n_points == 0:
            return pv.PolyData()
        radius = 0.002 * mesh.length if self.tube_radius is None else self.tube_radius
        return lines.tube(radius=radius) if radius > 0 else lines


#: Line colours for traces that don't set one.
TRACE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


class Trace:
    """
    A line chart of one value per frame, with a marker that moves to the current frame.

    A panel holds either traces or mesh layers, not both. Several traces in one panel share
    the axes. The values are computed from every frame of the panel's data when the figure is
    first drawn.

    Args:
        value: What to plot: a field name (plotted as its mean over the mesh, using the
            magnitude for vectors), a function `fn(mesh) -> float`, or an array with one value
            per frame.
        label (str, optional): Legend entry. The legend is shown if any trace has a label.
        color (str, optional): Line colour. Default is the next of `TRACE_COLORS`.
        width (float, optional): Line width in pixels. Default is `2`.
        x (array, optional): x value per frame. Default is the data's frame labels if they are
            numbers (e.g. the steps of a `VTUSequence`), else the frame index.
        x_label (str, optional): Default is `"frame"`.
        y_label (str, optional): Default is `value` when it is a field name.
    """

    is_chart = True
    scale_name = None

    def __init__(
        self, value, *, label=None, color=None, width=2.0, x=None, x_label="frame", y_label=None
    ):
        self.value = value
        self.label = label
        self.color = color
        self.width = width
        self.x = x
        self.x_label = x_label
        self.y_label = y_label if y_label is not None or not isinstance(value, str) else value

    def values(self, sequence):
        """The `(x, y)` arrays to plot, one entry per frame of `sequence`."""
        if isinstance(self.value, str) or callable(self.value):
            y = np.array([self._reduce(sequence[i]) for i in range(len(sequence))], dtype=float)
        else:
            y = _to_numpy(self.value).astype(float).ravel()
            if len(y) != len(sequence):
                raise ValueError(f"Trace has {len(y)} values for {len(sequence)} frames")

        if self.x is not None:
            x = _to_numpy(self.x).astype(float).ravel()
            if len(x) != len(y):
                raise ValueError(f"Trace has {len(x)} x values for {len(y)} frames")
        else:
            try:
                x = np.array([float(label) for label in sequence.labels])
            except ValueError:
                x = np.arange(len(y), dtype=float)
        return x, y

    def _reduce(self, mesh):
        if callable(self.value):
            return float(self.value(mesh))
        values = np.asarray(mesh[self.value], dtype=float)
        if values.ndim > 1:
            values = np.linalg.norm(values.reshape(len(values), -1), axis=1)
        return float(values.mean())
