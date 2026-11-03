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

"""Turning plot inputs into pyvista meshes, and sequences of them for animation."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

# Attributes of a PyTorch Geometric graph that describe its structure rather than node values.
_GRAPH_KEYS = {"pos", "coords", "edge_index", "edge_attr", "face", "batch", "ptr"}


def _import_pyvista():
    try:
        import pyvista as pv
    except ImportError as e:
        raise ImportError(
            'im2sim plotting needs pyvista. Install it with `pip install -e ".[mesh]"`.'
        ) from e
    return pv


def _to_numpy(x):
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def graph_to_mesh(points, edge_index=None, faces=None):
    """
    Build a pyvista mesh from node positions and, optionally, graph edges or triangle faces.

    Args:
        points (np.ndarray or torch.Tensor): `[N, 3]` node positions.
        edge_index (np.ndarray or torch.Tensor, optional): `[2, E]` edge index, as stored on a
            PyTorch Geometric `Data` object. Each undirected edge is drawn once even if it
            appears in both directions.
        faces (np.ndarray or torch.Tensor, optional): `[F, 3]` triangle node indices.

    Returns:
        pyvista.PolyData: lines for `edge_index`, triangles for `faces`, or vertices only if
        neither is given.
    """
    pv = _import_pyvista()
    points = _to_numpy(points).astype(float)
    mesh = pv.PolyData(points)

    if faces is not None:
        faces = _to_numpy(faces).astype(np.int64)
        mesh.faces = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()

    if edge_index is not None:
        edges = np.sort(_to_numpy(edge_index).astype(np.int64).T, axis=1)
        edges = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)
        mesh.lines = np.hstack([np.full((len(edges), 1), 2), edges]).ravel()

    if faces is None and edge_index is None:
        ids = np.arange(len(points))[:, None]
        mesh.verts = np.hstack([np.ones_like(ids), ids]).ravel()

    return mesh


def _from_pyg(data):
    attrs = data.to_dict()
    points = attrs.get("pos", attrs.get("coords"))
    if points is None:
        raise ValueError("graph has neither a `pos` nor a `coords` attribute")
    faces = attrs.get("face")  # PyG stores faces as [3, F]
    mesh = graph_to_mesh(
        points, attrs.get("edge_index"), None if faces is None else _to_numpy(faces).T
    )

    # Every other per-node attribute becomes a field on the mesh, e.g. `pressure` or `x`.
    for key, value in attrs.items():
        if key in _GRAPH_KEYS or not hasattr(value, "shape"):
            continue
        value = _to_numpy(value)
        if value.ndim >= 1 and value.shape[0] == mesh.n_points:
            value = value.reshape(mesh.n_points, -1)
            mesh.point_data[key] = value[:, 0] if value.shape[1] == 1 else value
    return mesh


def to_pyvista(obj):
    """
    Convert any supported plot input to a pyvista mesh.

    Accepts a pyvista dataset (returned as is), a PyTorch Geometric `Data` object with `pos`
    or `coords` (plus optional `edge_index`, `face` and per-node attributes, which become
    fields), a `(points, edge_index)` or `(points, edge_index, faces)` tuple, or an
    `[N, 3]` array of points.
    """
    pv = _import_pyvista()
    if isinstance(obj, pv.DataSet):
        return obj
    if isinstance(obj, tuple):
        return graph_to_mesh(*obj)
    if hasattr(obj, "to_dict") and hasattr(obj, "edge_index"):
        return _from_pyg(obj)
    array = _to_numpy(obj)
    if array.ndim == 2 and array.shape[1] == 3:
        return graph_to_mesh(array)
    raise TypeError(f"cannot plot an object of type {type(obj).__name__}")


def clean_field_names(mesh):
    """Strip surrounding whitespace from field names in place (some solvers pad them)."""
    for data in (mesh.point_data, mesh.cell_data):
        for name in list(data.keys()):
            if name != name.strip():
                data[name.strip()] = data.pop(name)
    return mesh


def vector_field(*names):
    """
    A derived field that stacks scalar fields into one `[N, k]` vector field.

    For example `fields={"U": vector_field("x-velocity", "y-velocity", "z-velocity")}`.
    """

    def stack(mesh):
        return np.stack([np.asarray(mesh[name]) for name in names], axis=1)

    return stack


class MeshSequence(Sequence):
    """
    A sequence of pyvista meshes, one per frame, loaded and prepared on first access.

    Each mesh has its field names cleaned (:func:`clean_field_names`) and then gets the
    derived `fields` added, in order, so later fields can use earlier ones.

    Args:
        items (list, optional): Inputs accepted by :func:`to_pyvista`, one per frame.
        loader (callable, optional): `loader(i)` returns the input for frame `i`. Use instead
            of `items` to load lazily; requires `length`.
        length (int, optional): Number of frames when using `loader`.
        fields (dict, optional): Derived fields, `{name: fn(mesh) -> array}`.
        labels (list of str, optional): A label per frame, e.g. the timestep. Default is the
            frame index.
        cache (bool, optional): Keep prepared meshes in memory. Default is `True`.
    """

    def __init__(
        self, items=None, *, loader=None, length=None, fields=None, labels=None, cache=True
    ):
        if (items is None) == (loader is None):
            raise ValueError("give exactly one of `items` or `loader`")
        if loader is not None and length is None:
            raise ValueError("`length` is required with `loader`")
        self._items = None if items is None else list(items)
        self._loader = loader
        self._length = len(self._items) if items is not None else int(length)
        self.fields = dict(fields or {})
        self.labels = (
            [str(i) for i in range(self._length)] if labels is None else [str(x) for x in labels]
        )
        if len(self.labels) != self._length:
            raise ValueError("labels must have one entry per frame")
        self.cache = cache
        self._cache = {}

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        if not isinstance(i, int | np.integer):
            raise TypeError("MeshSequence indices must be integers")
        if i < 0:
            i += self._length
        if not 0 <= i < self._length:
            raise IndexError(f"frame {i} out of range for {self._length} frames")
        if i in self._cache:
            return self._cache[i]

        raw = self._items[i] if self._items is not None else self._loader(i)
        mesh = to_pyvista(raw)
        if mesh is raw:
            mesh = mesh.copy()  # don't modify the caller's mesh
        clean_field_names(mesh)
        for name, fn in self.fields.items():
            mesh[name] = np.asarray(fn(mesh))

        if self.cache:
            self._cache[i] = mesh
        return mesh


class VTUSequence(MeshSequence):
    """
    A :class:`MeshSequence` of mesh files numbered by timestep, read lazily with pyvista.

    Args:
        pattern (str or Path): File path with a format field for the step, e.g.
            `"case-{:04d}.vtu"`.
        steps (iterable of int): The steps to load, one per frame. They also become the labels.
        fields (dict, optional): Derived fields, as in :class:`MeshSequence`.
        cache (bool, optional): Keep loaded meshes in memory. Default is `True`.
    """

    def __init__(self, pattern, steps, *, fields=None, cache=True):
        pv = _import_pyvista()
        self.pattern = str(pattern)
        self.steps = list(steps)
        super().__init__(
            loader=lambda i: pv.read(Path(self.pattern.format(self.steps[i]))),
            length=len(self.steps),
            fields=fields,
            labels=self.steps,
            cache=cache,
        )


def as_sequence(data):
    """
    Wrap plot input as a :class:`MeshSequence`.

    A `MeshSequence` is returned as is, a `list` is treated as one input per frame, and
    anything else (a mesh, a graph, a tuple) is a single frame.
    """
    if isinstance(data, MeshSequence):
        return data
    if isinstance(data, list):
        return MeshSequence(data)
    return MeshSequence([data])
