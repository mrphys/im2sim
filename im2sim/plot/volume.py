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
Slice-sweep plots of 3D images with overlaid masks, written as GIFs.

A :class:`VolumePlot` steps through a 3D image one slice at a time and draws each slice with
any number of mask layers on top. A mask is either a voxel array (:class:`Mask`) or a surface
mesh that is rasterised onto the image grid (:class:`MeshMask`), so a predicted mesh can be
checked against the image it came from. Uses matplotlib and Pillow only; rasterising meshes
also needs pyvista (`pip install -e ".[mesh]"`).
"""

import numpy as np

from im2sim.plot.data import _import_pyvista, _to_numpy, to_pyvista


def rasterise_nodes(mesh, shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    Fast, approximate rasterisation of a mesh's nodes with :class:`im2sim.layers.MaskRasterizer`.

    Each node marks the voxel it falls in and the result is dilated to close the gaps between
    nodes. This is the mask the models' `MaskRasterizer` layer produces, so it is what the
    network sees. It is only as accurate as the node spacing: use a volume mesh, as a surface
    mesh with few nodes leaves holes in the middle. Voxel `(i, j, k)` covers
    `origin + [i, i + 1) * spacing` here, as in `pointcloud_to_mask`.

    Args:
        mesh: Anything :func:`~im2sim.plot.data.to_pyvista` accepts.
        shape (tuple of int): Grid size `(X, Y, Z)`.
        spacing, origin (optional): Voxel size and the position of the grid's corner.

    Returns:
        np.ndarray: Boolean array of `shape`.
    """
    import torch
    from torch_geometric.data import Data

    from im2sim.layers import MaskRasterizer

    shape = tuple(int(s) for s in shape)
    points = np.asarray(to_pyvista(mesh).points, dtype=np.float32) - np.asarray(
        origin, dtype=np.float32
    )
    graph = Data(coords=torch.from_numpy(points), batch=torch.zeros(len(points), dtype=torch.long))
    layer = MaskRasterizer(voxel_sizes=tuple(float(v) for v in spacing))
    return layer(graph, torch.zeros(1, 1, *shape))[0, -1].numpy() > 0


def rasterise_mesh(mesh, shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), fill=True):
    """
    Rasterise a surface mesh onto a voxel grid.

    Voxel `(i, j, k)` is centred at `origin + (i, j, k) * spacing`, so with the defaults mesh
    coordinates are voxel indices, matching :func:`im2sim.layers.rasterization.pointcloud_to_mask`.

    Args:
        mesh: Anything :func:`~im2sim.plot.data.to_pyvista` accepts. Volume meshes are reduced
            to their outer surface.
        shape (tuple of int): Grid size `(X, Y, Z)`.
        spacing (tuple of float, optional): Voxel size along each axis. Default is `1`.
        origin (tuple of float, optional): Position of voxel `(0, 0, 0)`. Default is `0`.
        fill (bool, optional): Mark every voxel inside a closed surface (which must lie inside
            the grid). If False, only the voxels the surface passes through (a shell).
            Default is `True`.

    Returns:
        np.ndarray: Boolean array of `shape`.
    """
    pv = _import_pyvista()
    shape = tuple(int(s) for s in shape)
    if len(shape) != 3:
        raise ValueError("shape must have three entries")
    surface = to_pyvista(mesh)
    if not isinstance(surface, pv.PolyData):
        surface = surface.extract_surface(algorithm="dataset_surface")
    surface = surface.triangulate()
    if surface.n_cells == 0:
        return np.zeros(shape, dtype=bool)

    grid = pv.ImageData(dimensions=shape, spacing=tuple(spacing), origin=tuple(origin))
    grid.compute_implicit_distance(surface, inplace=True)
    signed = np.asarray(grid.point_data["implicit_distance"], dtype=float)
    # pyvista orders points with x varying fastest.
    signed = signed.reshape(shape[::-1]).transpose(2, 1, 0)
    half_diagonal = 0.5 * float(np.linalg.norm(spacing))
    shell = np.abs(signed) <= half_diagonal
    if not fill:
        return shell

    # Fill the shell rather than testing points against the surface, which depends on the
    # orientation of its normals. A shell cut off by the edge of the grid is left unfilled.
    from scipy.ndimage import binary_fill_holes

    filled = binary_fill_holes(shell)
    core = filled & ~shell
    if not core.any():  # thinner than the shell: nothing to tell inside from outside
        return filled
    # The shell is up to half a voxel too thick; keep only voxels whose centre is inside.
    # Which sign of the distance means inside follows from the core, whatever the normals do.
    inside_sign = 1.0 if np.median(signed[core]) >= 0 else -1.0
    return filled & (signed * inside_sign >= 0)


class Mask:
    """
    A voxel mask drawn over the image.

    Args:
        data (array or torch.Tensor): Mask with the same shape as the image. Non-zero is inside.
            Use `labels` for integer label maps.
        color (str, optional): Colour of the mask. Default is `"red"`.
        alpha (float, optional): Opacity of the filled mask, 0 to 1. Default is `0.4`.
        outline (bool, optional): Also draw the mask boundary, fully opaque. Default is `True`.
        labels (list of int, optional): Treat `data` as a label map and draw only these labels.
        name (str, optional): Legend entry.
    """

    def __init__(self, data, *, color="red", alpha=0.4, outline=True, labels=None, name=None):
        self.data = data
        self.color = color
        self.alpha = alpha
        self.outline = outline
        self.labels = labels
        self.name = name

    def voxels(self, shape, spacing, origin):
        """The boolean mask on the image grid."""
        data = _to_numpy(self.data)
        if data.shape != tuple(shape):
            raise ValueError(f"mask has shape {data.shape} but the image has shape {tuple(shape)}")
        if self.labels is not None:
            return np.isin(data, self.labels)
        return data != 0


class MeshMask(Mask):
    """
    A surface mesh rasterised onto the image grid and drawn as a mask.

    Mesh coordinates are taken to be in the same space as the image: voxel `(i, j, k)` sits at
    `origin + (i, j, k) * spacing` (the plot's `spacing` and `origin`, unless overridden
    here). The mesh is rasterised once and then reused for every slice.

    Args:
        mesh: A pyvista mesh, PyG `Data` with `pos` and `face`, or anything else
            :func:`~im2sim.plot.data.to_pyvista` accepts.
        method (str, optional): "nodes" (default) rasterises the mesh's nodes with the
            :class:`~im2sim.layers.MaskRasterizer` layer: fast, and the same mask the models
            see, but about a voxel fatter on every side, and it needs a volume mesh (or enough
            nodes) to fill the inside. "exact" rasterises the surface itself, see
            :func:`rasterise_mesh`: accurate but slow (tens of seconds at 128^3).
        fill (bool, optional): With "exact", fill the inside of a closed surface; False draws
            the shell only. Default is `True`.
        spacing, origin (optional): Override the image's voxel size / origin for this mesh.
        **kwargs: As for :class:`Mask`.
    """

    def __init__(self, mesh, *, method="nodes", fill=True, spacing=None, origin=None, **kwargs):
        super().__init__(None, **kwargs)
        if method not in ("nodes", "exact"):
            raise ValueError("method must be 'nodes' or 'exact'")
        self.mesh = mesh
        self.method = method
        self.fill = fill
        self.spacing = spacing
        self.origin = origin
        self._cache = None

    def voxels(self, shape, spacing, origin):
        key = (tuple(shape), tuple(spacing), tuple(origin))
        if self._cache is None or self._cache[0] != key:
            spacing, origin = self.spacing or spacing, self.origin or origin
            if self.method == "nodes":
                voxels = rasterise_nodes(self.mesh, shape, spacing, origin)
            else:
                voxels = rasterise_mesh(self.mesh, shape, spacing, origin, fill=self.fill)
            self._cache = (key, voxels)
        return self._cache[1]


def _boundary(mask2d):
    """Pixels of a boolean 2D mask that touch a pixel outside it (4-connected)."""
    padded = np.pad(mask2d, 1)
    inner = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:] & mask2d
    return mask2d & ~inner


def _take(volume, axis, index):
    return np.take(volume, index, axis=axis)


class VolumePlot:
    """
    Step through a 3D image slice by slice, with masks overlaid, and save it as a GIF.

    Args:
        image (array or torch.Tensor): `[X, Y, Z]` image, or `[C, X, Y, Z]` / `[1, C, X, Y, Z]`
            of which `channel` is shown.
        masks (list of Mask, optional): Drawn in order, later ones on top. Bare arrays or
            tensors are wrapped as :class:`Mask`, and pyvista meshes as :class:`MeshMask`.
        axis (int, optional): Axis to slide along: 0, 1 or 2. Default is `2`.
        channel (int, optional): Channel to show when `image` has channels. Default is `0`.
        spacing (tuple of float, optional): Voxel size, used to set the aspect ratio of each
            slice and to place meshes. Default is `(1, 1, 1)`.
        origin (tuple of float, optional): Position of voxel (0, 0, 0). Default is `(0, 0, 0)`.
        cmap (str, optional): Image colormap. Default is `"gray"`.
        clim (tuple, optional): Image intensity limits. Default is the 1st to 99th percentile of
            the whole volume, so brightness is comparable between slices.
        slices (iterable of int, optional): Which slice indices to draw. Default is all.
            Slices with no masks in them are still drawn unless `only_masked` is set.
        only_masked (bool, optional): Skip slices where no mask has any voxels. Default is `False`.
        title (str, optional): Title above each slice; the slice index is appended.
        legend (bool, optional): Show a legend for masks that have a `name`. Default is `True`.
        figsize (tuple, optional): Size in inches. Default is `(4, 4)`.
        dpi (int, optional): Default is `100`.
    """

    def __init__(
        self,
        image,
        masks=None,
        *,
        axis=2,
        channel=0,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        cmap="gray",
        clim=None,
        slices=None,
        only_masked=False,
        title=None,
        legend=True,
        figsize=(4.0, 4.0),
        dpi=100,
    ):
        image = _to_numpy(image).astype(float)
        if image.ndim == 5:
            if image.shape[0] != 1:
                raise ValueError("pass a single image, not a batch")
            image = image[0]
        if image.ndim == 4:
            image = image[channel]
        if image.ndim != 3:
            raise ValueError(f"expected a 3D image, got shape {image.shape}")
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")

        self.image = image
        self.axis = axis
        self.spacing = tuple(float(s) for s in spacing)
        self.origin = tuple(float(o) for o in origin)
        self.cmap = cmap
        if clim is None:
            lo, hi = np.percentile(image, [1, 99])
            clim = (float(lo), float(hi))
        self.clim = clim
        self.only_masked = only_masked
        self.title = title
        self.legend = legend
        self.figsize = figsize
        self.dpi = dpi
        self.masks = [self._wrap(m) for m in (masks or [])]

        n = image.shape[axis]
        self.slices = list(range(n)) if slices is None else [int(i) for i in slices]
        for i in self.slices:
            if not 0 <= i < n:
                raise ValueError(f"slice {i} out of range for axis {axis} with {n} slices")

    @staticmethod
    def _wrap(mask):
        if isinstance(mask, Mask):
            return mask
        try:
            pv = _import_pyvista()
        except ImportError:
            pv = None
        if (pv is not None and isinstance(mask, pv.DataSet)) or hasattr(mask, "edge_index"):
            return MeshMask(mask)
        return Mask(mask)

    def _volumes(self):
        shape = self.image.shape
        return [m.voxels(shape, self.spacing, self.origin) for m in self.masks]

    def _frame_indices(self, volumes):
        if not self.only_masked:
            return self.slices
        return [i for i in self.slices if any(_take(v, self.axis, i).any() for v in volumes)]

    def _plane(self, array, index):
        """A 2D slice with rows/columns arranged so the plot is oriented like the array."""
        return _take(array, self.axis, index).T

    def _aspect(self):
        axes = [a for a in range(3) if a != self.axis]  # (horizontal, vertical) after transpose
        return self.spacing[axes[1]] / self.spacing[axes[0]]

    def render(self, index, volumes=None):
        """Draw one slice and return it as an `[H, W, 3]` uint8 array."""
        import matplotlib

        matplotlib.use("Agg", force=False)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import to_rgb
        from matplotlib.figure import Figure as MplFigure
        from matplotlib.patches import Patch

        volumes = self._volumes() if volumes is None else volumes
        fig = MplFigure(figsize=self.figsize, dpi=self.dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 0.94 if self.title else 1.0])
        ax.set_axis_off()
        ax.imshow(
            self._plane(self.image, index),
            cmap=self.cmap,
            vmin=self.clim[0],
            vmax=self.clim[1],
            origin="lower",
            aspect=self._aspect(),
            interpolation="nearest",
        )

        for mask, volume in zip(self.masks, volumes, strict=True):
            plane = self._plane(volume, index)
            if not plane.any():
                continue
            overlay = np.zeros((*plane.shape, 4))
            overlay[plane, :3] = to_rgb(mask.color)
            overlay[plane, 3] = mask.alpha
            if mask.outline:
                overlay[_boundary(plane), 3] = 1.0
            ax.imshow(
                overlay,
                origin="lower",
                aspect=self._aspect(),
                interpolation="nearest",
            )

        if self.title:
            ax.set_title(f"{self.title}  [{index}]", fontsize=9)
        else:
            ax.text(
                0.02, 0.98, str(index), transform=ax.transAxes, va="top", color="white", fontsize=9
            )
        named = [m for m in self.masks if m.name]
        if self.legend and named:
            ax.legend(
                handles=[
                    Patch(color=m.color, alpha=max(m.alpha, 0.5), label=m.name) for m in named
                ],
                loc="lower right",
                fontsize=7,
                framealpha=0.6,
            )

        canvas.draw()
        frame = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        return frame

    def frames(self):
        """All frames as a list of `[H, W, 3]` uint8 arrays."""
        volumes = self._volumes()
        indices = self._frame_indices(volumes)
        if not indices:
            raise ValueError("no slices to draw (is every mask empty with only_masked=True?)")
        return [self.render(i, volumes) for i in indices]

    def save_gif(self, filename, fps=10, loop=0):
        """
        Write the slice sweep as an animated GIF.

        Args:
            filename (str or Path): Output path.
            fps (float, optional): Frames per second. Default is `10`.
            loop (int, optional): Times to repeat; 0 loops forever. Default is `0`.
        """
        from PIL import Image

        images = [Image.fromarray(f) for f in self.frames()]
        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            duration=max(1, round(1000 / fps)),
            loop=loop,
            disposal=2,
        )

    def save_image(self, filename, index=None):
        """Save a single slice as an image (the middle slice by default)."""
        from PIL import Image

        index = self.slices[len(self.slices) // 2] if index is None else index
        Image.fromarray(self.render(index)).save(filename)
