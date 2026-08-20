import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


class PointCloudPlot:
    """
    A class to create a grid of 3D scatter plots for visualizing point clouds.

    Args:
        nrows (int): Number of rows in the grid.
        ncols (int): Number of columns in the grid.
        point_sets (list of np.ndarray): List of point clouds to plot, each as an (N, 3) array of 3D coordinates.
        color_sets (list of np.ndarray, optional): List of color values for each point cloud. If None, all points will be colored uniformly.
        figsize (tuple, optional): Size of the figure. If None, it will be set based on nrows and ncols.
        cmap (str, optional): Colormap to use for coloring the points. Default is "Blues_r".
        norm_mode (str, optional): Normalization mode for color values. Options are 'all', 'row', 'col', or 'none'. Default is 'none'.
        bound_mode (str, optional): Mode for setting axis bounds. Default is 'all'.
        titles (list of str, optional): List of titles for each subplot. If None, no titles will be set.
        elev (float, optional): Elevation angle for the 3D view. Default is 20.
        azim (float, optional): Azimuth angle for the 3D view. Default is 90.
    """

    def __init__(
        self,
        nrows,
        ncols,
        point_sets,
        color_sets=None,
        figsize=None,
        cmap="Blues_r",
        norm_mode="none",  # 'all', 'row', 'col', 'none'
        bound_mode="all",
        titles=None,  # NEW: optional titles
        elev=20,
        azim=90,
    ):

        self.cmap = cmap
        self.elev = elev
        self.azim = azim
        self.nrows = nrows
        self.ncols = ncols
        self.norm_mode = norm_mode

        if color_sets is None:
            is_colored = False
            color_sets = [0.1 * np.ones(points.shape[0]) for points in point_sets]
        else:
            is_colored = True

        if figsize is None:
            figsize = (ncols * 3, nrows * 3)

        self.fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, subplot_kw={"projection": "3d"}
        )

        self.axes = np.array(axes).reshape(-1)
        self.scatters = []

        # -----------------------------
        # Titles handling
        # -----------------------------
        if titles is not None:
            if isinstance(titles, str):
                self.fig.suptitle(titles)
                titles = [None] * len(self.axes)
            elif len(titles) != len(self.axes):
                raise ValueError("titles must match number of subplots")

        # -----------------------------
        # Bounds (same as your logic)
        # -----------------------------
        maxs = np.max(point_sets[-1], axis=0)
        mins = np.min(point_sets[-1], axis=0)

        # -----------------------------
        # Normalization helpers
        # -----------------------------
        def compute_norm_ranges(color_sets):
            color_sets = list(color_sets)

            if norm_mode == "none":
                return [(None, None)] * len(color_sets)

            elif norm_mode == "all":
                all_vals = np.concatenate(color_sets)
                vmin, vmax = all_vals.min(), all_vals.max()
                return [(vmin, vmax)] * len(color_sets)

            elif norm_mode == "row":
                ranges = []
                for r in range(nrows):
                    row_vals = []
                    for c in range(ncols):
                        idx = r * ncols + c
                        row_vals.append(color_sets[idx])
                    row_vals = np.concatenate(row_vals)
                    vmin, vmax = row_vals.min(), row_vals.max()
                    for _ in range(ncols):
                        ranges.append((vmin, vmax))
                return ranges

            elif norm_mode == "col":
                ranges = [None] * len(color_sets)
                for c in range(ncols):
                    col_vals = []
                    for r in range(nrows):
                        idx = r * ncols + c
                        col_vals.append(color_sets[idx])
                    col_vals = np.concatenate(col_vals)
                    vmin, vmax = col_vals.min(), col_vals.max()
                    for r in range(nrows):
                        idx = r * ncols + c
                        ranges[idx] = (vmin, vmax)
                return ranges

            else:
                raise ValueError("norm_mode must be 'all', 'row', 'col', or 'none'")

        norm_ranges = compute_norm_ranges(color_sets)

        # -----------------------------
        # Plot creation
        # -----------------------------
        for i, (ax, points, colors) in enumerate(
            zip(self.axes, point_sets, color_sets, strict=True)
        ):
            vmin, vmax = norm_ranges[i]

            sc = ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            ax.view_init(elev=elev, azim=azim, vertical_axis="y")

            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])

            # Apply subplot title
            if titles is not None and titles[i] is not None:
                ax.set_title(titles[i])

            if is_colored:
                plt.colorbar(sc, ax=ax, shrink=0.5)

            self.scatters.append(sc)

        self.point_sets = point_sets
        self.color_sets = color_sets

    # ---------------------------------------------------------
    # DRAW ONE FRAME
    # ---------------------------------------------------------

    def draw_frame(self, point_sets=None, color_sets=None):

        if point_sets is None:
            point_sets = self.point_sets

        if color_sets is None:
            color_sets = self.color_sets

        # recompute normalization each frame
        def compute_norm_ranges(color_sets):
            color_sets = list(color_sets)

            if self.norm_mode == "none":
                return [(None, None)] * len(color_sets)

            elif self.norm_mode == "all":
                all_vals = np.concatenate(color_sets)
                vmin, vmax = all_vals.min(), all_vals.max()
                return [(vmin, vmax)] * len(color_sets)

            elif self.norm_mode == "row":
                ranges = []
                for r in range(self.nrows):
                    row_vals = []
                    for c in range(self.ncols):
                        idx = r * self.ncols + c
                        row_vals.append(color_sets[idx])
                    row_vals = np.concatenate(row_vals)
                    vmin, vmax = row_vals.min(), row_vals.max()
                    for _ in range(self.ncols):
                        ranges.append((vmin, vmax))
                return ranges

            elif self.norm_mode == "col":
                ranges = [None] * len(color_sets)
                for c in range(self.ncols):
                    col_vals = []
                    for r in range(self.nrows):
                        idx = r * self.ncols + c
                        col_vals.append(color_sets[idx])
                    col_vals = np.concatenate(col_vals)
                    vmin, vmax = col_vals.min(), col_vals.max()
                    for r in range(self.nrows):
                        idx = r * self.ncols + c
                        ranges[idx] = (vmin, vmax)
                return ranges

        norm_ranges = compute_norm_ranges(color_sets)

        new_scatters = []

        for i, (ax, sc, pts, colors) in enumerate(
            zip(self.axes, self.scatters, point_sets, color_sets, strict=True)
        ):
            vmin, vmax = norm_ranges[i]

            if sc is None or len(pts) != len(sc.get_offsets()):
                if sc is not None:
                    sc.remove()

                sc = ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    c=colors,
                    cmap=self.cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

            else:
                sc._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
                sc.set_array(colors)
                if vmin is not None:
                    sc.set_clim(vmin, vmax)

            new_scatters.append(sc)

        self.scatters = new_scatters
        return self.scatters

    # ---------------------------------------------------------
    # SAVE IMAGE
    # ---------------------------------------------------------

    def save_image(self, filename, dpi=200):
        plt.tight_layout()
        self.fig.savefig(filename, dpi=dpi)
        plt.close(self.fig)

    # ---------------------------------------------------------
    # ANIMATE
    # ---------------------------------------------------------

    def animate(
        self,
        point_sequence_sets,
        color_sequence_sets=None,
        filename="animation.gif",
        fps=15,
    ):

        n_frames = len(point_sequence_sets)

        def update(frame):

            colors = None if color_sequence_sets is None else color_sequence_sets[frame]

            return self.draw_frame(point_sets=point_sequence_sets[frame], color_sets=colors)

        ani = animation.FuncAnimation(self.fig, update, frames=n_frames, blit=False)

        if filename.endswith(".gif"):
            ani.save(filename, writer="pillow", fps=fps)
        else:
            ani.save(filename, writer="ffmpeg", fps=fps)

        plt.close(self.fig)
