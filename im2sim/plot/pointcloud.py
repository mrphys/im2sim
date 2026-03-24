import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PointCloudPlot():

    def __init__(self,
                 nrows,
                 ncols,
                 point_sets,
                 color_sets=None,
                 figsize=None,
                 cmap='Blues_r',
                 elev=20,
                 azim=90):

        self.cmap = cmap
        self.elev = elev
        self.azim = azim

        if color_sets is None:
            color_sets = [
                0.1 * np.ones(points.shape[0])
                for points in point_sets
            ]

        if figsize is None:
            figsize = (ncols * 3, nrows * 3)

        self.fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize,
            subplot_kw={'projection': '3d'}
        )

        self.axes = np.array(axes).reshape(-1)
        self.scatters = []

        maxs = np.max(point_sets[-1],axis=0)
        mins = np.min(point_sets[-1],axis=0)

        for i, (ax, points, colors) in enumerate(
                zip(self.axes, point_sets, color_sets)):
            print(points.shape, colors.shape)
            sc = ax.scatter(points[:, 0],
                            points[:, 1],
                            points[:, 2],
                            c=colors,
                            cmap=cmap,
                            vmin=colors.min(),
                            vmax=colors.max())

            ax.view_init(elev=elev,
                         azim=azim,
                         vertical_axis='y')

            # if i == 0:
            #     lims = ax.get_w_lims()
            # else:
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
            plt.colorbar(sc, ax=ax, shrink=0.5)

            self.scatters.append(sc)

        self.point_sets = point_sets
        self.color_sets = color_sets

    # ---------------------------------------------------------
    # DRAW ONE FRAME (works for static OR animation)
    # ---------------------------------------------------------

    def draw_frame(self, point_sets=None, color_sets=None):

        if point_sets is None:
            point_sets = self.point_sets

        if color_sets is None:
            color_sets = self.color_sets

        new_scatters = []

        for ax, sc, pts, colors in zip(
                self.axes,
                self.scatters,
                point_sets,
                color_sets):

            # If number of points changed → recreate scatter
            if sc is None or len(pts) != len(sc.get_offsets()):

                if sc is not None:
                    sc.remove()

                sc = ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    c=colors,
                    cmap=self.cmap,
                    vmin=0,
                    vmax=1
                )

            else:
                sc._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
                sc.set_array(colors)

            new_scatters.append(sc)

        self.scatters = new_scatters

        return self.scatters

    # ---------------------------------------------------------
    # SAVE SINGLE IMAGE
    # ---------------------------------------------------------

    def save_image(self, filename, dpi=200):
        plt.tight_layout()
        self.fig.savefig(filename, dpi=dpi)
        plt.close(self.fig)

    # ---------------------------------------------------------
    # ANIMATE
    # ---------------------------------------------------------

    def animate(self,
                point_sequence_sets,
                color_sequence_sets=None,
                filename="animation.gif",
                fps=15):

        n_frames = len(point_sequence_sets)

        def update(frame):

            if color_sequence_sets is None:
                colors = None
            else:
                colors = color_sequence_sets[frame]

            return self.draw_frame(
                point_sets=point_sequence_sets[frame],
                color_sets=colors
            )

        ani = animation.FuncAnimation(
            self.fig,
            update,
            frames=n_frames,
            blit=False
        )

        if filename.endswith(".gif"):
            ani.save(filename, writer="pillow", fps=fps)
        else:
            ani.save(filename, writer="ffmpeg", fps=fps)

        plt.close(self.fig)