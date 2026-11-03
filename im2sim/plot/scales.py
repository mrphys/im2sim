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

from dataclasses import dataclass

import numpy as np


@dataclass
class ColorScale:
    """
    A colour scale that any number of layers can share by name.

    Args:
        cmap (str): Colormap name. Default is `"viridis"`.
        clim (str or tuple): Colour limits. `"frame"` rescales to the values in each frame,
            `"global"` uses one range over every frame, and `(vmin, vmax)` fixes them.
            Layers sharing the scale share the range, across panels. Default is `"frame"`.
        title (str, optional): Colour bar title. Default is the scale's name.
        show_bar (bool): Draw a colour bar. Default is `True`.
        fmt (str): printf-style format for the colour bar labels. Default is `"%.3g"`.
    """

    cmap: str = "viridis"
    clim: str | tuple[float, float] = "frame"
    title: str | None = None
    show_bar: bool = True
    fmt: str = "%.3g"

    def __post_init__(self):
        if isinstance(self.clim, str):
            if self.clim not in ("frame", "global"):
                raise ValueError("clim must be 'frame', 'global' or a (vmin, vmax) pair")
        elif len(self.clim) != 2:
            raise ValueError("clim must be 'frame', 'global' or a (vmin, vmax) pair")
        else:
            self.clim = (float(self.clim[0]), float(self.clim[1]))


def value_range(arrays):
    """`(min, max)` over the finite values of all arrays, or None if there are none."""
    values = [np.ravel(a) for a in arrays if a is not None and np.size(a)]
    if not values:
        return None
    values = np.concatenate(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return float(values.min()), float(values.max())
