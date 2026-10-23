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

"""Result container shared by every lab experiment."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LabResult:
    """Outcome of one lab experiment on one model / graph / seed.

    Args:
        name (str): Experiment name, e.g. ``"jacobian_reach"``.
        scalars (dict[str, float]): Headline numbers (one value each).
        arrays (dict[str, np.ndarray]): Curves / per-node data used for plotting.
        model (str): Label of the model that was tested.
        graph (str): Label of the graph the experiment ran on.
        seed (int): Random seed.
        meta (dict): Free-form settings used to produce the result.
    """

    name: str
    scalars: dict[str, float] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    model: str = ""
    graph: str = ""
    seed: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.model or self.name

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "graph": self.graph,
            "seed": self.seed,
            "scalars": {k: float(v) for k, v in self.scalars.items()},
            "arrays": {k: np.asarray(v).tolist() for k, v in self.arrays.items()},
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LabResult:
        return cls(
            name=d["name"],
            model=d.get("model", ""),
            graph=d.get("graph", ""),
            seed=d.get("seed", 0),
            scalars=dict(d.get("scalars", {})),
            arrays={k: np.asarray(v) for k, v in d.get("arrays", {}).items()},
            meta=d.get("meta", {}),
        )


def save_results(results: Sequence[LabResult], path: str | Path) -> None:
    """Write results to a JSON file."""
    Path(path).write_text(json.dumps([r.to_dict() for r in results]))


def load_results(path: str | Path) -> list[LabResult]:
    """Read results written by :func:`save_results`."""
    return [LabResult.from_dict(d) for d in json.loads(Path(path).read_text())]


def results_to_frame(results: Sequence[LabResult]):
    """Tabulate the scalars of each result (one row per result) as a DataFrame."""
    import pandas as pd

    rows = [
        {"experiment": r.name, "model": r.model, "graph": r.graph, "seed": r.seed, **r.scalars}
        for r in results
    ]
    return pd.DataFrame(rows)


def experiment(trained: bool = False) -> Callable:
    """Mark a function as a lab experiment.

    Args:
        trained (bool): True if the experiment trains the model (and so needs a model
            factory rather than a ready-made adapter).
    """

    def deco(fn):
        fn.trained = trained
        return fn

    return deco


def is_trained(fn: Callable) -> bool:
    """Whether ``fn`` (possibly a ``functools.partial``) is a training experiment."""
    while hasattr(fn, "func") and not hasattr(fn, "trained"):
        fn = fn.func
    return bool(getattr(fn, "trained", False))
