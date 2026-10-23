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

"""Run a set of experiments over several models and seeds."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch

from im2sim.lab.adapters import ModelAdapter
from im2sim.lab.results import LabResult, is_trained


def run_suite(
    models: Mapping[str, ModelAdapter | Callable],
    experiments: Mapping[str, Callable],
    seeds: Sequence[int] = (0,),
    untrained_channels: tuple[int, int] = (1, 1),
    verbose: bool = True,
) -> list[LabResult]:
    """Run every experiment on every model for every seed.

    Args:
        models: ``label -> ModelAdapter`` (usable by untrained experiments only) or
            ``label -> factory(in_channels, out_channels) -> ModelAdapter`` (usable by all).
            Factories are called after ``torch.manual_seed(seed)``, so seeds vary the
            initialisation.
        experiments: ``label -> fn(model, seed=...) -> LabResult``; use ``functools.partial``
            to bind graphs and settings, e.g. ``partial(jacobian_reach, graph=g)``. The
            ``label`` becomes ``LabResult.name`` so one experiment can be run on several graphs.
        seeds: Random seeds.
        untrained_channels: ``(in, out)`` channels used to instantiate a factory for an untrained
            experiment (ignored for ready-made adapters).
        verbose: Print progress.

    Returns:
        list[LabResult]: One result per (model, experiment, seed) that could run. Trained
        experiments given a bare adapter are skipped.
    """
    results = []
    for mname, model in models.items():
        for ename, fn in experiments.items():
            for seed in seeds:
                is_adapter = isinstance(model, ModelAdapter)
                if is_trained(fn) and is_adapter:
                    if verbose:
                        print(f"skip {ename} on {mname}: needs a factory, got an adapter")
                    continue
                torch.manual_seed(seed)
                arg = model if is_adapter or is_trained(fn) else model(*untrained_channels)
                if verbose:
                    print(f"run  {ename} | {mname} | seed {seed}")
                res = fn(arg, seed=seed)
                res.name, res.model, res.seed = ename, mname, seed
                results.append(res)
    return results
