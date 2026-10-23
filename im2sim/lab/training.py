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

"""Minimal training loop shared by the trained lab experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch_geometric.data import Batch, Data

from im2sim.lab.adapters import ModelAdapter


def fit(
    adapter: ModelAdapter,
    graphs: Data | Sequence[Data],
    inputs: torch.Tensor | Sequence[torch.Tensor],
    targets: torch.Tensor | Sequence[torch.Tensor],
    masks: torch.Tensor | Sequence[torch.Tensor] | None = None,
    steps: int = 300,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
) -> list[float]:
    """Fit the adapter's model so ``adapter(graph, input) ~ target`` (masked MSE, Adam).

    Args:
        adapter (ModelAdapter): Model to train in place.
        graphs: One graph or a list; they are batched, so the list entries may differ in size.
        inputs: Node inputs matching ``graphs`` (one tensor, or one per graph).
        targets: Node targets ``[N]`` or ``[N, C]`` matching ``graphs``.
        masks: Optional bool masks selecting the nodes that enter the loss.
        steps (int): Optimiser steps (full batch).
        lr (float): Adam learning rate.
        weight_decay (float): Adam weight decay.

    Returns:
        list[float]: Loss per step.
    """

    def as_list(v):
        return list(v) if isinstance(v, Sequence) else [v]

    graphs, inputs, targets = as_list(graphs), as_list(inputs), as_list(targets)
    masks = as_list(masks) if masks is not None else None
    batch = Batch.from_data_list([g.clone() for g in graphs])
    x = torch.cat(inputs)
    y = torch.cat(targets)
    if y.dim() == 1:
        y = y[:, None]
    m = torch.cat(masks) if masks is not None else torch.ones(y.size(0), dtype=torch.bool)

    opt = torch.optim.Adam(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    adapter.model.train()
    history = []
    for _ in range(steps):
        opt.zero_grad()
        pred = adapter(batch, x)
        loss = torch.nn.functional.mse_loss(pred[m], y[m])
        loss.backward()
        opt.step()
        history.append(float(loss))
    adapter.model.eval()
    return history


@torch.no_grad()
def predict(adapter: ModelAdapter, graph: Data, x: torch.Tensor) -> torch.Tensor:
    """Eval-mode prediction ``[N, C]``."""
    with adapter.evaluating():
        return adapter(graph, x)
