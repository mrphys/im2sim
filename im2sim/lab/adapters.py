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

"""Adapters that give every GNN the same interface: ``adapter(graph, x=None) -> Tensor[N, C]``."""

from __future__ import annotations

import copy
import inspect
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager

import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

COORD_KEYS = ("pos", "coords")

#: ``factory(in_channels, out_channels) -> ModelAdapter``; builds a fresh, untrained model.
AdapterFactory = Callable[[int, int], "ModelAdapter"]


def dummy_image(shape: tuple[int, ...], channels: int = 1) -> torch.Tensor:
    """Zero image ``[1, channels, *shape]`` for im2sim models whose image input is irrelevant."""
    return torch.zeros(1, channels, *shape)


class ModelAdapter:
    """Wrap a model so lab experiments can treat it as a plain node-level function.

    Four calling conventions are supported (``kind``):

    * ``"pyg_xe"``: ``model(x, edge_index)`` returning a tensor (e.g. ``torch_geometric.nn.GCN``).
    * ``"pyg_data"``: ``model(data)`` returning a tensor or a ``Data``.
    * ``"graph"``: ``model(graph) -> Data`` (e.g. ``im2sim.models.SimpleGraphDecoder``,
      ``im2sim.layers.GraphConvBlock``).
    * ``"im2sim"``: ``model(image, graph) -> Data`` (``Im2SimBase`` / ``Im2SimGen2``).

    ``kind="auto"`` picks one from the model type / ``forward`` signature.

    Args:
        model (torch.nn.Module): The model to wrap.
        kind (str): Calling convention, or ``"auto"``.
        input_key (str): Graph attribute that experiments overwrite with test signals.
        output_key (str | None): Attribute of the returned ``Data`` holding the prediction
            (default ``"x"``). Ignored when the model returns a tensor.
        image (torch.Tensor | Callable | None): Image input for ``kind="im2sim"``; a tensor
            (expanded to the batch size) or ``fn(graph) -> tensor``. ``None`` lets a
            rasterizer-equipped model build it from the graph.
        call_fn (Callable | None): Custom ``fn(model, graph) -> Tensor`` overriding the above.
        hidden_filter (Callable | None): ``fn(name, module) -> bool`` choosing the layers
            reported by :meth:`hidden_states` (default: every ``MessagePassing`` layer).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        kind: str = "auto",
        input_key: str = "x",
        output_key: str | None = None,
        image=None,
        call_fn: Callable | None = None,
        hidden_filter: Callable[[str, torch.nn.Module], bool] | None = None,
    ):
        self.model = model
        self.kind = kind if kind != "auto" else self._detect(model)
        self.input_key = input_key
        self.output_key = output_key
        self.image = image
        self.call_fn = call_fn
        self.hidden_filter = hidden_filter or (lambda name, m: isinstance(m, MessagePassing))

    @staticmethod
    def _detect(model) -> str:
        from im2sim.models.im2sim_models import Im2SimBase

        if isinstance(model, Im2SimBase):
            return "im2sim"
        try:
            from im2sim.layers import GraphConvBlock
            from im2sim.models import SimpleGraphDecoder

            if isinstance(model, GraphConvBlock | SimpleGraphDecoder):
                return "graph"
        except ImportError:
            pass
        params = list(inspect.signature(model.forward).parameters)
        return "pyg_xe" if params[:2] == ["x", "edge_index"] else "pyg_data"

    # ---- graph manipulation -------------------------------------------------------
    def with_input(self, graph: Data, x: torch.Tensor) -> Data:
        """Shallow copy of ``graph`` with ``input_key`` replaced by ``x`` (autograd-friendly)."""
        g = copy.copy(graph)
        g[self.input_key] = x
        return g

    @staticmethod
    def set_coords(graph: Data, coords: torch.Tensor) -> Data:
        """Shallow copy of ``graph`` with every coordinate attribute (pos / coords) replaced."""
        g = copy.copy(graph)
        for key in COORD_KEYS:
            if key in g:
                g[key] = coords
        return g

    @staticmethod
    def _prepare(graph: Data) -> Data:
        if "batch" not in graph or graph.batch is None:
            graph = copy.copy(graph)
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
        return graph

    # ---- calling ------------------------------------------------------------------
    def __call__(self, graph: Data, x: torch.Tensor | None = None) -> torch.Tensor:
        """Run the model; ``x`` (if given) replaces ``graph.<input_key>``."""
        g = self._prepare(graph if x is None else self.with_input(graph, x))
        if self.call_fn is not None:
            return self.call_fn(self.model, g)
        if self.kind == "pyg_xe":
            out = self.model(g[self.input_key], g.edge_index)
        elif self.kind == "pyg_data":
            out = self.model(g)
        elif self.kind == "graph":
            out = self.model(g)
        elif self.kind == "im2sim":
            image = self.image(g) if callable(self.image) else self.image
            if torch.is_tensor(image) and image.size(0) != int(g.batch.max()) + 1:
                image = image.expand(int(g.batch.max()) + 1, *image.shape[1:])
            out = self.model(image, g)
        else:
            raise ValueError(f"unknown kind '{self.kind}'")
        if isinstance(out, list | tuple) and not torch.is_tensor(out):
            out = out[-1]
        if isinstance(out, Data):
            out = out[self.output_key or "x"]
        return out

    def parameters(self):
        return self.model.parameters()

    @contextmanager
    def evaluating(self):
        """Temporarily put the model in eval mode."""
        was = self.model.training
        self.model.eval()
        try:
            yield self
        finally:
            self.model.train(was)

    def hidden_states(self, graph: Data, x: torch.Tensor | None = None) -> OrderedDict:
        """Detached per-layer node embeddings, captured with forward hooks.

        Returns:
            OrderedDict[str, torch.Tensor]: ``"input"`` followed by the output of every layer
            selected by ``hidden_filter`` that produced a node-level tensor ``[N, C]``.
        """
        states: OrderedDict[str, torch.Tensor] = OrderedDict()
        g = graph if x is None else self.with_input(graph, x)
        states["input"] = g[self.input_key].detach()
        n = graph.num_nodes
        handles = []
        for name, module in self.model.named_modules():
            if name and self.hidden_filter(name, module):

                def hook(_m, _i, out, name=name):
                    if isinstance(out, tuple):
                        out = out[0]
                    if isinstance(out, Data):
                        out = out.x
                    if torch.is_tensor(out) and out.dim() == 2 and out.size(0) == n:
                        states[name] = out.detach()

                handles.append(module.register_forward_hook(hook))
        try:
            with torch.no_grad(), self.evaluating():
                self(graph, x)
        finally:
            for h in handles:
                h.remove()
        return states
