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
Lightweight layers for propagating dense boundary conditions across a mesh.

The layers avoid ``[E, C]`` edge tensors: neighbourhood terms come from one sparse matmul
(:class:`SparseOps`) and the virtual nodes are reached by pooling and cross-attention
(:class:`StructVirtualNode`). Everything derived from a graph's structure is built once per
forward pass in a :class:`GraphContext` and shared by all layers.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree, scatter

from im2sim.configs.core import LayerConfig
from im2sim.utils.layer_util import PYG_LAYERS, get_activation, register_pyg_layer


def _coords(graph: Data) -> torch.Tensor:
    """Node coordinates as float ``[N, 3]``, from ``coords`` (im2sim) or ``pos`` (PyG)."""
    coords = graph.coords if "coords" in graph else graph.pos
    return _pad_coords(coords)


def _pad_coords(coords: torch.Tensor) -> torch.Tensor:
    coords = coords.float()
    if coords.size(1) < 3:
        coords = F.pad(coords, (0, 3 - coords.size(1)))
    return coords


class SparseOps:
    """
    Sparse neighbourhood operators of a graph.

    ``feats(h)`` returns ``[h, mean_j h_j, d_x h, d_y h, d_z h]`` where ``d_k h`` is the mean over
    neighbours of ``rhat_k * (h_j - h_i)``: a scale-free directional difference, so the message
    pass is anisotropic and geometry-aware. One sparse matmul computes all four neighbourhood
    terms and no ``[E, C]`` tensor is ever stored.

    Args:
        coords (torch.Tensor): Node coordinates of shape (N, 3).
        edge_index (torch.Tensor): Edges of shape (2, E), messages flow from row 0 to row 1.
    """

    def __init__(self, coords: torch.Tensor, edge_index: torch.Tensor):
        n = coords.size(0)
        src, dst = edge_index
        w = 1.0 / degree(dst, n).clamp(min=1)[dst]
        r = coords[src] - coords[dst]
        rhat = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        vals = torch.cat([w, *(w * rhat[:, k] for k in range(3))])
        rows = torch.cat([dst + k * n for k in range(4)])
        cols = src.repeat(4)
        self.op = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (4 * n, n)).coalesce()
        self.bias = scatter(w[:, None] * rhat, dst, 0, dim_size=n, reduce="sum")  # (N, 3)
        self.n = n

    def feats(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h (torch.Tensor): Node features of shape (N, C).

        Returns:
            torch.Tensor: ``[h, mean, d_x, d_y, d_z]`` of shape (N, 5 * C).
        """
        out = torch.sparse.mm(self.op, h).view(4, self.n, -1)
        grads = out[1:] - self.bias.t()[:, :, None] * h[None]
        return torch.cat([h, out[0], *grads], dim=-1)


class GraphContext:
    """
    Everything derived from a graph's structure, built lazily and shared by all layers.

    Args:
        graph (pyg.data.Data): A graph, or a batch of graphs, with ``coords`` (or ``pos``) and
            ``edge_index``.
        structure_keys (Sequence[str]): Graph attributes holding the node indices of each
            structure, in order. Nodes in several structures belong to the last one (e.g. the
            rim where a cap meets the wall). Only needed by :attr:`assign`.
    """

    def __init__(self, graph: Data, structure_keys: Sequence[str]):
        self.graph = graph
        self.coords = _coords(graph)
        self.edge_index = graph.edge_index
        self.n = self.coords.size(0)
        self.structure_keys = tuple(structure_keys)
        self.batch = (
            graph.batch
            if "batch" in graph and graph.batch is not None
            else torch.zeros(self.n, dtype=torch.long, device=self.coords.device)
        ).long()
        num_graphs = getattr(graph, "num_graphs", None)
        self.num_graphs = int(num_graphs) if num_graphs is not None else int(self.batch.max()) + 1
        self._cache: dict = {}

    def _cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def ops(self) -> SparseOps:
        return self._cached("ops", lambda: SparseOps(self.coords, self.edge_index))

    @property
    def assign(self) -> torch.Tensor:
        """Structure id of every node, shape (N,); ``-1`` for nodes in no structure."""

        def build():
            assign = torch.full((self.n,), -1, dtype=torch.long, device=self.coords.device)
            for i, key in enumerate(self.structure_keys):
                if key not in self.graph:
                    raise ValueError(f"Structure key '{key}' not found in graph.")
                assign[self.graph[key].long()] = i
            return assign

        return self._cached("assign", build)

    def pos_features(self, n_fourier: int = 4) -> torch.Tensor:
        """
        Per-graph bounding-box normalised coordinates with Fourier features.

        Returns:
            torch.Tensor: Shape (N, 3 + 6 * n_fourier).
        """

        def build():
            lo = scatter(self.coords, self.batch, 0, reduce="min")[self.batch]
            hi = scatter(self.coords, self.batch, 0, reduce="max")[self.batch]
            half = ((hi - lo).max(1, keepdim=True).values / 2).clamp(min=1e-8)
            p = (self.coords - (hi + lo) / 2) / half
            freqs = 2.0 ** torch.arange(n_fourier, device=p.device) * math.pi
            ang = p[:, :, None] * freqs
            return torch.cat([p, ang.sin().flatten(1), ang.cos().flatten(1)], 1)

        return self._cached(("pos_features", n_fourier), build)


@register_pyg_layer(name="DirectionalConv")
class DirectionalConv(torch.nn.Linear):
    """
    Graph convolution on ``[h, mean, d_x h, d_y h, d_z h]`` neighbourhood features (see
    :class:`SparseOps`): a linear map of the node and its mean and directional differences.

    Registered as a graph layer, so it can be used as the ``conv_cfg`` of a
    :class:`~im2sim.configs.GraphConvBlockConfig` (the graph needs ``coords`` or ``pos``).
    Inside a larger model, call :meth:`apply` with shared :class:`SparseOps` instead, so the
    operator is only built once.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(5 * in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, coords: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Node features of shape (N, C_in).
            edge_index (torch.Tensor): Edges of shape (2, E).
            coords (torch.Tensor): Node coordinates of shape (N, 3).

        Returns:
            torch.Tensor: Node features of shape (N, C_out).
        """
        return self.apply(x, SparseOps(_pad_coords(coords), edge_index))

    def apply(self, x: torch.Tensor, ops: SparseOps) -> torch.Tensor:
        """Same as :meth:`forward` with precomputed operators."""
        return super().forward(ops.feats(x))


class DirectionalBlock(torch.nn.Module):
    """
    Residual local layer: ``h + out(act(norm(DirectionalConv(h))))``.

    Args:
        channels (int): Number of channels, kept by the block.
        activation (str | None): Activation function. Default is ``"GELU"``.
    """

    def __init__(self, channels: int, activation: str | None = "GELU"):
        super().__init__()
        self.mix = DirectionalConv(channels, channels)
        self.norm = torch.nn.LayerNorm(channels)
        self.act = get_activation(activation)
        self.out = torch.nn.Linear(channels, channels)

    def forward(self, h: torch.Tensor, ops: SparseOps) -> torch.Tensor:
        """
        Args:
            h (torch.Tensor): Node features of shape (N, C).
            ops (SparseOps): Neighbourhood operators of the graph.

        Returns:
            torch.Tensor: Node features of shape (N, C).
        """
        return h + self.out(self.act(self.norm(self.mix.apply(h, ops))))


class LocalBlock(torch.nn.Module):
    """
    Residual local layer with a configurable graph convolution: ``h + out(act(norm(conv(h))))``.

    The same block as :class:`DirectionalBlock`, but the convolution is looked up by name in the
    PyG layer registry, so any layer taking ``(x, edge_index)`` can be used, e.g. ``GATConv``,
    ``GCNConv`` or ``SAGEConv``, as well as ``DirectionalConv``.

    Args:
        channels (int): Number of channels, kept by the block.
        conv_cfg (LayerConfig | None): The convolution, built as ``conv(channels, channels,
            **conv_cfg.kwargs)``. It must output ``channels`` channels, so a multi-head layer needs
            e.g. ``kwargs={"heads": 4, "concat": False}``. Default is ``None``, which is
            ``DirectionalConv``.
        activation (str | None): Activation function. Default is ``"GELU"``.
    """

    def __init__(
        self,
        channels: int,
        conv_cfg: LayerConfig | None = None,
        activation: str | None = "GELU",
    ):
        super().__init__()
        name = "DirectionalConv" if conv_cfg is None else conv_cfg.name
        if name is None or name not in PYG_LAYERS:
            raise ValueError(f"Local convolution {name} not found in the PyG layers registry.")
        kwargs = {} if conv_cfg is None else conv_cfg.kwargs
        self.mix = PYG_LAYERS[name](channels, channels, **kwargs)
        self.norm = torch.nn.LayerNorm(channels)
        self.act = get_activation(activation)
        self.out = torch.nn.Linear(channels, channels)

    @staticmethod
    def uses_sparse_ops(conv_cfg: LayerConfig | None) -> bool:
        """Whether the convolution takes :class:`SparseOps` (``DirectionalConv``) rather than ``edge_index``."""
        return conv_cfg is None or issubclass(PYG_LAYERS[conv_cfg.name], DirectionalConv)

    def forward(self, h: torch.Tensor, neighbours: "SparseOps | torch.Tensor") -> torch.Tensor:
        """
        Args:
            h (torch.Tensor): Node features of shape (N, C).
            neighbours (SparseOps | torch.Tensor): The :class:`SparseOps` of the graph if
                :meth:`uses_sparse_ops`, otherwise its ``edge_index`` of shape (2, E).

        Returns:
            torch.Tensor: Node features of shape (N, C).
        """
        if isinstance(self.mix, DirectionalConv):
            mixed = self.mix.apply(h, neighbours)
        else:
            mixed = self.mix(h, neighbours)
        return h + self.out(self.act(self.norm(mixed)))


class StructVirtualNode(torch.nn.Module):
    """
    Nodes -> one virtual node (VN) per mesh structure -> nodes.

    Each VN is the mean of its structure's (projected) node features, so a dense boundary
    condition is summarised separately for the wall and each cap. The VNs exchange information
    (self-attention + MLP) and every node reads from them with cross-attention. Messages cost
    ``O(N * K)`` for ``K`` VNs and no ``[N * K, C]`` tensor is built for a single graph.

    Args:
        channels (int): Number of channels, kept by the layer.
        n_structures (int): Number of structures, i.e. VNs per graph.
        heads (int): Attention heads of the VN self-attention. Default is 4.
        activation (str | None): Activation function of the VN MLP. Default is ``"GELU"``.
    """

    def __init__(
        self, channels: int, n_structures: int, heads: int = 4, activation: str | None = "GELU"
    ):
        super().__init__()
        self.channels = channels
        self.n_structures = n_structures
        self.pool_v = torch.nn.Linear(channels, channels)
        self.emb = torch.nn.Parameter(torch.randn(n_structures, channels) * 0.1)
        self.norm1, self.norm2 = torch.nn.LayerNorm(channels), torch.nn.LayerNorm(channels)
        self.attn = torch.nn.MultiheadAttention(channels, heads, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(channels, 2 * channels),
            get_activation(activation),
            torch.nn.Linear(2 * channels, channels),
        )
        self.q = torch.nn.Linear(channels, channels)
        self.kk = torch.nn.Linear(channels, channels)
        self.v = torch.nn.Linear(channels, channels)
        self.out = torch.nn.Linear(channels, channels)

    def pool(self, h: torch.Tensor, ctx: GraphContext) -> torch.Tensor:
        """Mean of each structure's nodes per graph, shape (B, K, C). Empty structures give 0."""
        b, k = ctx.num_graphs, self.n_structures
        assign = ctx.assign
        keep = assign >= 0
        ids = ctx.batch * k + assign
        tokens = scatter(self.pool_v(h)[keep], ids[keep], 0, dim_size=b * k, reduce="mean")
        return tokens.view(b, k, self.channels)

    def forward(self, h: torch.Tensor, ctx: GraphContext) -> torch.Tensor:
        """
        Args:
            h (torch.Tensor): Node features of shape (N, C).
            ctx (GraphContext): Context of the graph, with ``len(structure_keys) == n_structures``.

        Returns:
            torch.Tensor: Node features of shape (N, C).
        """
        if len(ctx.structure_keys) != self.n_structures:
            raise ValueError(
                f"The layer has {self.n_structures} structures but the context defines "
                f"{len(ctx.structure_keys)} ({ctx.structure_keys})."
            )
        tok = self.pool(h, ctx) + self.emb
        t = self.norm1(tok)
        tok = tok + self.attn(t, t, t, need_weights=False)[0]
        tok = tok + self.mlp(self.norm2(tok))
        key, val = self.kk(tok), self.v(tok)  # (B, K, C)
        scale = math.sqrt(self.channels)
        if ctx.num_graphs == 1:  # plain matmuls, no per-node gather
            attn = torch.softmax(self.q(h) @ key[0].t() / scale, dim=1)
            return h + self.out(attn @ val[0])
        key, val = key[ctx.batch], val[ctx.batch]  # (N, K, C)
        attn = torch.softmax((self.q(h)[:, None] * key).sum(-1) / scale, dim=1)
        return h + self.out((attn[:, :, None] * val).sum(1))
