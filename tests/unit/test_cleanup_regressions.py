import subprocess
import sys

import numpy as np
import pytest
import torch

from im2sim.configs import GraphConvBlockConfig, ImageConvBlockConfig
from im2sim.layers import DefaultGraphNorm, DepthwiseConv, GraphConvBlock, ImageConvBlock
from im2sim.mesh_ops import make_padded_batch
from im2sim.utils.pyg_ops import knn_interpolate


def test_prelu_slopes_are_not_shared_between_image_block_layers():
    block = ImageConvBlock(2, 2, 2, ImageConvBlockConfig(depth=3, activation="PReLU"))
    prelus = [m for m in block.modules() if isinstance(m, torch.nn.PReLU)]
    assert len({id(m) for m in prelus}) == 3


def test_prelu_slopes_are_not_shared_between_graph_block_layers():
    block = GraphConvBlock(2, 2, GraphConvBlockConfig(depth=3, activation="PReLU"))
    prelus = [m for m in block.modules() if isinstance(m, torch.nn.PReLU)]
    assert len({id(m) for m in prelus}) == 2  # the last layer has no activation


def test_default_graph_norm_handles_single_value_graphs():
    norm = DefaultGraphNorm(1)
    out = norm(torch.rand(1, 1))
    assert out.shape == (1, 1) and torch.isfinite(out).all()

    # a batch mixing a single-node graph with a larger one
    x = torch.rand(4, 1)
    out = norm(x, torch.tensor([0, 1, 1, 1]))
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_invalid_block_configs_raise_value_errors():
    with pytest.raises(ValueError, match="Unsupported attention"):
        ImageConvBlock(2, 2, 2, ImageConvBlockConfig(depth=1).mod(attn_cfg=_layer("Nope")))


def _layer(name):
    from im2sim.configs.core import LayerConfig

    return LayerConfig(name=name, kwargs={})


def test_depthwise_conv_validates_channels_with_a_value_error():
    with pytest.raises(ValueError, match="multiple of in_channels"):
        DepthwiseConv(3, 4, rank=2)


def test_knn_interpolate_returns_zero_for_queries_without_neighbours():
    x = torch.ones(3, 2)
    pos_x = torch.rand(3, 3)
    pos_y = torch.rand(2, 3)
    out = knn_interpolate(
        x, pos_x, pos_y, batch_x=torch.zeros(3, dtype=torch.long), batch_y=torch.tensor([0, 1]), k=2
    )
    assert torch.isfinite(out).all()
    assert torch.equal(out[1], torch.zeros(2))  # batch element 1 has no source points
    assert torch.allclose(out[0], torch.ones(2))


def test_knn_interpolate_keeps_half_precision_dtype_and_exact_matches():
    pos = torch.rand(10, 3)
    x = torch.rand(10, 2).half()
    out = knn_interpolate(x, pos, pos, k=1)
    assert out.dtype == torch.half
    assert torch.allclose(out.float(), x.float())


def test_make_padded_batch_mask_follows_the_device_of_x():
    x = torch.randn(5, 2)
    padded, mask = make_padded_batch(x, torch.tensor([0, 0, 0, 1, 2]))
    assert padded.shape == (3, 3, 2) and mask.device == x.device
    assert mask.sum().item() == 5


def test_mesh_ops_imports_without_pyvista():
    code = "import numpy, sys; sys.modules['pyvista'] = None; import im2sim.mesh_ops"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_get_node_features_returns_float32_and_supports_vectors():
    pv = pytest.importorskip("pyvista")
    from im2sim.mesh_ops import get_node_features

    mesh = pv.Sphere()
    mesh.point_data["p"] = np.arange(mesh.n_points, dtype=np.float64)
    mesh.point_data["v"] = np.random.rand(mesh.n_points, 3)

    feats = get_node_features(mesh, ["p", "v"])
    assert feats.shape == (mesh.n_points, 4) and feats.dtype == torch.float32
    assert torch.equal(feats[:, 0], torch.arange(mesh.n_points, dtype=torch.float32))
