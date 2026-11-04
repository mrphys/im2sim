"""Regression tests for bugs found in review (data pipeline, transforms, losses and blocks)."""

import pytest
import torch
import torch_geometric as pyg

from im2sim.configs import GraphConvBlockConfig, ImageConvBlockConfig
from im2sim.data.core import Dataset, Pipeline, Transform
from im2sim.layers import GraphConvBlock, ImageConvBlock
from im2sim.losses.confusion_losses import DiceLoss
from im2sim.losses.mesh import FaceNormalLoss
from im2sim.losses.pointcloud import ChamferLoss
from im2sim.transforms import transforms as T
from im2sim.transforms.ops import FitZScoreOp, PowerScaleOp, RangeNormOp


def _channels_first_dataset(n=3):
    """(C=2, H, W) images whose channels have very different statistics."""
    gen = torch.Generator().manual_seed(0)
    imgs = [
        torch.stack([torch.randn(6, 6, generator=gen) + 10, torch.randn(6, 6, generator=gen) * 100])
        for _ in range(n)
    ]
    return Dataset(lambda i: {"img": imgs[int(i)]}, [str(i) for i in range(n)]), imgs


# -----------------------------------------------------------------------------
# 1. Fitting with a non-negative channel_dim
# -----------------------------------------------------------------------------


def test_pipeline_fit_per_channel_with_channels_first_layout():
    ds, imgs = _channels_first_dataset()
    p = Pipeline([T.FitZScore(keys="img", per_channel=True, channel_dim=0)])
    p.fit(ds)

    assert len(p.transforms[0].op) == 2  # one op per channel, not per batch element
    out = torch.stack([p(ds[i])["img"] for i in range(len(ds))])
    assert torch.allclose(out.mean(dim=(0, 2, 3)), torch.zeros(2), atol=1e-4)
    assert torch.allclose(out.std(dim=(0, 2, 3), unbiased=False), torch.ones(2), atol=1e-3)


def test_pipeline_fit_matches_for_channels_first_and_last():
    ds, imgs = _channels_first_dataset()
    ds_last = Dataset(lambda i: {"img": imgs[int(i)].permute(1, 2, 0)}, ds.cases)

    first = Pipeline([T.FitZScore(keys="img", per_channel=True, channel_dim=0)]).fit(ds)
    last = Pipeline([T.FitZScore(keys="img", per_channel=True, channel_dim=-1)]).fit(ds_last)

    for a, b in zip(first.transforms[0].op, last.transforms[0].op, strict=True):
        assert torch.allclose(a.mean, b.mean) and torch.allclose(a.std, b.std)


def test_explained_variance_with_channels_first_layout():
    ds, _ = _channels_first_dataset()
    ev = T.explained_variance(ds, "img", channel_dim=0)
    assert ev["explained_variance"].shape == (2,)
    assert torch.isclose(ev["cumulative_ratio"][-1], torch.tensor(1.0), atol=1e-5)


# -----------------------------------------------------------------------------
# 2. PowerScaleOp inverse
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("exp", [0.5, 2.0, 3.0])
def test_power_scale_roundtrip_near_zero(exp):
    op = PowerScaleOp(exp)
    x = torch.tensor([-1.0, -1e-3, -1e-6, 0.0, 1e-6, 1e-3, 1.0, 5.0], dtype=torch.float64)
    assert torch.allclose(op.inverse(op.forward(x)), x, rtol=1e-6, atol=1e-9)


# -----------------------------------------------------------------------------
# 3. Weighted confusion losses with explicit class weights
# -----------------------------------------------------------------------------


def test_weighted_dice_normalises_class_weights():
    y = torch.ones(2, 3, 4, 4)
    p = torch.full_like(y, 0.5)
    macro = DiceLoss()(y, p)

    assert torch.isclose(DiceLoss(average="weighted", class_weights=torch.ones(3))(y, p), macro)
    # scaling the weights must not change the loss
    w = torch.tensor([1.0, 2.0, 3.0])
    a = DiceLoss(average="weighted", class_weights=w)(y, p)
    b = DiceLoss(average="weighted", class_weights=10 * w)(y, p)
    assert torch.isclose(a, b)


# -----------------------------------------------------------------------------
# 4. "average" residuals with several sources
# -----------------------------------------------------------------------------


def test_graph_block_average_residual_weights_sources_equally():
    cfg = GraphConvBlockConfig(depth=3, residual_connections={2: [0, 1]}, residual_type="average")
    block = GraphConvBlock(2, 2, cfg)
    outputs = [torch.full((3, 2), 1.0), torch.full((3, 2), 2.0)]
    x = torch.full((3, 2), 6.0)
    out = block._check_and_apply_residual(x, 2, outputs)
    assert torch.allclose(out, torch.full((3, 2), 3.0))  # (1 + 2 + 6) / 3


def test_image_block_average_residual_weights_sources_equally():
    cfg = ImageConvBlockConfig(depth=3, residual_connections={2: [0, 1]}, residual_type="average")
    block = ImageConvBlock(2, 2, 2, cfg)
    outputs = [torch.full((1, 2, 4, 4), 1.0), torch.full((1, 2, 4, 4), 2.0)]
    x = torch.full((1, 2, 4, 4), 6.0)
    out = block._check_and_apply_residual(x, 2, outputs)
    assert torch.allclose(out, torch.full_like(x, 3.0))


def test_concat_residual_keeps_channel_order():
    cfg = GraphConvBlockConfig(depth=3, residual_connections={2: [0, 1]}, residual_type="concat")
    block = GraphConvBlock(1, 1, cfg)
    outputs = [torch.full((2, 1), 1.0), torch.full((2, 1), 2.0)]
    x = torch.full((2, 1), 3.0)
    out = block._check_and_apply_residual(x, 2, outputs)
    # same order as before the fix: last source first, then the layer input
    assert torch.equal(out[0], torch.tensor([2.0, 1.0, 3.0]))


# -----------------------------------------------------------------------------
# 5. Dataset with a single Transform
# -----------------------------------------------------------------------------


def test_dataset_accepts_single_transform():
    ds = Dataset(
        lambda c: {"a": torch.tensor([0.0, 5.0, 10.0])},
        ["c"],
        transforms=Transform(RangeNormOp(0, 1), keys="a"),
    )
    assert torch.allclose(ds[0]["a"], torch.tensor([0.0, 0.5, 1.0]), atol=1e-6)


def test_dataset_rejects_invalid_transforms():
    with pytest.raises(TypeError):
        Dataset(lambda c: {}, ["c"], transforms="not a transform")


# -----------------------------------------------------------------------------
# 6. Duplicate transform names
# -----------------------------------------------------------------------------


def test_pipeline_rename_does_not_clash_with_later_names():
    ops = [RangeNormOp(0, 1), RangeNormOp(0, 2), RangeNormOp(0, 3)]
    names = ["n", "n", "n_2"]
    p = Pipeline([Transform(op, keys="a", name=n) for op, n in zip(ops, names, strict=True)])

    assert len({t.name for t in p.transforms}) == 3
    assert p.transforms[2].name == "n_2"  # explicitly given names are kept
    assert len(p.state_dict()) == 3


def test_pipeline_fit_does_not_rename_transforms():
    ds, _ = _channels_first_dataset()
    ts = [Transform(FitZScoreOp(), keys="img", name=n) for n in ["z", "z", "z_2"]]
    p = Pipeline(ts)
    names = [t.name for t in p.transforms]
    p.fit(ds)
    assert [t.name for t in p.transforms] == names


# -----------------------------------------------------------------------------
# 7. Mesh losses on unbatched graphs
# -----------------------------------------------------------------------------


def _tri_graph(coords, batched):
    faces = torch.tensor([[0, 0], [1, 2], [2, 3]])
    g = pyg.data.Data(coords=coords, face_index=faces, num_nodes=coords.shape[0])
    return pyg.data.Batch.from_data_list([g]) if batched else g


def test_chamfer_loss_accepts_unbatched_graphs():
    torch.manual_seed(0)
    a = pyg.data.Data(coords=torch.rand(8, 3))
    b = pyg.data.Data(coords=torch.rand(6, 3))
    unbatched = ChamferLoss()(a, b)
    batched = ChamferLoss()(pyg.data.Batch.from_data_list([a]), pyg.data.Batch.from_data_list([b]))
    assert torch.isclose(unbatched, batched)


@pytest.mark.parametrize("supervised", [True, False])
def test_face_normal_loss_accepts_unbatched_graphs(supervised):
    torch.manual_seed(0)
    c1, c2 = torch.rand(4, 3), torch.rand(4, 3)
    loss = FaceNormalLoss("face_index", supervised=supervised)
    unbatched = loss(_tri_graph(c1, False), _tri_graph(c2, False))
    batched = loss(_tri_graph(c1, True), _tri_graph(c2, True))
    assert torch.allclose(unbatched, batched)


# -----------------------------------------------------------------------------
# 8. Channel-selective transforms on integer tensors
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("per_channel", [False, True])
def test_channel_transform_promotes_integer_input(per_channel):
    x = torch.tensor([[0, 7], [10, 7], [5, 7]], dtype=torch.int64)
    t = T.Norm(keys="img", channels=[0], per_channel=per_channel)
    out = t({"img": x})["img"]

    assert out.is_floating_point()
    assert torch.allclose(out[:, 0], torch.tensor([0.0, 1.0, 0.5]), atol=1e-6)
    assert torch.equal(out[:, 1], torch.full((3,), 7.0))  # other channels are unchanged
