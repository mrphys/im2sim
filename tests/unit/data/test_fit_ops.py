import pytest
import torch

from im2sim.transforms.ops import (
    FitNormOp,
    FitRangeNormOp,
    FitZScoreOp,
    NormOp,
    normtorange,
    standardise,
)


def test_fit_zscore_matches_torch_for_large_offset_float32():
    torch.manual_seed(0)
    x = torch.randn(100_000) * 1e-2 + 1e4  # std 0.01 around 1e4
    op = FitZScoreOp()
    op.fit_step(x)
    op.complete_fit()

    assert op.mean.item() == pytest.approx(x.double().mean().item(), rel=1e-9)
    assert op.std.item() == pytest.approx(x.double().std(unbiased=False).item(), rel=1e-3)


def test_fit_zscore_streaming_matches_single_pass():
    torch.manual_seed(0)
    x = torch.randn(1000) * 3 + 7
    streamed = FitZScoreOp()
    for chunk in x.chunk(7):
        streamed.fit_step(chunk)
    streamed.complete_fit()

    assert streamed.mean.item() == pytest.approx(x.double().mean().item(), rel=1e-9)
    assert streamed.std.item() == pytest.approx(x.double().std(unbiased=False).item(), rel=1e-9)


def test_fit_zscore_without_data_raises():
    with pytest.raises(RuntimeError):
        FitZScoreOp().complete_fit()


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-6])
def test_fit_norm_inverse_is_exact_for_small_ranges(scale):
    x = torch.rand(1000, dtype=torch.float64) * scale
    op = FitNormOp()
    op.fit_step(x)
    op.complete_fit()

    restored = op.inverse(op.forward(x))
    assert ((restored - x).abs().max() / (x.max() - x.min())).item() < 1e-9


def test_fit_range_norm_inverse_is_exact_for_small_ranges():
    x = torch.rand(1000, dtype=torch.float64) * 1e-6
    op = FitRangeNormOp(-1, 1)
    op.fit_step(x)
    op.complete_fit()

    restored = op.inverse(op.forward(x))
    assert ((restored - x).abs().max() / (x.max() - x.min())).item() < 1e-9


def test_fit_zscore_inverse_is_exact_for_small_std():
    x = torch.randn(1000, dtype=torch.float64) * 1e-5 + 3
    op = FitZScoreOp()
    op.fit_step(x)
    op.complete_fit()

    restored = op.inverse(op.forward(x))
    assert ((restored - x).abs().max() / x.std()).item() < 1e-9


def test_simple_ops_unchanged():
    x = torch.tensor([0.0, 1.0, 2.0])
    assert torch.allclose(NormOp().forward(x), normtorange(x))
    assert torch.allclose(standardise(x, 1.0, 1.0), (x - 1.0) / (1.0 + 1e-8))
