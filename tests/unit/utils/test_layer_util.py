import pytest
import torch
import torch.nn.functional as F

from im2sim.utils.layer_util import resize_nearest

SIZES = [1, 2, 3, 5, 7, 8, 15, 16, 17, 32, 33]


def _reference_6d(x, size):
    """Separable reference using F.interpolate: resize (D, H, W) per time step, then T."""
    n, c, t = x.shape[:3]
    out = F.interpolate(x.reshape(n, c * t, *x.shape[3:]), size=size[1:])
    s = out.shape[2:].numel()
    # (N, C*T, *S) -> (N*C, S, T) so the 1D interpolation runs over time
    out = out.reshape(n * c, t, s).transpose(1, 2)
    out = F.interpolate(out, size=size[0]).transpose(1, 2)
    return out.reshape(n, c, size[0], *size[1:])


@pytest.mark.parametrize("in_size", SIZES)
@pytest.mark.parametrize("out_size", SIZES)
def test_resize_nearest_matches_pytorch_index_rule(in_size, out_size):
    # Same rule F.interpolate uses, checked on a 6D input where each dim is resized on its own.
    x = torch.arange(in_size, dtype=torch.float32).view(1, 1, in_size, 1, 1, 1)
    expected = F.interpolate(x.view(1, 1, in_size), size=out_size).view(-1)
    assert torch.equal(resize_nearest(x, (out_size, 1, 1, 1)).view(-1), expected)


@pytest.mark.parametrize(
    "in_shape, size",
    [
        ((2, 3, 4), (7,)),
        ((2, 3, 5, 9), (10, 4)),
        ((1, 2, 3, 7, 6), (6, 14, 3)),
    ],
)
def test_resize_nearest_matches_interpolate_up_to_5d(in_shape, size):
    x = torch.randn(*in_shape)
    assert torch.equal(resize_nearest(x, size), F.interpolate(x, size=size))


@pytest.mark.parametrize(
    "in_shape, size",
    [
        ((2, 3, 4, 8, 8, 8), (4, 16, 16, 16)),
        ((2, 3, 4, 8, 8, 8), (2, 4, 4, 4)),
        ((1, 2, 3, 15, 17, 13), (2, 14, 16, 12)),
        ((1, 2, 2, 7, 6, 5), (5, 7, 6, 5)),
    ],
)
def test_resize_nearest_6d_matches_separable_reference(in_shape, size):
    x = torch.randn(*in_shape)
    out = resize_nearest(x, size)
    assert out.shape == (*in_shape[:2], *size)
    assert torch.equal(out, _reference_6d(x, size))


def test_resize_nearest_6d_same_size_is_identity():
    x = torch.randn(1, 2, 3, 4, 5, 6)
    assert torch.equal(resize_nearest(x, x.shape[2:]), x)


def test_resize_nearest_6d_gradient_flow():
    x = torch.randn(1, 2, 3, 4, 4, 4, requires_grad=True)
    resize_nearest(x, (6, 8, 8, 8)).sum().backward()
    assert x.grad is not None
    # Upsampling copies every input value at least once.
    assert (x.grad > 0).all()


def test_resize_nearest_rejects_wrong_number_of_dims():
    with pytest.raises(ValueError):
        resize_nearest(torch.randn(1, 1, 2, 4, 4, 4), (4, 4, 4))
