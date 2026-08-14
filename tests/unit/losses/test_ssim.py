import pytest
import torch

from im2sim.src.losses.ssim import SSIMLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_images(shape, dtype=torch.float32):
    """Create a pair of random images with values in [0, 1]."""
    y_true = torch.rand(shape, dtype=dtype)
    y_pred = torch.rand(shape, dtype=dtype)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_identical_images_have_zero_loss():
    """SSIM loss should be zero for identical images."""
    x = torch.rand(2, 1, 32, 32)

    loss = SSIMLoss()(x, x)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


@pytest.mark.parametrize(
    "shape,rank",
    [
        ((2, 1, 32, 32), 2),
        ((2, 3, 32, 32), 2),
        ((2, 1, 16, 32, 32), 3),
        ((2, 3, 16, 32, 32), 3),
    ],
)
def test_output_is_scalar(shape, rank):
    """SSIM loss should return a scalar."""
    y_true, y_pred = _make_images(shape)

    loss = SSIMLoss(rank=rank)(y_pred, y_true)

    assert loss.ndim == 0


@pytest.mark.parametrize(
    "shape,rank",
    [
        ((2, 1, 32, 32), 2),
        ((2, 3, 32, 32), 2),
        ((2, 1, 16, 32, 32), 3),
        ((2, 3, 16, 32, 32), 3),
    ],
)
def test_loss_is_finite(shape, rank):
    """SSIM loss should produce finite values."""
    y_true, y_pred = _make_images(shape)

    loss = SSIMLoss(rank=rank)(y_pred, y_true)

    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Rank / dimension handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,rank,batch_dims",
    [
        # [B, C, H, W]
        ((2, 1, 32, 32), 2, 1),

        # [B, C, D, H, W]
        ((2, 1, 8, 32, 32), 3, 1),

        # [B, D, C, H, W]
        # D is treated as an additional batch dimension.
        ((2, 8, 1, 32, 32), 2, 2),

        # [B, T, C, H, W]
        ((2, 4, 1, 32, 32), 2, 2),

        # [B, T, C, D, H, W]
        ((2, 4, 1, 8, 32, 32), 3, 2),
    ],
)
def test_batch_and_image_dimensions(
    shape,
    rank,
    batch_dims,
):
    """Different batch/image dimension configurations should work."""
    y_true, y_pred = _make_images(shape)

    loss = SSIMLoss(
        rank=rank,
        batch_dims=batch_dims,
    )(y_pred, y_true)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_2d_ssim_can_be_applied_to_3d_image():
    """2D SSIM should be calculable independently for every 3D slice."""
    y_true = torch.rand(2, 1, 8, 32, 32)
    y_pred = torch.rand(2, 1, 8, 32, 32)

    loss = SSIMLoss(
        rank=2,
        batch_dims=2,
    )(y_pred, y_true)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_2d_ssim_slice_batching_matches_explicit_loop():
    """Treating slices as batch dimensions should match an explicit loop."""
    y_true = torch.rand(2, 1, 5, 32, 32)
    y_pred = torch.rand(2, 1, 5, 32, 32)

    loss = SSIMLoss(
        rank=2,
        batch_dims=2,
    )(y_pred, y_true)

    slice_loss = torch.stack(
        [
            SSIMLoss(rank=2)(y_pred[:, :, i], y_true[:, :, i])
            for i in range(5)
        ]
    ).mean()

    assert torch.allclose(
        loss,
        slice_loss,
        atol=1e-6,
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Automatic dimension inference
# ---------------------------------------------------------------------------


def test_batch_dims_are_inferred_from_rank():
    """batch_dims should be inferred from rank."""
    y_true, y_pred = _make_images((2, 1, 32, 32))

    explicit = SSIMLoss(
        rank=2,
        batch_dims=1,
    )(y_pred, y_true)

    inferred = SSIMLoss(
        rank=2,
    )(y_pred, y_true)

    assert torch.allclose(
        explicit,
        inferred,
        atol=1e-6,
    )


def test_image_dims_are_inferred_from_batch_dims():
    """image_dims should be inferred from batch_dims."""
    y_true, y_pred = _make_images((2, 1, 32, 32))

    explicit = SSIMLoss(
        rank=2,
        batch_dims=1,
    )(y_pred, y_true)

    inferred = SSIMLoss(
        batch_dims=1,
    )(y_pred, y_true)

    assert torch.allclose(
        explicit,
        inferred,
        atol=1e-6,
    )


def test_default_is_2d_single_batch_dimension():
    """Default configuration should interpret input as [B, C, H, W]."""
    y_true, y_pred = _make_images((2, 1, 32, 32))

    default = SSIMLoss()(y_pred, y_true)

    explicit = SSIMLoss(
        rank=2,
        batch_dims=1,
    )(y_pred, y_true)

    assert torch.allclose(
        default,
        explicit,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_mismatched_shapes_raise():
    loss = SSIMLoss()

    y_true = torch.rand(2, 1, 32, 32)
    y_pred = torch.rand(2, 1, 16, 32)

    with pytest.raises(ValueError, match="identical shapes"):
        loss(y_pred, y_true)


@pytest.mark.parametrize("rank", [0, 1, 4, -1])
def test_invalid_rank_raises(rank):
    with pytest.raises(ValueError, match="rank"):
        SSIMLoss(rank=rank)


@pytest.mark.parametrize("filter_size", [0, 2, 4, 10])
def test_invalid_filter_size_raises(filter_size):
    with pytest.raises(ValueError, match="filter_size"):
        SSIMLoss(filter_size=filter_size)


def test_negative_filter_sigma_raises():
    with pytest.raises(ValueError, match="filter_sigma"):
        SSIMLoss(filter_sigma=-1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k1": -0.1},
        {"k2": -0.1},
    ],
)
def test_negative_regularisation_factors_raise(kwargs):
    with pytest.raises(ValueError, match="k1 and k2"):
        SSIMLoss(**kwargs)


def test_negative_batch_dims_raise():
    with pytest.raises(ValueError, match="batch_dims"):
        SSIMLoss(batch_dims=-1)


def test_zero_image_dims_raise():
    with pytest.raises(ValueError, match="image_dims"):
        SSIMLoss(image_dims=0)


def test_rank_and_image_dims_must_match():
    with pytest.raises(ValueError, match="rank and image_dims"):
        SSIMLoss(rank=2, image_dims=3)


def test_invalid_batch_image_configuration_raises():
    y_true = torch.rand(2, 1, 32, 32)
    y_pred = torch.rand(2, 1, 32, 32)

    with pytest.raises(ValueError):
        SSIMLoss(
            rank=3,
            batch_dims=1,
        )(y_pred, y_true)


# ---------------------------------------------------------------------------
# SSIM properties
# ---------------------------------------------------------------------------


def test_loss_is_zero_for_identical_images():
    for rank, shape in [
        (2, (2, 1, 32, 32)),
        (3, (2, 1, 8, 32, 32)),
    ]:
        x = torch.rand(shape)

        loss = SSIMLoss(rank=rank)(x, x)

        assert torch.isclose(
            loss,
            torch.tensor(0.0),
            atol=1e-6,
        )


def test_loss_is_symmetric():
    y_true, y_pred = _make_images((2, 1, 32, 32))

    loss_fn = SSIMLoss()

    loss_xy = loss_fn(y_pred, y_true)
    loss_yx = loss_fn(y_true, y_pred)

    assert torch.allclose(
        loss_xy,
        loss_yx,
        atol=1e-6,
        rtol=1e-5,
    )


def test_constant_identical_images_have_zero_loss():
    x = torch.full(
        (2, 1, 32, 32),
        0.5,
    )

    loss = SSIMLoss()(x, x)

    assert torch.isclose(
        loss,
        torch.tensor(0.0),
        atol=1e-6,
    )


def test_different_images_have_positive_loss():
    y_true = torch.zeros(2, 1, 32, 32)
    y_pred = torch.ones(2, 1, 32, 32)

    loss = SSIMLoss()(y_pred, y_true)

    assert loss > 0


# ---------------------------------------------------------------------------
# max_val
# ---------------------------------------------------------------------------


def test_default_max_val_for_float_is_one():
    y_true = torch.rand(2, 1, 32, 32)
    y_pred = torch.rand(2, 1, 32, 32)

    default = SSIMLoss()(y_pred, y_true)
    explicit = SSIMLoss(max_val=1.0)(
        y_pred,
        y_true,
    )

    assert torch.allclose(
        default,
        explicit,
        atol=1e-6,
    )


def test_explicit_max_val_is_used():
    y_true = torch.rand(2, 1, 32, 32) * 255
    y_pred = torch.rand(2, 1, 32, 32) * 255

    default = SSIMLoss(max_val=255.0)(
        y_pred,
        y_true,
    )

    same = SSIMLoss(max_val=255.0)(
        y_pred,
        y_true,
    )

    assert torch.allclose(
        default,
        same,
        atol=1e-6,
    )


def test_non_positive_max_val_raises():
    y_true = torch.rand(2, 1, 32, 32)
    y_pred = torch.rand(2, 1, 32, 32)

    with pytest.raises(ValueError, match="max_val"):
        SSIMLoss(max_val=0)(y_pred, y_true)


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------


def test_loss_supports_gradients():
    y_true = torch.rand(2, 1, 32, 32)
    y_pred = torch.rand(
        2,
        1,
        32,
        32,
        requires_grad=True,
    )

    loss = SSIMLoss()(y_pred, y_true)

    loss.backward()

    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()


def test_gradient_is_not_zero():
    y_true = torch.rand(2, 1, 32, 32)
    y_pred = torch.rand(
        2,
        1,
        32,
        32,
        requires_grad=True,
    )

    loss = SSIMLoss()(y_pred, y_true)

    loss.backward()

    assert torch.any(y_pred.grad != 0)


# ---------------------------------------------------------------------------
# Gaussian filter
# ---------------------------------------------------------------------------


def test_gaussian_kernel_is_normalised():
    loss_fn = SSIMLoss(
        rank=2,
        filter_size=11,
        filter_sigma=1.5,
    )

    x = torch.rand(2, 3, 32, 32)

    # Initialise the lazy kernel.
    loss_fn(x, x)

    kernel = loss_fn._kernel

    assert torch.allclose(
        kernel[0, 0].sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )


def test_kernel_is_depthwise():
    """Each channel should have its own identical Gaussian kernel."""
    loss_fn = SSIMLoss(rank=2)

    x = torch.rand(2, 3, 32, 32)

    loss_fn(x, x)

    kernel = loss_fn._kernel

    assert kernel.shape == (
        3,
        1,
        11,
        11,
    )

    assert torch.allclose(
        kernel[0],
        kernel[1],
    )

    assert torch.allclose(
        kernel[1],
        kernel[2],
    )


@pytest.mark.parametrize(
    "rank,shape,expected_kernel_shape",
    [
        (
            2,
            (2, 3, 32, 32),
            (3, 1, 11, 11),
        ),
        (
            3,
            (2, 3, 8, 32, 32),
            (3, 1, 11, 11, 11),
        ),
    ],
)
def test_kernel_shape(
    rank,
    shape,
    expected_kernel_shape,
):
    loss_fn = SSIMLoss(rank=rank)

    x = torch.rand(shape)

    loss_fn(x, x)

    assert loss_fn._kernel.shape == expected_kernel_shape