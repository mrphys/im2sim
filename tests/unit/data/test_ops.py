import numpy as np
import pytest

from im2sim.data.ops import *

eps = 1e-8


def test_normtorange_basic():
    x = np.array([0, 5, 10])
    result = normtorange(x, min=0, max=10, a=0, b=1)
    expected = np.array([0.0, 0.5, 1.0])
    assert np.allclose(result, expected)


def test_normtorange_auto_min_max():
    x = np.array([2, 4, 6])
    result = normtorange(x)
    expected = np.array([0.0, 0.5, 1.0])
    assert np.allclose(result, expected)


def test_normtorange_custom_range():
    x = np.array([0, 5, 10])
    result = normtorange(x, min=0, max=10, a=-1, b=1)
    expected = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(result, expected)


def test_inv_normtorange_basic():
    x = np.array([0.0, 0.5, 1.0])
    result = inv_normtorange(x, min=0, max=10, a=0, b=1)
    expected = np.array([0, 5, 10])
    assert np.allclose(result, expected)


def test_norm_inverse_consistency():
    x = np.random.rand(10)
    normed = normtorange(x)
    recovered = inv_normtorange(normed, min=x.min(), max=x.max())
    assert np.allclose(x, recovered)


def test_normalise_wrapper():
    x = np.array([1, 2, 3])
    assert np.allclose(normalise(x), normtorange(x))


def test_inv_normalise_wrapper():
    x = np.array([0.0, 0.5, 1.0])
    assert np.allclose(inv_normalise(x, 0, 10), inv_normtorange(x, 0, 10))


def test_standardise_basic():
    x = np.array([1, 2, 3])
    result = standardise(x)
    assert np.isclose(result.mean(), 0.0, atol=1e-7)
    assert np.isclose(result.std(), 1.0, atol=1e-7)


def test_standardise_with_given_params():
    x = np.array([1, 2, 3])
    result = standardise(x, mean=2, std=1)
    expected = np.array([-1, 0, 1])
    assert np.allclose(result, expected)


def test_inv_standardise_basic():
    x = np.array([-1, 0, 1])
    result = inv_standardise(x, mean=2, std=1)
    expected = np.array([1, 2, 3])
    assert np.allclose(result, expected)


def test_standardise_inverse_consistency():
    x = np.random.rand(10)
    mean = x.mean()
    std = x.std()
    standardised = standardise(x, mean, std)
    recovered = inv_standardise(standardised, mean, std)
    assert np.allclose(x, recovered)


def test_zero_variance_handling():
    x = np.array([5, 5, 5])
    result = standardise(x)
    # should not produce NaN due to eps
    assert not np.any(np.isnan(result))


################## FITTABLE OPS #################

