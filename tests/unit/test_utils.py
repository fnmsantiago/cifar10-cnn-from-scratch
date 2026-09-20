"""Unit tests for the data and batching helpers in ``src/utils.py``.

Run every test in the project:
    pytest -v

Run just this file:
    pytest tests/unit/test_utils.py -v
"""
import numpy as np
import pytest

from src.utils import load_cifar10, one_hot_encode, random_mini_batches


# ---------------------------------------------------------------------------
# load_cifar10() — normalized batch-first arrays: (m, 32, 32, 3) images
# ---------------------------------------------------------------------------
# The first run downloads CIFAR-10 (~170 MB) into ./data; later runs read the
# cached copy. Reading it is orders of magnitude slower than the arithmetic
# these tests assert on, so the dataset is loaded once per module through the
# fixture below rather than once per test.

@pytest.fixture(scope="module")
def cifar10():
    """The (x_train, x_test, y_train, y_test) tuple, loaded once per module."""
    return load_cifar10()


def test_load_cifar10_shapes(cifar10):
    """Images must be batch-first (m, 32, 32, 3); labels must be flat (m,)."""
    x_train, x_test, y_train, y_test = cifar10

    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000,)
    assert y_test.shape == (10000,)


def test_load_cifar10_returns_plain_numpy_arrays(cifar10):
    """All four splits must be numpy arrays, not torchvision Dataset objects."""
    for split in cifar10:
        assert isinstance(split, np.ndarray)


def test_load_cifar10_image_dtype_is_float_and_label_dtype_is_integer(cifar10):
    """Pixel data must be floating point; labels must stay integer-coded."""
    x_train, x_test, y_train, y_test = cifar10

    for images in (x_train, x_test):
        assert np.issubdtype(images.dtype, np.floating)

    for labels in (y_train, y_test):
        assert np.issubdtype(labels.dtype, np.integer)


def test_load_cifar10_pixel_values_are_scaled_to_unit_range(cifar10):
    """Every pixel must land in [0, 1]; the raw 0-255 range would explode training."""
    x_train, x_test, _, _ = cifar10

    for images in (x_train, x_test):
        assert images.min() >= 0.0
        assert images.max() <= 1.0

    # The scaled range must actually be used, not a narrower slice of it.
    np.testing.assert_allclose(x_train.min(), 0.0)
    np.testing.assert_allclose(x_train.max(), 1.0)


def test_load_cifar10_pixels_are_raw_bytes_divided_by_255(cifar10):
    """Multiplying pixels by 255 must recover whole numbers, catching /256 or standardization."""
    x_train, x_test, _, _ = cifar10

    for images in (x_train, x_test):
        raw = images * 255
        np.testing.assert_allclose(raw, np.round(raw), atol=1e-9)


def test_load_cifar10_labels_are_valid_class_indices(cifar10):
    """Labels must index one of the ten classes, with no class missing."""
    _, _, y_train, y_test = cifar10

    for labels in (y_train, y_test):
        assert labels.min() >= 0
        assert labels.max() < 10
        assert np.unique(labels).size == 10


def test_load_cifar10_splits_keep_their_official_class_balance(cifar10):
    """CIFAR-10 ships 5000 train / 1000 test images per class, so labels stay aligned."""
    _, _, y_train, y_test = cifar10

    expected_train = np.full(10, 5000)
    expected_test = np.full(10, 1000)

    np.testing.assert_array_equal(np.bincount(y_train, minlength=10), expected_train)
    np.testing.assert_array_equal(np.bincount(y_test, minlength=10), expected_test)

# ---------------------------------------------------------------------------
# one_hot_encode() / random_mini_batches() — roadmap item 3
# ---------------------------------------------------------------------------
# TODO: write these tests next. Both functions are still stubs in
# src/utils.py, so writing the tests first should produce a red suite:
# one_hot_encode() must return (m, n_classes) rows, and random_mini_batches()
# must shuffle while keeping each x_batch row-aligned with its y_batch.
