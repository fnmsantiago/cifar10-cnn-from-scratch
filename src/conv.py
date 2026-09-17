from typing import Tuple

import numpy as np


def conv2d_forward(
    a_prev: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> Tuple[np.ndarray, tuple]:
    """Applies a 2D convolution.

    Args:
        a_prev: Input batch of shape (m, h_prev, w_prev, c_prev).
        w: Filters of shape (f, f, c_prev, c).
        b: Biases of shape (1, 1, 1, c).
        stride: Stride used for both height and width.
        padding: Zero-padding applied to both height and width.

    Returns:
        A tuple (z, cache) where z has shape (m, h, w, c) and cache holds
        whatever backward() needs.
    """
    # TODO: implement
    pass


def conv2d_backward(
    dz: np.ndarray, cache: tuple
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagates through a 2D convolution.

    Args:
        dz: Gradient of the cost with respect to the conv output, same shape
            as z from forward().
        cache: The cache produced by conv2d_forward().

    Returns:
        A tuple (da_prev, dw, db) matching the shapes of a_prev, w, and b.
    """
    # TODO: implement
    pass


def max_pool_forward(
    a_prev: np.ndarray, pool_size: int = 2, stride: int = 2
) -> Tuple[np.ndarray, tuple]:
    """Applies max pooling.

    Args:
        a_prev: Input batch of shape (m, h_prev, w_prev, c).
        pool_size: Height and width of the pooling window.
        stride: Stride used for both height and width.

    Returns:
        A tuple (a, cache) where a has shape (m, h, w, c) and cache holds
        whatever backward() needs.
    """
    # TODO: implement
    pass


def max_pool_backward(da: np.ndarray, cache: tuple) -> np.ndarray:
    """Backpropagates through max pooling.

    Args:
        da: Gradient of the cost with respect to the pooling output, same
            shape as a from forward().
        cache: The cache produced by max_pool_forward().

    Returns:
        Gradient of the cost with respect to a_prev, same shape as the
        original input.
    """
    # TODO: implement
    pass
