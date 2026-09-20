"""Unit tests for the convolution and pooling layers in ``src/conv.py``.

The forward passes are covered by shape checks and hand-computed examples;
the backward passes are covered by numerical gradient checks (see the
testing philosophy in the roadmap).

Run every test in the project:
    pytest -v

Run just this file:
    pytest tests/unit/test_conv.py -v
"""
import numpy as np

from src.conv import conv2d_backward, conv2d_forward, max_pool_backward, max_pool_forward

# TODO: write tests for conv2d_forward, conv2d_backward, max_pool_forward,
# and max_pool_backward
