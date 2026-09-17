import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    """Applies the ReLU activation elementwise.

    Args:
        z: Pre-activation values of any shape.

    Returns:
        Post-activation values, same shape as z.
    """
    # TODO: implement
    pass


def relu_backward(da: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Computes the gradient of the cost with respect to a ReLU pre-activation.

    Args:
        da: Gradient of the cost with respect to the post-activation, same
            shape as z.
        z: Pre-activation values cached during the forward pass.

    Returns:
        Gradient of the cost with respect to z.
    """
    # TODO: implement
    pass


def softmax(z: np.ndarray) -> np.ndarray:
    """Applies the softmax activation to each row of z.

    Args:
        z: Pre-activation values of shape (m, num_classes).

    Returns:
        Softmax probabilities of shape (m, num_classes); each row sums to 1.
    """
    # TODO: implement
    pass
