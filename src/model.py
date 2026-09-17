from typing import Dict, List, Tuple

import numpy as np

from .activations import relu, relu_backward, softmax
from .conv import conv2d_backward, conv2d_forward, max_pool_backward, max_pool_forward
from .losses import categorical_cross_entropy
from .utils import one_hot_encode, random_mini_batches


class ConvNet:
    """A convolutional network for multi-class image classification.

    Architecture: [Conv -> ReLU -> MaxPool] x2 -> Flatten -> Dense -> Softmax.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        conv_filters: Tuple[int, int] = (8, 16),
        filter_size: int = 3,
        pool_size: int = 2,
    ) -> None:
        """Initializes the network's parameters.

        Args:
            input_shape: (height, width, channels) of a single input image.
            num_classes: Number of output classes.
            conv_filters: Number of filters in each of the two conv layers.
            filter_size: Height and width of each convolutional filter.
            pool_size: Height and width of each max-pooling window.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.parameters = self._initialize_parameters()

    def _initialize_parameters(self) -> Dict[str, np.ndarray]:
        """Initializes weights and biases for every layer.

        Returns:
            A dict with keys 'W1', 'b1', 'W2', 'b2' for the conv layers and
            'W3', 'b3' for the dense layer.
        """
        # TODO: implement
        pass

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Runs a forward pass through the network.

        Args:
            x: Input batch of shape (m, height, width, channels).

        Returns:
            A tuple (al, cache) where al is the softmax output of shape
            (m, num_classes) and cache holds whatever backward() needs.
        """
        # TODO: implement
        pass

    def compute_cost(self, al: np.ndarray, y: np.ndarray) -> float:
        """Computes the cost for a batch of predictions.

        Args:
            al: Softmax output of shape (m, num_classes).
            y: One-hot labels of shape (m, num_classes).

        Returns:
            The cost, as a scalar.
        """
        # TODO: implement
        pass

    def backward(
        self, al: np.ndarray, y: np.ndarray, cache: dict
    ) -> Dict[str, np.ndarray]:
        """Runs backpropagation and returns the parameter gradients.

        Args:
            al: Softmax output of shape (m, num_classes).
            y: One-hot labels of shape (m, num_classes).
            cache: Intermediate values produced by forward().

        Returns:
            A dict of gradients matching the keys in self.parameters.
        """
        # TODO: implement
        pass

    def update_parameters(
        self, grads: Dict[str, np.ndarray], learning_rate: float
    ) -> None:
        """Applies one gradient descent update to the network's parameters.

        Args:
            grads: Gradients produced by backward().
            learning_rate: Step size for the update.
        """
        # TODO: implement
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts class labels for a batch of inputs.

        Args:
            x: Input batch of shape (m, height, width, channels).

        Returns:
            Predicted integer class labels of shape (m,).
        """
        # TODO: implement
        pass

    def train(
        self,
        x: np.ndarray,
        y_int: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
        learning_rate: float = 0.1,
        print_cost: bool = True,
    ) -> List[float]:
        """Trains the network with mini-batch gradient descent.

        Args:
            x: Input data of shape (m, height, width, channels).
            y_int: Integer labels of shape (m,), not one-hot encoded.
            epochs: Number of passes over the full training set.
            batch_size: Number of examples per mini-batch.
            learning_rate: Step size for gradient descent.
            print_cost: Whether to print the cost periodically.

        Returns:
            The cost after each epoch, for plotting.
        """
        y = one_hot_encode(y_int, self.num_classes)
        costs = []

        for epoch in range(epochs):
            mini_batches = random_mini_batches(x, y, batch_size, seed=epoch)
            epoch_cost = 0.0

            for x_batch, y_batch in mini_batches:
                al, cache = self.forward(x_batch)
                epoch_cost += self.compute_cost(al, y_batch) * x_batch.shape[0]
                grads = self.backward(al, y_batch, cache)
                self.update_parameters(grads, learning_rate)

            epoch_cost /= x.shape[0]
            costs.append(epoch_cost)
            if print_cost and epoch % 5 == 0:
                print(f"Epoch {epoch}: cost = {epoch_cost:.4f}")

        return costs
