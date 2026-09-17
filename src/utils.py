from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


def load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads CIFAR-10 as normalized, batch-first arrays.

    Already implemented -- data-loading boilerplate, not an exercise.
    Downloads the dataset into ./data on first use.

    Returns:
        A tuple (x_train, x_test, y_train, y_test). x_train and x_test have
        shape (m, 32, 32, 3) with pixel values scaled to [0, 1]. y_train and
        y_test have shape (m,) with integer labels 0-9.
    """
    train_set = CIFAR10(root='./data', train=True, download=True)
    test_set = CIFAR10(root='./data', train=False, download=True)

    x_train = train_set.data / 255.0
    y_train = np.array(train_set.targets)
    x_test = test_set.data / 255.0
    y_test = np.array(test_set.targets)

    return x_train, x_test, y_train, y_test


def plot_examples(x: np.ndarray, y: np.ndarray, n: int = 8) -> None:
    """Plots a random sample of images with their labels.

    Already implemented -- plotting helper, not an exercise.

    Args:
        x: Image data of shape (m, 32, 32, 3).
        y: Integer labels of shape (m,).
        n: Number of examples to plot.
    """
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    idx = np.random.choice(x.shape[0], n, replace=False)
    for ax, i in zip(axes, idx):
        ax.imshow(x[i])
        ax.set_title(CLASS_NAMES[y[i]], fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Converts integer labels to one-hot vectors.

    Args:
        y: Integer labels of shape (m,), in the range [0, n_classes).
        n_classes: Number of classes.

    Returns:
        One-hot encoded labels of shape (m, n_classes).
    """
    # TODO: implement
    pass


def random_mini_batches(
    x: np.ndarray, y: np.ndarray, batch_size: int = 64, seed: int = 0
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Splits a dataset into shuffled mini-batches.

    Args:
        x: Input data with the example index as its first axis.
        y: One-hot labels of shape (m, n_classes).
        batch_size: Number of examples per batch.
        seed: Random seed; pass a different value each call (e.g. the
            epoch number) to reshuffle.

    Returns:
        A list of (x_batch, y_batch) tuples. The last batch may be smaller
        than batch_size.
    """
    # TODO: implement
    pass
