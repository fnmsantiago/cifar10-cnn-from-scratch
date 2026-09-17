# CIFAR-10 CNN — from scratch, then validated against a framework

A multi-class image classifier: a convolutional network implemented with
plain numpy (forward and backward propagation written by hand, including
convolution and pooling), later validated against an equivalent PyTorch
implementation.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Open `notebooks/01_train_from_scratch.ipynb` in VS Code and select the venv
as the kernel.

## Tests

```bash
pytest
```
