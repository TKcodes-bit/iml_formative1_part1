"""Sigmoid activation: squashes real values into (0, 1)."""

import numpy as np

from nn.module import Module


class Sigmoid(Module):
    """Sigmoid activation, applied elementwise: 1 / (1 + e^{-x})."""

    def __init__(self) -> None:
        """Initialize the Sigmoid module."""
        super().__init__()
        self.a = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid elementwise, and remember the output.

        Args:
            x (np.ndarray): input, any shape.

        Returns:
            np.ndarray: sigmoid(x), elementwise, same shape as x.
        """
        self.a = 1.0 / (1.0 + np.exp(-x))
        return self.a

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with
                respect to this layer's output, same shape as
                the original input to forward.

        Returns:
            np.ndarray: gradient of the loss with respect to
                this layer's input, same shape as grad_output.
        """
        return grad_output * self.a * (1.0 - self.a)
