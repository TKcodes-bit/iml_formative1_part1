"""ReLU activation: zeroes out negative values."""

import numpy as np
from nn.module import Module


class ReLU(Module):
    """ReLU activation, applied elementwise: max(0, x)."""

    def __init__(self) -> None:
        """Initialize the ReLU module."""
        super().__init__()
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute ReLU elementwise, and remember the mask.

        Args:
            x (np.ndarray): input, any shape.

        Returns:
            np.ndarray: max(0, x), elementwise, same shape as x.
        """
        self.mask = x > 0
        return np.maximum(0, x)

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
        return grad_output * self.mask
