"""Softmax activation: converts logits into a probability distribution."""

import numpy as np
from nn.module import Module


class Softmax(Module):
    """Softmax activation, applied row-wise to a batch of logits."""

    def __init__(self) -> None:
        """Initialize the Softmax module."""
        super().__init__()
        self.a = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities for a batch of logits.

        Args:
            x (np.ndarray): logits, shape (batch_size, C).

        Returns:
            np.ndarray: probabilities, shape (batch_size, C).
                Each row sums to 1.
        """
        # Max-subtraction trick for strict numerical stability against overflow
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        self.a = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.a

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with
                respect to this layer's output, shape (batch_size, C).

        Returns:
            np.ndarray: gradient of the loss with respect to
                this layer's input (the logits), shape (batch_size, C).
        """
        # Vectorized batch implementation of the row-wise Jacobian dot-product
        sum_grad_a = np.sum(grad_output * self.a, axis=1, keepdims=True)
        return self.a * (grad_output - sum_grad_a)
