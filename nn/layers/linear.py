"""Linear (fully connected) layer: z = xW + b."""

import numpy as np
from nn.module import Module


class Linear(Module):
    """A fully connected layer computing z = xW + b.

    Attributes:
        W (np.ndarray): weight matrix, shape
            (in_features, out_features).
        b (np.ndarray): bias vector, shape (out_features,).
        dW (np.ndarray): gradient buffer for self.W, same shape as self.W.
        db (np.ndarray): gradient buffer for self.b, same shape as self.b.
        x (np.ndarray): the input, saved unchanged so backward can use it.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the layer's weights, bias, and gradient buffers.

        Args:
            in_features (int): number of input features.
            out_features (int): number of output neurons.
        """
        super().__init__()
        # Xavier (Glorot) Uniform initialization
        bound = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-bound, bound, (in_features, out_features))
        self.b = np.zeros((out_features,))

        # Pre-allocate gradient buffers to ensure in-place modifications work
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute this layer's output for a batch of inputs.

        Args:
            x (np.ndarray): input, shape (batch_size, in_features).

        Returns:
            np.ndarray: output, shape (batch_size, out_features).
        """
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with
                respect to this layer's output, shape (batch_size, out_features).

        Returns:
            np.ndarray: gradient of the loss with respect to
                this layer's input, shape (batch_size, in_features).
        """
        # Gradients are accumulated over the batch axis using in-place assignment [...]
        self.dW[...] = self.x.T @ grad_output
        self.db[...] = np.sum(grad_output, axis=0)
        
        # dL/dX is not summed -- shape matches back to original input layout
        return grad_output @ self.W.T

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return this layer's learnable parameters.

        Returns:
            list[tuple[np.ndarray, np.ndarray]]: pairs of (parameter, gradient)
        """
        return [(self.W, self.dW), (self.b, self.db)]
