"""Categorical cross-entropy loss, for one-hot multi-class targets."""

import numpy as np


class CategoricalCrossEntropyLoss:
    """Categorical cross-entropy loss over C classes."""

    def __init__(self) -> None:
        """Initialize the loss module."""
        self.predictions = None
        self.targets = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the average categorical cross-entropy loss.

        Args:
            predictions (np.ndarray): softmax probabilities,
                shape (m, C). Clip away from exactly 0 before use.
            targets (np.ndarray): one-hot true labels, shape
                (m, C).

        Returns:
            float: the scalar loss, averaged over the batch.
        """
        self.predictions = np.clip(predictions, 1e-12, 1.0 - 1e-12)
        self.targets = targets
        loss = -np.sum(targets * np.log(self.predictions), axis=1)
        return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. predictions.

        Returns:
            np.ndarray: dL/da, shape (m, C), same shape as the
                predictions passed to forward. Use the same
                clipped predictions here as in forward.
        """
        m = self.predictions.shape[0]
        return -(1.0 / m) * (self.targets / self.predictions)
