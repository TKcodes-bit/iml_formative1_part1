"""Binary cross-entropy loss."""

import numpy as np


class CrossEntropyLoss:
    """Binary cross-entropy loss for a single output probability."""

    def __init__(self) -> None:
        """Initialize the loss module."""
        self.predictions = None
        self.targets = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the average binary cross-entropy loss.

        Args:
            predictions (np.ndarray): predicted probabilities,
                shape (m,) or (m, 1). Clip away from exactly
                0 or 1 before use.
            targets (np.ndarray): true labels, same shape as
                predictions, values 0 or 1.

        Returns:
            float: the scalar loss, averaged over the batch.
        """
        # Boundaries are clipped to prevent running log(0) which generates NaN values
        self.predictions = np.clip(predictions, 1e-12, 1.0 - 1e-12)
        self.targets = targets
        loss = -(targets * np.log(self.predictions) + (1.0 - targets) * np.log(1.0 - self.predictions))
        return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. predictions.

        Returns:
            np.ndarray: dL/da, same shape as the predictions
                passed to forward.
        """
        m = self.predictions.size
        return -(1.0 / m) * (self.targets / self.predictions - (1.0 - self.targets) / (1.0 - self.predictions))
