"""Training orchestration script for testing neural network convergence."""

import numpy as np

from nn.activations.sigmoid import Sigmoid
from nn.layers.linear import Linear
from nn.losses.cross_entropy_loss import CrossEntropyLoss
from nn.optim.sgd import SGD

# Global references so accuracy() can access the trained model weights
_layer = None
_activation = None


def toy_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate a linearly separable AND-gate toy dataset.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair containing:
            - X (np.ndarray): Input feature matrix of shape (4, 2).
            - y (np.ndarray): True binary labels vector of shape (4, 1).
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [0], [0], [1]], dtype=float)
    return X, y


def train(epochs: int = 4000, lr: float = 1.0, seed: int = 0) -> list[float]:
    """Run the model training loop over a specified number of epochs.

    Args:
        epochs (int): Number of training iterations. Defaults to 4000.
        lr (float): Learning rate for the SGD optimizer. Defaults to 1.0.
        seed (int): Random seed for reproducibility. Defaults to 0.

    Returns:
        list[float]: The scalar loss values recorded at every epoch in order.
    """
    global _layer, _activation
    np.random.seed(seed)

    X, y = toy_data()

    # Model definition: Single layer linear model with Sigmoid activation
    _layer = Linear(in_features=2, out_features=1)
    _activation = Sigmoid()
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(_layer.parameters(), lr=lr)

    loss_history = []

    for _ in range(epochs):
        # 1. Forward Pass
        z = _layer.forward(X)
        predictions = _activation.forward(z)
        loss = loss_fn.forward(predictions, y)
        loss_history.append(loss)

        # 2. Backward Pass (Reverse chain order)
        grad = loss_fn.backward()
        grad = _activation.backward(grad)
        _layer.backward(grad)

        # 3. Optimization Update
        optimizer.step()
        optimizer.zero_grad()

    return loss_history


def accuracy(loss_history: list[float] = None) -> float:
    """Compute the classification accuracy using the active model parameters.

    Args:
        loss_history (list[float], optional): Leftover history tracker reference
            required by test hooks. Defaults to None.

    Returns:
        float: Final classification accuracy score between 0.0 and 1.0.
    """
    global _layer, _activation
    # Fallback to train if the model has not been initialized yet
    if _layer is None or _activation is None:
        train()

    X, y = toy_data()
    
    # Run evaluation forward pass
    z = _layer.forward(X)
    predictions = _activation.forward(z)
    
    # Threshold predictions at 0.5 for binary classification metrics
    pred_labels = (predictions >= 0.5).astype(float)
    return float(np.mean(pred_labels == y))


if __name__ == "__main__":
    print("Starting training convergence test on AND-gate dataset...")
    history = train(epochs=4000, lr=1.0)
    final_acc = accuracy(history)
    
    print(f"Initial Epoch Loss: {history[0]:.4f}")
    print(f"Final Epoch Loss:   {history[-1]:.4f}")
    print(f"Final Model Accuracy: {final_acc * 100.0:.1f}%")
