"""Training orchestration script for testing neural network convergence."""

import csv
import numpy as np

from nn.activations.sigmoid import Sigmoid
from nn.layers.linear import Linear
from nn.losses import CrossEntropyLoss
from nn.optim import SGD

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
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)
    y = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=float)
    return X, y


def load_competition_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and parse the competition CSV file using pure NumPy and Python.

    Assumes the first row is a header, the last column contains the binary
    targets (0 or 1), and all leading columns are numerical features.

    Args:
        filepath (str): Path to the dataset relative to submission root.

    Returns:
        tuple[np.ndarray, np.ndarray]: Features (X) and labels (y) matrices.
    """
    features = []
    labels = []

    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row safely

        for row in reader:
            if not row:
                continue
            # Cast all row features to floats; separate the final label column
            numerical_row = [float(val) for val in row]
            features.append(numerical_row[:-1])
            labels.append([numerical_row[-1]])

    X = np.array(features, dtype=float)
    y = np.array(labels, dtype=float)

    # CRITICAL: Min-Max feature normalization to prevent exponential clipping
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    # Avoid zero-division errors for unvarying columns
    range_mask = (x_max - x_min) == 0
    divisor = np.where(range_mask, 1.0, x_max - x_min)
    X_normalized = (X - x_min) / divisor

    return X_normalized, y


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

        # 2. Backward Pass
        grad = loss_fn.backward()
        grad = _activation.backward(grad)
        _layer.forward_backward = _layer.backward(grad)

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
    if _layer is None or _activation is None:
        train()

    X, y = toy_data()

    z = _layer.forward(X)
    predictions = _activation.forward(z)

    pred_labels = (predictions >= 0.5).astype(float)
    return float(np.mean(pred_labels == y))


if __name__ == "__main__":
    print("Executing Custom Pipeline on Real Competition Data...")

    # 1. Load real data vectors from the assigned pathway
    try:
        X_real, y_real = load_competition_data("data/train.csv")
        num_features = X_real.shape[1]
        print(f"Dataset parsed! Rows: {X_real.shape[0]}, Features: {num_features}")

        # 2. Instantiate framework modules tailored to real input sizes
        np.random.seed(42)
        real_layer = Linear(in_features=num_features, out_features=1)
        real_activation = Sigmoid()
        real_loss_fn = CrossEntropyLoss()
        real_optimizer = SGD(real_layer.parameters(), lr=0.1)  # Lower LR for real data

        # 3. Run multi-epoch gradient updates
        for epoch in range(5000):
            z_real = real_layer.forward(X_real)
            preds = real_activation.forward(z_real)
            loss_val = real_loss_fn.forward(preds, y_real)

            grad_out = real_loss_fn.backward()
            grad_out = real_activation.backward(grad_out)
            real_layer.backward(grad_out)

            real_optimizer.step()
            real_optimizer.zero_grad()

            if epoch % 500 == 0:
                pred_classes = (preds >= 0.5).astype(float)
                acc = np.mean(pred_classes == y_real)
                print(f"Epoch {epoch:4d} | Loss: {loss_val:.6f} | Accuracy: {acc:.2%}")

    except FileNotFoundError:
        print("Warning: data/train.csv not located. Running toy layout fallback.")
        history = train()
        print(f"Final Toy Loss: {history[-1]:.6f} | Accuracy: {accuracy():.2%}")
