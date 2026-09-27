# Formative 1, Part 1 — Building a Neural Network from Scratch in NumPy

# Custom NumPy Neural Network Framework

A modular, lightweight deep learning framework implemented entirely from scratch using Python and NumPy. This project bypasses high-level deep learning frameworks to expose the underlying linear algebra, matrix vectorization, and calculus needed to orchestrate backpropagation and network optimization loops.

---

## Repository Architecture

```text
your_submission/
├── nn/
│   ├── __init__.py            # Package-level exports
│   ├── module.py              # Foundational Module abstract base contract
│   ├── layers/
│   │   ├── __init__.py
│   │   └── linear.py          # Fully-connected dense layer with Xavier Uniform initialization
│   ├── activations/
│   │   ├── __init__.py
│   │   ├── relu.py            # Elementwise Rectified Linear Unit activation with mask caching
│   │   ├── sigmoid.py         # Elementwise Sigmoid squash activation with cached probabilities
│   │   └── softmax.py         # Numerical-stable row-wise Softmax normalization
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── cross_entropy_loss.py              # Log-clipped Binary Cross-Entropy criterion
│   │   └── categorical_cross_entropy_loss.py  # Log-clipped Categorical Cross-Entropy criterion
│   └── optim/
│       ├── __init__.py
│       └── sgd.py             # Stochastic Gradient Descent in-place parameter updater
├── main.py                    # Training orchestration and pipeline convergence script
└── README.md                  # Project documentation notes
```

---

## Core Framework Modifications & Enhancements

To guarantee a stable runtime convergence and clear the strict constraints imposed by the automated validation hooks, several adjustments were made over the standard conceptual formulas:

### 1. In-Place Gradient Buffer Mutations (`[...]`)
A critical Python memory reference trap was resolved inside `Linear.backward()`. Standard array reassignments (`self.dW = ...`) break object pointer references, preventing the global optimizer instance from tracking updated derivative gradients. Using ellipsis buffer mutations (`self.dW[...] = ...`) forces in-place modifications to the existing memory, allowing the `SGD` step function to interact cleanly across training iterations.

### 2. Guarding Against Logarithmic Edge Crashes
In both `CrossEntropyLoss` and `CategoricalCrossEntropyLoss`, standard logarithmic functions risk throwing runtime `NaN` evaluation blocks if a prediction lands exactly on absolute bounds (0.0 or 1.0). To prevent numerical zero divisions or infinite errors, predictions are safely squeezed using boundary clamping:
```python
self.predictions = np.clip(predictions, 1e-12, 1.0 - 1e-12)
```

### 3. Exponentiation Max-Subtraction Normalization
To insulate the multi-class `Softmax` forward pass against floating-point numerical overflows (such as calculating e¹⁰⁰⁰ which yields systemic infinities), the row-wise scalar maxima are normalized out:
```python
x_max = np.max(x, axis=1, keepdims=True)
exp_x = np.exp(x - x_max)
```
Subtracting the maximum shifts exponents down to safe bounds (≤ 0) where output values can range predictably between 0 and 1 without altering the final algebraic evaluation.

### 4. Vectorized Jacobian Backpropagation
To eliminate sluggish, nested Python execution loops across batch evaluations, row-wise multi-class activations were vectorized through dot-product equivalents. The analytical vector implementation in `Softmax.backward` computes massive global gradients cleanly across the entire batch matrix simultaneously:
```python
sum_grad_a = np.sum(grad_output * self.a, axis=1, keepdims=True)
return self.a * (grad_output - sum_grad_a)
```

### 5. Google Linter Docstring Compliance
Every file was formatted to adhere to strict Google convention docstring requirements (`pydocstyle` convention D-rules) to satisfy `ruff check nn/` and `ruff check main.py` exit configurations. This includes absolute module descriptions, explicit variable annotations, structured `Args:` and `Returns:` formatting, tracking single-line line-length margins beneath 88 characters, and parsing dirty inline whitespaces from blank boundaries.

## 1. Environment

You need `conda` (Miniconda or Anaconda). From this folder, run once:

```bash
conda env create -f environment.yml
conda activate iml-formative1
```

Re-run `conda activate iml-formative1` in every new terminal. Check it worked:

```bash
python -c "import numpy, pytest; print('numpy', numpy.__version__)"
pytest --version
ruff --version
```

Python 3.11, NumPy 2.x, pytest 8.x, ruff. No deep-learning framework is
installed or permitted.

## 2. What's provided vs. what you build

**Provided — do not edit:**

```
guide.pdf            the assignment
environment.yml      the fixed dependency set
pyproject.toml       ruff + pytest configuration
conftest.py          test fixtures and the stage report
tests/               the public checks (a subset of what is graded)
```

**You build — everything under `nn/`, plus `main.py`:**

```
nn/
  __init__.py                       (empty)
  module.py                         Chapter 0.5
  layers/__init__.py  layers/linear.py
  activations/__init__.py  relu.py  sigmoid.py  softmax.py
  losses/__init__.py  cross_entropy_loss.py  categorical_cross_entropy_loss.py
  optim/__init__.py  sgd.py
main.py                             Chapter 10
README.md                           your own notes (you may overwrite this file)
```

Create these yourself, following the guide. Every `__init__.py` starts empty;
the guide tells you the one import line to add to each as you go.

## 3. Running the checks

Run everything from this folder (the submission root), with the environment
active. Each chapter ends with a "Validate before moving on" box in the guide —
run exactly what it says. In general:

```bash
pytest                       # all public checks (runs tests/ only)
pytest tests/layers/test_stage1_vector.py     # one stage
ruff check nn/               # documentation + style, required from Chapter 1 on
```

Each run prints a grouped **Stage** report and writes `stage<N>_report.json`.
`ruff check nn/` must exit 0 before you move past Chapter 1.

## 4. Important

The public `tests/` are a **subset**. Passing them is necessary, not
sufficient — grading runs a larger private suite plus a short technical
defense. The guide and the rubric list every property that is checked; read
both. Do not try to special-case the tests.

## 5. Submitting

Zip the submission root — `nn/`, `main.py`, your `README.md`, and the provided
files — exactly as laid out above. Do not rename files or move `tests/`.
