# Chapter 00: Prerequisites & Setup

> **Level**: Beginner | **Estimated Time**: 2–3 hours

---

## 0.1 What You Need to Know First

This tutorial assumes:
- Basic Python syntax (variables, loops, functions, lists)
- High school algebra (equations, functions, graphs)
- No prior machine learning knowledge required

---

## 0.2 Math Refresher

### Vectors and Matrices
A **vector** is an ordered list of numbers: `x = [x₁, x₂, ..., xₙ]`  
A **matrix** is a 2D array of numbers with rows and columns.

**Dot product** (used constantly in ML):
```
x · w = x₁w₁ + x₂w₂ + ... + xₙwₙ
```

### Summation Notation
```
Σᵢ xᵢ  means  x₁ + x₂ + ... + xₙ
```

### Derivatives (Calculus basics)
The derivative `df/dx` measures how much `f` changes when `x` changes slightly.  
Key rule: if `f(x) = x²`, then `df/dx = 2x`

### Logarithms
`log(x)` is the inverse of exponentiation. Used in probability and loss functions.  
Key property: `log(a × b) = log(a) + log(b)`

---

## 0.3 Python Setup

```bash
# Recommended: Python 3.9+
python --version

# We use ONLY standard library + math module in core chapters
# Optional: matplotlib for plotting (not required for algorithms)
pip install matplotlib
```

---

## 0.4 Building Our Own Math Utilities (From Scratch)

Throughout this tutorial we will **never use NumPy or scikit-learn** for core algorithms.  
Here are the basic utilities we will reuse in every chapter:

```python
# utils.py  — our from-scratch math toolkit
import math

# ── Vector operations ──────────────────────────────────────────────────────

def dot(v1, v2):
    """Dot product of two vectors."""
    return sum(a * b for a, b in zip(v1, v2))

def vec_add(v1, v2):
    """Element-wise addition."""
    return [a + b for a, b in zip(v1, v2)]

def vec_sub(v1, v2):
    """Element-wise subtraction."""
    return [a - b for a, b in zip(v1, v2)]

def scalar_mul(s, v):
    """Multiply vector by scalar."""
    return [s * x for x in v]

def norm(v):
    """Euclidean (L2) norm of a vector."""
    return math.sqrt(sum(x**2 for x in v))

# ── Matrix operations ──────────────────────────────────────────────────────

def mat_mul(A, B):
    """Matrix multiplication: A (m×k) × B (k×n) → C (m×n)."""
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    C = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

def transpose(M):
    """Transpose a matrix."""
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

def mat_vec_mul(M, v):
    """Multiply matrix M by vector v."""
    return [dot(row, v) for row in M]

# ── Statistics ─────────────────────────────────────────────────────────────

def mean(values):
    return sum(values) / len(values)

def variance(values):
    m = mean(values)
    return sum((x - m)**2 for x in values) / len(values)

def std_dev(values):
    return math.sqrt(variance(values))

def covariance(x, y):
    mx, my = mean(x), mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / len(x)

# ── Activation / math helpers ──────────────────────────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def relu(z):
    return max(0.0, z)

def softmax(z_list):
    max_z = max(z_list)   # numerical stability
    exp_z = [math.exp(z - max_z) for z in z_list]
    total = sum(exp_z)
    return [e / total for e in exp_z]
```

---

## 0.5 Understanding Data Representation

In ML, data is structured as:
- **Feature matrix X**: shape `(n_samples, n_features)` — each row is one observation
- **Target vector y**: shape `(n_samples,)` — what we want to predict

```python
# Example dataset: house size (sqft) vs price ($1000s)
X = [
    [800],
    [1200],
    [1500],
    [2000],
    [2500],
]
y = [150, 200, 250, 320, 400]  # prices
```

---

## 0.6 Train/Test Split (From Scratch)

```python
import random

def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Split dataset into training and testing sets."""
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx  = indices[split:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_test  = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test
```

---

## 0.7 Feature Normalization

Most ML algorithms work better when features are on the same scale.

**Min-Max Scaling** maps values to [0, 1]:
```
x_scaled = (x - x_min) / (x_max - x_min)
```

**Z-score Standardization** maps to mean=0, std=1:
```
x_std = (x - μ) / σ
```

```python
def min_max_scale(X):
    """Scale each feature column to [0, 1]."""
    n_features = len(X[0])
    mins = [min(row[j] for row in X) for j in range(n_features)]
    maxs = [max(row[j] for row in X) for j in range(n_features)]
    scaled = []
    for row in X:
        scaled_row = [
            (row[j] - mins[j]) / (maxs[j] - mins[j] + 1e-8)
            for j in range(n_features)
        ]
        scaled.append(scaled_row)
    return scaled, mins, maxs
```

---

## ✅ Chapter Summary

| Concept | Key Takeaway |
|---------|-------------|
| Vectors/Matrices | Core data structures for ML |
| Dot product | Foundation of most ML computations |
| Derivatives | Needed for gradient-based learning |
| Feature matrix | Rows = samples, Columns = features |
| Normalization | Essential preprocessing step |

---

## 📝 Exercises

1. Implement a `euclidean_distance(v1, v2)` function using only Python built-ins.
2. Write a function `normalize_column(data, col_idx)` that z-score normalizes a single column.
3. Create a 3×3 identity matrix using list comprehensions.

---

**Next Chapter →** [Chapter 01: Introduction to Machine Learning](chapter-01-intro-to-ml.md)
