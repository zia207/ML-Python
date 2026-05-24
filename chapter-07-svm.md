# Chapter 07: Support Vector Machines (SVM)

> **Level**: Intermediate | **Estimated Time**: 4–5 hours

---

## 7.1 Intuition

SVM finds the **hyperplane that maximizes the margin** between two classes.

Instead of any separating line, find the one that is *farthest from all training points* — this gives the best generalization.

**Support vectors**: The training points closest to the decision boundary that "support" (define) it.

---

## 7.2 Mathematical Formulation

### Linear SVM (Hard Margin)

A hyperplane: `wᵀx + b = 0`

Constraints (for linearly separable data):
```
yᵢ(wᵀxᵢ + b) ≥ 1  for all i
```

The margin width = `2 / ||w||`

**Objective**: Maximize margin = minimize `||w||²/2`

```
Minimize:   (1/2) ||w||²
Subject to: yᵢ(wᵀxᵢ + b) ≥ 1  ∀i
```

### Soft Margin SVM (allows misclassifications)

Introduce **slack variables** ξᵢ ≥ 0:
```
Minimize:   (1/2)||w||² + C Σᵢ ξᵢ
Subject to: yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ
```

**C** controls the bias-variance tradeoff:
- Small C: wider margin, more misclassifications (high bias)
- Large C: narrow margin, fewer misclassifications (high variance)

### Hinge Loss

The soft-margin objective is equivalent to minimizing:
```
J(w) = (1/n) Σᵢ max(0, 1 - yᵢ(wᵀxᵢ + b)) + λ||w||²
```

This is the **hinge loss** + L2 regularization, allowing stochastic gradient descent.

---

## 7.3 The Kernel Trick

For non-linearly separable data, map features to a higher-dimensional space where they *are* linearly separable:

```
x → φ(x)    (feature map to higher dimension)
```

The **kernel trick**: compute dot products in the transformed space without explicitly computing φ:

```
K(xᵢ, xⱼ) = φ(xᵢ)ᵀ φ(xⱼ)
```

| Kernel | Formula | Use case |
|--------|---------|----------|
| Linear | `xᵢᵀxⱼ` | Linearly separable data |
| Polynomial | `(xᵢᵀxⱼ + c)ᵈ` | Polynomial boundaries |
| RBF (Gaussian) | `exp(-γ||xᵢ-xⱼ||²)` | Most common choice |
| Sigmoid | `tanh(κxᵢᵀxⱼ + c)` | Neural net-like |

---

## 7.4 From-Scratch Python Implementation (Linear SVM via SGD)

```python
# svm.py
import math, random

class LinearSVM:
    """
    Linear SVM using Subgradient Descent on Hinge Loss.
    J(w,b) = C * Σ max(0, 1 - y(wᵀx + b)) + (1/2)||w||²
    Pure Python — no ML libraries.
    """

    def __init__(self, C=1.0, learning_rate=0.001, n_iterations=1000):
        self.C = C
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.w = None
        self.b = 0.0

    def _hinge_loss(self, X, y):
        total = 0.0
        for xi, yi in zip(X, y):
            margin = yi * (sum(wi*xij for wi,xij in zip(self.w, xi)) + self.b)
            total += max(0.0, 1.0 - margin)
        return self.C * total / len(y) + 0.5 * sum(wi**2 for wi in self.w)

    def fit(self, X, y):
        n, p = len(X), len(X[0])
        self.w = [0.0] * p
        self.b = 0.0

        for _ in range(self.n_iter):
            # Shuffle for stochastic updates
            indices = list(range(n))
            random.shuffle(indices)

            for i in indices:
                xi, yi = X[i], y[i]
                z = sum(wi*xij for wi,xij in zip(self.w, xi)) + self.b
                margin = yi * z

                if margin >= 1:
                    # Correctly classified with sufficient margin: only regularization gradient
                    self.w = [wi - self.lr * wi for wi in self.w]
                else:
                    # Hinge loss gradient
                    self.w = [wi - self.lr * (wi - self.C * yi * xij)
                               for wi, xij in zip(self.w, xi)]
                    self.b += self.lr * self.C * yi

        return self

    def decision_function(self, X):
        return [sum(wi*xij for wi,xij in zip(self.w, xi)) + self.b for xi in X]

    def predict(self, X):
        return [1 if score >= 0 else -1 for score in self.decision_function(X)]


# ── Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Linearly separable 2D data (labels: -1 and +1)
    X_train = [
        [2.0, 3.0], [3.0, 4.0], [3.5, 2.5], [4.0, 3.5],  # class +1
        [0.5, 0.5], [1.0, 1.5], [0.5, 2.0], [1.5, 0.5],  # class -1
    ]
    y_train = [1, 1, 1, 1, -1, -1, -1, -1]

    X_test = [[3.0, 3.0], [1.0, 1.0], [2.5, 1.0]]
    y_test = [1, -1, -1]

    svm = LinearSVM(C=1.0, learning_rate=0.001, n_iterations=2000)
    svm.fit(X_train, y_train)

    preds = svm.predict(X_test)
    acc = sum(yp==yt for yp,yt in zip(preds,y_test))/len(y_test)
    print(f"Predictions: {preds}")
    print(f"Accuracy: {acc:.2f}")
    print(f"Weights: {[f'{w:.3f}' for w in svm.w]}, Bias: {svm.b:.3f}")
```

---

## 7.5 Summary Table

| Aspect | Detail |
|--------|--------|
| Core idea | Maximize margin between classes |
| Decision boundary | `wᵀx + b = 0` |
| Support vectors | Points with `yᵢ(wᵀxᵢ+b) = 1` |
| C parameter | Regularization (small=wide margin) |
| Kernel trick | Non-linear boundaries without explicit mapping |

---

## 📝 Exercises

1. Implement a **kernel function** K(a,b) for RBF and modify SVM to use it.
2. Compare SVM with C=0.01 vs C=100 on noisy data.
3. Why does SVM only depend on the **support vectors** at test time?

---

**← Previous:** [Chapter 06: Decision Trees](chapter-06-decision-trees.md)  
**→ Next:** [Chapter 08: Ensemble Methods](chapter-08-ensemble-methods.md)
