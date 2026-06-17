# Chapter 05: Naive Bayes

> **Level**: Beginner–Intermediate | **Estimated Time**: 3 hours

---

## 5.1 Intuition

Naive Bayes is a **probabilistic classifier** based on Bayes' Theorem.  
It asks: *"Given these features, which class is most probable?"*

The "naive" part: it assumes all features are **conditionally independent** given the class — an unrealistic but surprisingly effective assumption.

---

## 5.2 Bayes' Theorem

```
P(y | x) = P(x | y) · P(y) / P(x)
```

Where:
- `P(y | x)` — **Posterior**: probability of class y given features x (what we want)
- `P(x | y)` — **Likelihood**: probability of seeing features x if class is y
- `P(y)` — **Prior**: how common is class y overall
- `P(x)` — **Evidence**: constant for a fixed x, so we often ignore it

**Decision rule** (MAP — Maximum A Posteriori):
```
ŷ = argmax_y [ P(y) · Π_j P(xⱼ | y) ]
```

In log space (to avoid underflow):
```
ŷ = argmax_y [ log P(y) + Σⱼ log P(xⱼ | y) ]
```

---

## 5.3 Variants of Naive Bayes

| Variant | Assumption about P(xⱼ | y) | Use case |
|---------|---------------------------|----------|
| **Gaussian NB** | Features follow Normal distribution | Continuous features |
| **Bernoulli NB** | Features are binary (0/1) | Binary text features |
| **Multinomial NB** | Features are counts | Word counts in text |

---

## 5.4 Gaussian Naive Bayes

For continuous features, model each feature as Gaussian (Normal):

```
P(xⱼ | y=c) = (1 / √(2πσ²)) · exp(-(xⱼ - μ)² / (2σ²))
```

**Training** (per class c, per feature j):
- μⱼ,c = mean of feature j among all samples with class c
- σ²ⱼ,c = variance of feature j among all samples with class c

---

## 5.5 From-Scratch Python Implementation

```python
# naive_bayes.py
import math

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes Classifier.
    Assumes features follow Normal distributions within each class.
    Pure Python — no ML libraries.
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.class_priors = {}      # log P(y=c)
        self.means = {}             # mean[c][j]
        self.variances = {}         # var[c][j]
        self.classes = []

    def fit(self, X, y):
        n = len(y)
        self.classes = sorted(set(y))
        n_features = len(X[0])

        for c in self.classes:
            # Samples belonging to class c
            X_c = [X[i] for i in range(n) if y[i] == c]
            n_c = len(X_c)

            # Prior: P(y=c)
            self.class_priors[c] = math.log(n_c / n)

            # Per-feature mean and variance
            self.means[c] = [
                sum(row[j] for row in X_c) / n_c
                for j in range(n_features)
            ]
            self.variances[c] = [
                sum((row[j] - self.means[c][j])**2 for row in X_c) / n_c + self.var_smoothing
                for j in range(n_features)
            ]

        return self

    def _log_gaussian(self, x, mu, var):
        """Log probability density of x under N(mu, var)."""
        return -0.5 * math.log(2 * math.pi * var) - (x - mu)**2 / (2 * var)

    def _log_likelihood(self, x, c):
        """Sum of log P(xⱼ | y=c) over all features j."""
        return sum(
            self._log_gaussian(xj, self.means[c][j], self.variances[c][j])
            for j, xj in enumerate(x)
        )

    def predict_log_proba(self, X):
        """Return log posterior for each class for each sample."""
        return [
            {c: self.class_priors[c] + self._log_likelihood(xi, c)
             for c in self.classes}
            for xi in X
        ]

    def predict(self, X):
        log_probs = self.predict_log_proba(X)
        return [max(lp, key=lp.get) for lp in log_probs]


# ── Multinomial Naive Bayes (for text) ─────────────────────────────────────

class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes for text classification.
    Features should be word/token counts.
    Uses Laplace (add-1) smoothing.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha   # Laplace smoothing parameter
        self.class_priors = {}
        self.feature_log_probs = {}
        self.classes = []

    def fit(self, X, y):
        """
        X: list of feature count vectors
        y: list of class labels
        """
        n = len(y)
        n_features = len(X[0])
        self.classes = sorted(set(y))

        for c in self.classes:
            X_c = [X[i] for i in range(n) if y[i] == c]
            self.class_priors[c] = math.log(len(X_c) / n)

            # Total count of each feature in class c
            feature_counts = [sum(row[j] for row in X_c) for j in range(n_features)]
            total_count = sum(feature_counts) + self.alpha * n_features

            self.feature_log_probs[c] = [
                math.log((feature_counts[j] + self.alpha) / total_count)
                for j in range(n_features)
            ]

        return self

    def predict(self, X):
        predictions = []
        for xi in X:
            scores = {}
            for c in self.classes:
                log_prob = self.class_priors[c]
                log_prob += sum(xij * lp for xij, lp in zip(xi, self.feature_log_probs[c]))
                scores[c] = log_prob
            predictions.append(max(scores, key=scores.get))
        return predictions


# ── Simple Text Vectorizer ─────────────────────────────────────────────────

class BagOfWords:
    """Convert text documents to word count vectors."""

    def __init__(self):
        self.vocab = {}

    def fit(self, documents):
        words = set()
        for doc in documents:
            words.update(doc.lower().split())
        self.vocab = {w: i for i, w in enumerate(sorted(words))}
        return self

    def transform(self, documents):
        n_vocab = len(self.vocab)
        vectors = []
        for doc in documents:
            vec = [0] * n_vocab
            for word in doc.lower().split():
                if word in self.vocab:
                    vec[self.vocab[word]] += 1
            vectors.append(vec)
        return vectors

    def fit_transform(self, documents):
        return self.fit(documents).transform(documents)


# ── Demo: Spam Detection ───────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Gaussian NB on Iris-like data ──
    print("=== Gaussian Naive Bayes ===")
    X_train = [
        [5.1, 3.5], [4.9, 3.0], [4.7, 3.2],   # class 0
        [7.0, 3.2], [6.4, 3.2], [6.9, 3.1],   # class 1
        [6.3, 3.3], [5.8, 2.7], [7.1, 3.0],   # class 2
    ]
    y_train = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)
    X_test = [[5.0, 3.4], [6.7, 3.1], [6.0, 2.9]]
    print("Predictions:", gnb.predict(X_test))  # expect [0, 1, 2]

    # ── Multinomial NB on text ──
    print("\n=== Multinomial Naive Bayes (Text) ===")
    train_docs = [
        "free money win prize lottery",
        "click here to claim your reward",
        "buy cheap pills online now",
        "meeting tomorrow at 9am",
        "please review the attached report",
        "quarterly results attached for review",
    ]
    y_text = ["spam", "spam", "spam", "ham", "ham", "ham"]

    bow = BagOfWords()
    X_text_train = bow.fit_transform(train_docs)

    mnb = MultinomialNaiveBayes(alpha=1.0)
    mnb.fit(X_text_train, y_text)

    test_docs = ["free reward click here", "please find attached report"]
    X_text_test = bow.transform(test_docs)
    print("Predictions:", mnb.predict(X_text_test))
    # Expected: ['spam', 'ham']
```

---

## 5.6 Laplace Smoothing

A problem: if a word never appeared in training for class "spam", then P(word|spam) = 0 and the entire product becomes 0!

**Laplace Smoothing** (add-α):
```
P(xⱼ | y=c) = (count(xⱼ, c) + α) / (total_count_in_c + α × n_features)
```

With α=1, every feature gets a pseudo-count of 1 to prevent zero probabilities.

---

## ✅ Chapter Summary

| Concept | Key Idea |
|---------|---------|
| Bayes' Theorem | `P(y|x) ∝ P(x|y) · P(y)` |
| Naive assumption | Features are conditionally independent given class |
| Gaussian NB | Models continuous features as Normal distributions |
| Multinomial NB | Models discrete counts (great for text) |
| Laplace smoothing | Prevents zero probabilities in unseen events |

---

## 📝 Exercises

1. Prove that argmax of log P(y|x) is the same as argmax of P(y|x).
2. Implement **Bernoulli Naive Bayes** for binary features (presence/absence of words).
3. Add **TF-IDF** weighting to the `BagOfWords` vectorizer.
4. Why might Naive Bayes outperform logistic regression on small datasets despite its strong independence assumption?

---

**← Previous:** [Chapter 04: KNN](chapter-04-knn.md)  
**→ Next:** [Chapter 06: Decision Trees](chapter-06-decision-trees.md)
