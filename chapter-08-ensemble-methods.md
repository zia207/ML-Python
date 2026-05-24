# Chapter 08: Ensemble Methods

> **Level**: Intermediate | **Estimated Time**: 4–5 hours

---

## 8.1 Intuition

> "The wisdom of the crowd" — combining many weak models produces a strong one.

Three main strategies:
1. **Bagging** — train many models on random subsets, average their outputs
2. **Boosting** — train models sequentially, each correcting the previous
3. **Stacking** — train a meta-model to combine predictions of base models

---

## 8.2 Bagging & Random Forest

### Bootstrap Aggregating (Bagging)
1. Create B bootstrap samples (sample n points **with replacement**)
2. Train one model per bootstrap sample
3. Aggregate predictions (vote for classification, mean for regression)

**Variance reduction**: Each model has high variance but they are de-correlated → average reduces variance without increasing bias.

### Random Forest
Random Forest = Bagging + **random feature subsets** at each split.

At each split, instead of searching all p features, randomly select √p features.
This further de-correlates the trees.

---

## 8.3 Boosting & AdaBoost

### Intuition
Boosting builds an ensemble **sequentially**. Each new model focuses on samples the previous ones got wrong.

### AdaBoost Algorithm
1. Initialize sample weights: wᵢ = 1/n
2. For t = 1 to T:
   a. Train weak learner hₜ on weighted data
   b. Compute error: εₜ = Σᵢ wᵢ · 𝟙[hₜ(xᵢ) ≠ yᵢ]
   c. Compute learner weight: αₜ = 0.5 · ln((1-εₜ)/εₜ)
   d. Update sample weights: wᵢ ← wᵢ · exp(-αₜ yᵢ hₜ(xᵢ))
   e. Normalize weights so they sum to 1
3. Final prediction: H(x) = sign(Σₜ αₜ hₜ(x))

---

## 8.4 From-Scratch Python Implementation

```python
# ensemble.py
import math, random
from collections import Counter

# ── We reuse DecisionTreeClassifier from Chapter 06 ──────────────────────
# (abbreviated here for brevity — use the full version from chapter-06)

def gini(y):
    n = len(y)
    if n == 0: return 0.0
    counts = Counter(y)
    return 1.0 - sum((c/n)**2 for c in counts.values())

class SimpleStump:
    """Depth-1 decision tree (decision stump) for use in AdaBoost."""
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, weights):
        n, n_features = len(X), len(X[0])
        best_error = math.inf
        for feat in range(n_features):
            values = sorted(set(row[feat] for row in X))
            thresholds = [(values[i]+values[i+1])/2 for i in range(len(values)-1)]
            for thr in thresholds:
                for polarity in [1, -1]:
                    preds = [polarity if X[i][feat] <= thr else -polarity for i in range(n)]
                    error = sum(weights[i] for i in range(n) if preds[i] != y[i])
                    if error < best_error:
                        best_error = error
                        self.feature_idx = feat
                        self.threshold = thr
                        self.polarity = polarity
        return best_error

    def predict(self, X):
        return [self.polarity if xi[self.feature_idx] <= self.threshold else -self.polarity
                for xi in X]


class AdaBoostClassifier:
    """AdaBoost with decision stumps as weak learners."""

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []

    def fit(self, X, y):
        n = len(y)
        weights = [1.0/n] * n

        for _ in range(self.n_estimators):
            stump = SimpleStump()
            error = stump.fit(X, y, weights)

            # Avoid division by zero
            error = max(error, 1e-10)
            error = min(error, 1 - 1e-10)

            # Learner weight
            alpha = 0.5 * math.log((1 - error) / error)
            stump.alpha = alpha

            # Update sample weights
            preds = stump.predict(X)
            new_weights = [
                weights[i] * math.exp(-alpha * y[i] * preds[i])
                for i in range(n)
            ]
            total = sum(new_weights)
            weights = [w/total for w in new_weights]
            self.stumps.append(stump)

        return self

    def predict(self, X):
        scores = [0.0] * len(X)
        for stump in self.stumps:
            preds = stump.predict(X)
            for i, p in enumerate(preds):
                scores[i] += stump.alpha * p
        return [1 if s >= 0 else -1 for s in scores]


# ── Random Forest ─────────────────────────────────────────────────────────

def bootstrap_sample(X, y, seed=None):
    """Sample n points with replacement."""
    if seed: random.seed(seed)
    n = len(X)
    indices = [random.randint(0, n-1) for _ in range(n)]
    return [X[i] for i in indices], [y[i] for i in indices]

class RandomForestClassifier:
    """Random Forest: Bagging + random feature subsets."""

    def __init__(self, n_estimators=10, max_depth=5, max_features='sqrt',
                 min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.trees = []
        self.feature_subsets = []

    def _get_n_features(self, n_total):
        if self.max_features == 'sqrt':
            return max(1, int(math.sqrt(n_total)))
        if self.max_features == 'log2':
            return max(1, int(math.log2(n_total)))
        return n_total

    def fit(self, X, y):
        n_total_features = len(X[0])
        n_select = self._get_n_features(n_total_features)

        for t in range(self.n_estimators):
            X_boot, y_boot = bootstrap_sample(X, y, seed=t)

            # Random feature subset
            feat_idx = sorted(random.sample(range(n_total_features), n_select))
            X_sub = [[row[j] for j in feat_idx] for row in X_boot]

            # Train a simple depth-limited tree (reuse logic from ch06)
            # For brevity, use a simple decision tree class
            tree = _SimpleDecisionTree(max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split)
            tree.fit(X_sub, y_boot)
            self.trees.append(tree)
            self.feature_subsets.append(feat_idx)

        return self

    def predict(self, X):
        # Collect votes from all trees
        all_preds = []
        for tree, feat_idx in zip(self.trees, self.feature_subsets):
            X_sub = [[row[j] for j in feat_idx] for row in X]
            all_preds.append(tree.predict(X_sub))

        # Majority vote
        final = []
        for i in range(len(X)):
            votes = [all_preds[t][i] for t in range(self.n_estimators)]
            final.append(Counter(votes).most_common(1)[0][0])
        return final


class _SimpleDecisionTree:
    """Minimal decision tree for use inside RandomForest."""

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)

    def _build(self, X, y, depth):
        if (len(set(y)) == 1 or len(y) < self.min_samples_split
                or depth >= self.max_depth):
            return Counter(y).most_common(1)[0][0]
        best_gain, best_feat, best_thr = -1, None, None
        for feat in range(len(X[0])):
            vals = sorted(set(row[feat] for row in X))
            for thr in [(vals[i]+vals[i+1])/2 for i in range(len(vals)-1)]:
                yl = [y[i] for i in range(len(X)) if X[i][feat] <= thr]
                yr = [y[i] for i in range(len(X)) if X[i][feat] >  thr]
                if not yl or not yr: continue
                gain = gini(y) - (len(yl)/len(y))*gini(yl) - (len(yr)/len(y))*gini(yr)
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, feat, thr
        if best_feat is None:
            return Counter(y).most_common(1)[0][0]
        li = [i for i in range(len(X)) if X[i][best_feat] <= best_thr]
        ri = [i for i in range(len(X)) if X[i][best_feat] >  best_thr]
        return {'feat': best_feat, 'thr': best_thr,
                'left': self._build([X[i] for i in li], [y[i] for i in li], depth+1),
                'right': self._build([X[i] for i in ri], [y[i] for i in ri], depth+1)}

    def _traverse(self, x, node):
        if not isinstance(node, dict):
            return node
        branch = node['left'] if x[node['feat']] <= node['thr'] else node['right']
        return self._traverse(x, branch)

    def predict(self, X):
        return [self._traverse(xi, self.tree) for xi in X]


# ── Demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X = [[2,3],[3,4],[1,2],[4,5],[0,1],[5,6],[2,1],[3,2],[4,1],[1,4]]
    y = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0]

    print("=== AdaBoost ===")
    ada = AdaBoostClassifier(n_estimators=20)
    ada.fit(X, [2*yi-1 for yi in y])  # convert 0/1 → -1/+1
    ada_preds = [(p+1)//2 for p in ada.predict(X)]
    print(f"Train accuracy: {sum(yp==yt for yp,yt in zip(ada_preds,y))/len(y):.2f}")

    print("\n=== Random Forest ===")
    rf = RandomForestClassifier(n_estimators=10, max_depth=3)
    rf.fit(X, y)
    rf_preds = rf.predict(X)
    print(f"Train accuracy: {sum(yp==yt for yp,yt in zip(rf_preds,y))/len(y):.2f}")
```

---

## 8.5 Comparison Table

| Method | Bias | Variance | Training | Interpretability |
|--------|------|----------|----------|-----------------|
| Single Tree | Low | High | Fast | High |
| Bagging | Low | Lower | Medium | Low |
| Random Forest | Low | Low | Medium | Low |
| AdaBoost | Lower | Low | Slower | Very Low |

---

## 📝 Exercises

1. Why does bagging reduce variance but not bias?
2. Implement **Gradient Boosted Trees** (GBT) where each tree fits the *residuals* of the previous.
3. What is **Out-of-Bag (OOB) error** and how can it be used instead of cross-validation?

---

**← Previous:** [Chapter 07: SVM](chapter-07-svm.md)  
**→ Next:** [Chapter 09: Neural Networks](chapter-09-neural-networks.md)
