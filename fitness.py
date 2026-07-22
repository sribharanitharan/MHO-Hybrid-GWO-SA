"""
fitness.py
==========
Dataset      : Wisconsin Breast Cancer (sklearn, 569 samples, 30 features)
Objective    : f(x) = alpha * ErrorRate(x) + beta * |S| / |D|
Variables    : x in {0,1}^30  (binary feature selection vector)
Constraints  : 1 <= |S| <= 30
alpha = 0.9  (classification error weight)
beta  = 0.1  (feature ratio penalty)
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ── Load & preprocess ─────────────────────────────────────────────────────────
_data = load_breast_cancer()
X_raw, y = _data.data, _data.target
X = MinMaxScaler().fit_transform(X_raw)
n_features = X.shape[1]   # 30

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

ALPHA = 0.9
BETA  = 0.1


def fitness(solution: np.ndarray) -> float:
    """
    Evaluate a continuous or binary solution vector.

    Parameters
    ----------
    solution : np.ndarray of shape (n_features,)
        Values > 0.5 treated as selected (1), others unselected (0).

    Returns
    -------
    float  — weighted fitness value; lower is better.

    Formula
    -------
    f(x) = ALPHA * (1 - Accuracy) + BETA * (|S| / |D|)

    where:
        |S| = number of selected features
        |D| = total features (30)
        Accuracy = KNN classifier accuracy on test set
    """
    selected = solution > 0.5

    # Constraint: at least 1 feature must be selected
    if selected.sum() == 0:
        return 1.0   # hard penalty

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train[:, selected], y_train)
    acc = clf.score(X_test[:, selected], y_test)

    error_rate    = 1.0 - acc
    feature_ratio = selected.sum() / n_features

    return ALPHA * error_rate + BETA * feature_ratio


# ── V-shaped binary transfer function ────────────────────────────────────────
def v_transfer(v: np.ndarray) -> np.ndarray:
    """
    V-shaped transfer function.
    Maps continuous velocity/position values to flip probabilities in [0, 1].

    Formula: T(v) = |tanh(v)|

    Higher |v| → higher probability of flipping the bit.
    """
    return np.abs(np.tanh(v))


def binarize(v_new: np.ndarray,
             current: np.ndarray,
             rng: np.random.RandomState) -> np.ndarray:
    """
    Stochastic binarization using the V-shaped transfer function.

    For each dimension d:
        probability of flip = v_transfer(v_new[d])
        new_bit[d] = 1 - current[d]  if rand() < prob
                   = current[d]       otherwise

    Parameters
    ----------
    v_new   : continuous velocity/position vector (shape: n_features,)
    current : current binary solution vector      (shape: n_features,)
    rng     : numpy RandomState for reproducibility

    Returns
    -------
    np.ndarray — updated binary solution vector
    """
    prob = v_transfer(v_new)
    flip = rng.rand(len(v_new)) < prob
    return np.where(flip, 1 - current, current).astype(float)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Dataset     : Wisconsin Breast Cancer")
    print(f"Samples     : {X_train.shape[0] + X_test.shape[0]}")
    print(f"Features    : {n_features}")
    print(f"Train/Test  : {X_train.shape[0]} / {X_test.shape[0]}")
    print()

    # Test with all features selected
    all_selected = np.ones(n_features)
    f_all = fitness(all_selected)
    print(f"Fitness (all 30 features)   : {f_all:.6f}")

    # Test with random selection
    rng = np.random.RandomState(42)
    rand_sol = rng.rand(n_features)
    f_rand = fitness(rand_sol)
    n_sel  = int((rand_sol > 0.5).sum())
    print(f"Fitness (random {n_sel:2d} features) : {f_rand:.6f}")

    # Test penalty case
    no_feat = np.zeros(n_features)
    f_none = fitness(no_feat)
    print(f"Fitness (no features)        : {f_none:.6f}  ← penalty applied")