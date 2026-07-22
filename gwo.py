"""
gwo.py
======
Algorithm A  : Binary Grey Wolf Optimizer (GWO)
Strength     : Strong global exploration via alpha/beta/delta pack hierarchy
Weakness     : Premature convergence in discrete / binary search spaces
Reference    : Mirjalili et al., Advances in Engineering Software, 2014
"""

import numpy as np
from fitness import fitness, binarize, n_features


def gwo(n_iter: int = 100, n_wolves: int = 20, seed: int = 0):
    """
    Binary GWO for Feature Selection.

    PACK HIERARCHY
    --------------
    alpha (α) — best wolf    : lowest fitness, leads the hunt
    beta  (β) — second best  : assists alpha
    delta (δ) — third best   : assists alpha and beta
    omega (ω) — remaining wolves : update positions guided by α, β, δ

    POSITION UPDATE (continuous domain)
    ------------------------------------
        D_alpha = |C1 * alpha_pos - pos_i|
        D_beta  = |C2 * beta_pos  - pos_i|
        D_delta = |C3 * delta_pos - pos_i|

        X1 = alpha_pos - A1 * D_alpha
        X2 = beta_pos  - A2 * D_beta
        X3 = delta_pos - A3 * D_delta

        v_new = (X1 + X2 + X3) / 3

    COEFFICIENT VECTORS
    --------------------
        a  = linearly decreases from 2 → 0 over iterations
        A  = 2 * a * r1 - a        (r1 ∈ [0,1] random)
        C  = 2 * r2                (r2 ∈ [0,1] random)

    BINARIZATION
    ------------
        Uses V-shaped transfer function: T(v) = |tanh(v)|
        Each bit flips with probability T(v_new[d])
        (see binarize() in fitness.py)

    Parameters
    ----------
    n_iter   : int   — total number of iterations        (default: 100)
    n_wolves : int   — pack size / population            (default: 20)
    seed     : int   — random seed for reproducibility   (default: 0)

    Returns
    -------
    best_sol    : np.ndarray  — best binary feature vector  (shape: n_features,)
    best_score  : float       — best fitness value achieved
    convergence : list[float] — best fitness value at each iteration
    """
    rng = np.random.RandomState(seed)

    # ── Step 1: Initialise population randomly in [0, 1]^n_features ──────────
    pos        = rng.rand(n_wolves, n_features)        # continuous positions
    binary_pos = (pos > 0.5).astype(float)             # binarized positions

    # ── Step 2: Evaluate initial fitness ─────────────────────────────────────
    fitness_vals = np.array([fitness(p) for p in binary_pos])

    # ── Step 3: Identify alpha, beta, delta ──────────────────────────────────
    sorted_idx  = np.argsort(fitness_vals)

    alpha_pos   = binary_pos[sorted_idx[0]].copy()   # best
    beta_pos    = binary_pos[sorted_idx[1]].copy()   # 2nd best
    delta_pos   = binary_pos[sorted_idx[2]].copy()   # 3rd best
    alpha_score = fitness_vals[sorted_idx[0]]

    convergence = []

    # ── Step 4: Main Optimization Loop ───────────────────────────────────────
    for t in range(n_iter):

        # Linearly decrease 'a' from 2 to 0 (controls exploration vs exploitation)
        a = 2.0 - t * (2.0 / n_iter)

        for i in range(n_wolves):

            # ── Update guided by Alpha ────────────────────────────────────────
            r1 = rng.rand(n_features)
            r2 = rng.rand(n_features)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha_pos - pos[i])
            X1      = alpha_pos - A1 * D_alpha

            # ── Update guided by Beta ─────────────────────────────────────────
            r1 = rng.rand(n_features)
            r2 = rng.rand(n_features)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta_pos - pos[i])
            X2     = beta_pos - A2 * D_beta

            # ── Update guided by Delta ────────────────────────────────────────
            r1 = rng.rand(n_features)
            r2 = rng.rand(n_features)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta_pos - pos[i])
            X3      = delta_pos - A3 * D_delta

            # ── Average the three positional updates ──────────────────────────
            v_new = (X1 + X2 + X3) / 3.0

            # ── Binarize using V-shaped transfer function ─────────────────────
            binary_pos[i] = binarize(v_new, binary_pos[i], rng)
            pos[i]        = v_new

            # ── Evaluate fitness of updated wolf ──────────────────────────────
            f = fitness(binary_pos[i])

            # ── Update alpha if better solution found ─────────────────────────
            if f < alpha_score:
                alpha_score = f
                alpha_pos   = binary_pos[i].copy()

        # Record best fitness at this iteration
        convergence.append(alpha_score)

    return alpha_pos.copy(), alpha_score, convergence


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Running Binary GWO (standalone test)...")
    print(f"  n_wolves = 20  |  n_iter = 100  |  seed = 0")
    print()

    best_sol, best_score, conv = gwo(n_iter=100, n_wolves=20, seed=0)

    n_selected = int((best_sol > 0.5).sum())
    print(f"  Best Fitness     : {best_score:.6f}")
    print(f"  Features Selected: {n_selected} / {n_features}")
    print(f"  Selected Indices : {list(np.where(best_sol > 0.5)[0])}")
    print()
    print(f"  Convergence (first 10 iters) : {[round(c,5) for c in conv[:10]]}")
    print(f"  Convergence (last  10 iters) : {[round(c,5) for c in conv[-10:]]}")