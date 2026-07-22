"""
sa.py
=====
Algorithm B  : Simulated Annealing (SA)
Strength     : Precise local exploitation via Metropolis probabilistic acceptance
Weakness     : Slow to reach global optimum from a random (cold) start
Reference    : Kirkpatrick et al., Science, 1983
"""

import numpy as np
from fitness import fitness, n_features


def simulated_annealing(init_solution=None,
                        T0: float = 100.0,
                        cooling_rate: float = 0.95,
                        n_iter: int = 100,
                        seed: int = 0):
    """
    Simulated Annealing for Binary Feature Selection.

    CORE IDEA
    ---------
    SA mimics the physical annealing process in metallurgy:
    - At HIGH temperature  → freely accepts worse solutions (exploration)
    - At LOW  temperature  → rarely accepts worse solutions (exploitation)
    This avoids getting permanently stuck in local optima.

    NEIGHBOURHOOD STRUCTURE
    -----------------------
    Single bit-flip mutation:
        Pick a random dimension d ∈ {0, 1, ..., n_features-1}
        Flip: neighbor[d] = 1 - current[d]
    One feature is toggled ON or OFF per iteration.

    METROPOLIS ACCEPTANCE CRITERION
    --------------------------------
    If neighbor_fit < current_fit  → ALWAYS accept (greedy improvement)
    If neighbor_fit >= current_fit → accept with probability:
        P(accept) = exp( -delta / T )
        where delta = neighbor_fit - current_fit  (> 0)

    At high T: exp(-delta/T) ≈ 1  → almost always accept worse
    At low  T: exp(-delta/T) ≈ 0  → almost never  accept worse

    COOLING SCHEDULE
    ----------------
    Geometric cooling:
        T_{t+1} = T_t * cooling_rate
    With T0=100 and cooling_rate=0.95:
        iter  1  → T = 100.00
        iter 10  → T =  59.87
        iter 50  → T =   7.69
        iter 100 → T =   0.59

    Parameters
    ----------
    init_solution : np.ndarray or None
        Starting binary vector (shape: n_features,).
        None  = random cold start.
        array = warm start from a previous algorithm (e.g., GWO best).
    T0            : float  — initial temperature            (default: 100.0)
    cooling_rate  : float  — geometric cooling factor       (default: 0.95)
    n_iter        : int    — number of iterations           (default: 100)
    seed          : int    — random seed for reproducibility(default: 0)

    Returns
    -------
    best_sol    : np.ndarray  — best binary feature vector found
    best_score  : float       — best fitness value achieved
    convergence : list[float] — best fitness value at each iteration
    """
    rng = np.random.RandomState(seed)

    # ── Step 1: Initialise current solution ──────────────────────────────────
    if init_solution is None:
        # Cold start: random binary vector
        current = (rng.rand(n_features) > 0.5).astype(float)
    else:
        # Warm start: use provided solution (e.g., GWO best output)
        current = init_solution.copy()

    current_fit = fitness(current)

    # Track global best separately (best ever seen, not just current)
    best     = current.copy()
    best_fit = current_fit

    T           = T0
    convergence = []

    # ── Step 2: Main Annealing Loop ───────────────────────────────────────────
    for iteration in range(n_iter):

        # ── Generate Neighbour via Single Bit-Flip ────────────────────────────
        neighbor          = current.copy()
        flip_idx          = rng.randint(0, n_features)     # random dimension
        neighbor[flip_idx] = 1.0 - neighbor[flip_idx]      # flip the bit
        neighbor_fit      = fitness(neighbor)

        # ── Metropolis Acceptance Criterion ───────────────────────────────────
        delta = neighbor_fit - current_fit                  # change in fitness

        if delta < 0:
            # Neighbor is BETTER → always accept
            current     = neighbor.copy()
            current_fit = neighbor_fit

        else:
            # Neighbor is WORSE → accept with probability exp(-delta / T)
            acceptance_prob = np.exp(-delta / max(T, 1e-10))
            if rng.rand() < acceptance_prob:
                current     = neighbor.copy()
                current_fit = neighbor_fit

        # ── Update Global Best ────────────────────────────────────────────────
        if current_fit < best_fit:
            best     = current.copy()
            best_fit = current_fit

        # ── Cool Down ─────────────────────────────────────────────────────────
        T *= cooling_rate

        # ── Record Best Fitness at This Iteration ─────────────────────────────
        convergence.append(best_fit)

    return best.copy(), best_fit, convergence


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Running Simulated Annealing (standalone test)...")
    print(f"  T0 = 100  |  cooling = 0.95  |  n_iter = 100  |  seed = 0")
    print()

    # ── Cold Start (random init) ──────────────────────────────────────────────
    best_sol, best_score, conv = simulated_annealing(
        init_solution=None,
        T0=100.0,
        cooling_rate=0.95,
        n_iter=100,
        seed=0
    )

    n_selected = int((best_sol > 0.5).sum())
    print(f"  [Cold Start]")
    print(f"  Best Fitness     : {best_score:.6f}")
    print(f"  Features Selected: {n_selected} / {n_features}")
    print(f"  Selected Indices : {list(np.where(best_sol > 0.5)[0])}")
    print()
    print(f"  Convergence (first 10 iters) : {[round(c,5) for c in conv[:10]]}")
    print(f"  Convergence (last  10 iters) : {[round(c,5) for c in conv[-10:]]}")
    print()

    # ── Warm Start (from a known solution) ───────────────────────────────────
    print(f"  [Warm Start — simulating GWO handoff]")
    warm_init = (np.random.RandomState(7).rand(n_features) > 0.5).astype(float)
    best_w, score_w, conv_w = simulated_annealing(
        init_solution=warm_init,
        T0=50.0,
        cooling_rate=0.95,
        n_iter=30,
        seed=1
    )
    print(f"  Best Fitness     : {score_w:.6f}")
    print(f"  Features Selected: {int((best_w > 0.5).sum())} / {n_features}")