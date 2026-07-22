"""
experiments.py
==============
Runs each algorithm N_RUNS independent times with different random seeds.
Computes  : Best, Worst, Mean, Standard Deviation of fitness values.
Saves to  : results/stats_results.csv  and  results/convergence.csv

Called by : main.py
Run standalone : python experiments.py
"""

import os
import numpy as np
import pandas as pd

from gwo    import gwo
from sa     import simulated_annealing
from hybrid import hybrid_gwo_sa

os.makedirs('results', exist_ok=True)

# ── Experiment Configuration ──────────────────────────────────────────────────
N_RUNS     = 10      # number of independent trials
N_ITER     = 100     # total iterations per algorithm
N_WOLVES   = 20      # GWO pack size
T0         = 100.0   # SA initial temperature
COOL       = 0.95    # SA cooling rate
N_ITER_GWO = 70      # Hybrid: GWO phase iterations
N_ITER_SA  = 30      # Hybrid: SA  phase iterations  (70 + 30 = 100)


# ─────────────────────────────────────────────────────────────────────────────
def run_experiments(n_runs: int = N_RUNS):
    """
    Run GWO, SA, and Hybrid GWO+SA for n_runs independent trials.

    Each trial uses a different random seed to ensure statistical
    independence. Seeds are deterministic (seed = i*13+7) so results
    are fully reproducible.

    Parameters
    ----------
    n_runs : int — number of independent trials  (default: 10)

    Returns
    -------
    results      : dict  {algo: [fitness_run1, ..., fitness_runN]}
    convergences : dict  {algo: [[curve_run1], ..., [curve_runN]]}
    """
    results = {
        'GWO'          : [],
        'SA'           : [],
        'Hybrid GWO+SA': [],
    }
    convergences = {
        'GWO'          : [],
        'SA'           : [],
        'Hybrid GWO+SA': [],
    }

    # ── Print table header ────────────────────────────────────────────────────
    print("=" * 62)
    print("  Running 10 Independent Trials — Each Algorithm")
    print("=" * 62)
    header = f"{'Run':<5} {'GWO':>12} {'SA':>12} {'Hybrid GWO+SA':>15}"
    print(header)
    print("─" * len(header))

    for i in range(n_runs):

        seed = i * 13 + 7     # deterministic but varied seeds
                               # i=0 → seed=7, i=1 → seed=20, etc.

        # ── Run GWO (Standalone) ──────────────────────────────────────────────
        _, f_gwo, c_gwo = gwo(
            n_iter=N_ITER,
            n_wolves=N_WOLVES,
            seed=seed
        )

        # ── Run SA (Standalone) ───────────────────────────────────────────────
        _, f_sa, c_sa = simulated_annealing(
            init_solution=None,     # cold start
            T0=T0,
            cooling_rate=COOL,
            n_iter=N_ITER,
            seed=seed
        )

        # ── Run Hybrid GWO + SA ───────────────────────────────────────────────
        _, f_hybrid, c_hybrid = hybrid_gwo_sa(
            n_iter_gwo=N_ITER_GWO,
            n_iter_sa=N_ITER_SA,
            n_wolves=N_WOLVES,
            seed=seed
        )

        # ── Store fitness values ──────────────────────────────────────────────
        results['GWO'].append(f_gwo)
        results['SA'].append(f_sa)
        results['Hybrid GWO+SA'].append(f_hybrid)

        # ── Store convergence curves ──────────────────────────────────────────
        convergences['GWO'].append(c_gwo)
        convergences['SA'].append(c_sa)
        convergences['Hybrid GWO+SA'].append(c_hybrid)

        # ── Print run result row ──────────────────────────────────────────────
        print(f"{i+1:<5} {f_gwo:>12.6f} {f_sa:>12.6f} {f_hybrid:>15.6f}")

    print("─" * len(header))
    print()
    return results, convergences


# ─────────────────────────────────────────────────────────────────────────────
def compute_stats(results: dict) -> pd.DataFrame:
    """
    Compute statistical metrics for each algorithm across all runs.

    Metrics
    -------
    Best    : minimum fitness value across all runs   (best solution found)
    Worst   : maximum fitness value across all runs   (worst  solution found)
    Mean    : average fitness across all runs         (central tendency)
    Std Dev : standard deviation across all runs      (stability / consistency)

    Lower values are better for all four metrics.

    Parameters
    ----------
    results : dict  {algo: [fitness_run1, ..., fitness_runN]}

    Returns
    -------
    pd.DataFrame with columns: Algorithm, Best, Worst, Mean, Std Dev
    """
    rows = []
    for algo, vals in results.items():
        v = np.array(vals)
        rows.append({
            'Algorithm': algo,
            'Best'     : round(float(np.min(v)),  6),
            'Worst'    : round(float(np.max(v)),  6),
            'Mean'     : round(float(np.mean(v)), 6),
            'Std Dev'  : round(float(np.std(v)),  6),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def save_convergence(convergences: dict):
    """
    Compute mean convergence curve per algorithm and save to CSV.

    The mean is taken across all N_RUNS independent trials so the
    convergence plot shows a smooth, representative curve.

    Output
    ------
    results/convergence.csv
        Columns : GWO, SA, Hybrid GWO+SA
        Rows    : one per iteration (100 rows)
        Index   : Iteration (1 to 100)
    """
    conv_data = {}
    for algo, curves in convergences.items():
        # curves is a list of N_RUNS lists, each of length N_ITER
        mean_curve = np.mean(curves, axis=0)    # shape: (N_ITER,)
        conv_data[algo] = mean_curve

    df = pd.DataFrame(conv_data)
    df.index      = range(1, len(df) + 1)
    df.index.name = 'Iteration'

    df.to_csv('results/convergence.csv')
    print("Saved → results/convergence.csv")


# ─────────────────────────────────────────────────────────────────────────────
def print_stats_table(stats_df: pd.DataFrame):
    """
    Print a formatted statistical results table to the terminal.

    Parameters
    ----------
    stats_df : pd.DataFrame — output of compute_stats()
    """
    print("=" * 62)
    print("  Statistical Results — 10 Independent Runs")
    print("=" * 62)
    print(f"  {'Algorithm':<18} {'Best':>10} {'Worst':>10} "
          f"{'Mean':>10} {'Std Dev':>10}")
    print("  " + "─" * 52)

    for _, row in stats_df.iterrows():
        marker = " ◄ BEST" if row['Algorithm'] == 'Hybrid GWO+SA' else ""
        print(f"  {row['Algorithm']:<18} "
              f"{row['Best']:>10.6f} "
              f"{row['Worst']:>10.6f} "
              f"{row['Mean']:>10.6f} "
              f"{row['Std Dev']:>10.6f}"
              f"{marker}")

    print("  " + "─" * 52)
    print()

    # ── Improvement summary ───────────────────────────────────────────────────
    gwo_mean    = float(stats_df.loc[stats_df['Algorithm'] == 'GWO',
                                     'Mean'].values[0])
    sa_mean     = float(stats_df.loc[stats_df['Algorithm'] == 'SA',
                                     'Mean'].values[0])
    hybrid_mean = float(stats_df.loc[stats_df['Algorithm'] == 'Hybrid GWO+SA',
                                     'Mean'].values[0])

    imp_gwo = ((gwo_mean - hybrid_mean) / gwo_mean) * 100
    imp_sa  = ((sa_mean  - hybrid_mean) / sa_mean)  * 100

    print(f"  Hybrid improvement over GWO : {imp_gwo:.2f}%")
    print(f"  Hybrid improvement over SA  : {imp_sa:.2f}%")
    print()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Step 1: Run all experiments
    results, convergences = run_experiments(n_runs=N_RUNS)

    # Step 2: Compute statistics
    stats_df = compute_stats(results)

    # Step 3: Print formatted table
    print_stats_table(stats_df)

    # Step 4: Save stats to CSV
    stats_df.to_csv('results/stats_results.csv', index=False)
    print("Saved → results/stats_results.csv")

    # Step 5: Save convergence curves to CSV
    save_convergence(convergences)

    # Step 6: Print all raw run values
    print()
    print("  Raw Fitness Values per Run")
    print("  " + "─" * 52)
    print(f"  {'Run':<6} {'GWO':>12} {'SA':>12} {'Hybrid GWO+SA':>15}")
    print("  " + "─" * 42)
    for i in range(N_RUNS):
        print(f"  {i+1:<6} "
              f"{results['GWO'][i]:>12.6f} "
              f"{results['SA'][i]:>12.6f} "
              f"{results['Hybrid GWO+SA'][i]:>15.6f}")
    print("  " + "─" * 52)