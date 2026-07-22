# Assignment IV — Hybrid Metaheuristic: GWO + SA
**Course:** Meta Heuristic Optimization Techniques (19MAM83)
**Dept:** Computing – AI & ML, CIT Coimbatore | AY 2025–26
**Author:** SRI BHARANITHARAN M
**Problem:** Feature Selection — Wisconsin Breast Cancer Dataset
**Hybrid:** Sequential GWO (Global Exploration) → SA (Local Exploitation)

---

## Project Structure

Assignment_IV/
├── fitness.py        # Dataset + objective function f(x)
├── gwo.py            # Algorithm A: Binary Grey Wolf Optimizer
├── sa.py             # Algorithm B: Simulated Annealing
├── hybrid.py         # Algorithm C: Sequential Hybrid GWO+SA
├── experiments.py    # 10-run statistical analysis
├── visualize.py      # 3 convergence & stats plots
├── main.py           # Entry point — runs everything
├── requirements.txt  # Python dependencies
└── results/          # Auto-created on run

---

## Setup & Run

# Install dependencies
pip install -r requirements.txt

# Run everything (CLI)
python main.py

---

## Objective Function

f(x) = 0.9 * ErrorRate(x) + 0.1 * |S| / |D|

x ∈ {0,1}^30   (binary feature selection vector)
Constraint: 1 ≤ |S| ≤ 30

---

## Hybrid Framework

| Phase   | Algorithm | Iterations | Role                        |
|---------|-----------|------------|-----------------------------|
| Phase 1 | GWO       | 1 – 70     | Global exploration (20 wolves) |
| Phase 2 | SA        | 71 – 100   | Local exploitation (warm-start from GWO best) |

Synergy: GWO eliminates SA's cold-start inefficiency.
         SA escapes GWO's premature convergence in binary spaces.

---

## Outputs (saved to results/)

| File                   | Description                        |
|------------------------|------------------------------------|
| stats_results.csv      | Best / Worst / Mean / Std Dev      |
| convergence.csv        | Iteration-wise mean fitness        |
| convergence_plot.png   | Convergence curves (3 algorithms)  |
| stats_bar_chart.png    | Mean & Std Dev bar comparison      |
| range_plot.png         | Best / Mean / Worst range chart    |

---

## Evaluation Rubric Coverage

| Criterion           | File(s)                          | Marks |
|---------------------|----------------------------------|-------|
| Problem Modeling    | fitness.py                       | 5m    |
| Hybrid Innovation   | hybrid.py, gwo.py, sa.py         | 8m    |
| Experimental Rigor  | experiments.py, stats_results.csv| 7m    |
| Critical Analysis   | visualize.py, main.py (inference)| 5m    |