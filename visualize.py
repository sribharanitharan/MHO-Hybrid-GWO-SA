"""
visualize.py
============
Generates three publication-quality plots from results/ CSVs:

  Plot 1 → results/convergence_plot.png
            Line chart: best fitness per iteration for all 3 algorithms
            Includes GWO → SA handoff marker at iteration 70

  Plot 2 → results/stats_bar_chart.png
            Side-by-side bar charts: Mean Fitness & Standard Deviation

  Plot 3 → results/range_plot.png
            Error-bar chart: Best / Mean / Worst per algorithm

Run standalone : python visualize.py
Called by      : main.py
Requires       : results/convergence.csv  and  results/stats_results.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

os.makedirs('results', exist_ok=True)

# ── Global Style Configuration ────────────────────────────────────────────────
COLORS = {
    'GWO'          : '#4f98a3',   # Teal
    'SA'           : '#fdab43',   # Orange
    'Hybrid GWO+SA': '#a86fdf',   # Purple
}
LABELS     = list(COLORS.keys())
BG_PAGE    = '#f7f6f2'
BG_PLOT    = '#f9f8f5'
TEXT_COLOR = '#28251d'
GRID_COLOR = '#dcd9d5'
MUTED      = '#7a7974'


def _style_ax(ax):
    """Apply consistent style to all plot axes."""
    ax.set_facecolor(BG_PLOT)
    ax.tick_params(colors=MUTED, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, linestyle='--', alpha=0.50, color=GRID_COLOR, zorder=0)


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1: CONVERGENCE CURVES
# ═════════════════════════════════════════════════════════════════════════════
def plot_convergence(conv_csv : str = 'results/convergence.csv',
                     save_path: str = 'results/convergence_plot.png'):
    """
    Line chart showing best fitness per iteration (mean over 10 runs)
    for GWO, SA, and Hybrid GWO+SA.

    Features
    --------
    - Shaded fill under each curve for visual clarity
    - Dashed vertical line marks GWO → SA handoff at iteration 70
    - Legend with algorithm names and final fitness values
    - Annotation label at the handoff marker

    Parameters
    ----------
    conv_csv  : path to results/convergence.csv
    save_path : output path for the PNG file
    """
    df    = pd.read_csv(conv_csv, index_col='Iteration')
    iters = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG_PAGE)
    _style_ax(ax)

    # ── Draw each algorithm's convergence curve ───────────────────────────────
    for algo, color in COLORS.items():
        y = df[algo].values

        # Main line
        ax.plot(iters, y,
                color=color, lw=2.4,
                label=f"{algo}  (final: {y[-1]:.5f})",
                zorder=3)

        # Shaded area under curve
        ax.fill_between(iters, y,
                         alpha=0.09, color=color, zorder=2)

    # ── GWO → SA handoff marker (Hybrid specific) ─────────────────────────────
    ax.axvline(x=70,
               color=TEXT_COLOR, lw=1.3,
               linestyle='--', alpha=0.50, zorder=4)

    # Handoff annotation text
    y_top = ax.get_ylim()[1]
    ax.text(71.5, y_top * 0.97,
            'GWO → SA\nHandoff\n(iter 70)',
            fontsize=8.5, color=TEXT_COLOR,
            alpha=0.65, va='top',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor=BG_PAGE,
                      edgecolor=GRID_COLOR,
                      alpha=0.8))

    # ── Axis labels & title ───────────────────────────────────────────────────
    ax.set_xlabel('Iteration', fontsize=12,
                  color=TEXT_COLOR, labelpad=8)
    ax.set_ylabel('Best Fitness Value  f(x)',
                  fontsize=12, color=TEXT_COLOR, labelpad=8)
    ax.set_title(
        'Convergence Comparison: GWO  vs  SA  vs  Hybrid GWO+SA\n'
        'Feature Selection — Wisconsin Breast Cancer Dataset  '
        '(30 features, 569 samples)',
        fontsize=12.5, color=TEXT_COLOR, pad=16, fontweight='semibold'
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.legend(fontsize=10.5,
              framealpha=0.93,
              facecolor=BG_PLOT,
              edgecolor=GRID_COLOR,
              loc='upper right')

    # ── X-axis ticks every 10 iterations ─────────────────────────────────────
    ax.set_xticks(range(0, 101, 10))
    ax.set_xlim(1, 100)

    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2: MEAN & STD DEV BAR CHART
# ═════════════════════════════════════════════════════════════════════════════
def plot_stats_bar(stats_csv : str = 'results/stats_results.csv',
                   save_path : str = 'results/stats_bar_chart.png'):
    """
    Side-by-side bar charts comparing Mean Fitness and Standard Deviation
    for all three algorithms across 10 independent runs.

    Lower values are better for BOTH metrics:
    - Mean    → measures average solution quality
    - Std Dev → measures consistency / stability

    Parameters
    ----------
    stats_csv : path to results/stats_results.csv
    save_path : output path for the PNG file
    """
    df = pd.read_csv(stats_csv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(BG_PAGE)

    metrics = ['Mean',    'Std Dev']
    titles  = [
        'Mean Fitness  (Lower = Better)',
        'Standard Deviation  (Lower = More Stable)'
    ]
    ylabels = ['Mean Fitness  f(x)', 'Standard Deviation']

    for ax, metric, title, ylabel in zip(axes, metrics, titles, ylabels):
        _style_ax(ax)

        vals  = [float(df.loc[df['Algorithm'] == a, metric].values[0])
                 for a in LABELS]
        clrs  = [COLORS[a] for a in LABELS]
        edgec = [COLORS[a] for a in LABELS]

        bars = ax.bar(
            LABELS, vals,
            color=[c + 'bb' for c in clrs],   # slight transparency
            edgecolor=edgec,
            linewidth=1.8,
            width=0.50,
            zorder=3
        )

        # ── Value labels on top of each bar ───────────────────────────────────
        ax.bar_label(bars,
                     fmt='%.5f',
                     padding=6,
                     fontsize=10,
                     color=TEXT_COLOR,
                     fontweight='semibold')

        # ── Highlight the best (lowest) bar ───────────────────────────────────
        min_idx = int(np.argmin(vals))
        bars[min_idx].set_edgecolor('#28251d')
        bars[min_idx].set_linewidth(2.5)

        # ── Axis formatting ───────────────────────────────────────────────────
        ax.set_title(title, fontsize=11.5,
                     color=TEXT_COLOR, pad=12, fontweight='semibold')
        ax.set_ylabel(ylabel, fontsize=10.5, color=MUTED)
        ax.set_ylim(0, max(vals) * 1.30)
        ax.tick_params(axis='x', labelsize=10, rotation=0)

        # ── Best label annotation ─────────────────────────────────────────────
        ax.text(min_idx, vals[min_idx] * 0.5, '★ BEST',
                ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')

    fig.suptitle(
        'Statistical Robustness — 10 Independent Runs\n'
        'Hybrid GWO+SA achieves lowest Mean and most stable Std Dev',
        fontsize=12.5, color=TEXT_COLOR,
        y=1.03, fontweight='semibold'
    )
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3: BEST / MEAN / WORST RANGE CHART
# ═════════════════════════════════════════════════════════════════════════════
def plot_range(stats_csv : str = 'results/stats_results.csv',
               save_path : str = 'results/range_plot.png'):
    """
    Error-bar chart showing the performance range across 10 runs:
        ▲  = Best  fitness  (lowest  fitness found)
        ●  = Mean  fitness  (central tendency)
        ▼  = Worst fitness  (highest fitness found)

    The vertical bar from Best to Worst shows how consistent each
    algorithm is. A shorter bar = more stable algorithm.

    Parameters
    ----------
    stats_csv : path to results/stats_results.csv
    save_path : output path for the PNG file
    """
    df = pd.read_csv(stats_csv)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG_PAGE)
    _style_ax(ax)

    for i, algo in enumerate(LABELS):
        row   = df[df['Algorithm'] == algo].iloc[0]
        b     = float(row['Best'])
        w     = float(row['Worst'])
        m     = float(row['Mean'])
        color = COLORS[algo]

        # ── Error bar: mean with asymmetric error (best to worst) ─────────────
        ax.errorbar(
            i, m,
            yerr=[[m - b], [w - m]],
            fmt='o',
            color=color,
            capsize=14,
            capthick=2.5,
            elinewidth=2.8,
            markersize=12,
            zorder=4,
            label=algo
        )

        # ── Best marker (triangle up) ─────────────────────────────────────────
        ax.scatter(i, b,
                   marker='^', color=color,
                   s=150, zorder=5)

        # ── Worst marker (triangle down) ──────────────────────────────────────
        ax.scatter(i, w,
                   marker='v', color=color,
                   s=150, zorder=5)

        # ── Value annotations ─────────────────────────────────────────────────
        ax.text(i, b - 0.0018,
                f'▲ {b:.5f}',
                ha='center', va='top',
                fontsize=8.5, color=color, fontweight='semibold')

        ax.text(i, m,
                f'  {m:.5f}',
                ha='left', va='center',
                fontsize=8.5, color=color)

        ax.text(i, w + 0.0012,
                f'▼ {w:.5f}',
                ha='center', va='bottom',
                fontsize=8.5, color=color, fontweight='semibold')

        # ── Range span annotation ─────────────────────────────────────────────
        span = w - b
        ax.text(i + 0.25, (b + w) / 2,
                f'range\n{span:.5f}',
                ha='left', va='center',
                fontsize=7.5, color=MUTED,
                style='italic')

    # ── Axis labels & title ───────────────────────────────────────────────────
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylabel('Fitness Value  f(x)', fontsize=12,
                  color=TEXT_COLOR, labelpad=8)
    ax.set_title(
        'Performance Range over 10 Independent Runs\n'
        '▲ = Best    ●  = Mean    ▼ = Worst',
        fontsize=12.5, color=TEXT_COLOR,
        pad=14, fontweight='semibold'
    )
    ax.set_xlim(-0.6, 2.8)

    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# STANDALONE RUN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 55)
    print("  Generating All Plots from results/ CSVs")
    print("=" * 55)
    print()

    # Check if required CSVs exist
    missing = []
    for f in ['results/convergence.csv', 'results/stats_results.csv']:
        if not os.path.exists(f):
            missing.append(f)

    if missing:
        print("  ERROR: Missing required CSV files:")
        for f in missing:
            print(f"    ✗  {f}")
        print()
        print("  Run experiments.py first to generate the CSVs:")
        print("    python experiments.py")
    else:
        print("  Plot 1: Convergence curves...")
        plot_convergence()

        print("  Plot 2: Mean & Std Dev bar chart...")
        plot_stats_bar()

        print("  Plot 3: Best / Mean / Worst range chart...")
        plot_range()

        print()
        print("  All 3 plots saved to results/")
        print("=" * 55)