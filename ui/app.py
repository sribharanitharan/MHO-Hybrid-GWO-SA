"""
app.py
======
Flask Web Dashboard for Assignment IV:
    "Design and Comparative Analysis of a Hybrid Metaheuristic"

Course   : Meta Heuristic Optimization Techniques (19MAM83)
Dept     : Computing – AI & ML, CIT Coimbatore  |  AY 2025-26
Author   : SRI BHARANITHARAN M

Routes
------
  GET  /              → Dashboard homepage (index.html)
  POST /run           → Run all experiments (AJAX call)
  GET  /results       → Fetch stats JSON for table rendering
  GET  /convergence   → Fetch convergence CSV data as JSON
  GET  /plots/<name>  → Serve generated plot images
  GET  /status        → Check if results are already available

Usage
-----
    cd ui/
    python app.py
    Open → http://127.0.0.1:5000
"""

import os
import sys
import json
import time
import threading
import subprocess

import numpy as np
import pandas as pd
from flask import (Flask, render_template, request,
                   jsonify, send_file, Response)

# ── Path setup: allow imports from parent directory ──────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))   # .../ui/
PARENT_DIR  = os.path.dirname(BASE_DIR)                    # .../Assignment_IV/
RESULTS_DIR = os.path.join(PARENT_DIR, 'results')
sys.path.insert(0, PARENT_DIR)

os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__)

# ── Global experiment state (thread-safe with lock) ──────────────────────────
experiment_state = {
    'running'   : False,
    'done'      : False,
    'progress'  : 0,           # 0 to 100
    'log'       : [],          # live log messages
    'error'     : None,
    'elapsed'   : 0.0,
}
state_lock = threading.Lock()


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE: Homepage
# ═════════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    """
    Serve the main dashboard page.
    Checks if previous results already exist so the UI can show them
    immediately without re-running experiments.
    """
    stats_exists = os.path.exists(
        os.path.join(RESULTS_DIR, 'stats_results.csv')
    )
    conv_exists = os.path.exists(
        os.path.join(RESULTS_DIR, 'convergence.csv')
    )
    plots_exist = all(
        os.path.exists(os.path.join(RESULTS_DIR, f))
        for f in ['convergence_plot.png',
                  'stats_bar_chart.png',
                  'range_plot.png']
    )
    return render_template(
        'index.html',
        results_ready=(stats_exists and conv_exists and plots_exist)
    )


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE: Run Experiments (AJAX — runs in background thread)
# ═════════════════════════════════════════════════════════════════════════════
@app.route('/run', methods=['POST'])
def run_experiments():
    """
    Accepts POST request with hyperparameter config from the UI form.
    Launches experiment in a background thread so the UI stays responsive.
    Progress is polled via /status route.

    JSON Body (optional overrides)
    --------------------------------
    {
        "n_runs"     : 10,
        "n_wolves"   : 20,
        "n_iter"     : 100,
        "t0"         : 100.0,
        "cooling"    : 0.95,
        "n_iter_gwo" : 70,
        "n_iter_sa"  : 30
    }
    """
    global experiment_state

    with state_lock:
        if experiment_state['running']:
            return jsonify({'status': 'already_running',
                            'message': 'Experiment is already running.'}), 409
        experiment_state = {
            'running' : True,
            'done'    : False,
            'progress': 0,
            'log'     : ['► Starting experiment...'],
            'error'   : None,
            'elapsed' : 0.0,
        }

    # Parse hyperparameters from request
    data       = request.get_json(silent=True) or {}
    n_runs     = int(data.get('n_runs',     10))
    n_wolves   = int(data.get('n_wolves',   20))
    n_iter     = int(data.get('n_iter',    100))
    t0         = float(data.get('t0',     100.0))
    cooling    = float(data.get('cooling',  0.95))
    n_iter_gwo = int(data.get('n_iter_gwo', 70))
    n_iter_sa  = int(data.get('n_iter_sa',  30))

    # Launch background thread
    thread = threading.Thread(
        target=_run_experiment_thread,
        args=(n_runs, n_wolves, n_iter, t0,
              cooling, n_iter_gwo, n_iter_sa),
        daemon=True
    )
    thread.start()

    return jsonify({'status': 'started',
                    'message': 'Experiment started successfully.'})


def _log(msg: str, progress: int = None):
    """Append a message to the live log and optionally update progress."""
    with state_lock:
        experiment_state['log'].append(msg)
        if progress is not None:
            experiment_state['progress'] = progress


def _run_experiment_thread(n_runs, n_wolves, n_iter,
                            t0, cooling, n_iter_gwo, n_iter_sa):
    """
    Background thread: runs all experiments, saves CSVs, generates plots.
    Updates experiment_state for live progress polling.
    """
    global experiment_state
    start = time.time()

    try:
        from gwo        import gwo
        from sa         import simulated_annealing
        from hybrid     import hybrid_gwo_sa
        from experiments import compute_stats, save_convergence
        from visualize  import (plot_convergence,
                                 plot_stats_bar, plot_range)

        # Change working dir to parent so relative paths work
        os.chdir(PARENT_DIR)

        results      = {'GWO': [], 'SA': [], 'Hybrid GWO+SA': []}
        convergences = {'GWO': [], 'SA': [], 'Hybrid GWO+SA': []}

        _log(f'► Configuration: {n_runs} runs | '
             f'{n_wolves} wolves | {n_iter} iters', 5)
        _log(f'  GWO phase: {n_iter_gwo} iters | '
             f'SA phase: {n_iter_sa} iters | '
             f'T0={t0} | cooling={cooling}')

        # ── Run all trials ────────────────────────────────────────────────────
        for i in range(n_runs):
            seed = i * 13 + 7
            prog = 10 + int((i / n_runs) * 70)   # progress: 10% → 80%

            _log(f'  Run {i+1}/{n_runs}  (seed={seed})', prog)

            _, f_gwo,    c_gwo    = gwo(
                n_iter=n_iter, n_wolves=n_wolves, seed=seed
            )
            _, f_sa,     c_sa     = simulated_annealing(
                T0=t0, cooling_rate=cooling,
                n_iter=n_iter, seed=seed
            )
            _, f_hybrid, c_hybrid = hybrid_gwo_sa(
                n_iter_gwo=n_iter_gwo,
                n_iter_sa=n_iter_sa,
                n_wolves=n_wolves,
                seed=seed
            )

            results['GWO'].append(f_gwo)
            results['SA'].append(f_sa)
            results['Hybrid GWO+SA'].append(f_hybrid)
            convergences['GWO'].append(c_gwo)
            convergences['SA'].append(c_sa)
            convergences['Hybrid GWO+SA'].append(c_hybrid)

            _log(f'    GWO={f_gwo:.6f}  SA={f_sa:.6f}  '
                 f'Hybrid={f_hybrid:.6f}')

        # ── Compute stats ─────────────────────────────────────────────────────
        _log('► Computing statistics...', 82)
        stats_df = compute_stats(results)
        stats_df.to_csv(
            os.path.join(RESULTS_DIR, 'stats_results.csv'), index=False
        )
        _log('  Saved → results/stats_results.csv')

        # ── Save convergence ──────────────────────────────────────────────────
        _log('► Saving convergence curves...', 86)
        conv_data = {
            algo: np.mean(curves, axis=0).tolist()
            for algo, curves in convergences.items()
        }
        conv_df             = pd.DataFrame(conv_data)
        conv_df.index       = range(1, len(conv_df) + 1)
        conv_df.index.name  = 'Iteration'
        conv_df.to_csv(os.path.join(RESULTS_DIR, 'convergence.csv'))
        _log('  Saved → results/convergence.csv')

        # ── Generate plots ────────────────────────────────────────────────────
        _log('► Generating Plot 1: Convergence curves...', 90)
        plot_convergence(
            conv_csv  = os.path.join(RESULTS_DIR, 'convergence.csv'),
            save_path = os.path.join(RESULTS_DIR, 'convergence_plot.png')
        )

        _log('► Generating Plot 2: Stats bar chart...', 94)
        plot_stats_bar(
            stats_csv = os.path.join(RESULTS_DIR, 'stats_results.csv'),
            save_path = os.path.join(RESULTS_DIR, 'stats_bar_chart.png')
        )

        _log('► Generating Plot 3: Range chart...', 97)
        plot_range(
            stats_csv = os.path.join(RESULTS_DIR, 'stats_results.csv'),
            save_path = os.path.join(RESULTS_DIR, 'range_plot.png')
        )

        elapsed = round(time.time() - start, 1)
        _log(f'✓ All done in {elapsed}s', 100)

        with state_lock:
            experiment_state['running'] = False
            experiment_state['done']    = True
            experiment_state['elapsed'] = elapsed

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        _log(f'✗ ERROR: {str(e)}', 100)
        with state_lock:
            experiment_state['running'] = False
            experiment_state['done']    = True
            experiment_state['error']   = err


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE: Poll experiment progress (AJAX polling every 1s)
# ═════════════════════════════════════════════════════════════════════════════
@app.route('/status')
def status():
    """
    Returns current experiment state as JSON.
    Polled by the frontend every second during a run.
    """
    with state_lock:
        return jsonify({
            'running' : experiment_state['running'],
            'done'    : experiment_state['done'],
            'progress': experiment_state['progress'],
            'log'     : experiment_state['log'][-20:],  # last 20 lines
            'error'   : experiment_state['error'],
            'elapsed' : experiment_state['elapsed'],
        })


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE: Fetch statistical results as JSON
# ═════════════════════════════════════════════════════════════════════════════
@app.route('/results')
def get_results():
    """
    Reads results/stats_results.csv and returns it as JSON.
    Called by the frontend to populate the stats table and bar charts.
    """
    path = os.path.join(RESULTS_DIR, 'stats_results.csv')
    if not os.path.exists(path):
        return jsonify({'error': 'No results yet. Run experiments first.'}), 404

    df   = pd.read_csv(path)
    data = df.to_dict(orient='records')

    # Compute improvement percentages
    algos = {row['Algorithm']: row for row in data}
    if 'Hybrid GWO+SA' in algos and 'GWO' in algos and 'SA' in algos:
        h = algos['Hybrid GWO+SA']['Mean']
        g = algos['GWO']['Mean']
        s = algos['SA']['Mean']
        imp_gwo = round(((g - h) / g) * 100, 2)
        imp_sa  = round(((s - h) / s) * 100, 2)
    else:
        imp_gwo = imp_sa = 0.0

    return jsonify({
        'stats'  : data,
        'imp_gwo': imp_gwo,
        'imp_sa' : imp_sa,
    })


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE: Fetch convergence data as JSON
# ═════════════════════════════════════════════════════════════════════════════
@app.route('/convergence')
def get_convergence():
    """
    Reads results/convergence.csv and returns it as JSON.
    Used by Chart.js in the frontend to draw convergence line charts.

    Returns
    -------
    JSON:
    {
        "iterations": [1, 2, ..., 100],
        "GWO"       : [...],
        "SA"        : [...],
        "Hybrid GWO+SA": [...]
    }
    """
    path = os.path.join(RESULTS_DIR, 'convergence.csv')
    if not os.path.exists(path):
        return jsonify({'error': 'No convergence data yet.'}), 404

    df = pd.read_csv(path, index_col='Iteration')
    return jsonify({
        'iterations'    : list(df.index.astype(int)),
        'GWO'           : df['GWO'].round(6).tolist(),
        'SA'            : df['SA'].round(6).tolist(),
        'Hybrid GWO+SA' : df['Hybrid GWO+SA'].round(6).tolist(),
    })


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE: Serve plot images
# ═════════════════════════════════════════════════════════════════════════════
@app.route('/plots/<filename>')
def serve_plot(filename: str):
    """
    Serve a generated PNG plot from results/ directory.

    Valid filenames
    ---------------
    - convergence_plot.png
    - stats_bar_chart.png
    - range_plot.png
    """
    allowed = {
        'convergence_plot.png',
        'stats_bar_chart.png',
        'range_plot.png',
    }
    if filename not in allowed:
        return jsonify({'error': 'Plot not found.'}), 404

    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return jsonify({'error': f'{filename} not generated yet.'}), 404

    return send_file(path, mimetype='image/png')


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE: Dataset info
# ═════════════════════════════════════════════════════════════════════════════
@app.route('/dataset')
def dataset_info():
    """
    Returns Wisconsin Breast Cancer dataset metadata as JSON.
    Used to populate the dataset KPI cards on the UI.
    """
    try:
        from fitness import (X_train, X_test, y_train,
                              y_test, n_features, ALPHA, BETA)
        total   = X_train.shape[0] + X_test.shape[0]
        classes = {
            'Malignant (0)': int((y_train == 0).sum() + (y_test == 0).sum()),
            'Benign (1)'   : int((y_train == 1).sum() + (y_test == 1).sum()),
        }
        return jsonify({
            'dataset'      : 'Wisconsin Breast Cancer',
            'total_samples': total,
            'train_samples': X_train.shape[0],
            'test_samples' : X_test.shape[0],
            'n_features'   : int(n_features),
            'classes'      : classes,
            'alpha'        : ALPHA,
            'beta'         : BETA,
            'objective'    : 'f(x) = α × ErrorRate(x) + β × |S| / |D|',
            'classifier'   : 'KNN (k=5)',
            'split'        : '70% train / 30% test',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE: Clear results (reset)
# ═════════════════════════════════════════════════════════════════════════════
@app.route('/clear', methods=['POST'])
def clear_results():
    """
    Deletes all generated CSVs and PNGs from results/.
    Resets the dashboard to fresh state.
    """
    files_to_clear = [
        'stats_results.csv', 'convergence.csv',
        'convergence_plot.png', 'stats_bar_chart.png', 'range_plot.png'
    ]
    cleared = []
    for f in files_to_clear:
        path = os.path.join(RESULTS_DIR, f)
        if os.path.exists(path):
            os.remove(path)
            cleared.append(f)

    with state_lock:
        experiment_state['done']     = False
        experiment_state['progress'] = 0
        experiment_state['log']      = []
        experiment_state['error']    = None

    return jsonify({'cleared': cleared,
                    'message': f'Cleared {len(cleared)} files.'})


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 55)
    print("  Assignment IV — MHO Dashboard")
    print("  Flask Web UI  |  19MAM83  |  CIT Coimbatore")
    print("=" * 55)
    print()
    print("  Open your browser at:")
    print("  → http://127.0.0.1:5000")
    print()
    print("  Routes available:")
    print("  GET  /              Dashboard homepage")
    print("  POST /run           Run experiments (AJAX)")
    print("  GET  /status        Poll experiment progress")
    print("  GET  /results       Stats JSON for table")
    print("  GET  /convergence   Convergence data JSON")
    print("  GET  /plots/<name>  Serve plot images")
    print("  GET  /dataset       Dataset metadata JSON")
    print("  POST /clear         Reset all results")
    print()
    print("  Press Ctrl+C to stop.")
    print("=" * 55)
    print()

    app.run(debug=True, host='127.0.0.1', port=5000)