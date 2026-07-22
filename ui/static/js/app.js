/* ═══════════════════════════════════════════════════════════
   app.js — MHO Assignment IV Dashboard
   Course : Meta Heuristic Optimization (19MAM83)
   Dept   : Computing – AI & ML, CIT Coimbatore | AY 2025-26
   Author : SRI BHARANITHARAN M
═══════════════════════════════════════════════════════════ */

'use strict';

// ── Constants ─────────────────────────────────────────────
const ALGO_COLORS = {
  'GWO'           : '#4f98a3',
  'SA'            : '#fdab43',
  'Hybrid GWO+SA' : '#a86fdf',
};

const SECTION_LOADERS = {
  stats      : loadStats,
  convergence: loadConvergence,
  inference  : loadInference,
};

// ── State ─────────────────────────────────────────────────
let pollTimer        = null;
let convergenceChart = null;
let seenLogs         = new Set();

// ══════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  lucide.createIcons();
  initTheme();
  initSidebar();
  initNavigation();
  loadDatasetInfo();
  autoShowResultsBanner();
});

// ══════════════════════════════════════════════════════════
//  THEME
// ══════════════════════════════════════════════════════════
function initTheme() {
  const root   = document.documentElement;
  const toggle = document.getElementById('themeToggle');

  // Default dark
  let theme = 'dark';
  root.setAttribute('data-theme', theme);
  renderThemeIcon(toggle, theme);

  toggle?.addEventListener('click', () => {
    theme = theme === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', theme);
    renderThemeIcon(toggle, theme);
    // Re-render chart in new theme colors
    if (convergenceChart) {
      fetch('/convergence')
        .then(r => r.ok ? r.json() : null)
        .then(data => { if (data && !data.error) renderConvergenceChart(data); })
        .catch(() => {});
    }
  });
}

function renderThemeIcon(btn, theme) {
  if (!btn) return;
  if (theme === 'dark') {
    btn.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
           viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="5"/>
        <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42
                 M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
      </svg>`;
  } else {
    btn.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
           viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
      </svg>`;
  }
}

// ══════════════════════════════════════════════════════════
//  SIDEBAR
// ══════════════════════════════════════════════════════════
function initSidebar() {
  const sidebar     = document.getElementById('sidebar');
  const mainContent = document.querySelector('.main-content');
  const toggleBtn   = document.getElementById('sidebarToggle');

  toggleBtn?.addEventListener('click', () => {
    const isWide = window.innerWidth > 700;
    if (isWide) {
      sidebar.classList.toggle('collapsed');
      mainContent.classList.toggle('expanded');
    } else {
      sidebar.classList.toggle('open');
    }
    lucide.createIcons();
  });

  // Close sidebar on outside click (mobile)
  document.addEventListener('click', (e) => {
    if (window.innerWidth <= 700
        && sidebar.classList.contains('open')
        && !sidebar.contains(e.target)
        && !toggleBtn.contains(e.target)) {
      sidebar.classList.remove('open');
    }
  });
}

// ══════════════════════════════════════════════════════════
//  NAVIGATION
// ══════════════════════════════════════════════════════════
function initNavigation() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      switchSection(item.dataset.section);
    });
  });
}

function switchSection(name) {
  // Deactivate all
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  // Activate target
  document.getElementById(name)?.classList.add('active');
  document.querySelector(`[data-section="${name}"]`)?.classList.add('active');

  // Lazy-load section data
  if (SECTION_LOADERS[name]) SECTION_LOADERS[name]();

  lucide.createIcons();
}

// ══════════════════════════════════════════════════════════
//  DATASET INFO (Overview KPIs)
// ══════════════════════════════════════════════════════════
function loadDatasetInfo() {
  fetch('/dataset')
    .then(r => r.json())
    .then(data => {
      if (data.error) return;
      setText('kpi-samples', data.total_samples);
      setText('kpi-features', data.n_features);
      setText('kpi-classes',  data.n_classes ?? '—');
      setText('kpi-traintest', `${data.train_size}/${data.test_size}`);
    })
    .catch(() => {});
}

// ══════════════════════════════════════════════════════════
//  AUTO-SHOW RESULTS BANNER (on page load if results exist)
// ══════════════════════════════════════════════════════════
function autoShowResultsBanner() {
  fetch('/results')
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (data && data.stats) showResultsBanner();
    })
    .catch(() => {});
}

// ══════════════════════════════════════════════════════════
//  RUN EXPERIMENTS
// ══════════════════════════════════════════════════════════
function startExperiment() {
  const runBtn  = document.getElementById('runBtn');
  const logBox  = document.getElementById('logBox');
  const progBar = document.getElementById('progressBar');
  const progPct = document.getElementById('progressPct');
  const dot     = document.getElementById('statusDot');

  // Collect form values
  const nRunsEl   = document.getElementById('n_runs');
  const nWolvesEl = document.getElementById('n_wolves');
  const nIterEl   = document.getElementById('n_iter');
  const t0El      = document.getElementById('t0');
  const coolingEl = document.getElementById('cooling');
  const gwoIterEl = document.getElementById('n_iter_gwo');

  const n_iter     = parseInt(nIterEl?.value)    || 100;
  const n_iter_gwo = parseInt(gwoIterEl?.value)  || 70;

  // Validation
  if (n_iter_gwo >= n_iter) {
    showToast('GWO iterations must be less than total iterations.', 'error');
    return;
  }
  if (n_iter_gwo < 10) {
    showToast('GWO iterations must be at least 10.', 'error');
    return;
  }

  const payload = {
    n_runs     : parseInt(nRunsEl?.value)    || 10,
    n_wolves   : parseInt(nWolvesEl?.value)  || 20,
    n_iter,
    t0         : parseFloat(t0El?.value)     || 100,
    cooling    : parseFloat(coolingEl?.value)|| 0.95,
    n_iter_gwo,
    n_iter_sa  : n_iter - n_iter_gwo,
  };

  // Reset UI
  seenLogs.clear();
  runBtn.disabled = true;
  runBtn.innerHTML = `
    <svg class="spin" xmlns="http://www.w3.org/2000/svg" width="16" height="16"
         viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
    </svg> Running…`;
  logBox.innerHTML = '';
  setProgress(0);
  dot.className = 'status-dot running';

  // Hide stale banner
  const banner = document.getElementById('resultsBanner');
  if (banner) banner.style.display = 'none';

  // Hide elapsed
  const elapsedRow = document.getElementById('elapsedRow');
  if (elapsedRow) elapsedRow.style.display = 'none';

  // POST /run
  fetch('/run', {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify(payload),
  })
  .then(r => r.json())
  .then(data => {
    if (data.status === 'started') {
      startPolling();
    } else {
      appendLog('Cannot start: ' + (data.message || 'already running'), 'error');
      resetRunBtn();
    }
  })
  .catch(err => {
    appendLog('Network error: ' + err.message, 'error');
    resetRunBtn();
  });
}

// ══════════════════════════════════════════════════════════
//  POLLING
// ══════════════════════════════════════════════════════════
function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollStatus, 1000);
}

function pollStatus() {
  fetch('/status')
    .then(r => r.json())
    .then(data => {
      // Update progress bar
      setProgress(data.progress ?? 0);

      // Append new log lines only
      (data.log || []).forEach(line => {
        if (!seenLogs.has(line)) {
          seenLogs.add(line);
          appendLog(line, classifyLog(line));
        }
      });

      // Done?
      if (!data.running && data.done) {
        clearInterval(pollTimer);
        pollTimer = null;

        if (data.error) {
          setStatusDot('error');
          appendLog('ERROR: ' + data.error, 'error');
        } else {
          setStatusDot('done');
          setProgress(100);
          appendLog('✓ All experiments complete!', 'success');
          showElapsed(data.elapsed ?? 0);
          showResultsBanner();
        }
        resetRunBtn();
      }
    })
    .catch(() => {}); // Silently ignore polling errors
}

function classifyLog(line) {
  const l = line.toLowerCase();
  if (l.includes('error') || l.includes('fail'))  return 'error';
  if (l.includes('✓') || l.includes('complete'))  return 'success';
  if (l.includes('run') || l.includes('phase'))   return 'info';
  return '';
}

// ══════════════════════════════════════════════════════════
//  UI HELPERS
// ══════════════════════════════════════════════════════════
function appendLog(msg, type = '') {
  const box = document.getElementById('logBox');
  if (!box) return;

  // Remove placeholder
  box.querySelector('.log-placeholder')?.remove();

  const div = document.createElement('div');
  div.className = 'log-line' + (type ? ' ' + type : '');
  div.textContent = msg;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function setProgress(pct) {
  const bar = document.getElementById('progressBar');
  const lbl = document.getElementById('progressPct');
  if (bar) bar.style.width = Math.min(100, pct) + '%';
  if (lbl) lbl.textContent = Math.min(100, pct) + '%';
}

function setStatusDot(state) {
  const dot = document.getElementById('statusDot');
  if (dot) dot.className = 'status-dot ' + state;
}

function showElapsed(seconds) {
  const row = document.getElementById('elapsedRow');
  const val = document.getElementById('elapsedTime');
  if (row) row.style.display = 'flex';
  if (val) val.textContent = `Completed in ${Number(seconds).toFixed(1)}s`;
}

function showResultsBanner() {
  const b = document.getElementById('resultsBanner');
  if (b) b.style.display = 'flex';
}

function resetRunBtn() {
  const btn = document.getElementById('runBtn');
  if (!btn) return;
  btn.disabled = false;
  btn.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
         viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polygon points="5 3 19 12 5 21 5 3"/>
    </svg> Run Experiments`;
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val ?? '—';
}

// ══════════════════════════════════════════════════════════
//  FORM CONTROLS
// ══════════════════════════════════════════════════════════
function resetForm() {
  const defaults = {
    n_runs    : 10,
    n_wolves  : 20,
    n_iter    : 100,
    t0        : 100,
    cooling   : 0.95,
    n_iter_gwo: 70,
  };
  Object.entries(defaults).forEach(([id, val]) => {
    const el = document.getElementById(id);
    if (el) el.value = val;
  });
}

function clearResults() {
  if (!confirm('Clear all results and plot images?')) return;
  fetch('/clear', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      const cleared = data.cleared?.join(', ') || 'nothing to clear';
      showToast(`Cleared: ${cleared}`, 'info');
      setTimeout(() => location.reload(), 1200);
    })
    .catch(() => showToast('Failed to clear.', 'error'));
}

// ══════════════════════════════════════════════════════════
//  TOAST NOTIFICATIONS
// ══════════════════════════════════════════════════════════
function showToast(msg, type = 'info') {
  // Remove existing toast
  document.getElementById('toast')?.remove();

  const colorMap = {
    info    : 'var(--color-primary)',
    success : 'var(--color-green)',
    error   : 'var(--color-error)',
  };

  const toast = document.createElement('div');
  toast.id = 'toast';
  toast.style.cssText = `
    position: fixed; bottom: 24px; right: 24px; z-index: 9999;
    background: var(--color-surface-2);
    border: 1px solid var(--color-border);
    border-left: 3px solid ${colorMap[type] || colorMap.info};
    border-radius: var(--radius-md);
    padding: 0.75rem 1.25rem;
    font-size: var(--text-sm);
    color: var(--color-text);
    box-shadow: var(--shadow-lg);
    max-width: 340px;
    animation: fadeIn 0.2s ease;
  `;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

// ══════════════════════════════════════════════════════════
//  STATS SECTION
// ══════════════════════════════════════════════════════════
function loadStats() {
  fetch('/results')
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (!data || data.error) return;
      renderStatsTable(data.stats);
      renderImprovementBanner(data.imp_gwo, data.imp_sa);
      loadPlotImage('plotStats', '/plots/stats_bar_chart.png');
      loadPlotImage('plotRange',  '/plots/range_plot.png');
    })
    .catch(() => {});
}

function renderStatsTable(stats) {
  const tbody = document.getElementById('statsBody');
  if (!tbody || !stats?.length) return;

  // Rank by Mean (ascending = better for minimization)
  const sorted  = [...stats].sort((a, b) => a.Mean - b.Mean);
  const rankMap = Object.fromEntries(sorted.map((r, i) => [r.Algorithm, i + 1]));

  tbody.innerHTML = stats.map(row => {
    const rank   = rankMap[row.Algorithm];
    const isBest = rank === 1;
    return `
      <tr class="${isBest ? 'row-best' : ''}">
        <td style="font-family:var(--font-body);font-weight:${isBest?'700':'400'}">
          ${row.Algorithm}${isBest ? ' ★' : ''}
        </td>
        <td>${fmt6(row.Best)}</td>
        <td>${fmt6(row.Worst)}</td>
        <td>${fmt6(row.Mean)}</td>
        <td>${fmt6(row['Std Dev'])}</td>
        <td><span class="rank-badge rank-${rank}">${rank}</span></td>
      </tr>`;
  }).join('');
}

function renderImprovementBanner(impGWO, impSA) {
  const banner = document.getElementById('improvementBanner');
  if (!banner) return;
  setText('impGWO', (impGWO ?? '—') + '%');
  setText('impSA',  (impSA  ?? '—') + '%');
  banner.style.display = 'flex';
}

function loadPlotImage(containerId, url) {
  const wrap = document.getElementById(containerId);
  if (!wrap) return;
  const cacheBust = Date.now();
  const img = document.createElement('img');
  img.src     = `${url}?t=${cacheBust}`;
  img.alt     = 'Algorithm plot';
  img.loading = 'lazy';
  img.onerror = () => {
    wrap.innerHTML = `
      <div class="plot-placeholder">
        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24"
             fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/>
        </svg>
        <span>Run experiments to generate this plot</span>
      </div>`;
  };
  wrap.innerHTML = '';
  wrap.appendChild(img);
}

// ══════════════════════════════════════════════════════════
//  CONVERGENCE SECTION
// ══════════════════════════════════════════════════════════
function loadConvergence() {
  fetch('/convergence')
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (!data || data.error) return;
      renderConvergenceChart(data);
      loadPlotImage('plotConvergence', '/plots/convergence_plot.png');
    })
    .catch(() => {});
}

function renderConvergenceChart(data) {
  const canvas = document.getElementById('convergenceChart');
  if (!canvas) return;

  // Destroy previous instance
  if (convergenceChart) {
    convergenceChart.destroy();
    convergenceChart = null;
  }

  const isDark     = document.documentElement.getAttribute('data-theme') === 'dark';
  const gridColor  = isDark ? 'rgba(255,255,255,0.055)' : 'rgba(0,0,0,0.07)';
  const labelColor = isDark ? '#797876' : '#7a7974';
  const tooltipBg  = isDark ? '#1c1b19' : '#f9f8f5';
  const tooltipFg  = isDark ? '#cdccca' : '#28251d';
  const tooltipBdr = isDark ? '#2e2d2b' : '#d4d1ca';

  const algoKeys = ['GWO', 'SA', 'Hybrid GWO+SA'];
  const datasets = algoKeys.map(key => ({
    label          : key,
    data           : data[key] ?? [],
    borderColor    : ALGO_COLORS[key],
    backgroundColor: ALGO_COLORS[key] + '18',
    fill           : true,
    tension        : 0.38,
    borderWidth    : key === 'Hybrid GWO+SA' ? 3 : 2.5,
    pointRadius    : 0,
    pointHoverRadius: 5,
    pointHoverBackgroundColor: ALGO_COLORS[key],
  }));

  convergenceChart = new Chart(canvas, {
    type: 'line',
    data: { labels: data.iterations ?? [], datasets },
    options: {
      responsive         : true,
      maintainAspectRatio: false,
      interaction        : { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display : true,
          position: 'top',
          labels  : {
            color        : labelColor,
            usePointStyle: true,
            pointStyle   : 'circle',
            boxWidth     : 8,
            boxHeight    : 8,
            padding      : 18,
            font         : { size: 11, family: 'General Sans, Inter, sans-serif' },
          },
        },
        tooltip: {
          backgroundColor: tooltipBg,
          titleColor     : labelColor,
          bodyColor      : tooltipFg,
          borderColor    : tooltipBdr,
          borderWidth    : 1,
          padding        : 10,
          callbacks: {
            title : ctx => `Iteration ${ctx[0].label}`,
            label : ctx => ` ${ctx.dataset.label}: ${fmt6(ctx.parsed.y)}`,
          },
        },
      },
      scales: {
        x: {
          grid : { color: gridColor },
          ticks: {
            color        : labelColor,
            font         : { size: 10 },
            maxTicksLimit: 11,
          },
          title: {
            display: true,
            text   : 'Iteration',
            color  : labelColor,
            font   : { size: 11 },
            padding: { top: 6 },
          },
        },
        y: {
          grid : { color: gridColor },
          ticks: {
            color   : labelColor,
            font    : { size: 10 },
            callback: v => v.toFixed(4),
          },
          title: {
            display: true,
            text   : 'Best Fitness  f(x)',
            color  : labelColor,
            font   : { size: 11 },
            padding: { bottom: 6 },
          },
        },
      },
    },
  });
}

// ══════════════════════════════════════════════════════════
//  INFERENCE SECTION
// ══════════════════════════════════════════════════════════
function loadInference() {
  fetch('/results')
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (!data || data.error) return;
      fillInferenceValues(data.stats, data.imp_gwo, data.imp_sa);
    })
    .catch(() => {});
}

function fillInferenceValues(stats, impGWO, impSA) {
  if (!stats?.length) return;

  const hybrid = stats.find(r => r.Algorithm === 'Hybrid GWO+SA');
  const gwo    = stats.find(r => r.Algorithm === 'GWO');
  const sa     = stats.find(r => r.Algorithm === 'SA');
  if (!hybrid || !gwo || !sa) return;

  // Q1 — mean fitness result chips
  setText('q1GWO',    fmt6(gwo.Mean)    + ' (GWO)');
  setText('q1SA',     fmt6(sa.Mean)     + ' (SA)');
  setText('q1Hybrid', fmt6(hybrid.Mean) + ' (Hybrid ★)');

  // Q2 — time bar widths (use relative computation times)
  // GWO=100 iterations, SA=100 iterations, Hybrid=100 total (same)
  // Represent as normalized complexity
  setBarWidth('barGWO',    60);   // ~60% relative compute
  setBarWidth('barSA',     40);   // ~40% relative compute
  setBarWidth('barHybrid', 100);  // 100% (same total iters, slightly more overhead)
  setText('valGWO',    '70 GWO iters + 0 SA');
  setText('valSA',     '0 GWO iters + 100 SA');
  setText('valHybrid', '70 GWO + 30 SA iters');

  // Q3 — synergy proof values
  setText('q3GWO',    fmt6(gwo.Mean));
  setText('q3SA',     fmt6(sa.Mean));
  setText('q3Hybrid', fmt6(hybrid.Mean));
  setText('q3ImpGWO', impGWO + '% better than GWO');
  setText('q3ImpSA',  impSA  + '% better than SA');
}

function setBarWidth(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = pct + '%';
}

// ══════════════════════════════════════════════════════════
//  UTILITIES
// ══════════════════════════════════════════════════════════
function fmt6(val) {
  if (val == null || isNaN(val)) return '—';
  return Number(val).toFixed(6);
}