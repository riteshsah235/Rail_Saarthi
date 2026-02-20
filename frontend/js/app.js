const API_BASE = (() => {
  const host = window.location.hostname;
  const port = window.location.port;
  const isLocal = !host || host === 'localhost' || host === '127.0.0.1';
  if (isLocal && (!port || port === '80' || port === '443')) return 'http://localhost:8000';
  if (isLocal && port !== '8000') return 'http://localhost:8000'; // frontend may be on 3000, 5500, etc.
  return `${window.location.protocol}//${host}${port === '80' || port === '443' ? '' : ':' + port}`;
})();

const $ = (id) => document.getElementById(id);

async function checkApi() {
  const dot = $('api-status-dot');
  const status = $('api-status');
  try {
    const r = await fetch(`${API_BASE}/`);
    const ok = r.ok;
    dot.className = 'w-2 h-2 rounded-full ' + (ok ? 'bg-emerald-400' : 'bg-amber-400');
    status.textContent = ok ? 'API connected' : 'API error';
  } catch (e) {
    dot.className = 'w-2 h-2 rounded-full bg-red-400';
    status.textContent = 'API offline — start backend (see README)';
  }
}

async function loadMetrics() {
  try {
    const r = await fetch(`${API_BASE}/api/metrics`);
    const data = await r.json();
    const cat = data.category || {};
    const sev = data.severity || {};
    $('metric-cat-acc').textContent = cat.accuracy != null ? (cat.accuracy * 100).toFixed(1) + '%' : '—';
    $('metric-cat-f1').textContent = cat.f1_weighted != null ? (cat.f1_weighted * 100).toFixed(1) + '%' : '—';
    $('metric-sev-acc').textContent = sev.accuracy != null ? (sev.accuracy * 100).toFixed(1) + '%' : '—';
    $('metric-sev-f1').textContent = sev.f1_weighted != null ? (sev.f1_weighted * 100).toFixed(1) + '%' : '—';
  } catch (e) {
    $('metric-cat-acc').textContent = '—';
    $('metric-cat-f1').textContent = '—';
    $('metric-sev-acc').textContent = '—';
    $('metric-sev-f1').textContent = '—';
  }
}

async function loadTrends() {
  try {
    const r = await fetch(`${API_BASE}/api/trends`);
    const data = await r.json();
    const byCat = data.by_category || [];
    const bySev = data.by_severity || [];
    const labelsCat = byCat.map((x) => x.category.replace(/_/g, ' '));
    const valuesCat = byCat.map((x) => x.count);
    const labelsSev = bySev.map((x) => x.severity);
    const valuesSev = bySev.map((x) => x.count);
    const colors = ['#1e3a5f', '#d4a853', '#0f172a', '#64748b', '#16a34a', '#b91c1c'];
    const sevColors = { low: '#16a34a', medium: '#d97706', high: '#ea580c', critical: '#b91c1c' };

    if (window.chartCategory) window.chartCategory.destroy();
    window.chartCategory = new Chart($('chart-category'), {
      type: 'bar',
      data: {
        labels: labelsCat,
        datasets: [{ label: 'Count', data: valuesCat, backgroundColor: colors.slice(0, labelsCat.length) }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, ticks: { stepSize: 1 } },
        },
      },
    });

    if (window.chartSeverity) window.chartSeverity.destroy();
    window.chartSeverity = new Chart($('chart-severity'), {
      type: 'doughnut',
      data: {
        labels: labelsSev,
        datasets: [{ data: valuesSev, backgroundColor: labelsSev.map((s) => sevColors[s] || '#64748b') }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
      },
    });
  } catch (e) {
    console.warn('Trends load failed', e);
  }
}

async function loadClusteringViz() {
  const container = $('cluster-viz-container');
  const placeholder = $('cluster-placeholder');
  const canvas = $('chart-cluster');
  try {
    const r = await fetch(`${API_BASE}/api/clustering-viz`);
    const data = await r.json();
    const points = data.points || [];
    placeholder.classList.add('hidden');
    if (points.length === 0) {
      placeholder.classList.remove('hidden');
      placeholder.textContent = 'No clustering data. Run training first.';
      return;
    }
    const nClusters = data.n_clusters || 3;
    const colors = ['#1e3a5f', '#d4a853', '#0f172a', '#64748b', '#16a34a', '#b91c1c'];
    const datasets = [];
    for (let c = 0; c < nClusters; c++) {
      const clusterPoints = points.filter((p) => p.cluster === c);
      datasets.push({
        label: `Cluster ${c + 1}`,
        data: clusterPoints.map((p) => ({ x: p.x, y: p.y })),
        backgroundColor: colors[c % colors.length] + '99',
        borderColor: colors[c % colors.length],
        borderWidth: 1,
        pointRadius: 6,
      });
    }
    if (window.chartCluster) window.chartCluster.destroy();
    window.chartCluster = new Chart(canvas, {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 2,
        plugins: {
          legend: { position: 'bottom' },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const p = points.find((p, i) => {
                  const d = ctx.raw;
                  return Math.abs(p.x - d.x) < 1e-5 && Math.abs(p.y - d.y) < 1e-5;
                });
                return p ? p.text + '…' : '';
              },
            },
          },
        },
        scales: {
          x: { title: { display: true, text: 'PC1' } },
          y: { title: { display: true, text: 'PC2' } },
        },
      },
    });
  } catch (e) {
    placeholder.classList.remove('hidden');
    placeholder.textContent = 'Clustering viz failed. Is the API running?';
    console.warn(e);
  }
}

async function analyzeComplaint() {
  const text = ($('complaint-input').value || '').trim();
  const resultEl = $('analyze-result');
  const errEl = $('analyze-error');
  resultEl.classList.add('hidden');
  errEl.classList.add('hidden');
  if (!text) {
    errEl.textContent = 'Please enter complaint text.';
    errEl.classList.remove('hidden');
    return;
  }
  $('btn-analyze').disabled = true;
  try {
    const r = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Analysis failed');
    $('result-category').textContent = (data.category || '').replace(/_/g, ' ');
    const sevEl = $('result-severity');
    sevEl.textContent = data.severity || '—';
    sevEl.className = 'inline-block px-2 py-0.5 rounded-full text-sm font-medium severity-' + (data.severity || 'low');
    $('result-cluster').textContent = 'Cluster #' + (data.cluster_id + 1);
    resultEl.classList.remove('hidden');
  } catch (e) {
    errEl.textContent = e.message || 'Request failed.';
    errEl.classList.remove('hidden');
  }
  $('btn-analyze').disabled = false;
}

async function loadCategories() {
  try {
    const r = await fetch(`${API_BASE}/api/categories`);
    const data = await r.json();
    const list = $('categories-list');
    list.innerHTML = (data.categories || []).map((c) => `<li class="capitalize">${c.replace(/_/g, ' ')}</li>`).join('');
  } catch (e) {
    $('categories-list').innerHTML = '<li class="text-slate-500">Could not load categories.</li>';
  }
}

function init() {
  checkApi();
  loadMetrics();
  loadTrends();
  loadClusteringViz();
  loadCategories();
  $('btn-analyze').addEventListener('click', analyzeComplaint);
  $('btn-clear').addEventListener('click', () => {
    $('complaint-input').value = '';
    $('analyze-result').classList.add('hidden');
    $('analyze-error').classList.add('hidden');
  });
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
