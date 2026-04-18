/* BillWise Dashboard — shared JavaScript */

// ── Chart rendering ───────────────────────────────────────────────────────────
function bwChart(elementId, jsonStr) {
  if (!jsonStr || !document.getElementById(elementId)) return;
  try {
    const fig = JSON.parse(jsonStr);
    Plotly.react(elementId, fig.data || [], fig.layout || {}, { responsive: true });
  } catch (e) {
    console.warn("Chart render error:", elementId, e);
  }
}

// ── Confidence badge ──────────────────────────────────────────────────────────
function bwConfBadge(score) {
  const pct = Math.round(score * 100);
  if (score >= 0.90) return `<span class="bw-conf high">${pct}%</span>`;
  if (score >= 0.70) return `<span class="bw-conf medium">${pct}%</span>`;
  return `<span class="bw-conf low">${pct}%</span>`;
}

// ── Build a table from an array of objects ────────────────────────────────────
function bwBuildTable(rows, columns, formatters) {
  if (!rows || rows.length === 0) {
    return '<p style="color:var(--muted);padding:16px 0">No data.</p>';
  }
  const cols = columns || Object.keys(rows[0]);
  let html = '<table class="bw-table"><thead><tr>';
  cols.forEach(c => {
    html += `<th>${c.replace(/_/g, ' ').toUpperCase()}</th>`;
  });
  html += '</tr></thead><tbody>';
  rows.forEach(row => {
    html += '<tr>';
    cols.forEach(c => {
      const fmt = formatters && formatters[c];
      const val = row[c] != null ? row[c] : '—';
      html += `<td>${fmt ? fmt(val, row) : val}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  return html;
}

// ── Loading overlay ───────────────────────────────────────────────────────────
function bwLoading(active) {
  const el = document.getElementById('bw-loading-overlay');
  if (el) el.classList.toggle('active', active);
}

// ── Tab switching ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.bw-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const group = tab.dataset.tabGroup;
      const target = tab.dataset.tabTarget;
      document.querySelectorAll(`.bw-tab[data-tab-group="${group}"]`).forEach(t => t.classList.remove('active'));
      document.querySelectorAll(`.bw-tab-pane[data-tab-group="${group}"]`).forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      const pane = document.querySelector(`.bw-tab-pane[data-tab-group="${group}"][data-tab-id="${target}"]`);
      if (pane) pane.classList.add('active');
    });
  });
});

// ── Generic AJAX filter helper ────────────────────────────────────────────────
async function bwFetch(url, body) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return resp.json();
}
