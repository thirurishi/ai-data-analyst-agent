"use strict";

const API = window.location.port === "5173" ? "http://127.0.0.1:8000" : "";
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const state = { files: [], current: null, insights: null, status: null, charts: new Map(), recs: [], recsFor: null };
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const ICON = {
  spark: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l1.9 4.6L18.5 9.5l-4.6 1.9L12 16l-1.9-4.6L5.5 9.5l4.6-1.9z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/></svg>',
  expand: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7"/></svg>',
  download: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11m0 0l-4-4m4 4l4-4M5 20h14"/></svg>',
  copy: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 012-2h8"/></svg>',
  alert: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 8v5m0 3.5v.01M10.3 3.9L2.4 18a2 2 0 001.7 3h15.8a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z"/></svg>',
};

/* ------------------------------------------------------------------ utilities */
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
function fmt(value) {
  if (value === null || value === undefined || value === "") return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  const abs = Math.abs(n);
  if (abs >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (abs >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (abs >= 1e4) return (n / 1e3).toFixed(1) + "K";
  return Number.isInteger(n) ? n.toLocaleString("en") : n.toLocaleString("en", { maximumFractionDigits: 2 });
}
function bytes(size) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(0)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}
function slug(text) { return String(text).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "chart"; }
function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.toggle("error", isError);
  el.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove("show"), 3800);
}
async function api(path, options = {}) {
  const init = { ...options };
  if (init.json !== undefined) {
    init.method = init.method || "POST";
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(init.json);
    delete init.json;
  }
  let response;
  try {
    response = await fetch(API + path, init);
  } catch {
    throw new Error("The API is not reachable. Check that the server is running.");
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body?.error?.message || (Array.isArray(body?.detail) ? body.detail[0]?.msg : body?.detail) || `Request failed (${response.status}).`;
    throw new Error(detail);
  }
  return body;
}

/* ------------------------------------------------------------------ theme */
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("theme", theme); } catch {}
  $("#theme-toggle").setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
  // Redraw charts so axis and grid colours follow the theme.
  state.charts.forEach((chart) => {
    const spec = chart.$spec, canvas = chart.canvas;
    if (spec && canvas?.isConnected) drawChart(canvas, spec);
  });
}

/* ------------------------------------------------------------------ charts */
// Chart.js loads on its own so a slow or blocked CDN never stops the rest of the app.
const chartReady = new Promise((resolve) => {
  const script = document.createElement("script");
  script.src = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js";
  script.onload = () => resolve(true);
  script.onerror = () => resolve(false);
  setTimeout(() => resolve(Boolean(window.Chart)), 8000);
  document.head.append(script);
});
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function palette() {
  // Muted, nature-inspired series colours: sage, olive, clay, slate, sand, rose, stone.
  return [css("--accent-strong"), "#8A9A5B", "#C0825F", "#6B7F8E", "#CDB487", "#B9787A", "#9A9A8F", "#4E6E78"];
}
async function drawChart(canvas, spec) {
  if (!(await chartReady) || !window.Chart) {
    canvas.parentElement.innerHTML = '<p class="hint">Charts need an internet connection to load Chart.js. The numbers are still accurate.</p>';
    return null;
  }
  const key = canvas.dataset.key || Math.random().toString(36).slice(2);
  canvas.dataset.key = key;
  state.charts.get(key)?.destroy();
  const ink2 = css("--ink-3"), line = css("--line"), accent = css("--accent-strong"), font = css("--font");
  const round = ["pie", "doughnut"].includes(spec.type);
  const type = spec.type === "histogram" ? "bar" : spec.type;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createLinearGradient(0, 0, 0, canvas.parentElement.clientHeight || 260);
  gradient.addColorStop(0, accent + "55");
  gradient.addColorStop(1, accent + "00");
  const data = spec.type === "scatter"
    ? { datasets: [{ label: spec.datasets[0].label, data: spec.points, backgroundColor: accent + "aa", pointRadius: 3.2, pointHoverRadius: 5 }] }
    : {
        labels: spec.labels,
        datasets: [{
          label: spec.datasets[0].label,
          data: spec.datasets[0].data,
          backgroundColor: round ? palette() : spec.type === "line" ? gradient : accent,
          hoverBackgroundColor: round ? undefined : css("--accent"),
          borderColor: round ? css("--surface") : accent,
          borderWidth: round ? 3 : spec.type === "line" ? 2.5 : 0,
          borderRadius: type === "bar" ? 6 : 0,
          borderSkipped: false,
          barPercentage: spec.type === "histogram" ? 1 : 0.72,
          categoryPercentage: spec.type === "histogram" ? 0.94 : 0.8,
          tension: 0.35,
          pointRadius: spec.type === "line" ? 0 : undefined,
          pointHoverRadius: 5,
          pointBackgroundColor: accent,
          fill: spec.type === "line",
          cutout: spec.type === "doughnut" ? "62%" : undefined,
        }],
      };
  const axis = (title) => ({
    title: { display: Boolean(title), text: title, color: ink2, font: { family: font, size: 12 } },
    ticks: { color: ink2, font: { family: font, size: 11 }, maxRotation: 40, autoSkip: true, callback(value) {
      const label = this.getLabelForValue ? this.getLabelForValue(value) : value;
      return typeof label === "number" ? fmt(label) : String(label).length > 16 ? String(label).slice(0, 15) + "…" : label;
    } },
    grid: { color: line, drawTicks: false },
    border: { display: false },
  });
  const chart = new Chart(canvas, {
    type,
    data,
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: reducedMotion ? 0 : 600, easing: "easeOutQuart" },
      interaction: { mode: round || spec.type === "scatter" ? "nearest" : "index", intersect: false },
      plugins: {
        legend: { display: round, position: "right", labels: { color: css("--ink-2"), boxWidth: 10, boxHeight: 10, useBorderRadius: true, borderRadius: 3, font: { family: font } } },
        tooltip: {
          backgroundColor: css("--surface-2"), titleColor: css("--ink"), bodyColor: css("--ink-2"), borderColor: css("--line-strong"), borderWidth: 1,
          padding: 10, cornerRadius: 8, titleFont: { family: font, weight: "600" }, bodyFont: { family: font },
          callbacks: { label: (c) => spec.type === "scatter" ? ` ${fmt(c.raw.x)}, ${fmt(c.raw.y)}` : ` ${c.dataset.label}: ${fmt(c.raw)}` },
        },
      },
      scales: round ? {} : {
        x: { ...axis(spec.x_label), ...(spec.type === "scatter" ? { type: "linear", ticks: { color: ink2, callback: (v) => fmt(v) } } : {}), grid: { display: spec.type === "scatter", color: line } },
        y: { ...axis(spec.y_label), beginAtZero: spec.type !== "scatter", ticks: { color: ink2, callback: (v) => fmt(v), maxTicksLimit: 6 } },
      },
    },
  });
  chart.$spec = spec;
  state.charts.set(key, chart);
  return chart;
}
function figure(spec, big = false) {
  const el = document.createElement("div");
  el.className = "figure card" + (big ? " big" : "");
  el.innerHTML = `
    <div class="figure-head"><h3>${esc(spec.title)}</h3>
      <div class="fig-actions">
        <button class="icon-btn" type="button" data-act="expand" aria-label="Enlarge chart" title="Enlarge">${ICON.expand}</button>
        <button class="icon-btn" type="button" data-act="download" aria-label="Download chart as PNG" title="Download PNG">${ICON.download}</button>
      </div></div>
    <div class="canvas"><canvas role="img" aria-label="${esc(spec.title)}"></canvas></div>${spec.insight ? `<p>${esc(spec.insight)}</p>` : ""}`;
  const canvas = $("canvas", el);
  requestAnimationFrame(() => drawChart(canvas, spec));
  $('[data-act="expand"]', el).addEventListener("click", () => openViewer(spec));
  $('[data-act="download"]', el).addEventListener("click", () => downloadChart(state.charts.get(canvas.dataset.key), spec.title));
  return el;
}
function downloadChart(chart, title) {
  if (!chart) return toast("The chart is still loading.", true);
  const link = document.createElement("a");
  link.href = chart.toBase64Image("image/png", 1);
  link.download = `${slug(title)}.png`;
  link.click();
}
let viewerChart = null;
async function openViewer(spec) {
  const dialog = $("#viewer");
  $("#viewer-title").textContent = spec.title;
  $("#viewer-note").textContent = spec.insight || "";
  dialog.showModal();
  viewerChart = await drawChart($("#viewer-canvas"), spec);
}
function resetCharts() { state.charts.forEach((c) => c.destroy()); state.charts.clear(); }

/* ------------------------------------------------------------------ status + files */
async function loadStatus() {
  try { state.status = await api("/status"); } catch { state.status = null; }
  const el = $("#engine"), pill = $("#mode-pill");
  const s = state.status;
  if (!s) {
    el.innerHTML = "<b>API offline</b><p>Start the server with uvicorn, then reload this page.</p>";
    pill.textContent = "API offline";
    return;
  }
  el.classList.toggle("on", s.llm_configured);
  pill.classList.toggle("on", s.llm_configured);
  pill.textContent = s.llm_configured ? "AI agent" : "Built-in analyst";
  el.innerHTML = s.llm_configured
    ? `<b>AI agent connected</b><p>${esc(s.llm_model)} plans each question; answers come from verified calculations.</p>`
    : `<b>Built-in analyst</b><p>Rule-based planner, no API key needed. Add <code>OPENROUTER_API_KEY</code> to <code>.env</code> to use an LLM.</p>`;
}
async function loadFiles(selectId) {
  try { state.files = (await api("/files")).files; } catch (error) { toast(error.message, true); return; }
  renderFiles();
  if (selectId) openFile(selectId);
}
function renderFiles() {
  const list = $("#files");
  const query = $("#file-search").value.trim().toLowerCase();
  const files = state.files.filter((f) => !query || f.original_filename.toLowerCase().includes(query));
  $("#file-count").textContent = state.files.length ? String(state.files.length) : "";
  $("#file-search-wrap").hidden = state.files.length < 6;
  list.innerHTML = "";
  if (!state.files.length) { list.innerHTML = '<li class="files-empty">No files yet. Upload one, or try the sample.</li>'; return; }
  if (!files.length) { list.innerHTML = '<li class="files-empty">No files match that filter.</li>'; return; }
  for (const file of files) {
    const li = document.createElement("li");
    li.innerHTML = `<button type="button" data-id="${file.file_id}" title="${esc(file.original_filename)}"><span class="ext ${file.kind === "document" ? "doc" : ""}">${esc(file.extension.toUpperCase())}</span><span><span class="fname">${esc(file.original_filename)}</span><span class="fsize">${bytes(file.file_size)}</span></span></button>`;
    $("button", li).addEventListener("click", () => openFile(file.file_id));
    list.append(li);
  }
  markCurrent();
}
function markCurrent() {
  $$(".files button").forEach((b) => b.setAttribute("aria-current", String(b.dataset.id === state.current?.file_id)));
}
async function upload(file) {
  const drop = $("#drop");
  drop.classList.add("busy");
  const form = new FormData();
  form.append("file", file, file.name);
  try {
    const response = await fetch(API + "/upload", { method: "POST", body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(body?.error?.message || "Upload failed.");
    toast(`Uploaded ${file.name}`);
    await loadFiles(body.file_id);
  } catch (error) {
    toast(error.message || "Upload failed.", true);
  } finally {
    drop.classList.remove("busy");
    $("#file-input").value = "";
  }
}
function setCrumb(name) {
  $("#crumb-sep").hidden = !name;
  $("#crumb-current").textContent = name || "";
}
function goHome() {
  state.current = null;
  resetCharts();
  markCurrent();
  setCrumb("");
  history.replaceState(null, "", location.pathname);
  $("#workspace").hidden = true;
  $("#docspace").hidden = true;
  $("#welcome").hidden = false;
  playDemo();
}

/* ------------------------------------------------------------------ open a dataset */
function skeleton() {
  return `<div class="sk-kpis">${'<div class="sk"></div>'.repeat(5)}</div>
    <div class="sk-row"><div class="sk"></div><div class="sk"></div></div>
    <div class="sk-charts"><div class="sk"></div><div class="sk"></div></div>`;
}
async function openFile(fileId) {
  const file = state.files.find((f) => f.file_id === fileId);
  if (!file) return;
  state.current = file;
  markCurrent();
  setCrumb(file.original_filename);
  history.replaceState(null, "", `#${fileId}`);
  resetCharts();
  $("#welcome").hidden = true;
  if (file.kind === "document") return openDocument(file);
  $("#docspace").hidden = true;
  $("#workspace").hidden = false;
  $("#ws-name").textContent = file.original_filename;
  $("#ws-meta").textContent = "Analysing…";
  $("#panel-overview").innerHTML = skeleton();
  $("#export-pdf").href = `${API}/report?file_id=${fileId}&format=pdf`;
  $("#export-md").href = `${API}/report?file_id=${fileId}&format=md`;
  $("#thread").innerHTML = "";
  $("#quality-badge").hidden = true;
  state.recs = []; state.recsFor = null;
  $("#panel-quality").innerHTML = "";
  selectTab("overview");
  try {
    state.insights = await api(`/insights?file_id=${fileId}`);
  } catch (error) {
    $("#panel-overview").innerHTML = errorCard(error.message);
    $("#ws-meta").textContent = "";
    return;
  }
  if (state.current?.file_id !== fileId) return;
  const ins = state.insights;
  $("#ws-meta").textContent = `${ins.rows.toLocaleString("en")} rows, ${ins.columns} columns, ${bytes(file.file_size)}`;
  const issues = ins.health.issue_count;
  $("#quality-badge").hidden = !issues;
  $("#quality-badge").textContent = issues;
  renderOverview(ins);
  renderAskChips(ins);
  renderThreadEmpty();
  renderBuilder(ins);
  renderData(ins);
}
function errorCard(message) {
  return `<div class="answer error reveal"><div class="avatar">${ICON.alert}</div><div class="answer-body card"><p>${esc(message)}</p></div></div>`;
}

function healthColor(score) { return score >= 80 ? "var(--good)" : score >= 65 ? "var(--warn)" : "var(--bad)"; }
function renderOverview(ins) {
  const panel = $("#panel-overview");
  const h = ins.health;
  const color = healthColor(h.score);
  panel.innerHTML = `
    <div class="kpis">${ins.kpis.map((k) => {
      const isHealth = k.label === "Data health";
      return `<div class="kpi card ${isHealth ? "health" : ""}" style="${isHealth ? `--kpi-color:${color}` : ""}"><span>${esc(k.label)}</span><strong>${esc(k.display)}</strong></div>`;
    }).join("")}</div>
    <div class="overview-grid">
      <section>
        <h2 class="section-title">What stands out</h2>
        <ul class="findings card">${ins.findings.length ? ins.findings.map((f) => `<li class="${f.severity}"><b>${esc(f.title)}</b><span>${esc(f.detail)}</span></li>`).join("") : "<li><b>Nothing unusual</b><span>The automatic scan found no notable patterns.</span></li>"}</ul>
      </section>
      <section>
        <h2 class="section-title">Data health</h2>
        <div class="health card">
          <div class="health-top">
            <div class="ring" style="--p:${h.score};--ring-color:${color}"><strong>${h.score}</strong></div>
            <div><b>${esc(h.label)}</b><p>${h.issue_count} issue${h.issue_count === 1 ? "" : "s"} found, ${fmt(h.missing_cells)} missing cells, ${fmt(h.duplicate_rows)} duplicate rows.</p></div>
          </div>
          <div class="dims">${Object.entries(h.dimensions).map(([name, value]) => `<div class="dim"><span>${name[0].toUpperCase() + name.slice(1)}</span><i style="--w:${value}%"></i><em>${value}%</em></div>`).join("")}</div>
          ${h.issue_count ? '<p style="margin:16px 0 0"><button class="btn small" type="button" id="go-quality">Review issues</button></p>' : ""}
        </div>
      </section>
    </div>
    ${ins.charts.length ? '<h2 class="section-title" style="margin-top:30px">Charts</h2>' : ""}
    <div class="charts" id="overview-charts" style="margin-top:0"></div>`;
  const grid = $("#overview-charts", panel);
  ins.charts.forEach((spec) => grid.append(figure(spec)));
  $("#go-quality", panel)?.addEventListener("click", () => selectTab("quality"));
}

/* ------------------------------------------------------------------ ask */
function renderAskChips(ins) {
  const chips = $("#ask-chips");
  chips.innerHTML = "";
  ins.suggested_questions.forEach((q) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.textContent = q;
    b.addEventListener("click", () => ask(q));
    chips.append(b);
  });
}
function renderThreadEmpty() {
  $("#thread").innerHTML = `<div class="chat-empty"><b>Ask anything about ${esc(state.current.original_filename)}</b>Totals, averages, rankings and breakdowns. Each answer shows the calculation behind it.</div>`;
}
async function ask(question) {
  question = question.trim();
  if (!question || !state.current) return;
  $("#ask-input").value = "";
  $(".chat-empty", $("#thread"))?.remove();
  const turn = document.createElement("div");
  turn.className = "turn reveal";
  turn.innerHTML = `<div class="bubble">${esc(question)}</div>
    <div class="answer"><div class="avatar">${ICON.spark}</div><div class="answer-body card"><div class="typing" aria-label="Working"><i></i><i></i><i></i></div></div></div>`;
  $("#thread").prepend(turn);
  const answer = $(".answer", turn), box = $(".answer-body", turn);
  try {
    const res = await api("/assistant/ask", { json: { file_id: state.current.file_id, question } });
    const args = Object.entries(res.tool.arguments).map(([k, v]) => `${k}: ${v}`).join(", ");
    box.innerHTML = `
      <p>${esc(res.answer)}</p>
      ${res.note ? `<p class="note">${esc(res.note)}</p>` : ""}
      ${res.chart ? '<div class="canvas"><canvas></canvas></div>' : ""}
      <div class="meta"><span class="tag ${res.mode}">${res.mode === "llm" ? "AI agent" : "Built-in analyst"}</span><span class="tag">${esc(res.tool.name)}</span>${args ? `<span class="tag">${esc(args)}</span>` : ""}
        <span class="spacer"></span><button class="btn ghost small" type="button" data-act="copy">${ICON.copy} Copy</button></div>
      <details><summary>Show the calculation result</summary><pre>${esc(JSON.stringify(res.result, null, 2))}</pre></details>`;
    if (res.chart) drawChart($("canvas", box), res.chart);
    $('[data-act="copy"]', box).addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(res.answer); toast("Answer copied"); } catch { toast("Copy is not available in this browser.", true); }
    });
  } catch (error) {
    answer.classList.add("error");
    $(".avatar", answer).innerHTML = ICON.alert;
    box.innerHTML = `<p>${esc(error.message)}</p>`;
  }
}

/* ------------------------------------------------------------------ builder + data */
function renderBuilder(ins) {
  const cols = ins.column_summary;
  const xs = cols.filter((c) => c.role !== "identifier");
  const nums = cols.filter((c) => c.role === "numeric");
  $("#builder-x").innerHTML = xs.map((c) => `<option value="${esc(c.name)}" ${c.name === (ins.primary_category || ins.primary_date) ? "selected" : ""}>${esc(c.name)}</option>`).join("");
  $("#builder-y").innerHTML = '<option value="">(row count)</option>' + nums.map((c) => `<option value="${esc(c.name)}" ${c.name === ins.primary_metric ? "selected" : ""}>${esc(c.name)}</option>`).join("");
  const out = $("#builder-out");
  out.className = "figure big empty";
  out.innerHTML = '<p class="hint">Pick a chart type and columns, then draw it.</p>';
}
function renderData(ins) {
  const roleRows = ins.column_summary.map((c) => `
    <tr><td><b>${esc(c.name)}</b></td><td><span class="role ${c.role}">${c.role}</span></td><td>${esc(c.dtype)}</td>
    <td><span class="miss" style="--w:${Math.min(100, c.missing_percentage)}%"></span>${c.missing_percentage}%</td>
    <td class="num">${fmt(c.unique)}</td><td>${c.sample.map(esc).join(", ")}</td></tr>`).join("");
  const preview = ins.preview;
  $("#panel-data").innerHTML = `
    <h2 class="section-title">Columns</h2>
    <div class="table-wrap card"><table><thead><tr><th>Column</th><th>Role</th><th>Type</th><th>Missing</th><th class="num">Unique</th><th>Examples</th></tr></thead><tbody>${roleRows}</tbody></table></div>
    <div class="subhead"><h2 class="section-title" style="margin:0">First rows</h2><a class="btn small" href="${API}/files/${state.current.file_id}/download" download>${ICON.download} Download CSV</a></div>
    <div class="table-wrap card"><table><thead><tr>${preview.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
    <tbody>${preview.rows.map((r) => `<tr>${r.map((v) => `<td class="${typeof v === "number" ? "num" : ""}">${v === null ? '<span class="hint">empty</span>' : esc(typeof v === "number" ? fmt(v) : v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

/* ------------------------------------------------------------------ quality */
async function loadQuality() {
  const panel = $("#panel-quality");
  if (!state.current || state.recsFor === state.current.file_id) return;
  panel.innerHTML = '<div class="card" style="padding:18px"><div class="sk sk-line" style="width:60%"></div><div class="sk sk-line" style="width:85%"></div><div class="sk sk-line" style="width:40%"></div></div>';
  try {
    const res = await api(`/data-quality/recommendations?file_id=${state.current.file_id}`);
    state.recs = res.recommendations;
    state.recsFor = state.current.file_id;
  } catch (error) {
    panel.innerHTML = errorCard(error.message);
    return;
  }
  renderQuality();
}
const AUTO_FIX = new Set(["duplicate_rows", "empty_column", "constant_column", "categorical_inconsistency"]);
function renderQuality() {
  const panel = $("#panel-quality");
  if (!state.recs.length) {
    panel.innerHTML = '<div class="card notice">No quality issues were found. This dataset is ready for analysis.</div>';
    return;
  }
  panel.innerHTML = `<p class="quality-intro">Nothing changes without your approval. Approve a recommendation, then apply it to create a cleaned copy of the file. Your original upload stays untouched.</p><div class="recs card"></div>`;
  const box = $(".recs", panel);
  state.recs.forEach((rec) => {
    const row = document.createElement("div");
    row.className = "rec";
    const title = rec.category.replace(/_/g, " ");
    const canApply = AUTO_FIX.has(rec.category);
    let actions = `<span class="status ${rec.status}">${rec.status}</span>`;
    if (rec.status === "pending") actions += `<button class="btn small good" data-act="approve">Approve</button><button class="btn small bad" data-act="reject">Reject</button>`;
    if (rec.status === "approved" && canApply) actions += `<button class="btn small primary" data-act="apply">Apply fix</button>`;
    if (rec.status === "approved" && !canApply) actions += `<span class="hint" style="font-size:13px">Needs manual handling</span>`;
    row.innerHTML = `
      <div><h3>${title[0].toUpperCase() + title.slice(1)}${rec.column ? ` <code>${esc(rec.column)}</code>` : ""}<span class="sev ${rec.severity}">${rec.severity}</span></h3>
      <p>${esc(rec.issue)}</p><p><b>Recommendation:</b> ${esc(rec.recommendation)}</p></div>
      <div class="act">${actions}</div>`;
    $$("[data-act]", row).forEach((b) => b.addEventListener("click", () => qualityAction(rec, b.dataset.act, b)));
    box.append(row);
  });
}
async function qualityAction(rec, action, button) {
  button.disabled = true;
  try {
    if (action === "apply") {
      const res = await api("/data-quality/apply", { json: { file_id: state.current.file_id, recommendation_id: rec.recommendation_id } });
      rec.status = "applied";
      if (res.cleaned_file_id) {
        toast(`${res.message} Opening the cleaned copy.`);
        await loadFiles(res.cleaned_file_id);
        return;
      }
      toast(res.message);
    } else {
      const res = await api(`/data-quality/recommendations/${rec.recommendation_id}/${action}`, { method: "POST" });
      rec.status = res.new_status;
      toast(action === "approve" ? "Recommendation approved" : "Recommendation rejected");
    }
  } catch (error) {
    toast(error.message, true);
  }
  renderQuality();
}

/* ------------------------------------------------------------------ documents */
async function openDocument(file) {
  $("#workspace").hidden = true;
  $("#docspace").hidden = false;
  $("#doc-name").textContent = file.original_filename;
  $("#doc-meta").textContent = "Reading the document…";
  $("#doc-text").innerHTML = '<div class="sk sk-line" style="width:70%"></div><div class="sk sk-line"></div><div class="sk sk-line" style="width:85%"></div>';
  $("#doc-text").classList.add("card");
  $("#doc-thread").innerHTML = state.status?.llm_configured && state.status?.embeddings_configured
    ? '<div class="chat-empty"><b>Ask about this document</b>Index it first, then ask. Answers cite the passages they came from.</div>'
    : '<div class="card notice">Document questions use embeddings and an LLM. Add <code>OPENROUTER_API_KEY</code> and <code>EMBEDDING_API_KEY</code> to your <code>.env</code> file and restart the server. You can still read the extracted text here.</div>';
  try {
    const doc = await api(`/documents/extract/${file.file_id}`, { method: "POST" });
    $("#doc-meta").textContent = `${doc.file_type.toUpperCase()}${doc.page_count ? `, ${doc.page_count} pages` : ""}, ${doc.character_count.toLocaleString("en")} characters`;
    $("#doc-text").textContent = doc.extracted_text;
  } catch (error) {
    $("#doc-meta").textContent = error.message;
    $("#doc-text").textContent = "";
  }
}
async function indexDocument() {
  const button = $("#doc-index");
  button.disabled = true; button.textContent = "Indexing…";
  try {
    const res = await api(`/documents/index/${state.current.file_id}`, { method: "POST" });
    toast(`Indexed ${res.chunk_count} passages.`);
  } catch (error) { toast(error.message, true); }
  button.disabled = false; button.textContent = "Index for questions";
}
async function askDocument(question) {
  question = question.trim();
  if (!question) return;
  $("#doc-input").value = "";
  $(".chat-empty", $("#doc-thread"))?.remove();
  const turn = document.createElement("div");
  turn.className = "turn reveal";
  turn.innerHTML = `<div class="bubble">${esc(question)}</div><div class="answer"><div class="avatar">${ICON.spark}</div><div class="answer-body card"><div class="typing"><i></i><i></i><i></i></div></div></div>`;
  $("#doc-thread").prepend(turn);
  const answer = $(".answer", turn), box = $(".answer-body", turn);
  try {
    const res = await api("/documents/ask", { json: { file_id: state.current.file_id, question } });
    box.innerHTML = `<p>${esc(res.answer)}</p>${res.citations.length ? `<p class="cites">Sources: passages ${res.citations.map((c) => c.chunk_index + 1).join(", ")}</p>` : ""}`;
  } catch (error) {
    answer.classList.add("error");
    $(".avatar", answer).innerHTML = ICON.alert;
    const hint = /configur/i.test(error.message) ? "" : " If you have not indexed this document yet, use “Index for questions” first.";
    box.innerHTML = `<p>${esc(error.message)}${hint}</p>`;
  }
}

/* ------------------------------------------------------------------ hero demo */
const DEMO_QUESTION = "Top 5 products by revenue";
let demoTimer = null;
function playDemo() {
  const typed = $("#demo-typed"), answer = $("#demo-a");
  if (!typed) return;
  clearTimeout(demoTimer);
  if (reducedMotion) { typed.textContent = DEMO_QUESTION; answer.classList.add("show"); return; }
  typed.textContent = ""; answer.classList.remove("show");
  let i = 0;
  const step = () => {
    typed.textContent = DEMO_QUESTION.slice(0, ++i);
    if (i < DEMO_QUESTION.length) demoTimer = setTimeout(step, 45 + Math.random() * 40);
    else demoTimer = setTimeout(() => answer.classList.add("show"), 380);
  };
  demoTimer = setTimeout(step, 500);
}

/* ------------------------------------------------------------------ tabs + wiring */
function selectTab(name) {
  $$("#tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$("#workspace .panel").forEach((p) => { p.hidden = p.dataset.panel !== name; });
  if (name === "quality") loadQuality();
  if (name === "ask") $("#ask-input").focus();
}
async function loadSample() {
  try {
    const response = await fetch(API + "/sample-data/retail_sales_2025.csv");
    if (!response.ok) throw new Error();
    const blob = await response.blob();
    upload(new File([blob], "retail_sales_2025.csv", { type: "text/csv" }));
  } catch { toast("The sample dataset could not be loaded.", true); }
}

function wire() {
  $("#theme-toggle").addEventListener("click", () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  $("#crumb-home").addEventListener("click", goHome);
  $("#home-link").addEventListener("click", (e) => { e.preventDefault(); goHome(); });
  $("#tabs").addEventListener("click", (e) => { const b = e.target.closest("button[data-tab]"); if (b) selectTab(b.dataset.tab); });
  $("#tabs").addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const tabs = $$("#tabs button");
    const i = tabs.findIndex((b) => b.getAttribute("aria-selected") === "true");
    const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    selectTab(next.dataset.tab); next.focus();
  });
  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName);
    if (e.key === "/" && !typing && !$("#workspace").hidden) { e.preventDefault(); selectTab("ask"); }
  });
  $("#file-input").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
  $("#file-search").addEventListener("input", renderFiles);
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });
  $("#load-sample").addEventListener("click", loadSample);
  $("#hero-upload").addEventListener("click", () => $("#file-input").click());
  $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask($("#ask-input").value); });
  $("#doc-form").addEventListener("submit", (e) => { e.preventDefault(); askDocument($("#doc-input").value); });
  $("#doc-index").addEventListener("click", indexDocument);
  $("#viewer-close").addEventListener("click", () => $("#viewer").close());
  $("#viewer").addEventListener("click", (e) => { if (e.target === e.currentTarget) e.currentTarget.close(); });
  $("#viewer").addEventListener("close", () => { viewerChart?.destroy(); viewerChart = null; });
  $("#viewer-download").addEventListener("click", () => downloadChart(viewerChart, $("#viewer-title").textContent));
  $("#builder").addEventListener("submit", async (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(e.target));
    if (!data.y) { data.y = null; data.aggregation = "count"; }
    const out = $("#builder-out");
    out.className = "figure big card";
    out.innerHTML = '<div class="sk" style="height:380px"></div>';
    try {
      const spec = await api("/charts", { json: { file_id: state.current.file_id, ...data } });
      const fig = figure(spec, true);
      fig.id = "builder-out";
      out.replaceWith(fig);
    } catch (error) {
      out.className = "figure big empty";
      out.innerHTML = `<p style="color:var(--bad)">${esc(error.message)}</p>`;
    }
  });
}

(async function init() {
  wire();
  $("#theme-toggle").setAttribute("aria-label", document.documentElement.dataset.theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
  playDemo();
  await loadStatus();
  const wanted = location.hash.slice(1);
  await loadFiles();
  if (wanted && state.files.some((f) => f.file_id === wanted)) openFile(wanted);
})();
