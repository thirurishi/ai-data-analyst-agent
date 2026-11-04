"use strict";

const API = window.location.port === "5173" ? "http://127.0.0.1:8000" : "";
const $ = (selector, root = document) => root.querySelector(selector);
const state = { files: [], current: null, insights: null, status: null, charts: new Map(), recs: [] };

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

/* ------------------------------------------------------------------ charts */
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function palette() {
  return [css("--accent"), "#16A3A3", "#E0912F", "#D2527F", "#7C5CD6", "#4F9D3A", "#8A93A8", "#C2372B"];
}
// Chart.js loads on its own so a slow or blocked CDN never stops the rest of the app.
const chartReady = new Promise((resolve) => {
  const script = document.createElement("script");
  script.src = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js";
  script.onload = () => resolve(true);
  script.onerror = () => resolve(false);
  setTimeout(() => resolve(Boolean(window.Chart)), 8000);
  document.head.append(script);
});
async function drawChart(canvas, spec) {
  if (!(await chartReady) || !window.Chart) {
    canvas.parentElement.innerHTML = '<p class="hint">Charts need an internet connection to load Chart.js. The numbers above are still accurate.</p>';
    return;
  }
  const key = canvas.dataset.key || Math.random().toString(36).slice(2);
  canvas.dataset.key = key;
  state.charts.get(key)?.destroy();
  const ink2 = css("--ink-2"), line = css("--line"), accent = css("--accent");
  const round = ["pie", "doughnut"].includes(spec.type);
  const type = spec.type === "histogram" ? "bar" : spec.type;
  const data = spec.type === "scatter"
    ? { datasets: [{ label: spec.datasets[0].label, data: spec.points, backgroundColor: accent + "99", pointRadius: 3 }] }
    : {
        labels: spec.labels,
        datasets: [{
          label: spec.datasets[0].label,
          data: spec.datasets[0].data,
          backgroundColor: round ? palette() : accent,
          borderColor: round ? css("--surface") : accent,
          borderWidth: round ? 2 : spec.type === "line" ? 2.5 : 0,
          borderRadius: type === "bar" ? 4 : 0,
          barPercentage: spec.type === "histogram" ? 1 : 0.8,
          categoryPercentage: spec.type === "histogram" ? 0.96 : 0.8,
          tension: 0.3,
          pointRadius: spec.type === "line" ? 2.5 : 0,
          fill: spec.type === "line" ? { target: "origin", above: accent + "18" } : false,
        }],
      };
  const axis = (title) => ({
    title: { display: Boolean(title), text: title, color: ink2, font: { family: css("--font"), size: 12 } },
    ticks: { color: ink2, font: { family: css("--font"), size: 11 }, maxRotation: 40, autoSkip: true, callback(value) {
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
      responsive: true, maintainAspectRatio: false, animation: { duration: 450 },
      plugins: {
        legend: { display: round, position: "right", labels: { color: ink2, boxWidth: 10, font: { family: css("--font") } } },
        tooltip: { callbacks: { label: (ctx) => spec.type === "scatter" ? `${fmt(ctx.raw.x)}, ${fmt(ctx.raw.y)}` : `${ctx.dataset.label}: ${fmt(ctx.raw)}` } },
      },
      scales: round ? {} : {
        x: { ...axis(spec.x_label), ...(spec.type === "scatter" ? { type: "linear", ticks: { color: ink2, callback: (v) => fmt(v) } } : {}) },
        y: { ...axis(spec.y_label), beginAtZero: spec.type !== "scatter", ticks: { color: ink2, callback: (v) => fmt(v) } },
      },
    },
  });
  state.charts.set(key, chart);
}
function figure(spec, big = false) {
  const el = document.createElement("div");
  el.className = "figure" + (big ? " big" : "");
  el.innerHTML = `<h3>${esc(spec.title)}</h3><div class="canvas"><canvas role="img" aria-label="${esc(spec.title)}"></canvas></div>${spec.insight ? `<p>${esc(spec.insight)}</p>` : ""}`;
  requestAnimationFrame(() => drawChart($("canvas", el), spec));
  return el;
}
function resetCharts() { state.charts.forEach((c) => c.destroy()); state.charts.clear(); }

/* ------------------------------------------------------------------ status + files */
async function loadStatus() {
  try {
    state.status = await api("/status");
  } catch { state.status = null; }
  const el = $("#engine");
  const s = state.status;
  if (!s) { el.innerHTML = "<b>API offline</b><p>Start the server with uvicorn.</p>"; return; }
  el.classList.toggle("on", s.llm_configured);
  el.innerHTML = s.llm_configured
    ? `<b>AI agent connected</b><p>Questions are planned by ${esc(s.llm_model)} and answered from verified calculations.</p>`
    : `<b>Built-in analyst</b><p>Questions are answered by the rule-based planner. Add <code>OPENROUTER_API_KEY</code> to <code>.env</code> to use an LLM.</p>`;
}
async function loadFiles(selectId) {
  try {
    state.files = (await api("/files")).files;
  } catch (error) { toast(error.message, true); return; }
  const list = $("#files");
  list.innerHTML = state.files.length ? "" : '<li class="files-empty">No files yet.</li>';
  for (const file of state.files) {
    const li = document.createElement("li");
    li.innerHTML = `<button type="button" data-id="${file.file_id}"><span class="ext">${esc(file.extension.toUpperCase())}</span><span><span class="fname">${esc(file.original_filename)}</span><span class="fsize">${bytes(file.file_size)}</span></span></button>`;
    $("button", li).title = file.original_filename;
    $("button", li).addEventListener("click", () => openFile(file.file_id));
    list.append(li);
  }
  markCurrent();
  if (selectId) openFile(selectId);
}
function markCurrent() {
  document.querySelectorAll(".files button").forEach((b) => b.setAttribute("aria-current", String(b.dataset.id === state.current?.file_id)));
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

/* ------------------------------------------------------------------ open a file */
async function openFile(fileId) {
  const file = state.files.find((f) => f.file_id === fileId);
  if (!file) return;
  state.current = file;
  markCurrent();
  history.replaceState(null, "", `#${fileId}`);
  resetCharts();
  $("#welcome").hidden = true;
  if (file.kind === "document") return openDocument(file);
  $("#docspace").hidden = true;
  $("#workspace").hidden = false;
  $("#ws-name").textContent = file.original_filename;
  $("#ws-meta").textContent = "Analysing…";
  $("#panel-overview").innerHTML = '<p class="pending">Profiling the dataset</p>';
  $("#export-pdf").href = `${API}/report?file_id=${fileId}&format=pdf`;
  $("#export-md").href = `${API}/report?file_id=${fileId}&format=md`;
  $("#thread").innerHTML = "";
  selectTab("overview");
  try {
    state.insights = await api(`/insights?file_id=${fileId}`);
  } catch (error) {
    $("#panel-overview").innerHTML = `<div class="answer error"><p>${esc(error.message)}</p></div>`;
    $("#ws-meta").textContent = "";
    return;
  }
  const ins = state.insights;
  $("#ws-meta").textContent = `${ins.rows.toLocaleString("en")} rows, ${ins.columns} columns, ${bytes(file.file_size)}`;
  renderOverview(ins);
  renderAskChips(ins);
  renderBuilder(ins);
  renderData(ins);
  state.recs = [];
  $("#panel-quality").innerHTML = "";
}

function renderOverview(ins) {
  const panel = $("#panel-overview");
  const h = ins.health;
  const ringColor = h.score >= 80 ? "var(--good)" : h.score >= 65 ? "var(--warn)" : "var(--bad)";
  panel.innerHTML = `
    <div class="kpis reveal">${ins.kpis.map((k) => `<div class="kpi"><span>${esc(k.label)}</span><strong>${esc(k.display)}</strong></div>`).join("")}</div>
    <div class="overview-grid">
      <section>
        <h2 class="section-title">What stands out</h2>
        <ul class="findings">${ins.findings.length ? ins.findings.map((f) => `<li class="${f.severity}"><b>${esc(f.title)}</b><span>${esc(f.detail)}</span></li>`).join("") : "<li><b>Nothing unusual</b><span>The automatic scan found no notable patterns.</span></li>"}</ul>
      </section>
      <section>
        <h2 class="section-title">Data health</h2>
        <div class="health">
          <div class="health-top">
            <div class="ring" style="--p:${h.score};--ring-color:${ringColor}"><strong>${h.score}</strong></div>
            <div><b>${esc(h.label)}</b><p>${h.issue_count} issue${h.issue_count === 1 ? "" : "s"} found, ${fmt(h.missing_cells)} missing cells, ${fmt(h.duplicate_rows)} duplicate rows.</p></div>
          </div>
          <div class="dims">${Object.entries(h.dimensions).map(([name, value]) => `<div class="dim"><span>${name[0].toUpperCase() + name.slice(1)}</span><i style="--w:${value}%"></i><em>${value}%</em></div>`).join("")}</div>
          ${h.issue_count ? '<p style="margin:16px 0 0"><button class="btn small" type="button" id="go-quality">Review issues</button></p>' : ""}
        </div>
      </section>
    </div>
    <div class="charts" id="overview-charts"></div>`;
  const grid = $("#overview-charts", panel);
  ins.charts.forEach((spec) => grid.append(figure(spec)));
  $("#go-quality", panel)?.addEventListener("click", () => selectTab("quality"));
}

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

async function ask(question) {
  question = question.trim();
  if (!question || !state.current) return;
  $("#ask-input").value = "";
  const turn = document.createElement("div");
  turn.className = "turn reveal";
  turn.innerHTML = `<div class="q">${esc(question)}</div><div class="answer"><p class="pending">Working it out</p></div>`;
  $("#thread").prepend(turn);
  const box = $(".answer", turn);
  try {
    const res = await api("/assistant/ask", { json: { file_id: state.current.file_id, question } });
    const args = Object.entries(res.tool.arguments).map(([k, v]) => `${k}: ${v}`).join(", ");
    box.innerHTML = `
      <p>${esc(res.answer)}</p>
      ${res.note ? `<p class="note">${esc(res.note)}</p>` : ""}
      ${res.chart ? '<div class="canvas"><canvas></canvas></div>' : ""}
      <div class="meta"><span class="tag ${res.mode}">${res.mode === "llm" ? "AI agent" : "Built-in analyst"}</span><span class="tag">${esc(res.tool.name)}</span>${args ? `<span class="tag">${esc(args)}</span>` : ""}</div>
      <details><summary>Show the calculation result</summary><pre>${esc(JSON.stringify(res.result, null, 2))}</pre></details>`;
    if (res.chart) drawChart($("canvas", box), res.chart);
  } catch (error) {
    box.classList.add("error");
    box.innerHTML = `<p>${esc(error.message)}</p>`;
  }
}

function renderBuilder(ins) {
  const cols = ins.column_summary;
  const xs = cols.filter((c) => c.role !== "identifier");
  const nums = cols.filter((c) => c.role === "numeric");
  $("#builder-x").innerHTML = xs.map((c) => `<option value="${esc(c.name)}" ${c.name === (ins.primary_category || ins.primary_date) ? "selected" : ""}>${esc(c.name)}</option>`).join("");
  $("#builder-y").innerHTML = '<option value="">(row count)</option>' + nums.map((c) => `<option value="${esc(c.name)}" ${c.name === ins.primary_metric ? "selected" : ""}>${esc(c.name)}</option>`).join("");
  $("#builder-out").innerHTML = '<p class="hint">Pick columns and draw a chart.</p>';
}

function renderData(ins) {
  const roleRows = ins.column_summary.map((c) => `
    <tr><td><b>${esc(c.name)}</b></td><td><span class="role ${c.role}">${c.role}</span></td><td>${esc(c.dtype)}</td>
    <td><span class="miss" style="--w:${Math.min(100, c.missing_percentage)}%"></span>${c.missing_percentage}%</td>
    <td class="num">${fmt(c.unique)}</td><td>${c.sample.map(esc).join(", ")}</td></tr>`).join("");
  const preview = ins.preview;
  $("#panel-data").innerHTML = `
    <h2 class="section-title">Columns</h2>
    <div class="table-wrap"><table><thead><tr><th>Column</th><th>Role</th><th>Type</th><th>Missing</th><th class="num">Unique</th><th>Examples</th></tr></thead><tbody>${roleRows}</tbody></table></div>
    <h2 class="section-title subhead">First rows</h2>
    <div class="table-wrap"><table><thead><tr>${preview.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
    <tbody>${preview.rows.map((r) => `<tr>${r.map((v) => `<td class="${typeof v === "number" ? "num" : ""}">${v === null ? '<span class="hint">empty</span>' : esc(typeof v === "number" ? fmt(v) : v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    <p style="margin-top:16px"><a class="btn" href="${API}/files/${state.current.file_id}/download" download>Download CSV</a></p>`;
}

/* ------------------------------------------------------------------ quality */
async function loadQuality() {
  const panel = $("#panel-quality");
  if (state.recs.length || !state.current) return;
  panel.innerHTML = '<p class="pending">Checking data quality</p>';
  try {
    const res = await api(`/data-quality/recommendations?file_id=${state.current.file_id}`);
    state.recs = res.recommendations;
  } catch (error) {
    panel.innerHTML = `<div class="answer error"><p>${esc(error.message)}</p></div>`;
    return;
  }
  renderQuality();
}
const AUTO_FIX = new Set(["duplicate_rows", "empty_column", "constant_column", "categorical_inconsistency"]);
function renderQuality() {
  const panel = $("#panel-quality");
  if (!state.recs.length) {
    panel.innerHTML = '<div class="answer"><p>No quality issues were found. This dataset is ready for analysis.</p></div>';
    return;
  }
  panel.innerHTML = `<p class="quality-intro">Nothing is changed without your approval. Approve a recommendation, then apply it to create a cleaned copy of the file. Your original upload stays untouched.</p><div class="recs"></div>`;
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
    row.querySelectorAll("[data-act]").forEach((b) => b.addEventListener("click", () => qualityAction(rec, b.dataset.act, b)));
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
        selectTab("quality");
        return;
      }
      toast(res.message);
    } else {
      const res = await api(`/data-quality/recommendations/${rec.recommendation_id}/${action}`, { method: "POST" });
      rec.status = res.new_status;
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
  $("#doc-text").textContent = "";
  $("#doc-thread").innerHTML = state.status?.llm_configured && state.status?.embeddings_configured
    ? '<p class="hint">Index the document, then ask a question. Answers cite the passages they came from.</p>'
    : '<div class="answer"><p>Document questions use embeddings and an LLM. Add <code>OPENROUTER_API_KEY</code> and <code>EMBEDDING_API_KEY</code> to your <code>.env</code> file and restart the server. You can still read the extracted text here.</p></div>';
  try {
    const doc = await api(`/documents/extract/${file.file_id}`, { method: "POST" });
    $("#doc-meta").textContent = `${doc.file_type.toUpperCase()}${doc.page_count ? `, ${doc.page_count} pages` : ""}, ${doc.character_count.toLocaleString("en")} characters`;
    $("#doc-text").textContent = doc.extracted_text;
  } catch (error) {
    $("#doc-meta").textContent = error.message;
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
  const turn = document.createElement("div");
  turn.className = "turn reveal";
  turn.innerHTML = `<div class="q">${esc(question)}</div><div class="answer"><p class="pending">Searching the document</p></div>`;
  $("#doc-thread").prepend(turn);
  const box = $(".answer", turn);
  try {
    const res = await api("/documents/ask", { json: { file_id: state.current.file_id, question } });
    box.innerHTML = `<p>${esc(res.answer)}</p>${res.citations.length ? `<p class="cites">Sources: passages ${res.citations.map((c) => c.chunk_index + 1).join(", ")}</p>` : ""}`;
  } catch (error) {
    box.classList.add("error");
    const hint = /configur/i.test(error.message) ? "" : " If you have not indexed this document yet, use “Index for questions” first.";
    box.innerHTML = `<p>${esc(error.message)}${hint}</p>`;
  }
}

/* ------------------------------------------------------------------ tabs + wiring */
function selectTab(name) {
  document.querySelectorAll("#tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.querySelectorAll("#workspace .panel").forEach((p) => { p.hidden = p.dataset.panel !== name; });
  if (name === "quality") loadQuality();
  if (name === "ask") $("#ask-input").focus();
}

function wire() {
  $("#tabs").addEventListener("click", (e) => { const b = e.target.closest("button[data-tab]"); if (b) selectTab(b.dataset.tab); });
  $("#tabs").addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const tabs = [...document.querySelectorAll("#tabs button")];
    const i = tabs.findIndex((b) => b.getAttribute("aria-selected") === "true");
    const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    selectTab(next.dataset.tab); next.focus();
  });
  $("#file-input").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });
  $("#load-sample").addEventListener("click", async () => {
    try {
      const response = await fetch(API + "/sample-data/retail_sales_2025.csv");
      if (!response.ok) throw new Error();
      const blob = await response.blob();
      upload(new File([blob], "retail_sales_2025.csv", { type: "text/csv" }));
    } catch { toast("The sample dataset could not be loaded.", true); }
  });
  $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask($("#ask-input").value); });
  $("#doc-form").addEventListener("submit", (e) => { e.preventDefault(); askDocument($("#doc-input").value); });
  $("#doc-index").addEventListener("click", indexDocument);
  $("#builder").addEventListener("submit", async (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(e.target));
    if (!data.y) { data.y = null; data.aggregation = "count"; }
    const out = $("#builder-out");
    out.innerHTML = '<p class="pending">Drawing</p>';
    try {
      const spec = await api("/charts", { json: { file_id: state.current.file_id, ...data } });
      const fig = figure(spec, true);
      out.replaceWith(fig); fig.id = "builder-out";
    } catch (error) {
      out.innerHTML = `<p style="color:var(--bad)">${esc(error.message)}</p>`;
    }
  });
}

(async function init() {
  wire();
  await loadStatus();
  const wanted = location.hash.slice(1);
  await loadFiles();
  if (wanted && state.files.some((f) => f.file_id === wanted)) openFile(wanted);
})();
