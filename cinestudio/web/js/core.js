// cinestudio web app - Shared helpers and app frame: DOM helpers, icons, the API client, toasts, polling, deletes, tabs, health, meters, the queue, LoRA lists and the paged grids.
// Plain scripts (no modules), loaded in order by index.html; they share their top-level names.
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ICON = {
  trash: `<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><path fill="currentColor"
    d="M9 3h6l1 2h4v2H4V5h4l1-2zm-3 6h12l-1 12H7L6 9zm4 2v8h2v-8h-2zm4 0v8h2v-8h-2z"/></svg>`,
  redo: `<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor"
    d="M17.65 6.35A8 8 0 1 0 19.73 14h-2.08A6 6 0 1 1 12 6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>`,
  undo: `<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor"
    d="M12.5 8c-2.6 0-5 1-6.9 2.6L2 7v9h9l-3.6-3.6c1.4-1.2 3.2-1.9 5.1-1.9 3.5 0 6.5 2.3 7.6 5.5l2.4-.8C21 11 17.1 8 12.5 8z"/></svg>`,
  close: `<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path fill="currentColor"
    d="M6.4 5 5 6.4 10.6 12 5 17.6 6.4 19l5.6-5.6 5.6 5.6 1.4-1.4-5.6-5.6L19 6.4 17.6 5 12 10.6z"/></svg>`,
  cube: `<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="none" stroke="currentColor"
    stroke-width="2" stroke-linejoin="round" d="M12 2.8 20 7.2v9.6L12 21.2 4 16.8V7.2L12 2.8zM4 7.2l8 4.4 8-4.4M12 11.6v9.6"/></svg>`,
  music: `<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor"
    d="M12 3v10.55A4 4 0 1 0 14 17V7h4V3h-6z"/></svg>`,
  stop: `<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/></svg>`,
  person: `<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor"
    d="M12 12a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9zm0 2c-4.4 0-8 2.2-8 5v2h16v-2c0-2.8-3.6-5-8-5z"/></svg>`,
  people: `<svg viewBox="0 0 24 24" width="13" height="13" aria-hidden="true"><path fill="currentColor"
    d="M9 11a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7zm7.5 0a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM9 13c-3.3 0-6 1.7-6 3.8V19h12v-2.2C15 14.7 12.3 13 9 13zm7.5 0c-.6 0-1.2.1-1.7.2 1.3.9 2.2 2.1 2.2 3.6V19h4v-2.2c0-2.1-2-3.8-4.5-3.8z"/></svg>`,
  film: `<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor"
    d="M4 4h16v16H4V4zm2 2v2h2V6H6zm10 0v2h2V6h-2zM6 10v4h12v-4H6zm0 6v2h2v-2H6zm10 0v2h2v-2h-2zM10 6v2h4V6h-4zm0 10v2h4v-2h-4z"/></svg>`,
  image: `<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor"
    d="M4 5h16v14H4V5zm2 2v8.6l3.5-3.5 2.5 2.5 3.5-3.5L18 13.6V7H6zm2.5 1.5a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3z"/></svg>`,
};
// how many characters a template has (its Mannequin* layers, /api/shots and the template cards' "mannequins"):
// "👥 2 characters"; a template whose layers were never read says so (one is assumed)
const MAX_VIDEO_CHARACTERS = 4;                        // runs take --character once per slot (runner.MAX_CHARACTERS)
function slotChip(t) {
  const slots = (t && t.mannequins) || [];
  if (!slots.length) return "";
  const n = slots.length, unscanned = t.slots_scanned === false;
  const sides = slots.map((s) => `${s.label}${s.side && n > 1 ? ` (${s.side})` : ""}: ${s.collection || "the stand-in"}`).join(" · ");
  const tip = unscanned ? "The template's layers were not read yet: one character (Mannequin) is assumed. Templates tab > Layers reads them."
    : `${sides}${n > MAX_VIDEO_CHARACTERS ? ` · videos support up to ${MAX_VIDEO_CHARACTERS} characters for now` : ""}`;
  return `<span class="chip slot-chip${n > MAX_VIDEO_CHARACTERS ? " fig-warn" : ""}${unscanned ? " unscanned" : ""}" title="${esc(tip)}">${ICON.people}${n} character${n === 1 ? "" : "s"}${unscanned ? " · not scanned" : ""}</span>`;
}
const trashButton = (kind, id) =>
  `<button class="icon-del" data-del-kind="${kind}" data-del-id="${esc(id)}" title="Delete" aria-label="Delete ${esc(id)}">${ICON.trash}</button>`;

const state = { media: {}, q: {}, imgOpts: null, chars: [], char: null, view: null, sel: null, loras: [], runs: [], run: null, polls: {}, queue: { items: [] } };

async function api(path, opts = {}) {
  // every request carries the app header: the server refuses changes without it (other sites cannot add it)
  const r = await fetch(path, { ...opts, headers: { "X-Cinestudio": "1", ...(opts.headers || {}) } });
  let body;
  try { body = await r.json(); } catch { throw new Error(`${r.status} ${r.statusText}`); }
  if (!body.ok) throw new Error(body.error || "request failed");
  return body.data;
}
const postJSON = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data || {}) });

function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (err ? " err" : "");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.add("hidden"), err ? 9000 : 3500);
}
const fail = (e) => toast(e.message || String(e), true);

const fmtTime = (s) => { s = Math.max(0, Math.round(s)); const m = Math.floor(s / 60);
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : m ? `${m} min` : `${s} s`; };
function progressBar(frac, elapsed) {
  const pct = Math.max(0, Math.min(100, Math.round(frac * 100)));
  const eta = frac > 0.02 && elapsed ? elapsed / frac - elapsed : null;
  return `<div class="pbar"><div class="progress"><div style="width:${pct}%"></div></div></div>
    <span class="ptext" title="${fmtTime(elapsed || 0)} so far">${pct}%${eta !== null ? ` · ${fmtTime(eta)} left` : ""}</span>`;
}

function poll(key, fn, ms = 3000) { clearInterval(state.polls[key]); state.polls[key] = setInterval(fn, ms); }
function stopPoll(key) { clearInterval(state.polls[key]); delete state.polls[key]; }

async function deleteItem(kind, id) {
  const what = kind === "character" ? `the character "${id}"` : kind === "prop" ? `the prop "${id}" (runs keep their own copy)` : kind === "song" ? `the song "${id}"`
    : kind === "template" ? `the template "${id}"` : kind === "background" ? `the background "${id}" (runs keep their own copy)`
    : "this video";
  if (!confirm(`Delete ${what}? This cannot be undone.`)) return false;
  try {
    const base = { character: "/api/characters", prop: "/api/props", song: "/api/songs", template: "/api/templates",
                   background: "/api/backgrounds" }[kind] || "/api/runs";
    await postJSON(`${base}/${encodeURIComponent(id)}/delete`);
    toast("Deleted.");
    return true;
  } catch (e) { fail(e); return false; }
}
function bindDeletes(root, after) {
  $$(".icon-del", root).forEach((b) => b.addEventListener("click", async (ev) => {
    ev.stopPropagation();
    if (await deleteItem(b.dataset.delKind, b.dataset.delId)) after(b.dataset.delId);
  }));
}

// ------------------------------------------------------------------ tabs, health, meters
function showTab(name) {
  $$(".rail .tab[data-tab]").forEach((x) => x.classList.toggle("active", x.dataset.tab === name));
  $$(".pane").forEach((p) => p.classList.toggle("hidden", p.id !== "tab-" + name));
  if (location.hash.split("/")[0] !== "#" + name) history.replaceState(null, "", "#" + name);
  if (name === "video") loadVideoTab(); else stopPoll("runs");
  if (name === "characters") poll("chars", () => { if (!document.hidden) refreshCharStates(); }, 5000);
  else stopPoll("chars");
  if (name === "music") { loadSongs(); poll("songs", () => { if (!document.hidden) loadSongs(); }, 3000); }
  else stopPoll("songs");
  if (name === "templates") { loadTemplates(); poll("templates", () => { if (!document.hidden) loadTemplates(); }, 3000); }
  else stopPoll("templates");
  if (name === "backgrounds") { loadBackgroundsTab(); poll("backgrounds", () => { if (!document.hidden) loadBackgroundsTab(); }, 4000); }
  else stopPoll("backgrounds");
  if (name === "styles") { loadStylesTab(); poll("styles", () => { if (!document.hidden) loadStylesTab(); }, 10000); }
  else stopPoll("styles");
  if (name === "settings") { loadSettings(); poll("settings", () => { if (!document.hidden) loadSettings(true); }, 10000); }
  else stopPoll("settings");
  if (name === "workflows") loadWorkflowsTab();                // the advanced mode's graphs (workflows.js)
}
$$(".rail .tab[data-tab]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// a link from one tab's card to another tab's card (a template <-> its backplate): the tab opens on the card's page,
// the card scrolls into view and flashes. The tab's loader calls focusPage before it slices its page, focusFlash after.
const pendingFocus = {};
function focusCard(tab, name) { pendingFocus[tab] = name; showTab(tab); }
function focusPage(tab, pager, names) {
  const want = pendingFocus[tab];
  if (!want) return;
  const i = names.indexOf(want);
  if (i < 0) { delete pendingFocus[tab]; return; }
  if (pager.mode === "pages") pager.page = Math.floor(i / pager.per()) + 1;
}
function focusFlash(tab, grid, attr) {
  const want = pendingFocus[tab];
  if (!want) return;
  const card = $$(`[data-${attr}]`, grid).find((c) => c.dataset[attr] === want);
  if (!card) return;
  delete pendingFocus[tab];
  card.scrollIntoView({ block: "center", behavior: "smooth" });
  card.classList.add("focus-flash");
  setTimeout(() => card.classList.remove("focus-flash"), 2400);
}

async function loadHealth(report = false) {
  const h = $("#health");
  try {
    const d = await api("/api/status");
    const bad = Object.entries(d.checks).filter(([, v]) => !v.ok);
    h.className = "health " + (bad.length ? "bad" : "ok");
    h.title = bad.length ? bad.map(([k, v]) => `${k}: ${v.error || (v.missing || []).join(", ")}`).join("\n")
      : Object.keys(d.checks).join(", ") + ": all OK";
    if (report) toast(h.title, bad.length > 0);
  } catch (e) { h.className = "health bad"; h.title = e.message; if (report) fail(e); }
}
$("#health").addEventListener("click", () => loadHealth(true));

const GB = (b) => (b / 1024 ** 3).toFixed(1);
// a meter turns orange or red only when its value is high
function meter(label, frac, value, title, warn = 0.85, crit = 0.95) {
  const f = Math.max(0, Math.min(1, frac));
  const level = f >= crit ? "crit" : f >= warn ? "warn" : "";
  return `<span class="sbm ${label.toLowerCase()} ${level}" title="${esc(title)}"><span>${label}</span><span class="bar"><i style="width:${Math.round(f * 100)}%"></i></span>
    <span class="v">${esc(value)}</span></span>`;
}
async function loadMeters() {
  if (document.hidden) return;
  let s;
  try { s = await api("/api/system"); } catch { return; }
  const out = [];
  if (s.cpu !== null) out.push(meter("CPU", s.cpu / 100, `${Math.round(s.cpu)}%`, `CPU ${Math.round(s.cpu)}%`));
  if (s.ram_total) {
    const f = s.ram_used / s.ram_total;
    out.push(meter("RAM", f, `${Math.round(f * 100)}%`, `RAM ${GB(s.ram_used)} of ${GB(s.ram_total)} GB`, 0.9, 0.97));
  }
  const g = s.gpu;
  if (g) {
    out.push(meter("GPU", g.util / 100, `${Math.round(g.util)}%`, `${g.name}: ${Math.round(g.util)}%`, 2, 2));   // a busy GPU is normal
    const f = g.vram_used / g.vram_total;
    out.push(meter("VRAM", f, `${Math.round(f * 100)}%`, `VRAM ${GB(g.vram_used)} of ${GB(g.vram_total)} GB`, 0.92, 0.98));
    if (g.temp !== null) out.push(meter("TEMP", g.temp / 100, `${Math.round(g.temp)}°`, `GPU ${Math.round(g.temp)} °C`, 0.8, 0.87));
  }
  $("#meters").innerHTML = out.join("");
}

// ------------------------------------------------------------------ queue (one line for all GPU work)
async function loadQueue() {
  try { state.queue = await api("/api/queue"); } catch { return; }
  const n = state.queue.items.length, badge = $("#queue-count");
  const run = state.queue.items.find((x) => x.status === "running") || state.queue.items[0];
  const waiting = n - (run ? 1 : 0);
  badge.textContent = `+${waiting}`; badge.classList.toggle("hidden", waiting < 1);
  $("#queue-text").textContent = !run ? "Idle"
    : run.status === "running" ? `${run.label} · ${Math.round((run.frac || 0) * 100)}%` : `${run.label} · starting`;
  $("#btn-queue").classList.toggle("busy", !!run);
  $("#btn-queue").title = n ? `Queue: ${n} (click to see all)` : "Queue: empty";
  if (!$("#queue-pop").classList.contains("hidden")) renderQueue();
}
function renderQueue() {
  const q = state.queue, pop = $("#queue-pop");
  const others = q.other_comfy_jobs ? `<span class="muted" title="ComfyUI jobs of other programs">+${q.other_comfy_jobs} other</span>` : "";
  pop.innerHTML = `<div class="qhead">Queue${others}</div>` + (q.items.length ? q.items.map((it) => {
    const running = it.status === "running";
    const since = it.step_started || it.started;
    const line = running ? `${it.text ? `<span>${esc(it.text)}</span>` : ""}${progressBar(it.frac || 0, since ? q.now - since : 0)}`
      : `<span>${it.position === 0 ? "starting" : it.position === 1 ? "next" : `#${it.position} in line`}</span>`;
    return `<div class="qitem ${running ? "running" : ""}" data-kind="${it.kind}" data-ref="${esc(it.ref)}">
      <span class="qicon">${it.kind === "video" ? ICON.film : it.kind === "song" ? ICON.music : it.kind === "template" ? ICON.cube : it.kind === "lab" && String(it.ref).startsWith("bg-") ? ICON.image : ICON.person}</span>
      <span class="qlabel" title="${esc(it.label)}">${esc(it.label)}</span>
      <button class="qstop" title="${running ? "Stop" : "Remove from the queue"}" aria-label="Stop">${ICON.stop}</button>
      <div class="qline">${line}</div></div>`;
  }).join("") : `<div class="qempty">Nothing in the queue</div>`);
  $$(".qitem", pop).forEach((row) => {
    const it = q.items.find((x) => x.kind === row.dataset.kind && x.ref === row.dataset.ref);
    row.addEventListener("click", () => {
      closeQueue();
      if (it.kind === "video") { showTab("video"); openRun(it.ref); }
      else if (it.kind === "song") showTab("music");
      else if (it.kind === "template") showTab("templates");
      else if (it.kind === "lab" && String(it.ref).startsWith("bg-")) showTab("backgrounds");
      else if (it.character) { showTab("characters"); setCastMode("characters"); openChar(it.character); }
      else if (it.prop) { showTab("characters"); setCastMode("props"); openProp(it.prop); }   // the Cast tab's Props
    });
    $(".qstop", row).addEventListener("click", async (ev) => {
      ev.stopPropagation();
      const msg = it.status === "running" ? `Stop "${it.label}"? The work of this step is lost.` : `Remove "${it.label}" from the queue?`;
      if (!confirm(msg)) return;
      try { await postJSON("/api/queue/cancel", { kind: it.kind, ref: it.ref }); toast("Stop requested."); loadQueue(); }
      catch (e) { fail(e); }
    });
  });
}
function closeQueue() { $("#queue-pop").classList.add("hidden"); $("#btn-queue").classList.remove("active"); }
$("#btn-queue").addEventListener("click", (ev) => {
  ev.stopPropagation();
  const pop = $("#queue-pop");
  if (pop.classList.contains("hidden")) { pop.classList.remove("hidden"); $("#btn-queue").classList.add("active"); renderQueue(); loadQueue(); }
  else closeQueue();
});
document.addEventListener("click", (ev) => { if (!ev.target.closest("#queue-pop")) closeQueue(); });
document.addEventListener("keydown", (ev) => {
  if (ev.key !== "Escape") return;
  closeQueue();
  if (state.sel && $("#player").classList.contains("hidden") && !(ev.target.closest && ev.target.closest("input, textarea"))) {
    state.sel = null; renderChar();
  }
});
const queued = (what) => { toast(`${what}: added to the queue`); loadQueue(); };
// a dialog box (class "overlay"): hide it; false when it was not open
function closeOverlay(id) {
  const box = $(id);
  if (!box || box.classList.contains("hidden")) return false;
  box.classList.add("hidden");
  return true;
}

// the LTX video style LoRAs: not the control LoRAs (IC-LoRA, Layout-To-Render) and not the image-model LoRAs (Qwen-Image-Edit:
// AnyPose, Multiple-Angles, Lightning; Qwen-Image: 360 diffusion), which cannot load into LTX. The LTX 2.5 distilled
// LoRA is an internal second-stage refinement component, not a selectable style.
const NOT_STYLE_LORA = /ic-lora|lightning|multiple-angles|anypose|minimax|qwen|360-diffusion|layout-to-render|(?:^|[\\/])ltx-2\.5-22b-distilled-lora-450-bf16\.safetensors$/i;
const styleLoras = () => (state.loras || []).filter((l) => !NOT_STYLE_LORA.test(l));

async function loadLoras() {
  try { state.loras = await api("/api/loras"); } catch { state.loras = []; }
  const style = styleLoras();
  $$(".lora-select").forEach((s) => {
    const cur = s.value;
    s.innerHTML = `<option value="">No style LoRA</option>` + style.map((l) => `<option>${esc(l)}</option>`).join("");
    if (cur) s.value = cur;            // default: no style LoRA (the 2.3 anime LoRA hurt LTX-2.5: softer, mask lost)
  });
}


// ------------------------------------------------------------------ paged grids (Music, Templates, Video)
// A page is FULL ROWS: rows x the grid's real column count, minus 1 for the "New" tile that starts every page, so no
// square hangs alone in a row - except on the last page. The column count comes from the CSS grid itself (it follows
// the window width); a resize keeps the first item of the page in view. "All" shows everything in one scrolling list.
const ROW_CHOICES = [3, 6, 12];
function gridCols(el) {
  const t = getComputedStyle(el).gridTemplateColumns;
  const n = t && t !== "none" ? t.split(" ").filter(Boolean).length : 0;
  return Math.max(1, n || 4);
}
function makePager(key, reload) {
  const p = { key, mode: "pages", rows: 6, page: 1, cols: 4, reload };
  try {
    const s = JSON.parse(localStorage.getItem(`grid.${key}`) || "{}");
    if (s.mode === "all" || s.mode === "pages") p.mode = s.mode;
    if (ROW_CHOICES.includes(s.rows)) p.rows = s.rows;
  } catch { /* defaults */ }
  p.save = () => { try { localStorage.setItem(`grid.${key}`, JSON.stringify({ mode: p.mode, rows: p.rows })); } catch { /* not kept */ } };
  p.per = () => (p.mode === "all" ? 0 : Math.max(1, p.rows * p.cols - 1));
  p.offset = () => (p.mode === "all" ? 0 : (p.page - 1) * p.per());
  p.measure = (grid) => { p.cols = gridCols(grid); };
  p.watch = (grid) => {                        // a new column count: the same first item stays on the page
    if (p.observer) return;
    p.observer = new ResizeObserver(() => {
      if (!grid.offsetParent) return;          // the tab is hidden
      const cols = gridCols(grid);
      if (cols === p.cols) return;
      const first = p.offset();
      p.cols = cols;
      if (p.mode === "pages") p.page = Math.floor(first / p.per()) + 1;
      p.reload(true);
    });
    p.observer.observe(grid);
  };
  return p;
}
function pageNumbers(cur, pages) {            // 1 … 4 5 6 … 11
  const want = new Set([1, pages, cur - 1, cur, cur + 1].filter((n) => n >= 1 && n <= pages));
  const out = []; let last = 0;
  [...want].sort((x, y) => x - y).forEach((n) => { if (n - last > 1) out.push("…"); out.push(n); last = n; });
  return out;
}
// the toolbar (count, rows per page, Pages / All) and the page buttons; returns false when the page was out of range
function drawPaging(p, total, shown, noun, bar, nav, grid) {
  const all = p.mode === "all", per = p.per();
  const pages = all ? 1 : Math.max(1, Math.ceil(total / per));
  if (!all && p.page > pages) { p.page = pages; return false; }
  const from = total ? p.offset() + 1 : 0, to = total ? p.offset() + shown : 0;
  const sig = JSON.stringify([total, shown, p.mode, p.rows, p.cols, p.page]);
  if (bar.dataset.sig === sig) return true;          // unchanged: keep the toolbar (an open menu stays open)
  bar.dataset.sig = sig;
  bar.innerHTML = `<span class="count">${total} ${noun}${total === 1 ? "" : "s"}${!all && total > per ? ` · ${from}–${to}` : ""}</span>
    <div class="grow"></div>
    ${!all ? `<label class="per" title="Each page is full rows; the number of squares per row follows the window width">Rows per page
      <select data-rows>${ROW_CHOICES.map((n) => `<option value="${n}" ${n === p.rows ? "selected" : ""}>${n}</option>`).join("")}</select></label>` : ""}
    <div class="seg-switch" role="group" aria-label="Show">
      <button type="button" data-mode="pages" class="${all ? "" : "on"}" aria-pressed="${!all}">Pages</button>
      <button type="button" data-mode="all" class="${all ? "on" : ""}" aria-pressed="${all}">All</button></div>`;
  nav.innerHTML = all || pages <= 1 ? "" : `<button type="button" data-page="${p.page - 1}" ${p.page <= 1 ? "disabled" : ""} aria-label="Previous page">‹ Previous</button>
    ${pageNumbers(p.page, pages).map((n) => n === "…" ? `<span class="gap">…</span>`
      : `<button type="button" data-page="${n}" class="${n === p.page ? "on" : ""}" ${n === p.page ? 'aria-current="page"' : ""}>${n}</button>`).join("")}
    <button type="button" data-page="${p.page + 1}" ${p.page >= pages ? "disabled" : ""} aria-label="Next page">Next ›</button>`;
  nav.classList.toggle("hidden", !nav.innerHTML.trim());
  const go = (fn) => { fn(); p.save(); grid.scrollTop = 0; p.reload(true); };
  $$("[data-page]", nav).forEach((b) => b.addEventListener("click", () => go(() => { p.page = +b.dataset.page; })));
  $$("[data-mode]", bar).forEach((b) => b.addEventListener("click", () => go(() => { p.mode = b.dataset.mode; p.page = 1; })));
  const rows = $("[data-rows]", bar);
  if (rows) rows.addEventListener("change", () => go(() => { const first = p.offset(); p.rows = +rows.value; p.page = Math.floor(first / p.per()) + 1; }));
  return true;
}


// Seed boxes: a dice button next to every seed box, also those drawn later (user, 2026-10-27: "we need a randomizer
// button for the seeds box"). A single seed gets a new random number; a "seeds" list (comma-separated) gets one more.
// The retake form keeps its own "New seed" button.
const DICE_ICON = `<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><rect x="3.5" y="3.5" width="17" height="17"
  rx="3.5" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="8.3" cy="8.3" r="1.5" fill="currentColor"/>
  <circle cx="12" cy="12" r="1.5" fill="currentColor"/><circle cx="15.7" cy="15.7" r="1.5" fill="currentColor"/></svg>`;
function addSeedDice(root = document) {
  $$('input[name="seed"], input[name="seeds"]', root).forEach((inp) => {
    if (inp.dataset.dice || inp.closest(".retake-form")) return;
    inp.dataset.dice = "1";
    const wrap = document.createElement("span");
    wrap.className = "seed-wrap";
    inp.parentNode.insertBefore(wrap, inp);
    wrap.appendChild(inp);
    const b = document.createElement("button");
    b.type = "button";
    b.className = "seed-dice";
    b.title = inp.name === "seeds" ? "Add a random seed" : "Random seed";
    b.setAttribute("aria-label", b.title);
    b.innerHTML = DICE_ICON;
    b.addEventListener("click", (e) => {
      e.preventDefault();
      const r = Math.floor(Math.random() * 1e6);
      if (inp.name === "seeds") {
        const cur = inp.value.split(",").map((s) => s.trim()).filter(Boolean);
        cur.push(String(r));
        inp.value = cur.join(",");
      } else {
        inp.value = r;
      }
      inp.dispatchEvent(new Event("input", { bubbles: true }));
      inp.dispatchEvent(new Event("change", { bubbles: true }));
    });
    wrap.appendChild(b);
  });
}
function watchSeedBoxes() {
  addSeedDice();
  new MutationObserver((muts) => {
    for (const m of muts) for (const n of m.addedNodes) if (n.nodeType === 1) addSeedDice(n);
  }).observe(document.body, { childList: true, subtree: true });
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", watchSeedBoxes);
else watchSeedBoxes();
