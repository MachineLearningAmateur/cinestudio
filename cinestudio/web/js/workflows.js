// cinestudio web app - The Workflows view (an ADVANCED mode, off by default): the ComfyUI graphs behind each feature,
// drawn like ComfyUI's nodes, and the exact graphs a video run submitted (nodes added or changed at run time marked).
// The switch: Settings > Advanced (saved in the app settings, mirrored in localStorage so the page does not flicker),
// or Ctrl+Shift+G. Off: the rail button, the status chip and every "graph" link are hidden (CSS: body.wf-advanced).
// Plain scripts (no modules), loaded in order by index.html; they share their top-level names (all start with "wf").
"use strict";

const wfState = { list: null, recent: [], run: null, runGraphs: null, cur: null, lay: null, sel: null,
                  view: { x: 0, y: 0, k: 1 }, filter: "", search: "", hits: [], hitAt: -1, advanced: false };
const WF_W = 270, WF_HEAD = 36, WF_ROW = 18, WF_GAP_X = 110, WF_GAP_Y = 26;
const WF_KINDS = [
  ["loader", "Loaders (models, LoRAs, VAE, CLIP)"], ["text", "Conditioning and text"], ["sampling", "Sampling (samplers, guiders, sigmas, noise)"],
  ["latent", "Latents"], ["image", "Images, videos, sound (load, save, scale, encode, decode)"], ["guide", "LTX guides"], ["other", "Other"],
];

// ------------------------------------------------------------------ advanced mode
function wfSetAdvanced(on, save = false) {
  wfState.advanced = !!on;
  document.body.classList.toggle("wf-advanced", wfState.advanced);
  try { localStorage.setItem("wf.advanced", wfState.advanced ? "1" : "0"); } catch { /* not kept: fine */ }
  const box = $("#wf-adv-toggle");
  if (box) box.checked = wfState.advanced;
  if (!wfState.advanced && !$("#tab-workflows").classList.contains("hidden")) showTab("characters");
  if (save) postJSON("/api/settings/advanced", { advanced: wfState.advanced }).catch(fail);
}
try { wfSetAdvanced(localStorage.getItem("wf.advanced") === "1"); } catch { /* private window: off */ }
api("/api/settings/advanced").then((d) => wfSetAdvanced(d.advanced)).catch(() => {});
$("#wf-adv-toggle").addEventListener("change", (ev) => {
  wfSetAdvanced(ev.target.checked, true);
  toast(ev.target.checked ? "Advanced mode on: the Workflows button is in the left rail, above Settings." : "Advanced mode off.");
});
$("#wf-adv-chip").addEventListener("click", () => { showTab("settings"); showSettingsSection("advanced"); });
document.addEventListener("keydown", (ev) => {
  if (ev.ctrlKey && ev.shiftKey && !ev.altKey && (ev.key === "G" || ev.key === "g")) {
    ev.preventDefault();
    wfSetAdvanced(!wfState.advanced, true);
    toast(wfState.advanced ? "Advanced mode on (Ctrl+Shift+G turns it off)." : "Advanced mode off.");
  } else if (ev.key === "Escape" && wfState.sel && !$("#tab-workflows").classList.contains("hidden")) {
    wfSelect(null);
  }
});
// "graph" links anywhere (the run details, each clip and take): open this run's graphs
document.addEventListener("click", (ev) => {
  const b = ev.target.closest && ev.target.closest("[data-wf-run]");
  if (!b) return;
  ev.preventDefault();
  ev.stopPropagation();
  if (typeof closeRun === "function") closeRun();
  showTab("workflows");
  wfOpenRun(b.dataset.wfRun, b.dataset.wfJob || null);
});

// ------------------------------------------------------------------ the list
async function loadWorkflowsTab() {
  try {
    if (!wfState.list) wfState.list = await api("/api/workflows");
    wfState.recent = (await api("/api/graphs/recent")).items;
  } catch (e) { fail(e); return; }
  wfRenderList();
  if (!wfState.cur && !wfState.run) {
    const first = wfState.list.items.find((x) => x.name === "ltx25_layouting") || wfState.list.items[0];
    if (first) wfOpen({ kind: "base", name: first.name });
  }
}
async function wfOpenRun(id, job) {
  wfState.run = id;
  try {
    if (!wfState.list) wfState.list = await api("/api/workflows");
    wfState.runGraphs = await api(`/api/runs/${encodeURIComponent(id)}/graphs`);
  } catch (e) { fail(e); return; }
  wfRenderList();
  const sub = wfState.runGraphs.submitted;
  const pick = (job && sub.find((s) => s.job === job)) || sub[0];
  if (job && !sub.find((s) => s.job === job)) toast(`No saved graph for "${wfJobLabel(job)}" yet (runs made before this view have none).`);
  if (pick) wfOpen({ kind: "run", run: id, name: pick.job });
  else if (wfState.runGraphs.frozen.length) wfOpen({ kind: "frozen", run: id, name: wfState.runGraphs.frozen[0].name });
}
function wfJobLabel(job) {
  let m;
  if (job === "clip") return "Clip 1";
  if ((m = /^clip(\d+)$/.exec(job))) return `Clip ${m[1]}`;
  if ((m = /^take(\d+)_(\d+)$/.exec(job))) return `Clip ${m[1]} · take ${m[2]}`;
  return { look: "Look image", look_scene: "Look image: the place", background: "Background", depth: "360 depth",
           restyle: "Sheet in the run's style" }[job] || job;
}
const wfAgo = (t) => { const s = Date.now() / 1000 - t; return s < 90 ? "just now" : s < 5400 ? `${Math.round(s / 60)} min ago`
  : s < 172800 ? `${Math.round(s / 3600)} h ago` : new Date(t * 1000).toLocaleDateString(); };
function wfItem(src, label, sub, title, cls = "", badge = "") {
  const c = wfState.cur && wfState.cur.src;
  const on = c && c.kind === src.kind && c.name === src.name && c.run === src.run && c.copy === src.copy;
  return `<button type="button" class="wf-item ${cls} ${on ? "on" : ""}" data-src="${esc(JSON.stringify(src))}" title="${esc(title || label)}">
    <b>${esc(label)}${badge}</b>${sub ? `<span>${esc(sub)}</span>` : ""}</button>`;
}
// a built-in graph and, under it, the user's copies of it (the one new jobs use is marked)
function wfBaseItem(x, label) {
  const copies = x.copies || [];
  const use = `<i class="wf-use" title="New jobs use this graph">in use</i>`;
  let h = wfItem({ kind: "base", name: x.name }, label, `${x.name} · ${x.nodes} nodes`,
    `${x.name}: ${x.features.join("; ")}`, "", `<i class="wf-builtin">built-in</i>${copies.length && !x.active ? use : ""}`);
  h += copies.map((c) => wfItem({ kind: "copy", name: x.name, copy: c.id }, c.label,
    `copy of ${x.name}${c.stale ? " · made from an older built-in" : ""}`,
    `Your copy of ${x.name} (${c.id}); click to edit its values`, "wf-copy", x.active === c.id ? use : "")).join("");
  return h;
}
function wfRenderList() {
  const f = wfState.filter.toLowerCase();
  const match = (...xs) => !f || xs.join(" ").toLowerCase().includes(f);
  const out = [];
  if (wfState.run && wfState.runGraphs) {
    const g = wfState.runGraphs;
    const subs = g.submitted.filter((s) => match(s.job, wfJobLabel(s.job), s.profile, s.variant));
    const frozen = g.frozen.filter((s) => match(s.name));
    out.push(`<div class="wf-group run"><div class="wf-gtitle">This run's graphs
        <button type="button" class="linkbtn" id="wf-close-run" title="Show only the feature graphs">Hide</button></div>
      <div class="wf-runline"><span class="rid">${esc(wfState.run)}</span>
        <button type="button" class="linkbtn" id="wf-back-run" title="Open this video again">Back to the video</button></div>
      ${subs.length ? subs.map((s) => wfItem({ kind: "run", run: wfState.run, name: s.job }, wfJobLabel(s.job),
          `${s.profile || "?"}_${s.variant || "?"} · ${s.nodes ?? "?"} nodes · ${wfAgo(s.time)}`,
          "The exact graph this job sent to ComfyUI")).join("")
        : `<div class="sub">No submitted graph saved yet. Graphs are saved from now on, for each job a run sends.</div>`}
      ${frozen.length ? `<details class="wf-frozen"><summary>Frozen copies (${frozen.length})</summary>
        ${frozen.map((s) => wfItem({ kind: "frozen", run: wfState.run, name: s.name }, s.name, `${s.nodes} nodes`,
          "The run's own copy of the graph, made at 'run new' (before the run-time changes)")).join("")}</details>` : ""}</div>`);
  }
  const recent = wfState.recent.filter((r) => match(r.job, r.profile, r.variant));
  if (recent.length) {
    out.push(`<details class="wf-group" ${wfState.recentOpen ? "open" : ""} id="wf-recent"><summary class="wf-gtitle">Recent jobs <span class="muted">${recent.length}</span></summary>
      ${recent.slice(0, 60).map((r) => wfItem({ kind: "recent", name: r.job }, r.job,
        `${r.profile || "?"}_${r.variant || "?"} · ${wfAgo(r.time)}`, "A character, background, song or template job's exact graph")).join("")}</details>`);
  }
  const items = (wfState.list ? wfState.list.items : []).filter((x) => match(x.name, x.group, x.features.join(" "), x.model_family || "",
    (x.also || []).map((a) => a.label).join(" "), (x.copies || []).map((c) => c.label).join(" ")));
  for (const grp of wfState.list ? wfState.list.groups : []) {
    const rows = [];                                   // a graph also shows in the other groups it serves ("also")
    items.forEach((x) => {
      if (x.group === grp) rows.push([x, x.features[0] || x.name]);
      (x.also || []).forEach((a) => { if (a.group === grp) rows.push([x, a.label]); });
    });
    if (!rows.length) continue;
    const older = grp === wfState.list.groups[wfState.list.groups.length - 1];
    const body = rows.map(([x, label]) => wfBaseItem(x, label)).join("");
    out.push(older ? `<details class="wf-group" ${f ? "open" : ""}><summary class="wf-gtitle">${esc(grp)} <span class="muted">${rows.length}</span></summary>${body}</details>`
      : `<div class="wf-group"><div class="wf-gtitle">${esc(grp)}</div>${body}</div>`);
  }
  const box = $("#wf-list");
  box.innerHTML = out.join("") || `<div class="sub">No graph matches.</div>`;
  $$(".wf-item", box).forEach((b) => b.addEventListener("click", () => wfOpen(JSON.parse(b.dataset.src))));
  const rec = $("#wf-recent", box);
  if (rec) rec.addEventListener("toggle", () => { wfState.recentOpen = rec.open; });
  const back = $("#wf-back-run", box);
  if (back) back.addEventListener("click", () => { const id = wfState.run; showTab("video"); openRun(id); });
  const hide = $("#wf-close-run", box);
  if (hide) hide.addEventListener("click", () => { wfState.run = null; wfState.runGraphs = null; wfRenderList(); });
}
$("#wf-filter").addEventListener("input", (ev) => { wfState.filter = ev.target.value.trim(); wfRenderList(); });

// ------------------------------------------------------------------ open one graph
async function wfOpen(src, keepEdits = false) {
  if (!keepEdits && wfPendingCount() && !confirm("Leave this copy without saving your changes?")) return;
  let d, title, sub;
  try {
    if (src.kind === "base") {
      d = await api(`/api/workflows/${encodeURIComponent(src.name)}${src.tier ? `?tier=${encodeURIComponent(src.tier)}` : ""}`);
      title = d.features[0] || d.name;
      sub = `${d.name} · ${d.model_family || d.profile} · ${d.nodes} nodes · built-in (read-only)${d.tier ? ` · as the ${d.tier} tier freezes it` : ""}`;
    } else if (src.kind === "copy") {
      d = await api(`/api/workflows/${encodeURIComponent(src.name)}/copies/${encodeURIComponent(src.copy)}`);
      title = d.meta.label;
      sub = `your copy of ${d.builtin} · ${Object.keys(d.graph).length} nodes${d.active ? " · new jobs use it" : ""}`;
    } else if (src.kind === "run" || src.kind === "frozen") {
      d = await api(`/api/runs/${encodeURIComponent(src.run)}/graphs/${encodeURIComponent(src.name)}${src.kind === "frozen" ? "?frozen=1" : ""}`);
      title = src.kind === "run" ? `${wfJobLabel(src.name)}: the graph sent` : `${src.name} (the run's frozen copy)`;
      sub = src.kind === "run" ? `${d.base_name || "?"} · run ${src.run}${d.meta && d.meta.time ? ` · ${wfAgo(d.meta.time)}` : ""}` : `run ${src.run}`;
    } else {
      d = await api(`/api/graphs/recent/${encodeURIComponent(src.name)}`);
      title = `${src.name}: the graph sent`;
      sub = `${d.base_name || "?"}${d.meta && d.meta.time ? ` · ${wfAgo(d.meta.time)}` : ""}`;
    }
  } catch (e) { fail(e); return; }
  const same = wfState.cur && wfState.cur.src.kind === src.kind && wfState.cur.src.name === src.name
    && wfState.cur.src.copy === src.copy && wfState.cur.src.run === src.run;
  const file = src.kind === "base" ? `${src.name}${d.tier ? "_" + d.tier : ""}_api.json`
    : src.kind === "copy" ? `${src.name}__${String(d.meta.label).replace(/[^A-Za-z0-9_-]+/g, "-")}_api.json`
    : `${src.run ? src.run + "_" : ""}${src.name}.json`;
  wfState.cur = { src, d, title, file };
  if (!keepEdits) wfState.pending = {};
  if (!same) wfState.sel = null;
  $("#wf-title").innerHTML = `<b>${esc(title)}</b><span class="sub">${esc(sub)}</span>`;
  wfRenderList();
  const view = { ...wfState.view };
  wfState.lay = wfLayout(d.graph, d.titles || {});
  wfDraw();
  if (same) { wfState.view = view; wfApplyView(); } else wfFit(true);
  wfSelect(wfState.sel);
  wfLegend();
  wfActions();
  if (src.kind === "copy") wfLoadCombos(d.graph);
}

// ------------------------------------------------------------------ the user's copies: choose, edit, save
// Built-in graphs are read-only. A copy's literal values can be changed (the inputs the job sets and the links cannot:
// the job's prompt, seed, pictures and models still fill the copy). Changes wait in wfState.pending until Save.
wfState.pending = {};
wfState.combos = {};
wfState.combosOk = null;
const wfPendingCount = () => Object.values(wfState.pending || {}).reduce((s, f) => s + Object.keys(f).length, 0);
const wfIsBound = (id, k) => !!(wfState.cur && (wfState.cur.d.bound || []).some(([n, f]) => n === id && f === k));
const wfEditable = (v) => ["boolean", "number", "string"].includes(typeof v);
function wfValueOf(id, k) {
  const p = (wfState.pending[id] || {});
  return Object.prototype.hasOwnProperty.call(p, k) ? p[k] : ((wfState.cur.d.graph[id] || {}).inputs || {})[k];
}
function wfCustomFields(id) {
  // the inputs of a copy's node whose value (saved or about to be saved) differs from the built-in
  const d = wfState.cur && wfState.cur.d;
  if (!d || wfState.cur.src.kind !== "copy") return [];
  const b = ((d.builtin_graph || {})[id] || {}).inputs, own = (d.graph[id] || {}).inputs || {};
  if (!b) return [];
  return Object.keys(own).filter((k) => wfEditable(own[k]) && k in b && JSON.stringify(wfValueOf(id, k)) !== JSON.stringify(b[k]));
}
async function wfLoadCombos(graph) {
  const classes = [...new Set(Object.values(graph).map((n) => n.class_type).filter((c) => c && !(c in wfState.combos)))];
  if (!classes.length) return;
  try {
    const r = await api(`/api/object_info?classes=${encodeURIComponent(classes.join(","))}`);
    Object.assign(wfState.combos, r.combos || {});
    wfState.combosOk = !!r.available;
  } catch { wfState.combosOk = false; }
  if (wfState.sel) wfSelect(wfState.sel);
}
function wfRefreshMarks() {
  $$("#wf-svg .wf-node").forEach((g) => {
    const m = wfMark(g.dataset.id);
    g.classList.toggle("custom", m === "custom");
    const badge = $(".wf-badge", g);
    if (badge) badge.textContent = m === "custom" ? "● custom" : badge.textContent;
  });
  wfActions();
}
function wfActions() {
  const box = $("#wf-actions"), c = wfState.cur;
  if (!c || !["base", "copy"].includes(c.src.kind)) { box.classList.add("hidden"); box.innerHTML = ""; return; }
  const d = c.d, name = c.src.name;
  const item = (wfState.list && wfState.list.items.find((x) => x.name === name)) || d;
  const copies = item.copies || d.copies || [];
  const active = c.src.kind === "copy" ? (d.active ? c.src.copy : (item.active || null)) : (d.active || null);
  const use = `<label class="wf-usesel" title="Which graph NEW jobs of this feature use (jobs already queued keep theirs)">New jobs use
      <select id="wf-use"><option value="">Built-in</option>${copies.map((x) => `<option value="${esc(x.id)}" ${x.id === active ? "selected" : ""}>${esc(x.label)}${x.stale ? " (older built-in)" : ""}</option>`).join("")}</select></label>`;
  let h = use;
  if (c.src.kind === "base") {
    const tiers = d.tiers || [];
    h += `<button type="button" id="wf-make-copy" title="Make your own copy of this graph to change its values">Make a copy</button>`;
    if (tiers.length > 1) h += `<label class="wf-usesel" title="The graph as a run on this model tier would freeze it (model files, decode tiles); the marked nodes differ from the ${esc(tiers[0])} tier">Show for
        <select id="wf-tier"><option value="">the file as saved</option>${tiers.map((t) => `<option value="${esc(t)}" ${d.tier === t ? "selected" : ""}>the ${esc(t)} tier</option>`).join("")}</select></label>`;
    h += `<span class="sub wf-act-note">Built-in graphs are read-only. Make a copy to change values; then choose it under "New jobs use".</span>`;
  } else {
    const n = Object.keys(d.custom || {}).length ? Object.values(d.custom).reduce((s, f) => s + f.length, 0) : 0;
    const pend = wfPendingCount(), hist = (d.meta && d.meta.history) || 0;
    h += `<span class="wf-count" title="Inputs whose saved value differs from the built-in">${n} value${n === 1 ? "" : "s"} changed from the built-in</span>
      ${pend ? `<button type="button" class="primary" id="wf-save">Save ${pend} change${pend === 1 ? "" : "s"}</button><button type="button" id="wf-cancel">Cancel</button>` : ""}
      <button type="button" id="wf-undo" ${hist ? "" : "disabled"} title="Go back to the version before the last save (${hist} kept)">Undo last save</button>
      <button type="button" id="wf-reset-all" ${n ? "" : "disabled"} title="Put the built-in's values back into every node">Reset all to built-in</button>
      <button type="button" id="wf-dup" title="Make another copy of this copy">Duplicate</button>
      <button type="button" id="wf-rename">Rename</button>
      <button type="button" id="wf-delete" class="danger">Delete</button>
      <span class="sub wf-act-note">Changes apply to new jobs only. Custom values can make results worse.</span>
      ${(d.problems || []).length ? `<div class="wf-act-warn">This copy was made from an older built-in and does not fit it: ${esc(d.problems.slice(0, 3).join("; "))}. New jobs cannot use it; make a new copy.</div>`
        : d.meta && d.meta.stale ? `<div class="wf-act-warn soft">Made from an older version of the built-in graph. It still fits; compare it with the built-in.</div>` : ""}`;
  }
  box.innerHTML = h;
  box.classList.remove("hidden");
  const enc = encodeURIComponent, base = `/api/workflows/${enc(name)}`;
  const after = async (msg, open) => {
    wfState.list = await api("/api/workflows");
    if (msg) toast(msg);
    wfOpen(open || c.src, true);
  };
  $("#wf-use", box).addEventListener("change", async (ev) => {
    try { await postJSON(`${base}/use`, { copy: ev.target.value || null }); after(ev.target.value ? "New jobs of this feature use your copy." : "New jobs use the built-in graph."); }
    catch (e) { fail(e); wfActions(); }
  });
  const on = (id, fn) => { const b = $(id, box); if (b) b.addEventListener("click", async () => { try { await fn(); } catch (e) { fail(e); } }); };
  on("#wf-make-copy", async () => {
    const r = await postJSON(`${base}/copies`, {});
    after(`Made "${r.meta.label}". Change its values in the node panel, then Save.`, { kind: "copy", name, copy: r.id });
  });
  const tier = $("#wf-tier", box);
  if (tier) tier.addEventListener("change", () => wfOpen({ kind: "base", name, tier: tier.value || undefined }, true));
  if (c.src.kind !== "copy") return;
  const cu = `${base}/copies/${enc(c.src.copy)}`;
  on("#wf-save", async () => {
    const r = await postJSON(cu, { values: wfState.pending, version: d.meta.version });
    wfState.pending = {};
    (r.warnings || []).forEach((w) => toast(w, true));
    after(r.warnings && r.warnings.length ? "" : `Saved ${r.changed} value${r.changed === 1 ? "" : "s"}. New jobs that use this copy get them.`);
  });
  on("#wf-cancel", async () => { wfState.pending = {}; wfRefreshMarks(); if (wfState.sel) wfSelect(wfState.sel); });
  on("#wf-undo", async () => { await postJSON(`${cu}/undo`); after("The last save is undone."); });
  on("#wf-reset-all", async () => {
    if (!confirm("Put the built-in's values back into every node of this copy?")) return;
    await postJSON(`${cu}/reset`, {}); wfState.pending = {}; after("All values are the built-in's again (Undo last save brings yours back).");
  });
  on("#wf-dup", async () => {
    const r = await postJSON(`${base}/copies`, { from: c.src.copy });
    after(`Made "${r.meta.label}".`, { kind: "copy", name, copy: r.id });
  });
  on("#wf-rename", async () => {
    const label = prompt("A name for this copy:", d.meta.label);
    if (!label || label === d.meta.label) return;
    await postJSON(`${cu}/rename`, { label }); after("Renamed.");
  });
  on("#wf-delete", async () => {
    if (!confirm(`Delete the copy "${d.meta.label}"? This cannot be undone.${d.active ? "\n\nNew jobs use the built-in graph again." : ""}`)) return;
    await postJSON(`${cu}/delete`); wfState.pending = {};
    after("Deleted.", { kind: "base", name });
  });
}
// one input field of a copy's node (the type stays: number, text, choice, true/false)
function wfField(id, k, v, cls) {
  const cur = wfValueOf(id, k), attrs = `class="wf-edit" data-node="${esc(id)}" data-field="${esc(k)}"`;
  if (typeof v === "boolean") return `<label class="wf-bool"><input type="checkbox" ${attrs} data-type="bool" ${cur ? "checked" : ""}> ${cur ? "true" : "false"}</label>`;
  if (typeof v === "number") {
    const t = ((wfState.cur.d.types || {})[id] || {})[k] || (Number.isInteger(v) ? "int" : "float");
    return `<input type="number" ${attrs} data-type="${t}" step="${t === "int" ? 1 : "any"}" value="${esc(String(cur))}">`;
  }
  const opts = ((wfState.combos[cls] || {})[k]) || null;
  if (opts && opts.length) {
    const list = opts.includes(cur) ? opts : [cur, ...opts];
    return `<select ${attrs} data-type="str">${list.map((o) => `<option ${o === cur ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>`;
  }
  if (String(cur).length > 60 || String(cur).includes("\n")) return `<textarea ${attrs} data-type="str" rows="4">${esc(cur)}</textarea>`;
  return `<input type="text" ${attrs} data-type="str" value="${esc(cur)}">`;
}
function wfBindEdits(panel, id) {
  const d = wfState.cur.d;
  $$(".wf-edit", panel).forEach((el) => {
    const set = () => {
      const k = el.dataset.field, t = el.dataset.type, saved = d.graph[id].inputs[k];
      let v;
      if (t === "bool") { v = el.checked; el.parentElement.lastChild.textContent = ` ${v}`; }
      else if (t === "int" || t === "float") {
        v = Number(el.value);
        const bad = el.value.trim() === "" || !Number.isFinite(v) || (t === "int" && !Number.isInteger(v));
        el.classList.toggle("bad", bad);
        el.title = bad ? (t === "int" ? "A whole number" : "A number") : "";
        if (bad) return;
      } else v = el.value;
      const p = (wfState.pending[id] = wfState.pending[id] || {});
      if (JSON.stringify(v) === JSON.stringify(saved)) delete p[k]; else p[k] = v;
      if (!Object.keys(p).length) delete wfState.pending[id];
      wfRefreshMarks();
    };
    el.addEventListener(el.tagName === "SELECT" || el.type === "checkbox" ? "change" : "input", set);
  });
  const reset = $("#wf-node-reset", panel);
  if (reset) reset.addEventListener("click", () => {
    const b = (d.builtin_graph[id] || {}).inputs || {};
    for (const k of wfCustomFields(id)) {
      const saved = d.graph[id].inputs[k];
      const p = (wfState.pending[id] = wfState.pending[id] || {});
      if (JSON.stringify(b[k]) === JSON.stringify(saved)) delete p[k]; else p[k] = b[k];
      if (!Object.keys(p).length) delete wfState.pending[id];
    }
    wfRefreshMarks();
    wfSelect(id);
    toast("The built-in values are back in the fields: Save to keep them.");
  });
}

// ------------------------------------------------------------------ kinds and layout
function wfKind(cls) {
  if (/^LTXAdd|^LTXV.*(Guide|ImgToVideo)|ICLoRAParameters/.test(cls)) return "guide";
  if (/Loader(ModelOnly)?$|Loader_|LoaderGGUF|^LoadMoGe|DownloadAndLoad/.test(cls)) return "loader";
  if (/TextEncode|Conditioning|CLIPText|Reference/.test(cls)) return "text";
  if (/Sampler|Guider|Sigmas|Scheduler|Noise|CFG|ModelSampling/.test(cls)) return "sampling";
  if (/Latent|AdaIN/i.test(cls)) return "latent";
  if (/Image|Video|VAE|Save|^Load|Audio|GLB|Mesh|PointMap|Pose|Resize|Scale|Mask/.test(cls)) return "image";
  return "other";
}
const wfIsLink = (v, g) => Array.isArray(v) && v.length === 2 && Object.prototype.hasOwnProperty.call(g, String(v[0])) && Number.isInteger(v[1]);
function wfClamp(v, n = 34) {
  let s = typeof v === "string" ? v.replace(/\s*\n\s*/g, " ⏎ ") : JSON.stringify(v);
  if (s === undefined) s = String(v);
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}
function wfLayout(graph, titles) {
  const ids = Object.keys(graph).filter((id) => graph[id] && typeof graph[id] === "object");
  const nodes = {}, links = [];
  for (const id of ids) {
    const n = graph[id], ins = [], widgets = [];
    for (const [k, v] of Object.entries(n.inputs || {})) {
      if (wfIsLink(v, graph)) { ins.push({ name: k, from: String(v[0]), out: v[1] }); links.push({ from: String(v[0]), out: v[1], to: id, input: k }); }
      else widgets.push([k, v]);
    }
    nodes[id] = { id, cls: n.class_type || "?", title: titles[id] || (n._meta && n._meta.title) || "", ins, widgets, outs: [], kind: wfKind(n.class_type || "") };
  }
  for (const l of links) if (!nodes[l.from].outs.includes(l.out)) nodes[l.from].outs.push(l.out);
  for (const n of Object.values(nodes)) {
    n.outs.sort((a, b) => a - b);
    n.h = WF_HEAD + Math.max(n.ins.length, n.outs.length, 0) * WF_ROW + n.widgets.length * (WF_ROW - 1) + 12;
    n.w = WF_W;
  }
  // the column of a node = its longest path from a node without inputs (a cycle stops at the node already on the path)
  const depth = {}, onPath = new Set();
  const dep = (id) => {
    if (depth[id] !== undefined) return depth[id];
    if (onPath.has(id)) return 0;
    onPath.add(id);
    const d = nodes[id].ins.reduce((m, i) => Math.max(m, dep(i.from) + 1), 0);
    onPath.delete(id);
    return (depth[id] = d);
  };
  ids.forEach(dep);
  // a node without inputs (a loader, a number) stands just left of its first user, not in a tall first column
  const usersOf = {};
  links.forEach((l) => (usersOf[l.from] = usersOf[l.from] || []).push(l.to));
  for (const id of ids) {
    if (!nodes[id].ins.length && usersOf[id]) depth[id] = Math.max(0, Math.min(...usersOf[id].map((t) => depth[t])) - 1);
  }
  const cols = [];
  for (const id of ids) (cols[depth[id]] = cols[depth[id]] || []).push(id);
  for (let i = 0; i < cols.length; i++) cols[i] = cols[i] || [];
  // fewer crossings: barycenter sweeps (by the sources, then by the targets)
  const pos = {};
  const setPos = () => cols.forEach((c) => c.forEach((id, i) => { pos[id] = (i + 0.5) / c.length; }));
  setPos();
  const outsOf = usersOf;
  const avg = (xs, own) => (xs.length ? xs.reduce((s, x) => s + pos[x], 0) / xs.length : own);
  for (let pass = 0; pass < 4; pass++) {
    const down = pass % 2 === 0;
    const order = down ? cols.map((_, i) => i) : cols.map((_, i) => cols.length - 1 - i);
    for (const ci of order) {
      const key = {};
      cols[ci].forEach((id) => { key[id] = avg(down ? nodes[id].ins.map((i) => i.from) : (outsOf[id] || []), pos[id]); });
      cols[ci].sort((a, b) => key[a] - key[b]);
      cols[ci].forEach((id, i) => { pos[id] = (i + 0.5) / cols[ci].length; });
    }
  }
  const heights = cols.map((c) => c.reduce((s, id) => s + nodes[id].h + WF_GAP_Y, -WF_GAP_Y));
  const maxH = Math.max(0, ...heights);
  cols.forEach((c, ci) => {
    let y = (maxH - heights[ci]) / 2;
    c.forEach((id) => { nodes[id].x = ci * (WF_W + WF_GAP_X); nodes[id].y = y; y += nodes[id].h + WF_GAP_Y; });
  });
  return { nodes, links, w: cols.length * (WF_W + WF_GAP_X) - WF_GAP_X, h: maxH };
}

// ------------------------------------------------------------------ drawing (SVG)
function wfMark(id) {
  const d = wfState.cur ? wfState.cur.d : {};
  if (wfState.cur && wfState.cur.src.kind === "copy") return wfCustomFields(id).length ? "custom" : "";
  if ((d.added || []).includes(id)) return "added";
  if ((d.changed || []).includes(id)) return "changed";
  if ((d.filled || []).includes(id)) return "filled";
  return "";
}
const wfInY = (n, i) => n.y + WF_HEAD + i * WF_ROW + WF_ROW / 2 + 4;
const wfOutY = (n, out) => n.y + WF_HEAD + Math.max(0, n.outs.indexOf(out)) * WF_ROW + WF_ROW / 2 + 4;
function wfDraw() {
  const L = wfState.lay;
  if (!L) return;
  const paths = L.links.map((l) => {
    const a = L.nodes[l.from], b = L.nodes[l.to];
    const x1 = a.x + a.w, y1 = wfOutY(a, l.out), x2 = b.x, y2 = wfInY(b, b.ins.findIndex((i) => i.name === l.input));
    const dx = Math.max(50, Math.abs(x2 - x1) / 2);
    return `<path class="wf-link k-${a.kind}" data-from="${esc(l.from)}" data-to="${esc(l.to)}" d="M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}"/>`;
  }).join("");
  const boxes = Object.values(L.nodes).map((n) => {
    const mark = wfMark(n.id);
    const head = n.title && n.title !== n.cls ? n.title : n.cls;
    const rows = Math.max(n.ins.length, n.outs.length);
    let t = `<g class="wf-node k-${n.kind} ${mark}" data-id="${esc(n.id)}" transform="translate(${n.x},${n.y})">
      <rect class="wf-body" width="${n.w}" height="${n.h}" rx="9"/>
      <path class="wf-head" d="M0,9 a9,9 0 0 1 9,-9 h${n.w - 18} a9,9 0 0 1 9,9 v${WF_HEAD - 9} h-${n.w} z"/>
      <text class="wf-t1" x="10" y="15">${esc(wfClamp(head, 30))}</text>
      <text class="wf-t2" x="10" y="29">${esc(wfClamp(`${n.cls} · #${n.id}`, 42))}</text>
      ${mark ? `<text class="wf-badge" x="${n.w - 8}" y="15" text-anchor="end">${mark === "added" ? "+ added" : mark === "custom" ? "● custom" : mark}</text>` : ""}`;
    n.ins.forEach((i, k) => {
      const y = WF_HEAD + k * WF_ROW + WF_ROW / 2 + 4;
      t += `<circle class="wf-sock" cx="0" cy="${y}" r="4"/><text class="wf-in" x="9" y="${y + 4}">${esc(wfClamp(i.name, 22))}</text>`;
    });
    n.outs.forEach((o, k) => {
      const y = WF_HEAD + k * WF_ROW + WF_ROW / 2 + 4;
      t += `<circle class="wf-sock out" cx="${n.w}" cy="${y}" r="4"/><text class="wf-out" x="${n.w - 9}" y="${y + 4}" text-anchor="end">out ${o}</text>`;
    });
    n.widgets.forEach(([k, v], j) => {
      const y = WF_HEAD + rows * WF_ROW + j * (WF_ROW - 1) + 16;
      t += `<text class="wf-w" x="10" y="${y}"><tspan class="wf-wk">${esc(wfClamp(k, 18))}</tspan> ${esc(wfClamp(v, Math.max(8, 40 - Math.min(k.length, 18))))}</text>`;
    });
    return t + "</g>";
  }).join("");
  $("#wf-svg").innerHTML = `<g id="wf-view">${paths}${boxes}</g>`;
  $("#wf-empty").classList.toggle("hidden", !!Object.keys(L.nodes).length);
  wfApplyView();
  wfHighlight();
  if (wfState.sel) wfSelect(wfState.sel, false);
}
function wfApplyView() {
  const v = wfState.view, g = $("#wf-view");
  if (g) g.setAttribute("transform", `translate(${v.x},${v.y}) scale(${v.k})`);
  $("#wf-zoom-v").textContent = `${Math.round(v.k * 100)}%`;
}
function wfFit(readable = false) {
  // readable (a graph just opened): the whole graph when it fits at 35% or more, else 55% from the graph's start
  const L = wfState.lay, svg = $("#wf-svg");
  if (!L || !svg.clientWidth) return;
  const W = svg.clientWidth, H = svg.clientHeight, pad = 40;
  const k = Math.max(0.08, Math.min(1.1, (W - pad * 2) / Math.max(1, L.w), (H - pad * 2) / Math.max(1, L.h)));
  if (readable && k < 0.35) {
    const k2 = 0.55;
    wfState.view = { k: k2, x: pad, y: (H - L.h * k2) / 2 };
    if (!wfFit.told) { wfFit.told = true; toast("A large graph: drag to move it, the wheel zooms, Fit shows all of it."); }
  } else wfState.view = { k, x: (W - L.w * k) / 2, y: (H - L.h * k) / 2 };
  wfApplyView();
}
function wfZoom(f, cx, cy) {
  const v = wfState.view, svg = $("#wf-svg");
  if (cx === undefined) { cx = svg.clientWidth / 2; cy = svg.clientHeight / 2; }
  const k = Math.max(0.08, Math.min(3, v.k * f));
  v.x = cx - (cx - v.x) * (k / v.k); v.y = cy - (cy - v.y) * (k / v.k); v.k = k;
  wfApplyView();
}
function wfCenterOn(id) {
  const n = wfState.lay && wfState.lay.nodes[id], svg = $("#wf-svg");
  if (!n) return;
  const v = wfState.view;
  if (v.k < 0.6) v.k = 0.8;
  v.x = svg.clientWidth / 2 - (n.x + n.w / 2) * v.k; v.y = svg.clientHeight / 2 - (n.y + n.h / 2) * v.k;
  wfApplyView();
}

// pan (drag the background or a node), zoom (wheel), click a node = its inputs
(() => {
  const svg = $("#wf-svg");
  let drag = null;
  svg.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    const node = ev.target.closest(".wf-node");
    drag = { x: ev.clientX, y: ev.clientY, vx: wfState.view.x, vy: wfState.view.y, node: node && node.dataset.id, moved: false };
    svg.setPointerCapture(ev.pointerId);
  });
  svg.addEventListener("pointermove", (ev) => {
    if (!drag) return;
    const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) { drag.moved = true; svg.classList.add("panning"); }
    if (drag.moved) { wfState.view.x = drag.vx + dx; wfState.view.y = drag.vy + dy; wfApplyView(); }
  });
  const end = () => {
    if (drag && !drag.moved) wfSelect(drag.node || null);
    drag = null; svg.classList.remove("panning");
  };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", () => { drag = null; svg.classList.remove("panning"); });
  svg.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const r = svg.getBoundingClientRect();
    wfZoom(Math.exp(-ev.deltaY * 0.0015), ev.clientX - r.left, ev.clientY - r.top);
  }, { passive: false });
})();
$("#wf-zoom-in").addEventListener("click", () => wfZoom(1.25));
$("#wf-zoom-out").addEventListener("click", () => wfZoom(0.8));
$("#wf-fit").addEventListener("click", () => wfFit());

// ------------------------------------------------------------------ search
function wfHighlight() {
  const q = wfState.search.toLowerCase(), L = wfState.lay;
  wfState.hits = !q || !L ? [] : Object.values(L.nodes).filter((n) => `${n.cls} ${n.title} ${n.id}`.toLowerCase().includes(q)).map((n) => n.id);
  $("#wf-svg").classList.toggle("searching", !!q);
  $$("#wf-svg .wf-node").forEach((g) => g.classList.toggle("hit", wfState.hits.includes(g.dataset.id)));
  $("#wf-hits").textContent = q ? `${wfState.hits.length} found` : "";
}
$("#wf-search").addEventListener("input", (ev) => { wfState.search = ev.target.value.trim(); wfState.hitAt = -1; wfHighlight(); });
$("#wf-search").addEventListener("keydown", (ev) => {
  if (ev.key !== "Enter" || !wfState.hits.length) return;
  wfState.hitAt = (wfState.hitAt + 1) % wfState.hits.length;
  const id = wfState.hits[wfState.hitAt];
  wfCenterOn(id); wfSelect(id, false);
});

// ------------------------------------------------------------------ the node panel
function wfBindInfo(id) {
  // which tool-level value fills which input of this node (from the binding map), and the node's optional rule
  const b = (wfState.cur && wfState.cur.d.bind) || {}, by = {};
  for (const [name, targets] of Object.entries(b.inputs || {}))
    for (const t of targets) if (String(t[0]) === id) (by[t[1]] = by[t[1]] || []).push(t[2] === "not" ? `not ${name}` : name);
  for (const [slot, targets] of Object.entries(b.models || {}))
    for (const t of targets) if (String(t[0]) === id) (by[t[1]] = by[t[1]] || []).push(`model slot ${slot}`);
  const notes = [];
  const opt = (b.optional_nodes || {})[id];
  if (opt) notes.push(`Optional: kept only when "${opt.when}" is given (else the link goes around it).`);
  for (const [name, o] of Object.entries(b.optional_images || {})) {
    if ((o.nodes || []).includes(id)) notes.push(`Optional picture "${name}": removed when no file is given.`);
    for (const [n, f] of o.inputs || []) if (String(n) === id) (by[f] = by[f] || []).push(`only with "${name}"`);
  }
  if (b.outputs && Object.values(b.outputs).includes(id)) notes.push("The job's result is read from this node.");
  return { by, notes };
}
function wfSelect(id, center = false) {
  wfState.sel = id;
  $$("#wf-svg .wf-node").forEach((g) => g.classList.toggle("sel", g.dataset.id === id));
  $$("#wf-svg .wf-link").forEach((p) => p.classList.toggle("lit", !!id && (p.dataset.from === id || p.dataset.to === id)));
  const panel = $("#wf-panel");
  const L = wfState.lay, n = id && L && L.nodes[id];
  if (!n) { panel.classList.add("hidden"); return; }
  if (center) wfCenterOn(id);
  const d = wfState.cur.d, raw = (d.graph || {})[id] || {};
  const { by, notes } = wfBindInfo(id);
  const files = (d.meta && d.meta.files) || {};
  const diffFields = (d.fields || {})[id] || [];
  const mark = wfMark(id);
  const base = d.base && d.base[id];
  const isCopy = wfState.cur.src.kind === "copy";
  const custom = isCopy ? wfCustomFields(id) : [];
  const bi = isCopy ? (((d.builtin_graph || {})[id] || {}).inputs || null) : null;
  const rows = Object.entries(raw.inputs || {}).map(([k, v]) => {
    const link = wfIsLink(v, d.graph);
    const src = link && L.nodes[String(v[0])];
    const bound = wfIsBound(id, k);
    const val = link ? `<button type="button" class="linkbtn wf-jump" data-jump="${esc(String(v[0]))}">from node ${esc(String(v[0]))}${src && src.title ? ` (${esc(src.title)})` : src ? ` (${esc(src.cls)})` : ""} output ${v[1]}</button>`
      : isCopy && !bound && wfEditable(v) ? wfField(id, k, v, n.cls)
      : `<pre>${esc(typeof v === "string" ? v : JSON.stringify(v, null, 1))}</pre>${isCopy && bound ? `<div class="wf-was">set by the job: not editable</div>` : ""}`;
    const was = diffFields.includes(k) && base ? `<div class="wf-was">before: ${esc(wfClamp(base.inputs ? base.inputs[k] : undefined, 120))}</div>` : "";
    const fromBuiltin = custom.includes(k) && bi ? `<div class="wf-was">built-in: ${esc(wfClamp(bi[k], 120))}</div>` : "";
    const file = files[`${id}.${k}`] ? `<div class="wf-file" title="${esc(files[`${id}.${k}`])}">file sent: ${esc(files[`${id}.${k}`].split(/[\\/]/).pop())}</div>` : "";
    return `<tr class="${diffFields.includes(k) ? "diff" : ""} ${custom.includes(k) ? "custom" : ""}"><th>${esc(k)}${(by[k] || []).map((x) => `<span class="wf-tag">${esc(x)}</span>`).join("")}</th>
      <td>${val}${file}${was}${fromBuiltin}</td></tr>`;
  }).join("");
  const editNote = isCopy ? `<div class="wf-note custom">Your copy: change the white fields, then Save above.${custom.length ? ` ${custom.length} value${custom.length === 1 ? " differs" : "s differ"} from the built-in.` : ""}
      ${wfState.combosOk === false ? " ComfyUI is not running: type choices by hand (the lists appear when it runs)." : ""}
      ${bi && custom.length ? `<button type="button" class="linkbtn" id="wf-node-reset">Reset this node to the built-in values</button>` : ""}
      ${!bi ? " This node is not in the built-in graph." : ""}</div>`
    : wfState.cur.src.kind === "base" ? `<div class="wf-note">Built-in graphs are read-only. Use "Make a copy" above to change values.</div>` : "";
  const extraFiles = Object.entries(files).filter(([key]) => key.startsWith(id + ".") && !((raw.inputs || {})[key.slice(id.length + 1)] !== undefined))
    .map(([key, p]) => `<div class="wf-file" title="${esc(p)}">${esc(key.slice(id.length + 1))}: file sent ${esc(p.split(/[\\/]/).pop())}</div>`).join("");
  panel.innerHTML = `<div class="wf-ph"><div><b>${esc(n.title || n.cls)}</b><span class="sub">${esc(n.cls)} · node #${esc(id)}</span></div>
      <button type="button" class="close" id="wf-panel-close" title="Close (Esc)" aria-label="Close">${ICON.close}</button></div>
    ${mark && mark !== "custom" ? `<div class="wf-note ${mark}">${mark === "added" ? "Added at run time (not in the base graph)."
      : mark === "changed" ? `${esc(d.diff_label || "Changed at run time")}: ${esc(diffFields.join(", "))}.` : `Filled with the job's values: ${esc(diffFields.join(", "))}.`}</div>` : ""}
    ${editNote}
    ${notes.map((x) => `<div class="wf-note">${esc(x)}</div>`).join("")}
    <table class="wf-kv">${rows || `<tr><td class="muted">No inputs.</td></tr>`}</table>${extraFiles}
    <div class="wf-outs">${n.outs.length ? `Outputs used: ${n.outs.map((o) => `out ${o} → ${L.links.filter((l) => l.from === id && l.out === o)
      .map((l) => `<button type="button" class="linkbtn wf-jump" data-jump="${esc(l.to)}">#${esc(l.to)}.${esc(l.input)}</button>`).join(", ")}`).join("; ")}` : "No output used by another node."}</div>`;
  panel.classList.remove("hidden");
  $("#wf-panel-close").addEventListener("click", () => wfSelect(null));
  $$(".wf-jump", panel).forEach((b) => b.addEventListener("click", () => wfSelect(b.dataset.jump, true)));
  if (isCopy) wfBindEdits(panel, id);
}

// ------------------------------------------------------------------ legend, download, copy
function wfLegend() {
  const d = wfState.cur ? wfState.cur.d : {};
  const diff = (d.added || []).length + (d.changed || []).length + (d.filled || []).length + (d.removed || []).length;
  const kinds = WF_KINDS.map(([k, label]) => `<span class="wf-lg k-${k}" title="${esc(label)}"><i></i>${esc(label.split(" (")[0])}</span>`).join("");
  const isCopy = wfState.cur && wfState.cur.src.kind === "copy";
  const marks = isCopy ? `<span class="wf-lg custom"><i></i>Value differs from the built-in</span>`
    : d.diff_label ? `<span class="wf-lg changed"><i></i>${esc(d.diff_label)} (${(d.changed || []).length})</span>`
    : d.base ? `<span class="wf-lg added"><i></i>Added at run time (${(d.added || []).length})</span>
    <span class="wf-lg changed"><i></i>Changed at run time (${(d.changed || []).length})</span>
    <span class="wf-lg filled"><i></i>Filled with the job's values (${(d.filled || []).length})</span>
    ${(d.removed || []).length ? `<span class="wf-lg removed" title="${esc(d.removed.join(", "))}">Removed: ${esc(wfClamp(d.removed.join(", "), 60))}</span>` : ""}` : "";
  const files = d.meta && d.meta.files && Object.keys(d.meta.files).length
    ? `<span class="wf-lg note" title="ComfyUI received these files as uploads; in a downloaded graph the Load nodes keep placeholder names">${Object.keys(d.meta.files).length === 1 ? "1 file" : `${Object.keys(d.meta.files).length} files`} uploaded with the job</span>` : "";
  $("#wf-legend").innerHTML = `<div class="wf-lg-row">${kinds}</div>${diff || files || isCopy ? `<div class="wf-lg-row">${marks}${files}</div>` : ""}`;
}
function wfJsonText() { return wfState.cur ? JSON.stringify(wfState.cur.d.graph, null, 2) : ""; }
$("#wf-download").addEventListener("click", () => {
  if (!wfState.cur) return;
  const url = URL.createObjectURL(new Blob([wfJsonText()], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url; a.download = wfState.cur.file;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
  toast("Saved. Drag the file onto ComfyUI (or use its Open) to load the graph.");
});
$("#wf-copy").addEventListener("click", async () => {
  if (!wfState.cur) return;
  const text = wfJsonText();
  try { await navigator.clipboard.writeText(text); }
  catch {
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } finally { ta.remove(); }
  }
  toast("The graph's JSON is copied.");
});
window.addEventListener("resize", () => { if (wfState.lay && !$("#tab-workflows").classList.contains("hidden")) wfApplyView(); });
