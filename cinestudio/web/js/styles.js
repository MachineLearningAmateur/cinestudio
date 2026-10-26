// cinestudio web app - The Styles tab: the style library (one set of look words for characters, backgrounds
// and videos: cinestudio/styles.py): cards, the New / Edit style dialog, default, duplicate, delete; the floating skin
// menu of the Characters tab with the New skin dialog; the small Style selects of the New background and New video forms.
// Plain scripts (no modules), loaded in order by index.html; they share their top-level names.
"use strict";

const styleLib = { list: [], def: null, at: 0, loading: null };
const STYLE_KINDS = [["characters", "character"], ["backgrounds", "background"]];   // a template has no look of its own
const stylePlural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const styleSame = (a, b) => (a || "").split(/\s+/).join(" ").trim().replace(/\.$/, "") === (b || "").split(/\s+/).join(" ").trim().replace(/\.$/, "");

async function loadStyleLib(force = false) {
  if (!force && styleLib.at && Date.now() - styleLib.at < 4000) return styleLib;
  if (styleLib.loading) return styleLib.loading;
  styleLib.loading = api("/api/styles").then((d) => {
    styleLib.list = d.styles || []; styleLib.def = d.default; styleLib.at = Date.now();
    return styleLib;
  }).finally(() => { styleLib.loading = null; });
  return styleLib.loading;
}
const styleByName = (n) => styleLib.list.find((s) => s.name === n);
// the options of a Style select: "none" first, then the library (a missing current style stays visible)
function styleOptionsHtml(cur, none) {
  return `<option value="">${esc(none)}</option>` + styleLib.list.map((s) =>
    `<option value="${esc(s.name)}" ${s.name === cur ? "selected" : ""}>${esc(s.label || s.name)}${s.default ? " (default)" : ""}</option>`).join("")
    + (cur && !styleByName(cur) ? `<option value="${esc(cur)}" selected>${esc(cur)} (not in the library)</option>` : "");
}

// ------------------------------------------------------------------ the tab: one card per style
const styleSig = (s) => JSON.stringify([s.label, s.words, s.light, s.lora, s.reference_url, s.reference_crop, s.version,
                                        s.builtin, s.has_builtin, s.default, s.used_by, s.stale, s.negative_image]);

function styleUsed(s) {
  const n = STYLE_KINDS.map(([k, w]) => stylePlural((s.used_by[k] || []).length, w)).join(", ");
  const names = STYLE_KINDS.filter(([k]) => (s.used_by[k] || []).length)
    .map(([k, w]) => `${w}s: ${s.used_by[k].join(", ")}`).join("\n");
  return `<span class="fact" title="${esc(names || "Nothing uses this style yet")}">Used by: ${esc(n)}</span>`;
}

function styleStale(s) {
  const all = STYLE_KINDS.flatMap(([k]) => s.stale[k] || []);
  if (!all.length) return "";
  const names = STYLE_KINDS.filter(([k]) => (s.stale[k] || []).length).map(([k, w]) => `${w}s: ${s.stale[k].join(", ")}`).join("\n");
  return `<span class="fact warn" title="Made with an older version of this style (their pictures do not change by themselves):\n${esc(names)}">${all.length} made with an older version</span>`;
}

function styleCard(s) {
  const badges = [];
  if (s.builtin) badges.push(`<span class="chip builtin" title="Ships with cinestudio: Edit makes your own copy">Built-in</span>`);
  else if (s.has_builtin) badges.push(`<span class="chip st-copy" title="Your own copy of a built-in style: it wins over the built-in">Your copy</span>`);
  if (s.default) badges.push(`<span class="chip ready" title="The library's default style (the CLI and the MCP tools report it)">Default</span>`);
  const ref = s.reference_url
    ? `<img class="st-ref" src="${esc(s.reference_url)}" alt="Reference picture of ${esc(s.label || s.name)}" title="Reference picture (click to view large)" data-st-ref="${esc(s.reference_url)}">`
    : `<span class="st-ref none" title="No reference picture">${ICON.image}</span>`;
  const lora = s.lora ? `<span class="fact" title="The video LoRA of this style">LoRA ${esc(s.lora.name)} × ${esc(s.lora.strength)}</span>`
    : `<span class="fact muted" title="No video LoRA">no LoRA</span>`;
  const del = s.builtin ? "" : s.has_builtin
    ? `<button type="button" data-st-del="${esc(s.name)}" data-st-copy="1" title="Delete your copy: the built-in style comes back">Back to the built-in</button>`
    : `<button type="button" class="st-danger" data-st-del="${esc(s.name)}" title="Delete this style (assets keep their words)">Delete</button>`;
  return `<div class="st-top">${ref}
      <div class="st-titles"><div class="sc-head"><span class="sc-name" title="${esc(s.label || s.name)}">${esc(s.label || s.name)}</span></div>
        <div class="st-id"><code>${esc(s.name)}</code> · version ${esc(s.version)}</div>
        <div class="st-badges">${badges.join("")}</div></div></div>
    <div class="sc-tags clamp" title="Click to show all">${esc(s.words)}</div>
    ${s.light ? `<div class="st-line" title="Light words (added after the words)"><b>Light</b> ${esc(s.light)}</div>` : ""}
    <div class="sc-facts">${lora}${styleUsed(s)}${styleStale(s)}</div>
    <div class="bg-actions">
      <button type="button" data-st-edit="${esc(s.name)}" title="${s.builtin ? "Change it: this makes your own copy" : "Change the words, light, LoRA or picture"}">Edit</button>
      ${s.default ? "" : `<button type="button" data-st-default="${esc(s.name)}" title="Make this the library's default style">Set as default</button>`}
      <button type="button" data-st-dup="${esc(s.name)}" title="A new style that starts from this one">Duplicate</button>
      <button type="button" data-st-mkref="${esc(s.name)}" title="${s.lora ? "The video model draws pictures in this style with its LoRA; pick one as the reference picture"
        : "The video model draws pictures for these words (this style has no LoRA); pick one as the reference picture"}">${s.lora ? "Make a reference with the video LoRA" : "Make a reference with LTX"}</button>
      ${del}
    </div>`;
}

function bindStyleCard(card) {
  const t = $(".sc-tags", card);
  if (t) t.addEventListener("click", () => t.classList.toggle("clamp"));
  const img = $("[data-st-ref]", card);
  if (img) img.addEventListener("click", () => openPanoViewer(img.dataset.stRef, img.alt, true));
  const name = card.dataset.style;
  $$("[data-st-edit]", card).forEach((b) => b.addEventListener("click", () => openStyleDlg("edit", name)));
  $$("[data-st-dup]", card).forEach((b) => b.addEventListener("click", () => openStyleDlg("dup", name)));
  $$("[data-st-mkref]", card).forEach((b) => b.addEventListener("click", () => openStyleRefDlg(name)));
  $$("[data-st-default]", card).forEach((b) => b.addEventListener("click", async () => {
    try { await postJSON(`/api/styles/${encodeURIComponent(name)}/default`); toast("Default style set."); loadStylesTab(); }
    catch (e) { fail(e); }
  }));
  $$("[data-st-del]", card).forEach((b) => b.addEventListener("click", async () => {
    const s = styleByName(name) || { label: name };
    const msg = b.dataset.stCopy ? `Go back to the built-in style "${s.label || name}"? Your changes to it are lost.`
      : `Delete the style "${s.label || name}"? Characters and backgrounds keep their words. This cannot be undone.`;
    if (!confirm(msg)) return;
    try {
      await postJSON(`/api/styles/${encodeURIComponent(name)}/delete`);
      toast(b.dataset.stCopy ? "Back to the built-in style." : "Style deleted.");
      loadStylesTab();
    } catch (e) { fail(e); }
  }));
}

async function loadStylesTab() {
  try { await loadStyleLib(true); } catch { return; }
  const list = styleLib.list, grid = $("#style-grid"), bar = $("#style-bar");
  const sig = String(list.length);
  if (bar.dataset.sig !== sig) {
    bar.dataset.sig = sig;
    bar.innerHTML = `<span class="count">${stylePlural(list.length, "style")}</span><div class="grow"></div>
      <span class="muted small st-bar-hint">One set of look words for characters, backgrounds and videos</span>`;
  }
  if (!$("#btn-new-style", grid)) {
    grid.innerHTML = `<button type="button" class="song-card new-tile" id="btn-new-style" title="New style" aria-label="New style">
        <span class="plus-box"><svg viewBox="0 0 24 24" width="30" height="30" aria-hidden="true"><path fill="currentColor" d="M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6z"/></svg></span>
        <span class="plus-label">New style</span></button>`;
    $("#btn-new-style").addEventListener("click", () => openStyleDlg("new"));
  }
  const names = new Set(list.map((s) => s.name));
  $$(".style-card[data-style]", grid).forEach((c) => { if (!names.has(c.dataset.style)) c.remove(); });
  let after = $("#btn-new-style", grid);
  for (const s of list) {
    let card = $$(".style-card[data-style]", grid).find((c) => c.dataset.style === s.name);
    if (!card || card.dataset.sig !== styleSig(s)) {
      const fresh = document.createElement("div");
      fresh.className = "song-card style-card" + (s.default ? " default" : "");
      fresh.dataset.style = s.name;
      fresh.dataset.sig = styleSig(s);
      fresh.innerHTML = styleCard(s);
      if (card) card.replaceWith(fresh); else after.after(fresh);
      card = fresh;
      bindStyleCard(card);
    }
    if (card.previousElementSibling !== after) after.after(card);
    after = card;
  }
  fillStylePresets();
}

// ------------------------------------------------------------------ the New / Edit / Duplicate dialog
const styleDlg = { mode: "new", name: null, from: null, removeRef: false, url: null };

function styleRefPreview(url, hint) {
  const img = $("#style-dlg-ref");
  if (url) { img.src = url; img.classList.remove("hidden"); } else { img.removeAttribute("src"); img.classList.add("hidden"); }
  $("#style-dlg-ref-hint").textContent = hint || "Choose a PNG, JPG or WEBP picture";
  $("#btn-style-ref-remove").classList.toggle("hidden", !url);
}

function freeStyleName(base) {
  const root = base.replace(/_copy(_\d+)?$/, "").slice(0, 56);
  let n = `${root}_copy`, k = 2;
  while (styleByName(n)) n = `${root}_copy_${k++}`;
  return n;
}

async function openStyleDlg(mode, name) {
  try { await loadStyleLib(true); } catch (e) { return fail(e); }
  const s = name ? styleByName(name) : null;
  if (name && !s) return toast(`The style "${name}" is gone.`, true);
  const f = $("#form-style-lib");
  f.reset();
  Object.assign(styleDlg, { mode, name: mode === "edit" ? name : null, from: mode === "dup" ? name : null, removeRef: false, url: null });
  const title = { new: "New style", edit: "Edit style", dup: "Duplicate style" }[mode];
  $("#style-dlg-title").firstChild.textContent = title + " ";
  $("#style-dlg-name").textContent = mode === "edit" ? s.label || s.name : mode === "dup" ? `from ${s.label || s.name}` : "";
  f.name.value = mode === "edit" ? s.name : mode === "dup" ? freeStyleName(s.name) : "";
  f.name.readOnly = mode === "edit";
  $("#style-dlg-namefield").classList.toggle("hidden", mode === "edit");
  f.label.value = s ? (mode === "dup" ? `${s.label || s.name} copy` : s.label || "") : "";
  f.words.value = s ? s.words || "" : "";
  f.light.value = s ? s.light || "" : "";
  f.negative_image.value = s ? s.negative_image || "" : "";
  await loadLoras();
  const sel = $("#style-dlg-lora"), lora = s && s.lora;
  if (lora && ![...sel.options].some((o) => o.value === lora.name)) sel.insertAdjacentHTML("beforeend", `<option value="${esc(lora.name)}">${esc(lora.name)} (not found)</option>`);
  sel.value = lora ? lora.name : "";
  f.lora_strength.value = lora ? lora.strength : 0.4;
  styleRefPreview(s && s.reference_url, s && s.reference_url ? (mode === "dup" ? "The new style gets its own copy of this picture" : "The current picture · choose a file to replace it") : "");
  const note = $("#style-dlg-note"), stale = s ? STYLE_KINDS.reduce((n, [k]) => n + (s.used_by[k] || []).length, 0) : 0;
  const notes = [];
  if (mode === "edit" && s.builtin) notes.push("This is a built-in style: saving makes your own copy with the same name. Deleting the copy later brings the built-in back.");
  if (mode === "edit" && stale) notes.push(`Saving makes version ${s.version + 1}. What already uses it keeps its pictures and shows as "made with an older version".`);
  note.textContent = notes.join(" ");
  note.classList.toggle("hidden", !notes.length);
  $("#style-dlg-foot").textContent = mode === "edit" ? `Version ${s.version} → ${s.version + 1}` : "";
  $("#style-dlg").classList.remove("hidden");
  setTimeout(() => (mode === "edit" ? f.words : f.name).focus(), 30);
}

$("#form-style-lib").reference.addEventListener("change", (ev) => {
  const file = ev.target.files[0];
  if (styleDlg.url) { URL.revokeObjectURL(styleDlg.url); styleDlg.url = null; }
  if (!file) {
    const s = styleByName(styleDlg.name || styleDlg.from);
    return styleRefPreview(!styleDlg.removeRef && s && s.reference_url, "");
  }
  styleDlg.url = URL.createObjectURL(file);
  styleDlg.removeRef = false;
  styleRefPreview(styleDlg.url, file.name);
});
$("#btn-style-ref-remove").addEventListener("click", () => {
  const f = $("#form-style-lib");
  f.reference.value = "";
  if (styleDlg.url) { URL.revokeObjectURL(styleDlg.url); styleDlg.url = null; }
  styleDlg.removeRef = true;
  styleRefPreview(null, styleDlg.mode === "edit" ? "The picture is removed when you save" : "");
});
$("#btn-close-style-dlg").addEventListener("click", () => closeOverlay("#style-dlg"));
$("#style-dlg").addEventListener("click", (ev) => { if (ev.target.id === "style-dlg") closeOverlay("#style-dlg"); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeOverlay("#style-dlg"); });

$("#form-style-lib").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = ev.target, mode = styleDlg.mode;
  if (!f.words.value.trim()) return toast("A style needs its words (the look).", true);
  const name = f.name.value.trim();
  if (mode !== "edit" && !/^[a-z0-9][a-z0-9_-]{0,63}$/.test(name)) return toast("Name: lowercase letters, digits, - and _ (at most 64).", true);
  const fd = new FormData();
  if (mode !== "edit") fd.append("name", name);
  for (const k of ["label", "words", "light", "negative_image"]) fd.append(k, f[k].value.trim());
  fd.append("lora_name", f.lora_name.value);
  fd.append("lora_strength", f.lora_strength.value || "0.4");
  if (f.reference.files.length) fd.append("reference", f.reference.files[0]);
  else if (styleDlg.removeRef && mode === "edit") fd.append("remove_reference", "1");
  else if (mode === "dup" && !styleDlg.removeRef) fd.append("copy_reference_from", styleDlg.from);
  const url = mode === "edit" ? `/api/styles/${encodeURIComponent(styleDlg.name)}/edit` : "/api/styles/new";
  try {
    const r = await api(url, { method: "POST", body: fd });
    toast(mode === "edit" ? `Saved: version ${r.style.version}.` : "Style made.");
    closeOverlay("#style-dlg");
    loadStylesTab();
  } catch (e) { fail(e); }
});

// ------------------------------------------------------------------ a reference picture made by the video model
// (cinestudio/style_ref.py): LTX-2.5 + the style's LoRA draw a still picture from text; the user picks one and Qwen's
// style conversions then get a tight crop of it as Picture 2.
const styleRef = { name: null };

async function openStyleRefDlg(name) {
  try { await loadStyleLib(true); } catch (e) { return fail(e); }
  const s = styleByName(name);
  if (!s) return toast(`The style "${name}" is gone.`, true);
  styleRef.name = name;
  const f = $("#form-style-ref");
  f.reset();
  $("#style-ref-name").textContent = s.label || s.name;
  $("#style-ref-what").textContent = s.lora
    ? `The video model draws the subject with ${s.lora.name} in this style's words. Pick one: style conversions then use a crop of it.`
    : "This style has no video LoRA: the pictures show the video model's own look for its words. Pick one: style conversions then use a crop of it.";
  $("#style-ref-cands").innerHTML = "";
  $("#style-ref-status").textContent = "";
  $("#style-ref-dlg").classList.remove("hidden");
  await refreshStyleRef(true);
}

function styleRefStatus(job) {
  if (!job) return { text: "", busy: false };
  if (job.status === "queued") return { text: job.position ? `Waiting in the queue · ${job.position} ahead` : "Waiting in the queue", busy: true };
  if (job.status === "running") return { text: `Making pictures… ${Math.round((job.progress || 0) * 100)}%`, busy: true };
  if (job.status === "stopping") return { text: "Stopping…", busy: true };
  if (job.status === "failed") return { text: `The last try failed: ${job.error || "unknown error"}`, busy: false };
  if (job.status === "interrupted") return { text: "The last try stopped before it ended.", busy: false };
  return { text: "", busy: false };
}

async function refreshStyleRef(fill = false) {
  const name = styleRef.name, dlg = $("#style-ref-dlg");
  if (!name || dlg.classList.contains("hidden")) return stopPoll("styleref");
  let d;
  try { d = await api(`/api/styles/${encodeURIComponent(name)}/reference-candidates`); } catch (e) { stopPoll("styleref"); return fail(e); }
  if (styleRef.name !== name) return;
  const f = $("#form-style-ref"), s = styleByName(name) || {};
  if (fill) {
    f.subject.value = (d.job && d.job.subject) || d.subject;
    f.seeds.value = (d.seeds || []).join(", ");
  }
  const st = styleRefStatus(d.job);
  $("#style-ref-status").textContent = st.text;
  $("#btn-style-ref-make").disabled = st.busy;
  const cur = s.reference_source;
  $("#style-ref-cands").innerHTML = d.candidates.map((c) => `<div class="st-cand${c.file === cur ? " current" : ""}">
      <img src="${esc(c.url)}" alt="Seed ${esc(c.seed)}" title="Click to view large" data-url="${esc(c.url)}" loading="lazy">
      <div class="st-cand-foot"><span class="muted">seed ${esc(c.seed)}</span>${c.file === cur ? `<span class="chip ready">In use</span>`
        : `<button type="button" data-file="${esc(c.file)}">Use this one</button>`}</div></div>`).join("");
  $$("#style-ref-cands img").forEach((im) => im.addEventListener("click", () => openPanoViewer(im.dataset.url, im.alt, true)));
  $$("#style-ref-cands button[data-file]").forEach((b) => b.addEventListener("click", async () => {
    try {
      await postJSON(`/api/styles/${encodeURIComponent(name)}/use-reference`, { file: b.dataset.file });
      toast("Reference picture set.");
      await loadStyleLib(true);
      loadStylesTab();
      refreshStyleRef();
    } catch (e) { fail(e); }
  }));
  if (st.busy) poll("styleref", () => refreshStyleRef(), 5000); else stopPoll("styleref");
}

$("#form-style-ref").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = ev.target, name = styleRef.name;
  const seeds = f.seeds.value.split(/[\s,]+/).filter(Boolean).map(Number);
  if (!seeds.length || seeds.some((n) => !Number.isInteger(n))) return toast("Seeds: whole numbers, for example 202, 303.", true);
  try {
    await postJSON(`/api/styles/${encodeURIComponent(name)}/make-reference`, { subject: f.subject.value.trim(), seeds });
    toast("Queued.");
    refreshStyleRef();
  } catch (e) { fail(e); }
});
$("#btn-close-style-ref").addEventListener("click", () => { stopPoll("styleref"); closeOverlay("#style-ref-dlg"); });
$("#style-ref-dlg").addEventListener("click", (ev) => { if (ev.target.id === "style-ref-dlg") { stopPoll("styleref"); closeOverlay("#style-ref-dlg"); } });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") { stopPoll("styleref"); closeOverlay("#style-ref-dlg"); } });

// ------------------------------------------------------------------ the character's skins (Characters tab)
// Skins (user, 2026-10-25: "think of it as like switching skins for a character"): a small glass button floats over
// the top-left corner of the canvas (user: "a drop down that hovers over our sheets ... that way it doesnt offset our
// sheet"; it takes no room in the layout); it opens a menu of the skins (picture, style, status, the one on show
// checked; a skin being made shows its job's progress), "+ New skin…" (a dialog of the library styles) and the rare
// actions. docs/skin_picker_research.md
const skinUi = { open: false, focusUntil: 0, backTo: null };
const SKIN_STEPS = 4;                                // the front redrawn + right, back, left (an unposed front: one more)
// the skin menu serves both kinds of the Cast tab (2026-10-25: Characters | Props): what differs per kind. A prop's
// skin is one picture (one Qwen job) and has no views, sheet or mannequin.
function skinCtx() {
  if (state.castMode === "props" && state.prop) {
    return { kind: "prop", cur: state.prop, api: "/api/props", noun: "prop", steps: 1,
             open: (n) => openProp(n), reload: (sel) => loadProps(sel), clear: () => clearProp(),
             done: "New prop skin made: check its picture." };
  }
  return { kind: "character", cur: state.char, api: "/api/characters", noun: "character", steps: SKIN_STEPS,
           open: (n) => openChar(n), reload: (sel) => loadChars(sel), clear: () => clearChar(), done: SKIN_DONE };
}
const skinReady = (x) => x.status === "ready" && !(x.missing || []).length;
const SKIN_SVG = {
  check: `<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path fill="currentColor" d="M9.5 16.2 5.3 12l-1.4 1.4 5.6 5.6L20.1 8.4 18.7 7z"/></svg>`,
  caret: `<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path fill="currentColor" d="M7 9.5h10L12 15z"/></svg>`,
  plus: `<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6z"/></svg>`,
  search: `<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M10 4a6 6 0 0 1 4.8 9.6l4.8 4.8-1.4 1.4-4.8-4.8A6 6 0 1 1 10 4zm0 2a4 4 0 1 0 0 8 4 4 0 0 0 0-8z"/></svg>`,
};
function skinSwatch(name) {                          // a steady colour for a style without a picture
  let h = 0;
  for (const ch of String(name || "")) h = (h * 31 + ch.codePointAt(0)) % 360;
  return `hsl(${h} 38% 30%)`;
}
const firstChar = (t) => [...String(t || "?").trim()][0] || "?";    // a whole character (CJK, accents, emoji)
// a picture, or the same-size placeholder: the initial on a colour of the style
const skinPic = (url, label, key, cls) => url
  ? `<span class="${cls}"><img src="${esc(url)}" alt="" loading="lazy" draggable="false"></span>`
  : `<span class="${cls} none" style="background:${skinSwatch(key || label)}"><b>${esc(firstChar(label))}</b></span>`;
const styleLabelOf = (name) => { const s = styleByName(name); return s ? s.label || s.name : name || "new skin"; };

// the progress line of a new_skin job (queued: the place in the queue; running: which picture of the job)
function skinJobText(j) {
  if (!j || j.status === "queued") return { text: j && j.queue_position ? `waiting · ${j.queue_position} ahead` : "waiting in the queue", frac: 0 };
  const n = Math.max(1, (j.events || []).filter((e) => e.event === "comfy_job").length);
  const p = j.progress, step = p && p.max ? p.value / p.max : 0;
  const total = Math.max(skinCtx().steps, n);
  return { text: `picture ${n} of ~${total}`, frac: Math.min(0.98, (n - 1 + step) / total) };
}

// the skin families's view: the skins, the jobs that make new ones (a record may exist already), this skin's menu
function skinModel(c) {
  const fam = c.skins && c.skins.length ? c.skins : [c];
  const main = fam.find((x) => !x.skin_of) || fam[0];
  const live = c.skin_jobs || [];
  const making = new Map(), pending = [];
  for (const j of live) {
    const rec = fam.find((x) => x.skin_of && x.style_ref && x.style_ref === j.style && !skinReady(x));
    if (rec && !making.has(rec.name)) making.set(rec.name, j); else pending.push(j);
  }
  const actions = [];
  if (!c.style_ref) actions.push(["tag", "Set which style this skin is drawn in…"]);
  if (c.skin_of) actions.push(["delete", "Delete this skin…"]);
  return { fam, main, live, making, pending, actions };
}

// renderChar draws the canvas, then this: a box of no height at the top of the canvas holds the floating button, so
// nothing in the canvas moves. Hidden on the Mannequin view (its own layout has a note line there) and on the
// candidates (a skin needs a picked front; the button would cover the first candidate's head).
function charSkinSwitch(c) {
  const canvas = $("#char-canvas");
  if (!canvas || !c) return;
  let host = $("#skin-switch", canvas);
  if (!host) { host = document.createElement("div"); host.id = "skin-switch"; host.className = "skin-switch"; canvas.prepend(host); }
  host.dataset.char = c.name;
  const prop = skinCtx().kind === "prop";
  const hide = prop ? state.propView === "candidates" || !(c.urls || {}).front
    : state.view === "mannequin" || state.view === "candidates";   // its own layout / before a front exists
  host.classList.toggle("hidden", hide);
  host.classList.toggle("compact", !prop && (state.view === "views" || state.view === "front"));   // narrow pictures: the avatar and the caret only
  if (hide) { skinUi.open = false; stopPoll("skins"); host.innerHTML = ""; return; }
  const draw = () => {
    if (host.dataset.char !== c.name || !host.isConnected) return;
    const m = skinModel(c), label = skinStyleLabel(c);
    const thumb = (c.urls || {}).front;
    const keep = skinUi.open && document.activeElement && host.contains(document.activeElement) ? document.activeElement.dataset.key : null;
    const hadTrigger = !!(document.activeElement && document.activeElement.id === "skin-trigger");   // a redraw keeps the focus
    host.innerHTML = `<button type="button" class="skin-trigger" id="skin-trigger" aria-haspopup="menu" aria-expanded="${skinUi.open}"
        aria-controls="skin-pop" title="${esc(`Skin: ${label}${m.fam.length > 1 ? ` (${m.fam.length} skins)` : ""}. The same ${skinCtx().noun} in other styles: switch or make a new one.`)}">
        ${thumb ? `<span class="skin-av" style="background-image:url('${esc(thumb)}')"></span>` : `<span class="skin-av none" style="background:${skinSwatch(c.style_ref || label)}">${esc(firstChar(label))}</span>`}
        <span class="skin-tlabel">${esc(label)}</span>${m.fam.length > 1 ? `<span class="skin-tn" aria-label="${m.fam.length} skins">${m.fam.length}</span>` : ""}
        ${m.live.length ? `<span class="spinner" title="A new skin is being made"></span>` : ""}${SKIN_SVG.caret}</button>
      ${skinUi.open ? skinPopHtml(c, m) : ""}`;
    bindSkinSwitch(host, c, m);
    placeSkinTrigger();
    $$("#char-canvas .hero img, #char-canvas .views .view:first-child img").forEach((im) => { if (!im.complete) im.addEventListener("load", placeSkinTrigger, { once: true }); });
    if (skinUi.open) { placeSkinPop(); const f = keep && $(`[data-key="${CSS.escape(keep)}"]`, host); if (f) f.focus(); }
    // back on the button after a switch (the canvas is drawn again, maybe more than once) or a redraw
    const lost = !document.activeElement || document.activeElement === document.body;
    if (!skinUi.open && (hadTrigger || Date.now() < skinUi.focusUntil && lost)) $("#skin-trigger", host).focus();
    if (m.live.length) { skinJobsTick(c); poll("skins", () => skinJobsTick(c), 4000); } else stopPoll("skins");
  };
  draw();
  loadStyleLib().then(draw).catch(() => {});
}

// the button sits on the displayed picture's own top-left corner (user: "on top of the sheet's top left corner"), over
// its black margin: the sheet's display frame keeps room for it above the bodies (fitSheetView); it follows the picture
// on load, resize (the canvas's ResizeObserver) and view changes
const SKIN_INSET = 12;
const skinTriggerRoom = () => { const b = $("#skin-trigger"); return SKIN_INSET + (b && b.offsetHeight || 34) + 10; };
function placeSkinTrigger() {
  const host = $("#skin-switch"), btn = $("#skin-trigger");
  if (!host || !btn || host.classList.contains("hidden")) return;
  const grid = skinCtx().kind === "character" && state.view === "views";
  const pic = grid ? $("#char-canvas .views .view") : $("#char-canvas .hero img");
  const r = pic && pic.getBoundingClientRect(), h = host.getBoundingClientRect();
  if (!r || !r.width || !r.height) { btn.style.left = ""; btn.style.top = ""; btn.style.maxWidth = ""; }
  else {
    const inset = grid ? 6 : SKIN_INSET;
    btn.style.left = `${Math.round(r.left - h.left + inset)}px`;
    btn.style.top = `${Math.round(r.top - h.top + inset)}px`;
    btn.style.maxWidth = `${Math.max(48, Math.min(260, Math.floor(r.width - 2 * inset)))}px`;
  }
  if (skinUi.open) placeSkinPop();
}

function skinPopHtml(c, m) {
  const s = c.style_ref ? styleByName(c.style_ref) : null;
  const stale = s && (+c.style_version || 0) < (+s.version || 1);
  const many = m.fam.length > 1;
  const row = (x) => {
    const cur = x.name === c.name, job = m.making.get(x.name), label = skinStyleLabel(x);
    const sub = [many && !x.skin_of ? "main" : "", job ? "" : x.status === "ready" ? readyText(x) : "not finished yet"].filter(Boolean).join(" · ");
    return `<div class="skin-opt${cur ? " current" : ""}" role="menuitemradio" aria-checked="${cur}" tabindex="-1" data-key="skin:${esc(x.name)}"
        data-skin="${esc(x.name)}" title="${esc(`${label}${x.skin_of ? "" : " (the main skin)"}\n${job ? "being made" : readyText(x)}`)}">
      ${skinPic(x.name === c.name ? (c.urls || {}).front || x.thumb_url : x.thumb_url, label, x.style_ref || x.name, "skin-opt-pic")}
      <span class="skin-opt-text"><span class="skin-opt-label"><span class="state ${skinReady(x) ? "ready" : ""}" aria-hidden="true"></span>${esc(label)}</span>
        ${job ? skinProgress(job) : `<small>${esc(sub)}</small>`}</span>
      <span class="skin-opt-check" aria-hidden="true">${cur ? SKIN_SVG.check : ""}</span></div>`;
  };
  const wait = (j) => `<div class="skin-opt making" role="menuitem" aria-disabled="true" tabindex="-1" data-key="job:${esc(j.id)}"
      title="${esc(`New skin: ${styleLabelOf(j.style)}. The app opens it when it is done.`)}">
      <span class="skin-opt-pic none"><span class="spinner"></span></span>
      <span class="skin-opt-text"><span class="skin-opt-label">${esc(styleLabelOf(j.style))}</span>${skinProgress(j)}</span><span class="skin-opt-check"></span></div>`;
  return `<div class="skin-pop" id="skin-pop" role="menu" aria-labelledby="skin-pop-head">
      <div class="skin-pop-head" id="skin-pop-head" title="${esc(shownName(m.main))}">Skins of ${esc(shownName(m.main))}</div>
      <div class="skin-pop-list">${m.fam.map(row).join("")}${m.pending.map(wait).join("")}</div>
      ${stale ? `<div class="skin-pop-note">This skin was made with an older version of its style (v${esc(c.style_version || 0)} of v${esc(s.version)}).</div>` : ""}
      ${c.style_ref && !s && styleLib.list.length ? `<div class="skin-pop-note">Its style "${esc(c.style_ref)}" is not in the library.</div>` : ""}
      <div class="skin-pop-foot">
        <div role="separator" class="skin-sep"></div>
        <div class="skin-opt act" role="menuitem" tabindex="-1" data-key="act:new" data-act="new" title="The same ${esc(skinCtx().noun)} in another style: a GPU job">
          <span class="skin-opt-pic plus">${SKIN_SVG.plus}</span><span class="skin-opt-text"><span class="skin-opt-label">New skin…</span></span><span class="skin-opt-check"></span></div>
        ${m.actions.length ? `<div role="separator" class="skin-sep"></div>${m.actions.map(([k, t]) =>
          `<div class="skin-opt small${k === "delete" ? " danger" : ""}" role="menuitem" tabindex="-1" data-key="act:${k}" data-act="${k}">${esc(t)}</div>`).join("")}` : ""}
      </div></div>`;
}
const skinProgress = (j) => { const t = skinJobText(j); return `<span class="skin-prog" data-job="${esc(j.id)}"><small class="skin-ptext">${esc(t.text)}</small>
    <span class="skin-pbar"><i style="width:${Math.round(t.frac * 100)}%"></i></span></span>`; };

// the menu hovers over the canvas (fixed: the canvas scrolls and would clip it); it stays inside the window, flips up
// when there is no room below, and its list scrolls (New skin and the actions stay at the bottom)
function placeSkinPop() {
  const pop = $("#skin-pop"), btn = $("#skin-trigger");
  if (!pop || !btn) return;
  const r = btn.getBoundingClientRect(), sb = $(".statusbar");
  const bottom = Math.min(window.innerHeight, sb ? sb.getBoundingClientRect().top : window.innerHeight) - 8;
  const w = Math.min(320, window.innerWidth - 16);
  pop.style.width = `${w}px`;
  pop.style.left = `${Math.round(Math.max(8, Math.min(r.left, window.innerWidth - w - 8)))}px`;
  const below = bottom - (r.bottom + 6), above = r.top - 8 - 8;
  const up = below < 240 && above > below;
  pop.style.maxHeight = `${Math.round(Math.max(160, up ? above : below))}px`;
  pop.style.top = up ? "" : `${Math.round(r.bottom + 6)}px`;
  pop.style.bottom = up ? `${Math.round(window.innerHeight - r.top + 6)}px` : "";
  const cur = $(".skin-opt.current", pop);
  if (cur) { const list = $(".skin-pop-list", pop); list.scrollTop = Math.max(0, cur.offsetTop - list.clientHeight / 2 + cur.offsetHeight / 2); }
}

function setSkinPop(open, focus) {                    // focus: "current" | "last" | null
  skinUi.open = open;
  const c = skinCtx().cur;
  if (c) charSkinSwitch(c);
  if (!open) return;
  const items = $$("#skin-pop [role^=menuitem]");
  const f = focus === "last" ? items[items.length - 1] : $("#skin-pop .skin-opt.current") || items[0];
  if (f) f.focus();
}
function closeSkinPop(refocus) {
  if (!skinUi.open) return false;
  skinUi.open = false;
  const pop = $("#skin-pop"), btn = $("#skin-trigger");
  if (pop) pop.remove();
  if (btn) { btn.setAttribute("aria-expanded", "false"); if (refocus) btn.focus(); }
  return true;
}
document.addEventListener("click", (ev) => { if (skinUi.open && !ev.target.closest("#skin-switch")) closeSkinPop(false); });
window.addEventListener("resize", () => { if (skinUi.open) placeSkinPop(); });
document.addEventListener("scroll", (ev) => {                       // the page moved under the fixed menu: close it
  if (skinUi.open && !(ev.target.closest && ev.target.closest("#skin-pop"))) closeSkinPop(false);
}, true);

function bindSkinSwitch(host, c, m) {
  const btn = $("#skin-trigger", host);
  btn.addEventListener("click", (ev) => { ev.stopPropagation(); if (skinUi.open) closeSkinPop(false); else setSkinPop(true, "current"); });
  btn.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowDown" || ev.key === "ArrowUp") { ev.preventDefault(); setSkinPop(true, ev.key === "ArrowUp" ? "last" : "current"); }
  });
  const pop = $("#skin-pop", host);
  if (!pop) return;
  const items = $$("[role^=menuitem]", pop);
  const act = async (el) => {
    if (el.getAttribute("aria-disabled") === "true") return;
    const name = el.dataset.skin, what = el.dataset.act, ctx = skinCtx();
    if (name) {
      closeSkinPop(name === c.name);
      if (name !== c.name) { skinUi.focusUntil = Date.now() + 3000; ctx.open(name); }
      return;
    }
    closeSkinPop(false);
    if (what === "new") return openSkinDlg("new", c, btn);
    if (what === "tag") return openSkinDlg("tag", c, btn);
    if (what === "delete") { if (await deleteItem(ctx.kind, c.name)) { ctx.clear(); ctx.reload(charBase(c)); } else btn.focus(); }
  };
  items.forEach((el, i) => {
    el.addEventListener("click", (ev) => { ev.stopPropagation(); act(el); });
    el.addEventListener("keydown", (ev) => {
      const k = ev.key;
      if (k === "Enter" || k === " ") { ev.preventDefault(); act(el); return; }
      if (k === "Escape") { ev.preventDefault(); ev.stopPropagation(); closeSkinPop(true); return; }
      if (k === "Tab") { closeSkinPop(false); return; }
      const to = k === "ArrowDown" ? (i + 1) % items.length : k === "ArrowUp" ? (i - 1 + items.length) % items.length
        : k === "Home" ? 0 : k === "End" ? items.length - 1 : -1;
      if (to < 0) return;
      ev.preventDefault();
      items[to].focus();                             // a focused row in the list scrolls into view by itself
    });
  });
}

// the skins being made: their job's progress in place; at the end the new skin opens
async function skinJobsTick(c) {
  const ctx = skinCtx(), cur = () => skinCtx().cur;
  if (!cur() || charBase(cur()) !== charBase(c) || !$("#skin-switch")) { stopPoll("skins"); return; }
  for (const sj of c.skin_jobs || []) {
    let j;
    try { j = await api(`/api/jobs/${encodeURIComponent(sj.id)}`); } catch { continue; }
    if (!cur() || charBase(cur()) !== charBase(c)) return;
    if (j.status === "queued" || j.status === "running") {
      const t = skinJobText(j);
      $$(`#skin-switch [data-job="${sj.id}"]`).forEach((box) => {
        $(".skin-ptext", box).textContent = t.text;
        $(".skin-pbar i", box).style.width = `${Math.round(t.frac * 100)}%`;
      });
      continue;
    }
    // ended: the open skin's own job bar (watchJob) handles a job of the skin on show; this handles the others
    if (cur().name === (j.character || j.prop) && state.polls.job) return;
    stopPoll("skins");
    toast(j.status === "done" ? ctx.done : `New skin: ${j.status}${j.error ? ` — ${j.error}` : ""}`, j.status !== "done");
    await ctx.reload();
    if (cur() && charBase(cur()) === charBase(c)) ctx.open((j.status === "done" && j.result && j.result.skin) || cur().name);
    return;
  }
}
const SKIN_DONE = "New skin made: check its sheet, then press \"Use as is\".";

// ------------------------------------------------------------------ the New skin dialog (also: name a skin's style)
function skinDlgEl() {
  let d = $("#skin-dlg");
  if (d) return d;
  d = document.createElement("div");
  d.id = "skin-dlg"; d.className = "overlay hidden";
  d.setAttribute("role", "dialog"); d.setAttribute("aria-modal", "true"); d.setAttribute("aria-labelledby", "skin-dlg-title");
  d.innerHTML = `<div class="modal"><div class="modal-head"><h2 id="skin-dlg-title"></h2>
      <button type="button" class="close" data-close title="Close (Esc)" aria-label="Close">${ICON.close}</button></div>
    <div class="skin-dlg-body"></div></div>`;
  document.body.append(d);
  d.addEventListener("click", (ev) => { if (ev.target === d || ev.target.closest("[data-close]")) closeSkinDlg(); });
  d.addEventListener("keydown", (ev) => { if (ev.key === "Escape") { ev.stopPropagation(); closeSkinDlg(); } });
  return d;
}
function closeSkinDlg() {
  if (!closeOverlay("#skin-dlg")) return;
  if (skinUi.backTo && skinUi.backTo.isConnected) skinUi.backTo.focus();
  skinUi.backTo = null;
}
const SKIN_SEARCH_FROM = 9;                          // more styles than this: a filter field

async function openSkinDlg(mode, c, from) {
  try { await loadStyleLib(true); } catch (e) { return fail(e); }
  const d = skinDlgEl(), body = $(".skin-dlg-body", d);
  skinUi.backTo = from || null;
  const m = skinModel(c);
  const taken = new Map();                           // style -> why it cannot be chosen (one skin per style)
  m.fam.forEach((x) => { if (x.style_ref) taken.set(x.style_ref, x.name === c.name ? (mode === "new" ? "this skin" : null) : "already a skin"); });
  if (mode === "new") m.live.forEach((j) => { if (j.style && !taken.get(j.style)) taken.set(j.style, "being made"); });
  for (const [k, v] of [...taken]) if (!v) taken.delete(k);
  const list = styleLib.list;
  const free = list.filter((s) => !taken.has(s.name)), used = list.filter((s) => taken.has(s.name));
  let sel = (free.find((s) => s.default) || free[0] || {}).name || null;
  const noFront = mode === "new" && !(c.urls || {}).front;
  const ahead = (state.queue.items || []).length;
  const who = shownName(m.main);
  $("#skin-dlg-title").innerHTML = mode === "new" ? `<span class="t">New skin</span> <span class="muted" title="${esc(who)}">of ${esc(who)}</span>`
    : `<span class="t">Which style is this skin drawn in?</span>`;
  const card = (s) => {
    const why = taken.get(s.name), label = s.label || s.name;
    return `<div class="skin-style${why ? " off" : ""}" role="radio" aria-checked="${s.name === sel}" aria-disabled="${!!why}" tabindex="${s.name === sel ? 0 : -1}"
        data-style="${esc(s.name)}" data-find="${esc(`${label} ${s.name} ${s.words || ""}`.toLowerCase())}" title="${esc(`${label}\n${s.words || ""}`)}">
      <span class="ss-pic-wrap">${skinPic(s.reference_url, label, s.name, "ss-pic")}<span class="skin-check">${SKIN_SVG.check}</span></span>
      <span class="ss-text"><b>${esc(label)}</b>
        <span class="ss-words">${esc(s.words || "")}</span>
        <span class="ss-tags">${why ? `<span class="chip">${esc(why)}</span>` : ""}${s.default ? `<span class="chip ready">default</span>` : ""}${s.lora ? `<span class="chip" title="The video LoRA of this style">LoRA</span>` : ""}</span></span></div>`;
  };
  const shown = mode === "new" && !free.length ? [] : list;    // nothing to choose: the empty state alone
  const cols = Math.max(1, Math.min(3, Math.max(free.length, shown.length ? used.length : 0)));   // 1-3 columns: the dialog fits them
  const search = shown.length >= SKIN_SEARCH_FROM;
  const empty = !list.length ? `<div class="skin-empty"><b>The style library is empty.</b><span>Make a style first; it can then be a skin.</span></div>`
    : mode === "new" && !free.length ? `<div class="skin-empty"><b>Every style is already a skin of this character.</b><span>Make another style, then come back.</span></div>` : "";
  body.innerHTML = `
    <p class="skin-intro">${mode === "new" ? (skinCtx().kind === "prop" ? "Redraws this prop in another style. Its shape stays." : "Redraws this character in another style. The pose stays.")
      : "Only the label and the style words change; the pictures stay."}</p>
    ${search ? `<label class="skin-search">${SKIN_SVG.search}<input type="search" placeholder="Filter ${shown.length} styles" aria-label="Filter the styles" autocomplete="off" spellcheck="false"></label>` : ""}
    <div class="skin-scroll">${empty}
      ${shown.length ? `<div role="radiogroup" aria-label="Styles">
        ${free.length ? `<div class="skin-styles" style="--cols:${cols}">${free.map(card).join("")}</div>` : ""}
        ${used.length ? `${free.length ? `<div class="skin-group-label">${mode === "new" ? "Already skins of this character" : "Used by other skins"}</div>` : ""}
          <div class="skin-styles" style="--cols:${cols}">${used.map(card).join("")}</div>` : ""}</div>
        <p class="skin-nomatch hidden">No style matches.</p>` : ""}
      ${noFront ? `<p class="hint-box warn">This skin has no front view to redraw: pick a front first.</p>` : ""}
    </div>
    <div class="skin-foot">
      ${mode === "new" && free.length ? `<div class="skin-cost">${skinCtx().steps > 1 ? `About ${skinCtx().steps} image jobs` : "One image job"} · added to the queue${ahead ? ` (${ahead} ahead)` : ""}</div>` : ""}
      <div class="skin-btns">${free.length || mode === "tag" ? `<button type="button" class="ghost" data-styles-tab title="The style library">Manage styles</button>` : ""}
        <span class="skin-picked" aria-live="polite"></span>
        <button type="button" data-close>Cancel</button>
        ${shown.length ? `<button type="button" class="primary" data-go>${mode === "new" ? "Make the skin" : "Set the style"}</button>`
          : `<button type="button" class="primary" data-styles-tab>Make a style</button>`}</div>
    </div>`;
  const cards = $$(".skin-style", body), go = $("[data-go]", body), picked = $(".skin-picked", body);
  const sync = () => {                                // the choice, without drawing again (keeps the scroll and the focus)
    cards.forEach((el) => {
      const on = el.dataset.style === sel;
      el.setAttribute("aria-checked", on); el.tabIndex = on ? 0 : -1;
    });
    if (go) go.disabled = !sel || noFront;
    const s = styleByName(sel);
    picked.textContent = s ? s.label || s.name : "";
    picked.title = picked.textContent;
  };
  const visible = () => cards.filter((el) => !el.classList.contains("hidden") && el.getAttribute("aria-disabled") !== "true");
  const choose = (el) => {
    if (!el || el.getAttribute("aria-disabled") === "true") return;
    sel = el.dataset.style; sync();
    el.focus(); el.scrollIntoView({ block: "nearest" });
  };
  const gridCols = () => { const g = $(".skin-styles", body); return g ? getComputedStyle(g).gridTemplateColumns.split(" ").filter(Boolean).length : 1; };
  cards.forEach((el) => {
    el.addEventListener("click", () => choose(el));
    el.addEventListener("keydown", (ev) => {
      if (ev.key === " " || ev.key === "Enter") { ev.preventDefault(); choose(el); return; }     // Enter chooses; only the button makes
      const v = visible(), i = v.indexOf(el), n = gridCols();
      const step = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: n, ArrowUp: -n }[ev.key];
      let to = -1;
      if (step) to = Math.max(0, Math.min(v.length - 1, (i < 0 ? 0 : i) + step));
      if (ev.key === "Home") to = 0;
      if (ev.key === "End") to = v.length - 1;
      if (to < 0 || !v.length) return;
      ev.preventDefault();                           // arrows move AND choose (APG radio group); taken styles are skipped
      choose(v[to]);
    });
  });
  const input = $(".skin-search input", body);
  if (input) {
    input.addEventListener("input", () => {
      const q = input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
      let n = 0;
      cards.forEach((el) => { const hit = q.every((w) => el.dataset.find.includes(w)); el.classList.toggle("hidden", !hit); if (hit) n++; });
      const lab = $(".skin-group-label", body);
      const seen = (off) => cards.some((el) => !el.classList.contains("hidden") && el.classList.contains("off") === off);
      if (lab) lab.classList.toggle("hidden", !(seen(true) && seen(false)));   // the heading only between both groups
      $(".skin-nomatch", body).classList.toggle("hidden", n > 0);
    });
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowDown") { ev.preventDefault(); const v = visible(); choose(v.find((el) => el.dataset.style === sel) || v[0]); }
      if (ev.key === "Enter") ev.preventDefault();   // never a submit from the filter
    });
  }
  $$("[data-styles-tab]", body).forEach((b) => b.addEventListener("click", () => { closeSkinDlg(); showTab("styles"); }));
  if (go) go.addEventListener("click", () => (mode === "new" ? makeSkin(c, sel) : tagSkin(c, sel)));
  sync();
  d.classList.remove("hidden");
  setTimeout(() => {
    const s = cards.find((el) => el.dataset.style === sel);
    if (s) s.scrollIntoView({ block: "nearest" });
    const f = input || s || $("[data-go]", body) || $("[data-styles-tab]", body) || $("[data-close]", body);
    if (f) f.focus();
  }, 30);
}

async function makeSkin(c, style) {
  const go = $("#skin-dlg [data-go]");
  if (go) go.disabled = true;
  const fd = new FormData(); fd.append("style", style);
  const ctx = skinCtx();
  try {
    await api(`${ctx.api}/${encodeURIComponent(c.name)}/new-skin`, { method: "POST", body: fd });
    closeSkinDlg();
    queued(`New skin (${styleLabelOf(style)})`);
    ctx.open(c.name);                                 // the menu gets the job (its row) and the job bar
  } catch (e) { fail(e); if (go) go.disabled = false; }
}

async function tagSkin(c, style) {
  const ctx = skinCtx();
  try {
    await postJSON(`${ctx.api}/${encodeURIComponent(c.name)}/use-style`, { style });
    closeSkinDlg();
    await ctx.reload();
    if (ctx.cur && ctx.cur.name === c.name) ctx.open(c.name);           // fresh skins: the new label
    toast(`This skin is marked as "${styleLabelOf(style)}".`);
  } catch (e) { fail(e); }
}

// ------------------------------------------------------------------ the Style selects of the other tabs
// the New background and New video forms: a Style select that fills the style words
function fillStylePresets() {
  $$(".style-preset-select").forEach((sel) => {
    if (document.activeElement === sel) return;
    const cur = sel.value;
    sel.innerHTML = styleOptionsHtml(cur, sel.id === "video-style-preset" ? "Own words (the Style text)" : "None (own words)");
    sel.value = styleByName(cur) ? cur : "";
  });
}
const styleTextOf = (name) => (styleByName(name) || {}).text || "";

$("#bg-style-preset").addEventListener("change", (ev) => {
  const t = styleTextOf(ev.target.value);
  if (t) $("#form-bg").style.value = t;
});

function videoStyleHint() {
  const f = $("#form-video"), s = styleByName(f.style_preset.value), hint = $("#video-style-hint");
  if (!s) { hint.textContent = ""; return; }
  if (!styleSame(f.style.value, s.text) && f.style.value.trim()) {
    hint.textContent = "The Style text was changed: your own words are used, not the style.";
    hint.classList.add("warn");
    return;
  }
  hint.classList.remove("warn");
  hint.textContent = `Uses "${s.label || s.name}" (version ${s.version})` + (s.lora ? ` with its LoRA ${s.lora.name} × ${s.lora.strength} unless you choose another LoRA.` : ".");
}
// a new character in the form: its own library style, unless the user chose one
function videoStyleSync() {
  const f = $("#form-video"), sel = f.style_preset;
  fillStylePresets();
  if (sel.dataset.user !== "1") {
    const c = (state.videoChars || []).find((x) => x.name === f.character.value);
    const ref = c && c.style_ref && styleByName(c.style_ref) ? c.style_ref : "";
    sel.value = ref;
    if (ref && (!f.style.value.trim() || f.style.value === f.style.dataset.auto)) {
      f.style.value = styleTextOf(ref); f.style.dataset.auto = f.style.value;
    }
  }
  videoStyleHint();
}
$("#video-style-preset").addEventListener("change", (ev) => {
  const f = $("#form-video"), sel = ev.target;
  sel.dataset.user = "1";
  const t = styleTextOf(sel.value);
  if (t) { f.style.value = t; f.style.dataset.auto = ""; }       // a chosen style is not replaced by a character change
  videoStyleHint();
});
$("#form-video").style.addEventListener("input", videoStyleHint);
$("#video-char-select").addEventListener("change", videoStyleSync);

// a dialog that opens: fresh style lists (the New video form also takes the character's style)
function watchStyleDialogs() {
  const obs = (id, fn) => new MutationObserver(() => { if (!$(id).classList.contains("hidden")) loadStyleLib().then(fn).catch(() => {}); })
    .observe($(id), { attributes: true, attributeFilter: ["class"] });
  obs("#new-bg", fillStylePresets);
  obs("#new-video", videoStyleSync);
}
watchStyleDialogs();
loadStyleLib().then(fillStylePresets).catch(() => {});

// the run's style warnings (assets that name another style than the run) beside the run's note
function runStyleWarnings(d) {
  const w = (d && d.style_warnings) || [];
  return w.length ? `<div class="note st-warn"><b>Style</b> ${w.map(esc).join("; ")}</div>` : "";
}
