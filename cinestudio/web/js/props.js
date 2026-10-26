// cinestudio web app - The Cast tab's Props (2026-10-25, user: "change the tab's name to something more broad, like
// actors"): a Characters | Props switch at the top of the library column; props use the same tiles, the same floating
// skin menu (styles.js skinCtx), the same edit tools (characters.js: the region box, the quality settings, the history)
// and the same job bar. A prop is one picture (front, optionally a side) + words: no views, sheet or mannequin.
// Plain scripts (no modules), loaded in order by index.html; they share their top-level names.
"use strict";

state.castMode = "characters";                       // the app opens on the characters (main.js opens the first one)
state.props = [];
state.prop = null;

// ------------------------------------------------------------------ the switch
function castRender() { if (state.castMode === "props" && state.prop) renderProp(); else if (state.char) renderChar(); }

function setCastMode(mode, open = true) {
  if (mode === state.castMode) return;
  if (mode === "props" && state.char && state.view === "mannequin" && !mqLeaveOk()) return;   // unsaved body changes
  state.castMode = mode;
  $$(".cast-switch [data-cast]").forEach((b) => {
    const on = b.dataset.cast === mode;
    b.classList.toggle("on", on); b.setAttribute("aria-selected", String(on));
  });
  const props = mode === "props";
  $("#btn-new-char").classList.toggle("hidden", props); $("#char-list").classList.toggle("hidden", props);
  $("#btn-new-prop").classList.toggle("hidden", !props); $("#prop-list").classList.toggle("hidden", !props);
  skinUi.open = false; state.sel = null; state.region = null; state.regionMode = false;
  if (props) {
    if (state.char) state.lastChar = state.char.name;
    clearChar();
    clearProp();
    loadProps().then(() => {
      if (!open || state.castMode !== "props" || state.prop) return;
      const back = state.lastProp && state.props.find((x) => x.name === state.lastProp);
      const first = back || state.props.find((x) => !x.skin_of) || state.props[0];
      if (first) openProp(first.name);
    });
  } else {
    if (state.prop) state.lastProp = state.prop.name;
    clearProp();
    clearChar();
    if (!open) return;
    const back = state.lastChar && (state.chars || []).find((x) => x.name === state.lastChar);
    const first = back || (state.chars || []).find((x) => !x.skin_of);
    if (first) openChar(first.name);
  }
}
$$(".cast-switch [data-cast]").forEach((b) => b.addEventListener("click", () => setCastMode(b.dataset.cast)));

// ------------------------------------------------------------------ the library column
const propFamily = (base) => (state.props || []).filter((x) => charBase(x) === base);
const propSkinKey = (base) => `cs.propskin.${base}`;
function activePropSkin(base) {                      // the skin this viewer looked at last (a browser convenience)
  try {
    const n = localStorage.getItem(propSkinKey(base));
    if (n && propFamily(base).some((x) => x.name === n)) return n;
  } catch { /* no storage: the main record */ }
  return base;
}
function rememberPropSkin(p) { try { localStorage.setItem(propSkinKey(charBase(p)), p.name); } catch { /* no storage */ } }
const propReady = (p) => p.status === "ready";
const propStatusText = (p) => (propReady(p) ? "ready for videos (Video tab > Objects in this template)" : p.status === "candidates" ? "pick a candidate" : p.status);

async function loadProps(select) {
  let rows;
  try { rows = await api("/api/props"); } catch (e) { return fail(e); }
  state.props = rows;
  state.videoProps = undefined;                      // the Video form's object rows read the library again
  const curBase = state.prop ? charBase(state.prop) : null;
  const names = new Set(rows.map((p) => p.name));
  const mains = rows.filter((p) => !p.skin_of || !names.has(p.skin_of));
  $("#prop-list").innerHTML = mains.map((m) => {
    const fam = propFamily(charBase(m)), n = fam.length;
    const want = curBase === charBase(m) ? state.prop.name : activePropSkin(charBase(m));
    const p = fam.find((x) => x.name === want) || m;
    const skinsNote = n > 1 ? ` · ${n} skins: ${fam.map(skinStyleLabel).join(", ")}` : "";
    return `<div class="tile prop ${curBase === charBase(m) ? "active" : ""}" data-name="${esc(p.name)}" data-base="${esc(charBase(m))}" role="button" tabindex="0" title="${esc(shownName(m))}${shownName(m) !== m.name ? ` (ID: ${esc(m.name)})` : ""}${esc(skinsNote)}\n${esc(m.words || "")}">
       ${p.thumb_url ? `<img src="${p.thumb_url}" alt="" loading="lazy">` : `<span class="initial">${esc(shownName(m)[0])}</span>`}
       <span class="state ${propReady(p) ? "ready" : ""}" title="${esc(propStatusText(p))}"></span>
       ${n > 1 ? `<span class="skins-n" title="${n} skins (the skin menu over the picture switches them)">${n}</span>` : ""}
       <span class="label">${esc(shownName(m))}</span>${trashButton("prop", m.name)}</div>`;
  }).join("");
  $$("#prop-list .tile").forEach((t) => {
    t.addEventListener("click", () => openProp(t.dataset.name));
    t.addEventListener("keydown", (ev) => { if (ev.key === "Enter") openProp(t.dataset.name); });
  });
  bindDeletes($("#prop-list"), (name) => {
    if (state.prop && state.prop.name === name) clearProp();
    loadProps();
  });
  if (select) openProp(select);
}

function clearProp() {
  state.prop = null; state.propView = null; state.sel = null; stopPoll("job"); stopPoll("skins");
  charPane().classList.add("no-panel");
  $("#char-head").innerHTML = ""; $("#char-desc").innerHTML = ""; $("#jobbar").innerHTML = ""; $("#char-panel").innerHTML = "";
  $("#char-canvas").innerHTML = `<div class="empty">${state.castMode === "props" ? "Choose a prop or press +" : "Choose a character or press +"}</div>`;
  $("#btn-new-prop").classList.remove("active");
}

async function openProp(name) {
  if (state.castMode !== "props") setCastMode("props", false);
  const seq = (state.propSeq = (state.propSeq || 0) + 1);
  let p;
  try { p = await api(`/api/props/${encodeURIComponent(name)}`); } catch (e) { return fail(e); }
  if (seq !== state.propSeq || state.castMode !== "props") return;   // another prop (or the characters) meanwhile
  if (!state.prop || state.prop.name !== name) { state.propView = null; state.sel = null; $("#jobbar").innerHTML = ""; }
  state.prop = p;
  rememberPropSkin(p);
  $("#btn-new-prop").classList.remove("active");
  const tile = $$("#prop-list .tile").find((t) => t.dataset.base === charBase(p));
  $$("#prop-list .tile").forEach((t) => t.classList.toggle("active", t === tile));
  const thumb = (p.urls || {}).front, img = tile && $("img", tile);
  if (!tile || tile.dataset.name !== name) loadProps();             // a new prop, or another skin of the same tile
  else if (thumb && img && img.getAttribute("src") !== thumb) img.setAttribute("src", thumb);
  else if (thumb && !img) loadProps();
  renderProp();
  const live = (p.jobs || []).find((j) => j.status === "queued" || j.status === "running");
  if (live) watchPropJob(live.id, name); else stopPoll("job");
}

// ------------------------------------------------------------------ the prop view
function propViews(p) {
  const u = p.urls || {}, v = [["front", "Picture", !!u.front]];
  if (u.side) v.push(["side", "Side", true]);
  if ((p.candidates || []).length) v.push(["candidates", "Candidates", true]);
  return v;
}

function renderProp() {
  const p = state.prop, u = p.urls || {};
  const views = propViews(p);
  if (!state.propView || !views.some(([k, , ok]) => k === state.propView && ok)) state.propView = u.front ? "front" : "candidates";
  const fam = (p.skins && p.skins.length > 1) ? p.skins : null;
  const main = fam ? fam.find((x) => !x.skin_of) || p : p;
  $("#char-head").innerHTML = `<span class="name ${main === p ? "editable" : ""}" title="${main === p ? `Click to rename (ID: ${esc(p.name)}; the ID does not change)` : `A skin of ${esc(shownName(main))} (ID: ${esc(p.name)}): rename the prop on its main skin`}">${esc(shownName(main))}</span><span class="chip ${esc(p.status)}">${esc(p.status)}</span>
    <div class="seg-switch">${views.map(([k, label, ok]) =>
      `<button data-view="${k}" class="${state.propView === k ? "on" : ""}" ${ok ? "" : "disabled"}>${label}</button>`).join("")}</div>`;
  $$("#char-head [data-view]").forEach((b) => b.addEventListener("click", () => {
    state.propView = b.dataset.view; state.sel = null; state.regionMode = false; renderProp();
  }));
  if (main === p) $("#char-head .name").addEventListener("click", startPropRename);
  $("#char-desc").innerHTML = `<div class="desc" title="The prop's words: every video prompt that uses it says them (click to show all)">${esc(p.words || "")}</div>`;
  $("#char-desc .desc").addEventListener("click", (ev) => ev.currentTarget.classList.toggle("open"));

  const sel = state.sel;
  if (sel && !u[sel]) state.sel = null;
  const img = (k) => `<img src="${u[k]}" alt="${esc(shownName(p))}: ${k}" data-target="${k}" class="pickable prop-pic-big ${state.sel === k ? "selected" : ""}" title="Click to edit">`;
  $("#char-canvas").innerHTML = state.propView === "candidates"
    ? `<div class="cands">${(p.candidates || []).map((x) =>
        `<button class="cand ${x.seed === p.seed ? "on" : ""}" data-seed="${x.seed}" title="Use seed ${x.seed}">
           <img src="${x.url}" alt="seed ${x.seed}" loading="lazy"><span>${x.seed}</span></button>`).join("")}</div>`
    : u[state.propView] ? `<div class="hero prop-hero">${img(state.propView)}</div>` : `<div class="empty">Making the candidates…</div>`;
  charSkinSwitch(p);                                 // the floating skin menu over the picture (styles.js)
  placeSkinTrigger();
  $$("#char-canvas .cand").forEach((t) => t.addEventListener("click", async () => {
    if (+t.dataset.seed === p.seed && u.front) return;
    if (u.front && !confirm(`Use seed ${t.dataset.seed} as the prop's picture?\nThe current picture goes to the history (undo brings it back).`)) return;
    try {
      state.prop = (await postJSON(`/api/props/${encodeURIComponent(p.name)}/pick`, { seed: +t.dataset.seed })).prop;
      state.propView = "front"; renderProp(); loadProps();
    } catch (e) { fail(e); }
  }));
  const panel = propPanel(p, u);
  $("#char-panel").innerHTML = panel;
  charPane().classList.toggle("no-panel", !panel);
  bindPropPanel(p);
}

function startPropRename() {
  const p = state.prop, span = $("#char-head .name");
  const inp = document.createElement("input");
  inp.className = "rename"; inp.value = shownName(p); inp.spellcheck = false; inp.maxLength = 80;
  inp.title = "Any name (spaces and capitals are fine). Enter to save, Esc to stop. The ID stays the same.";
  span.replaceWith(inp); inp.focus(); inp.select();
  let done = false;
  const finish = async (save) => {
    if (done) return; done = true;
    const v = inp.value.trim();
    if (!save || !v || v === shownName(p)) return renderProp();
    try {
      state.prop = (await postJSON(`/api/props/${encodeURIComponent(p.name)}/rename`, { new_name: v })).prop;
      toast(`Renamed to ${shownName(state.prop)}.`); await loadProps(); renderProp();
    } catch (e) { fail(e); renderProp(); }
  };
  inp.addEventListener("keydown", (ev) => { if (ev.key === "Enter") finish(true); if (ev.key === "Escape") finish(false); });
  inp.addEventListener("blur", () => finish(true));
}

const PROP_TARGET_LABEL = { front: "Picture", side: "Side picture" };
function propPanel(p, u) {
  const sel = state.sel;
  if (state.propView === "candidates") return "";    // click a candidate to use it
  if (sel) {                                         // a picture is selected: its edit tools (characters.js helpers)
    const last = lastChange(p, sel);
    return `<div class="block"><h2>${PROP_TARGET_LABEL[sel] || sel}<span class="muted" id="q-eta" title="About how long one change takes">${etaText()}</span>
        <button type="button" class="close" id="btn-desel" title="Close (Esc)" aria-label="Close">${ICON.close}</button></h2>
      <form id="form-edit" class="form">
        <input type="hidden" name="target" value="${sel}">
        <textarea name="instruction" rows="4" required
          placeholder="One change, for example: make the seam a little thicker. Keep everything else exactly the same.">${esc(draftOf(`prop:${p.name}`, sel))}</textarea>
        <label class="refbtn" title="Optional: ONE detail, tightly cropped. Call it &quot;Picture 2&quot; in the text (the prop is Picture 1).">${ICON.image}<span class="reftext">Reference image</span>
          <input type="file" name="reference" accept="image/png,image/jpeg,image/webp"></label>
        ${regionBlock(p, sel)}
        ${qualityBlock(false)}
        <div class="row"><button class="primary grow" type="submit">Apply</button>
          ${last ? `<button type="button" id="btn-undo" class="icon-btn" title="${esc(`Undo: ${last.instruction}`)}" aria-label="Undo">${ICON.undo}</button>` : ""}</div>
      </form></div>${historyBlock(p, sel)}`;
  }
  const s = p.style_ref && typeof styleByName === "function" ? styleByName(p.style_ref) : null;
  const stale = s && (+p.style_version || 0) < (+s.version || 1);
  const edits = (p.edits || []).filter((e) => (e.target || "front") === state.propView).length;
  return `<div class="block prop-info"><h2>Prop</h2>
      <form id="form-prop-words" class="form">
        <label class="fl" title="A few concrete words: every video prompt that uses this prop says them, and the run sheet's Prop line"><span>Words <i>every video prompt uses them</i></span>
          <input name="words" maxlength="300" required autocomplete="off" value="${esc(p.words || "")}"></label>
        <label class="fl" title="More detail for new pictures (a new skin redraws the picture with it). Finished pictures do not change."><span>Description <i>optional · the picture's detail</i></span>
          <textarea name="description" rows="3" maxlength="1200" placeholder="empty = the words">${esc(p.description || "")}</textarea></label>
        <div class="row"><button class="primary grow" type="submit" disabled>Save</button></div>
      </form>
      <div class="sub small prop-facts">
        <div><b>Style</b> ${esc(s ? s.label || s.name : p.style_ref || "own look")}${stale ? ` <span class="warnline">made with an older version (v${esc(p.style_version || 0)} of v${esc(s.version)})</span>` : ""}</div>
        <div>Click the picture to change it (one change per edit; every change can be undone)${edits ? ` · ${edits} change${edits === 1 ? "" : "s"} so far` : ""}.</div>
        <div>In a video: choose it in the Video tab's <i>Objects in this template</i>. Its picture goes into a prop panel of the run's sheet and its words into the clip texts.</div>
      </div></div>`;
}

function propSelect(target) {
  state.sel = state.sel === target ? null : target;
  state.regionMode = false;
  renderProp();
  const t = $("#form-edit textarea");
  if (t) t.focus();
}

function bindPropPanel(p) {
  const name = encodeURIComponent(p.name);
  $$("#char-canvas img.pickable").forEach((im) => im.addEventListener("click", () => {
    if (im.dataset.target === state.sel && (state.regionMode || Date.now() - (state.regionUp || 0) < 400)) return;   // a box drag
    propSelect(im.dataset.target);
  }));
  bindHistory(p);
  bindRegion(p);
  bindQuality();
  const desel = $("#btn-desel");
  if (desel) desel.addEventListener("click", () => propSelect(state.sel));
  const fe = $("#form-edit");
  if (fe) {
    fe.instruction.addEventListener("input", () => setDraft(`prop:${p.name}`, state.sel, fe.instruction.value));
    fe.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const target = state.sel, fd = new FormData(fe);
      if (!fd.get("region")) fd.delete("region");
      startPropJob(`/api/props/${name}/edit`, fd).then((ok) => {
        if (!ok) return;
        setDraft(`prop:${p.name}`, target, "");
        if (state.region && state.region.target === target) { state.region = null; state.regionMode = false; }
      });
    });
    const ref = fe.reference;
    ref.addEventListener("change", () => {
      const lab = ref.closest(".refbtn"), f = ref.files[0];
      $(".reftext", lab).textContent = f ? f.name : "Reference image";
      const old = $("img", lab); if (old) old.remove();
      if (f) { const img = new Image(); img.src = URL.createObjectURL(f); lab.prepend(img); $("svg", lab).classList.add("hidden"); }
      else $("svg", lab).classList.remove("hidden");
    });
  }
  const undo = $("#btn-undo");
  if (undo) undo.addEventListener("click", async () => {
    try {
      state.prop = (await postJSON(`/api/props/${name}/undo`, { target: state.sel })).prop;
      toast("Undone."); renderProp(); loadProps();
    } catch (e) { fail(e); }
  });
  const fw = $("#form-prop-words");
  if (fw) {
    const save = $("button[type=submit]", fw);
    const dirty = () => { save.disabled = fw.words.value.trim() === (p.words || "") && fw.description.value.trim() === (p.description || ""); };
    fw.addEventListener("input", dirty);
    fw.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      try {
        state.prop = (await postJSON(`/api/props/${name}/words`, { words: fw.words.value, description: fw.description.value })).prop;
        toast("Saved: new runs use these words."); renderProp(); loadProps();
      } catch (e) { fail(e); }
    });
  }
}

// ------------------------------------------------------------------ jobs (create, edit, new skin)
const PROP_JOB_LABEL = { prop_create: "New prop", prop_edit: "Edit", prop_skin: "New skin" };
async function startPropJob(url, form) {
  try {
    const d = await api(url, { method: "POST", body: form });
    if (d.job) { watchPropJob(d.job.id, state.prop ? state.prop.name : d.job.prop); queued(PROP_JOB_LABEL[d.job.action] || d.job.action); }
    return true;
  } catch (e) { fail(e); return false; }
}

function watchPropJob(id, name) {
  const tick = async () => {
    let j;
    try { j = await api(`/api/jobs/${id}`); } catch (e) { stopPoll("job"); return fail(e); }
    const what = PROP_JOB_LABEL[j.action] || j.action;
    if (state.castMode === "props" && state.prop && state.prop.name === name) {
      const p = j.progress;
      const frac = p && p.max ? ((p.item || 1) - 1 + p.value / p.max) / (p.items || 1) : 0;
      const elapsed = j.started ? Date.now() / 1000 - j.started : 0;
      const part = p && p.items > 1 ? `<span class="ptext">${p.item} of ${p.items}</span>` : "";
      $("#jobbar").innerHTML = j.status === "queued" || j.status === "running"
        ? `<div class="jobbar"><span class="spinner"></span><b>${esc(what)}</b>${part}
             ${j.status === "running" ? progressBar(frac, elapsed)
               : `<span class="ptext">${j.queue_position ? `waiting · ${j.queue_position} ahead` : "waiting"}</span>`}</div>`
        : j.status === "failed" ? `<div class="jobbar err">${esc(what)} failed: ${esc(j.error)}</div>` : "";
    }
    if (j.status !== "queued" && j.status !== "running") {
      stopPoll("job");
      if (j.status === "done") toast(j.action === "prop_skin" ? "New prop skin made: check its picture." : j.action === "prop_create" ? "The candidates are ready: click one to use it." : `${what}: done`);
      else toast(`${what}: ${j.status}${j.error ? ` — ${j.error}` : ""}`, true);
      await loadProps();
      const skin = j.status === "done" && j.result && j.result.skin;
      if (state.castMode === "props" && state.prop && state.prop.name === name) openProp(skin || name);
    }
  };
  tick();
  poll("job", tick, 4000);
}

// ------------------------------------------------------------------ the New prop form
$("#btn-new-prop").addEventListener("click", async () => {
  clearProp();
  $("#btn-new-prop").classList.add("active");
  $$("#prop-list .tile").forEach((t) => t.classList.remove("active"));
  $("#char-head").innerHTML = `<span class="name">New prop</span>`;
  const canvas = $("#char-canvas");
  canvas.innerHTML = "";
  canvas.append($("#tpl-new-prop").content.cloneNode(true));
  const f = $("#form-new-prop");
  const fillStyles = () => {                         // the library's default style is chosen (use the characters' style)
    const cur = f.style.value || (styleLib.list.find((s) => s.default) || {}).name || "";
    f.style.innerHTML = styleOptionsHtml(cur, "No library style");
  };
  fillStyles();
  loadStyleLib().then(fillStyles).catch(() => {});
  const sync = () => {
    const src = f.source.value;
    $$(".only-text", f).forEach((e) => e.classList.toggle("hidden", src !== "text"));
    $$(".only-image", f).forEach((e) => e.classList.toggle("hidden", src !== "image"));
  };
  $$("input[name=source]", f).forEach((r) => r.addEventListener("change", sync));
  sync();
  $$(".drop input", f).forEach((inp) => inp.addEventListener("change", () => {
    const d = inp.closest(".drop"), file = inp.files[0];
    const old = $("img", d); if (old) old.remove();
    d.classList.toggle("filled", !!file);
    if (file) { const img = new Image(); img.src = URL.createObjectURL(file); d.append(img); }
  }));
  f.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const src = f.source.value;
    if (src === "image" && !f.image.files.length) return toast("Choose the picture of the prop.", true);
    const form = new FormData(f);
    for (const [k, v] of [...form.entries()]) if (v instanceof File && !v.size) form.delete(k);
    try {
      const d = await api("/api/props", { method: "POST", body: form });
      if (d.prop) { await loadProps(d.prop.name); toast("Prop added."); return; }
      queued("New prop");
      const name = d.name;                           // the ID the server made from the name
      const wait = setInterval(async () => {         // the prop's folder appears when the job starts
        let list, job;
        try { [list, job] = await Promise.all([api("/api/props"), api(`/api/jobs/${d.job.id}`)]); } catch { return; }
        if (list.some((x) => x.name === name)) { clearInterval(wait); await loadProps(name); watchPropJob(d.job.id, name); }
        else if (!["queued", "running"].includes(job.status)) {
          clearInterval(wait);
          toast(`New prop: ${job.status}${job.error ? ` — ${job.error}` : ""}`, job.status !== "done");
        }
      }, 2000);
    } catch (e) { fail(e); }
  });
  f.name.focus();
});
