// cinestudio web app - The Cast tab's characters (the tab id stays "characters"; its Props: props.js): the library, the character view, edits, quality settings and the New character form.
// Plain scripts (no modules), loaded in order by index.html; they share their top-level names.
"use strict";

// ------------------------------------------------------------------ characters
const charPane = () => $("#tab-characters");
// the name the app shows (any text, 2026-10-24); c.name is the ID (folder, files, jobs, runs, the API)
const shownName = (c) => (c && (c.display_name || c.name)) || "";
// skins (user, 2026-10-25: "think of it as like switching skins for a character"): a skin is its own record with
// skin_of = the main record's ID; the library shows one tile per character, the skin menu (styles.js) switches its skins
const charBase = (c) => (c && (c.skin_of || c.name)) || "";
const charFamily = (base) => (state.chars || []).filter((x) => charBase(x) === base);
function skinStyleLabel(c) {
  const s = c && c.style_ref && typeof styleByName === "function" ? styleByName(c.style_ref) : null;
  return s ? s.label || s.name : (c && c.style_ref) || (c && c.skin_of ? "Own style" : "Original");   // no library style
}
function charLabel(c) {                              // "kitsune · 3D animated" when the character has more than one skin
  const fam = charFamily(charBase(c));
  if (fam.length < 2) return shownName(c);
  return `${shownName(fam.find((x) => !x.skin_of) || c)} · ${skinStyleLabel(c)}`;
}
const skinKey = (base) => `cs.skin.${base}`;
function activeSkin(base) {                          // the skin this viewer looked at last (a browser convenience)
  try {
    const n = localStorage.getItem(skinKey(base));
    if (n && charFamily(base).some((x) => x.name === n)) return n;
  } catch { /* no storage: the main record */ }
  return base;
}
function rememberSkin(c) { try { localStorage.setItem(skinKey(charBase(c)), c.name); } catch { /* no storage */ } }

async function loadChars(select) {
  try { state.chars = await api("/api/characters"); } catch (e) { return fail(e); }
  const curBase = state.char ? charBase(state.char) : null;
  const names = new Set(state.chars.map((c) => c.name));
  const mains = state.chars.filter((c) => !c.skin_of || !names.has(c.skin_of));   // a skin whose main is gone shows alone
  $("#char-list").innerHTML = mains.map((m) => {
    const fam = charFamily(charBase(m)), n = fam.length;
    // the list row of the skin on show (the open one, else the last one looked at): it has the thumbnail
    const want = curBase === charBase(m) ? state.char.name : activeSkin(charBase(m));
    const c = fam.find((x) => x.name === want) || m;
    const skinsNote = n > 1 ? ` · ${n} skins: ${fam.map(skinStyleLabel).join(", ")}` : "";
    return `<div class="tile ${curBase === charBase(m) ? "active" : ""}" data-name="${esc(c.name)}" data-base="${esc(charBase(m))}" role="button" tabindex="0" title="${esc(shownName(m))}${shownName(m) !== m.name ? ` (ID: ${esc(m.name)})` : ""}${esc(skinsNote)}">
       ${c.thumb_url ? `<img src="${c.thumb_url}" alt="" loading="lazy">` : `<span class="initial">${esc(shownName(m)[0])}</span>`}
       <span class="state ${c.status === "ready" && !(c.missing || []).length ? "ready" : ""}" title="${esc(readyText(c))}"></span>
       ${n > 1 ? `<span class="skins-n" title="${n} skins (the skin menu over the picture switches them)">${n}</span>` : ""}
       <span class="label">${esc(shownName(m))}</span>${trashButton("character", m.name)}</div>`;
  }).join("");
  $$("#char-list .tile").forEach((t) => {
    t.addEventListener("click", () => openChar(t.dataset.name));
    t.addEventListener("keydown", (ev) => { if (ev.key === "Enter") openChar(t.dataset.name); });
  });
  bindDeletes($("#char-list"), (name) => {
    if (state.char && state.char.name === name) clearChar();
    loadChars();
  });
  if (select) openChar(select);
}

// the tiles' status dots follow changes made outside this page (a CLI job, a cancel from the queue panel): redraw the
// list only when a name or a status changed (user, 2026-10-23: "why is masked3d_rest indicator still orange")
async function refreshCharStates() {
  let fresh;
  try { fresh = await api("/api/characters"); } catch { return; }
  const sig = (list) => (list || []).map((c) => `${c.name}:${shownName(c)}:${c.status}:${c.thumb_url || ""}:${(c.missing || []).join()}:${c.skin_of || ""}:${c.style_ref || ""}`).join("|");
  if (sig(fresh) !== sig(state.chars)) loadChars();
}

// what a character still needs before a Blender video can use it (user, 2026-10-24: "a character isnt ready to be
// used until a mannequin is set for the character with its respective proxy kit")
const MISSING_TEXT = { sheet: "the views and the sheet", mannequin: "its mannequin body", proxy: "its stand-in parts" };
function readyText(c) {
  if (c.status !== "ready") return c.status;
  const m = (c.missing || []).filter((k) => k !== "sheet");
  return m.length ? `needs ${m.map((k) => MISSING_TEXT[k]).join(" and ")} before a video` : "ready for videos";
}

function clearChar() {
  state.char = null; state.view = null; state.sel = null; stopPoll("job");
  charPane().classList.add("no-panel");
  $("#char-head").innerHTML = ""; $("#char-desc").innerHTML = ""; $("#jobbar").innerHTML = "";
  $("#char-canvas").innerHTML = `<div class="empty">Choose a character or press +</div>`;
  $("#btn-new-char").classList.remove("active");
}

async function openChar(name) {
  if (state.char && state.char.name !== name && state.view === "mannequin" && !mqLeaveOk()) return;
  if (state.mq && state.mq.name !== name) state.mq = null;            // another character: a fresh editor
  const seq = (state.openSeq = (state.openSeq || 0) + 1);
  let c;
  try { c = await api(`/api/characters/${encodeURIComponent(name)}`); } catch (e) { return fail(e); }
  if (seq !== state.openSeq) return;                  // the user chose another character meanwhile
  if (!state.char || state.char.name !== name) { state.view = null; state.sel = null; $("#jobbar").innerHTML = ""; }
  state.char = c;
  rememberSkin(c);
  $("#btn-new-char").classList.remove("active");
  $$("#char-list .tile").forEach((t) => t.classList.toggle("active", t.dataset.base === charBase(c)));
  // the library tile follows the front (an edit, a touch-up or an undo changes it; the list is not drawn again)
  const thumb = (c.urls || {}).front || (c.urls || {}).sheet;
  const tile = $$("#char-list .tile").find((t) => t.dataset.base === charBase(c));
  if (tile && tile.dataset.name !== name) { tile.dataset.name = name; loadChars(); }   // another skin of the same tile
  const img = tile && $("img", tile);
  if (thumb && img && img.getAttribute("src") !== thumb) img.setAttribute("src", thumb);
  else if (thumb && tile && !img) loadChars();
  renderChar();
  const live = (c.jobs || []).find((j) => j.status === "queued" || j.status === "running");
  if (live) watchJob(live.id, name); else stopPoll("job");
}

function charViews(c) {
  const u = c.urls || {};
  const v = [];
  if (c.source === "text" && c.candidates.length && !u.front) v.push(["candidates", "Candidates", true]);
  if (c.source !== "sheet") { v.push(["front", "Front", !!u.front]); v.push(["views", "Views", !!u.front]); }
  v.push(["sheet", "Sheet", !!u.sheet]);
  if (u.sheet_styled) v.push(["styled", "Styled", true]);
  v.push(["mannequin", "Mannequin", !!(u.front || u.sheet)]);   // its own mannequin (body, parts) at one scale
  return v;
}
function defaultView(c) {
  const u = c.urls || {};
  if (!u.front && c.candidates.length) return "candidates";
  if (u.sheet_styled) return "styled";
  if (u.sheet) return "sheet";
  return c.source === "sheet" ? "sheet" : "front";
}

function renderChar() {
  const c = state.char, u = c.urls || {};
  const views = charViews(c);
  const choosing = state.view === "candidates" && c.candidates.length;   // also opened by "Other candidates"
  if (!choosing && (!state.view || !views.some(([k, , ok]) => k === state.view && ok))) state.view = defaultView(c);

  // one profile per character: a skin shows the main record's name (the floating skin menu shows its style)
  const fam = (c.skins && c.skins.length > 1) ? c.skins : null;
  const main = fam ? fam.find((x) => !x.skin_of) || c : c;
  $("#char-head").innerHTML = `<span class="name ${main === c ? "editable" : ""}" title="${main === c ? `Click to rename (ID: ${esc(c.name)}; the ID does not change)` : `A skin of ${esc(shownName(main))} (ID: ${esc(c.name)}): rename the character on its main skin`}">${esc(shownName(main))}</span><span class="chip ${esc(c.status)}">${esc(c.status)}</span>
    <div class="seg-switch">${views.map(([k, label, ok]) =>
      `<button data-view="${k}" class="${state.view === k ? "on" : ""}" ${ok ? "" : "disabled"}>${label}</button>`).join("")}</div>`;
  $$("#char-head [data-view]").forEach((b) => b.addEventListener("click", () => {
    if (state.view === "mannequin" && b.dataset.view !== "mannequin" && !mqLeaveOk()) return;   // unsaved body changes
    if (state.view === "mannequin" && b.dataset.view !== "mannequin" && state.mq) state.mq.draft = { ...state.mq.saved };
    state.view = b.dataset.view; state.sel = null; renderChar();
  }));
  if (main === c) $("#char-head .name").addEventListener("click", startRename);
  $("#char-desc").innerHTML = `<div class="desc" title="Click to show all">${esc(c.description)}</div>`;
  $("#char-desc .desc").addEventListener("click", (ev) => ev.currentTarget.classList.toggle("open"));

  $("#char-canvas").innerHTML = renderCanvas(c, u);
  charSkinSwitch(c);                                 // the floating skin menu over the canvas (styles.js; no layout room)
  fitSheetView();                                    // the sheet framed to its content (display only)
  $$("#char-canvas .cand").forEach((t) => t.addEventListener("click", async () => {
    if (+t.dataset.seed === c.seed && u.front) return;
    if (u.front && !confirm(`Use seed ${t.dataset.seed} as the front view?\nThe current front goes to the history (undo brings it back). The views and the sheet must be made again.`)) return;
    try {
      state.char = (await postJSON(`/api/characters/${encodeURIComponent(c.name)}/pick`, { seed: +t.dataset.seed })).character;
      state.view = "front"; renderChar(); loadChars();
    } catch (e) { fail(e); }
  }));

  if (state.sel && !(u[state.sel] && c.source !== "sheet")) state.sel = null;
  if (state.view === "mannequin") { mqEnsure(c); mqLoadSchema(); }
  const panel = renderPanel(c, u);
  $("#char-panel").innerHTML = panel;
  charPane().classList.toggle("no-panel", !panel);
  loadLoras();
  bindPanel(c);
  if (state.view === "mannequin") {                  // the Blender preview of the draft (cached on the server)
    drawMqStage();
    if (!state.mq.preview && !state.mq.loading && !state.mq.error) mqPreview(true);
  }
}

function startRename() {
  const c = state.char, span = $("#char-head .name");
  const inp = document.createElement("input");
  inp.className = "rename"; inp.value = shownName(c); inp.spellcheck = false; inp.maxLength = 80;
  inp.title = "Any name (spaces and capitals are fine). Enter to save, Esc to stop. The ID stays the same.";
  span.replaceWith(inp); inp.focus(); inp.select();
  let done = false;
  const finish = async (save) => {
    if (done) return; done = true;
    const v = inp.value.trim();
    if (!save || !v || v === shownName(c)) return renderChar();
    try {
      state.char = (await postJSON(`/api/characters/${encodeURIComponent(c.name)}/rename`, { new_name: v })).character;
      toast(`Renamed to ${shownName(state.char)}.`); await loadChars(); renderChar();
    } catch (e) { fail(e); renderChar(); }
  };
  inp.addEventListener("keydown", (ev) => { if (ev.key === "Enter") finish(true); if (ev.key === "Escape") finish(false); });
  inp.addEventListener("blur", () => finish(true));
}

const TARGET_LABEL = { front: "Front view", right: "Right view", back: "Back view", left: "Left view", face: "Face" };
const EXAMPLE = {
  front: "make the scarf red",
  right: "give the hand five fingers",
  back: "make the cape a bit longer",
  left: "give the left hand five fingers",
  face: "make the eye slits thinner",
};

function renderCanvas(c, u) {
  const sel = state.sel;
  const pickable = (k) => c.source !== "sheet" && !!u.front;          // images that can be edited
  const img = (k, src) => `<img src="${src}" alt="${k}" data-target="${k}"
      class="${pickable(k) ? "pickable" : ""} ${sel === k ? "selected" : ""}" ${pickable(k) ? `title="Click to edit"` : ""}>`;
  const hero = (inner, extra = "") => `<div class="hero">${inner}${extra}</div>`;
  switch (state.view) {
    case "candidates":
      return c.candidates.length ? `<div class="cands">${c.candidates.map((x) =>
        `<button class="cand ${x.seed === c.seed ? "on" : ""}" data-seed="${x.seed}" title="Use seed ${x.seed}">
           <img src="${x.url}" alt="seed ${x.seed}" loading="lazy"><span>${x.seed}</span></button>`).join("")}</div>`
        : `<div class="empty">Making the candidates…</div>`;
    case "front":
      return u.front ? hero(img("front", u.front), c.source !== "sheet" && !u.right && !u.sheet
        ? `<div class="next-step"><button id="btn-views" class="primary" title="Right, back, left and face, made from the front view (about 45 minutes)">Make the views</button></div>` : "")
        : `<div class="empty">Pick a candidate first</div>`;
    case "views": {
      const given = new Set(c.provided || []);
      return `<div class="views">${["front", "right", "back", "left", "face"].map((k) => u[k]
        ? `<div class="view ${sel === k ? "selected" : ""}">${img(k, u[k])}<span class="tag ${given.has(k) ? "yours" : ""}"
             ${given.has(k) ? `title="Your image"` : ""}>${k}</span></div>`
        : `<div class="view missing" title="${k}: not made yet">${k}</div>`).join("")}</div>`;
    }
    case "styled":
      return hero(`<img src="${u.sheet_styled}" alt="styled sheet">`);
    case "mannequin":
      return `<div id="mq-stage" class="mq-stage"></div>`;
    default: {
      const w = (c.checks && c.checks.warnings) || [];
      // the colour check (charsheet.palette_check): a main colour of the front that mostly vanishes in a view often means
      // a lost part (a cape, a scarf, a glove); some drops are normal (a mask is not seen from the back): a hint, not a fail
      const views = [...new Set(w.map((x) => String(x).split(":")[0].trim()).filter(Boolean))];
      const why = "A main colour of the front view mostly vanishes in these views. That often means a lost part (a cape, a scarf, a glove); some drops are normal (a mask is not seen from the back).";
      const note = w.length ? `<div class="warnline" title="${esc(why + "\n\n" + w.join("\n"))}">Check ${esc(views.join(", ") || "the views")}: ${views.length === 1 ? "it may miss" : "they may miss"} a colour of the front (a lost part?)</div>` : "";
      return u.sheet ? hero(`<img src="${u.sheet}" alt="sheet">`, note) : `<div class="empty">No sheet yet</div>`;
    }
  }
}

// the Sheet view shows the sheet CROPPED FOR DISPLAY to its content (user, 2026-10-25: "the extra gap feels off"): the
// file keeps its empty space on purpose (one scale for every sheet: AGENTS.md 4.5), only the picture on screen is
// framed: the box of non-black pixels + an even margin, via CSS object-view-box (Chromium); without it, or when the box
// cannot be measured, the whole picture shows. Only the sheets: the Front and Views keep the full picture (the region
// tool's box is in the full image's pixels).
const PIC_BOX = new Map();                              // picture URL -> its content box in image pixels (or null)
const VIEWBOX_OK = typeof CSS !== "undefined" && CSS.supports && CSS.supports("object-view-box", "inset(1px)");
function contentBox(url) {
  if (PIC_BOX.has(url)) return PIC_BOX.get(url);
  const p = new Promise((resolve) => {
    const im = new Image();
    im.onload = () => {
      try {
        const W = im.naturalWidth, H = im.naturalHeight, k = Math.min(1, 480 / W);
        const w = Math.max(1, Math.round(W * k)), h = Math.max(1, Math.round(H * k));
        const cv = document.createElement("canvas");
        cv.width = w; cv.height = h;
        const g = cv.getContext("2d", { willReadFrequently: true });
        g.drawImage(im, 0, 0, w, h);
        const d = g.getImageData(0, 0, w, h).data, rows = new Uint32Array(h), cols = new Uint32Array(w);
        for (let y = 0, i = 0; y < h; y++) for (let x = 0; x < w; x++, i += 4) {
          if (Math.max(d[i], d[i + 1], d[i + 2]) > 12) { rows[y]++; cols[x]++; }
        }
        const minR = Math.max(2, w * 0.004), minC = Math.max(2, h * 0.004);   // isolated specks do not count
        const y0 = rows.findIndex((n) => n >= minR), x0 = cols.findIndex((n) => n >= minC);
        if (y0 < 0 || x0 < 0) return resolve(null);
        let y1 = h - 1, x1 = w - 1;
        while (rows[y1] < minR) y1--;
        while (cols[x1] < minC) x1--;
        resolve({ W, H, x0: x0 / k, y0: y0 / k, x1: (x1 + 1) / k, y1: (y1 + 1) / k });
      } catch { resolve(null); }                     // a picture the canvas cannot read: shown whole
    };
    im.onerror = () => resolve(null);
    im.src = url;
  });
  PIC_BOX.set(url, p);
  return p;
}
async function fitSheetView() {
  const img = (state.view === "sheet" || state.view === "styled") && $("#char-canvas .hero img");
  if (img && VIEWBOX_OK) {
    const url = img.getAttribute("src"), box = await contentBox(url);
    if (box && img.isConnected && img.getAttribute("src") === url) {
      if (!img.complete) await img.decode().catch(() => {});
      const { W, H } = box, ch = box.y1 - box.y0;
      const room = typeof skinTriggerRoom === "function" ? skinTriggerRoom() : 0;   // the skin button sits in the top margin
      const apply = (top, m) => {
        const t = Math.max(0, box.y0 - top), b = Math.max(0, H - box.y1 - m), l = Math.max(0, box.x0 - m), r = Math.max(0, W - box.x1 - m);
        const v = `inset(${Math.round(t)}px ${Math.round(r)}px ${Math.round(b)}px ${Math.round(l)}px)`;
        if (img.style.getPropertyValue("object-view-box") !== v) img.style.setProperty("object-view-box", v);
        return H - t - b;
      };
      let vh = apply(ch * 0.04, ch * 0.04);
      for (let i = 0; i < 2; i++) {                  // the margin in screen pixels depends on the scale: two passes
        const s = img.getBoundingClientRect().height / vh;
        if (!s) break;
        const m = Math.max(ch * 0.04, 16 / s);
        vh = apply(Math.max(m, room / s), m);
      }
      img.classList.add("framed");
    }
  }
  if (typeof placeSkinTrigger === "function") placeSkinTrigger();
}
// a new size of the canvas (the window, the panel): frame the sheet again and move the skin button with the picture
if (typeof ResizeObserver !== "undefined") {
  let raf = 0;
  new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => {
    if (state.char) fitSheetView(); else if (state.prop) placeSkinTrigger();   // a prop (props.js): the skin button follows
  }); })
    .observe($("#char-canvas"));
}

// drafts: the text typed for each character and image, kept until it is applied
const draftOf = (name, target) => (state.drafts || {})[`${name}:${target}`] || "";
function setDraft(name, target, text) { (state.drafts = state.drafts || {})[`${name}:${target}`] = text; }

// the last change of one image that can be undone (an edit, a redo or a pick)
const lastChange = (c, target) => (c.edits || []).filter((e) => !e.undone && (e.target || "front") === target).slice(-1)[0];

function renderPanel(c, u) {
  const sel = state.sel;
  if (state.view === "candidates") return "";        // the grid needs no panel: click a candidate to use it
  if (state.view === "mannequin") return mqEditor(c);
  if (sel) {                                         // an image is selected: only its actions
    const last = lastChange(c, sel), isView = sel !== "front";
    return `<div class="block"><h2>${TARGET_LABEL[sel]}<span class="muted" id="q-eta" title="About how long one change takes">${etaText()}</span>
        <button type="button" class="close" id="btn-desel" title="Close (Esc)" aria-label="Close">${ICON.close}</button></h2>
      <form id="form-edit" class="form">
        <input type="hidden" name="target" value="${sel}">
        <textarea name="instruction" rows="4" required
          placeholder="One change, for example: ${EXAMPLE[sel]}. Keep everything else the same.">${esc(draftOf(c.name, sel))}</textarea>
        <label class="refbtn" title="Optional: ONE detail, tightly cropped. Call it &quot;Picture 2&quot; in the text (the character is Picture 1).">${ICON.image}<span class="reftext">Reference image</span>
          <input type="file" name="reference" accept="image/png,image/jpeg,image/webp"></label>
        ${regionBlock(c, sel)}
        ${qualityBlock(isView)}
        <div class="row"><button class="primary grow" type="submit">Apply</button>
          ${isView ? `<button type="button" id="btn-redo" class="icon-btn" title="Make this view again with a new seed" aria-label="Redo">${ICON.redo}</button>` : ""}
          ${last ? `<button type="button" id="btn-undo" class="icon-btn" title="${esc(`Undo: ${last.instruction}`)}" aria-label="Undo">${ICON.undo}</button>` : ""}</div>
      </form>
      ${sel === "front" && c.source === "text" && c.candidates.length > 1
        ? `<button type="button" id="btn-cands" class="ghost" title="Start again from another front view (asks first; undo brings this one back)">Other candidates</button>` : ""}
      </div>${historyBlock(c, sel)}`;
  }
  // nothing selected: only the next step, when there is one
  if (c.status === "ready") return "";              // the Mannequin tab at the top holds the body (no card here)
  if (u.sheet && c.status !== "ready") {
    return `<div class="block"><h2>Finish</h2>
      <form id="form-style" class="form">
        <div class="row2"><select name="style_lora" class="lora-select" title="The style LoRA of your videos"></select>
          <input name="keep" type="number" min="0" max="1" step="0.05" value="0.5" title="How much of the sheet stays (0 to 1)"></div>
        <div class="row"><button class="primary grow" type="submit" title="Redraw the sheet in the LoRA style with LTX">Style pass</button>
          <button type="button" id="btn-finish" class="grow" title="Ready for videos without a style pass">Use as is</button></div>
      </form></div>`;
  }
  return "";
}

// ------------------------------------------------------------------ the Mannequin view and editor (phase 1, 2026-10-24)
// User: "some kind of rendered live image of what the mannequin would look like in the mannequin config", "this side
// bar is no where near intuitive"; plan: docs/mannequin_preview_research.md. The canvas shows the character's own front
// beside its mannequin (front, side, back) rendered by Blender at ONE scale; the panel edits the BODY (sliders, presets,
// one Save). The stand-in parts (proxy kit) are only shown here: an AI agent sets them with the MCP tool set_proxy_kit
// (user: "anything with proxy kit use should be handled by ai").
const MQ_FALLBACK = [["scale", "Overall height", 0.5, 1.3], ["head", "Head size", 0.7, 2.0], ["neck", "Neck length", 0.2, 1.5],
                     ["torso", "Torso length", 0.6, 1.4], ["legs", "Leg length", 0.5, 1.4], ["arms", "Arm length", 0.6, 1.4]];
const mqNorm = (body) => {                          // the stored form: values of 1.0 dropped, 4 decimals
  const out = {};
  Object.keys(body || {}).sort().forEach((k) => { const v = Math.round(+body[k] * 1e4) / 1e4; if (Number.isFinite(v) && Math.abs(v - 1) > 1e-9) out[k] = v; });
  return out;
};
const mqSame = (a, b) => JSON.stringify(mqNorm(a)) === JSON.stringify(mqNorm(b));
const mqDirty = () => !!(state.mq && !mqSame(state.mq.draft, state.mq.saved));
function mqEnsure(c) {
  const saved = mqNorm(c.mannequin || {});
  if (!state.mq || state.mq.name !== c.name) state.mq = { name: c.name, saved, draft: { ...saved }, preview: null, loading: false, error: "", seq: 0 };
  else if (!mqDirty()) { state.mq.saved = saved; state.mq.draft = { ...saved }; }
  else state.mq.saved = saved;
  return state.mq;
}
async function mqLoadSchema() {
  if (state.mqSchema || state.mqSchemaLoading) return;
  state.mqSchemaLoading = true;
  try { state.mqSchema = await api("/api/mannequin/schema"); } catch { state.mqSchema = null; }
  state.mqSchemaLoading = false;
  if (state.char && state.view === "mannequin") renderChar();
}
const mqControls = () => (state.mqSchema && state.mqSchema.body) ||
  MQ_FALLBACK.map(([key, label, min, max]) => ({ key, label, min, max, step: 0.01, help: "" }));
const mqLeaveOk = () => !mqDirty() || confirm("Discard the unsaved mannequin changes?");
window.addEventListener("beforeunload", (ev) => { if (mqDirty()) { ev.preventDefault(); ev.returnValue = ""; } });

// linear RGB 0..1 (the kit's colours) -> an sRGB hex swatch
function linHex(v) {
  const rgb = Array.isArray(v) ? v : [v, v, v];
  return "#" + rgb.map((x) => { x = Math.max(0, Math.min(1, +x || 0)); const s = x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055;
    return Math.round(s * 255).toString(16).padStart(2, "0"); }).join("");
}
function partsSummary(p) {
  const sw = (part) => { const v = part.color ?? part.shade; return v === undefined ? "" : `<i class="swatch" style="background:${linHex(v)}"></i>`; };
  const rows = [];
  if (p.mask) rows.push(["Mask", [p.mask.shape === "round" ? "round" : "pointed", p.mask.ears ? "fox ears" : "", p.mask.eyes ? "eyes" : "",
    p.mask.marks ? `${typeof p.mask.marks === "string" ? p.mask.marks : "own"} marks` : ""].filter(Boolean).join(", "), sw(p.mask)]);
  if (p.hair) rows.push(["Hair", [p.hair.length || "short", p.hair.locks ? `side locks to the ${p.hair.locks}` : "", p.hair.back ? `back to the ${p.hair.back}` : ""].filter(Boolean).join(", "), sw(p.hair)]);
  if (p.cape) rows.push(["Cape", `to the ${p.cape.length || "thigh"}`, sw(p.cape)]);
  if (p.jacket) rows.push(["Jacket", "short jacket", sw(p.jacket)]);
  if (Array.isArray(p.bands) && p.bands.length) rows.push(["Bands", `${p.bands.length} band${p.bands.length > 1 ? "s" : ""}`, ""]);
  if (p.suit) rows.push(["Suit", "body colour", sw(p.suit)]);
  if (p.shoes) rows.push(["Shoes", "shoe colour", sw(p.shoes)]);
  return rows;
}

// plain-words status of the character and of the draft (research 5.6)
function mqStatus(c) {
  const miss = (c.missing || []).filter((k) => k !== "sheet");
  const p = state.mq && state.mq.preview;
  if (c.sheet_stale && !mqDirty()) return ["warn", "The sheet still shows another height (its last redraw did not finish): press Draw the sheet again."];
  if (mqDirty()) return ["", p ? `Saving draws the sheet again at ${p.height_m.toFixed(2)} m (${Math.round(p.ratio * 100)}% of the template). The sheet's pose changes only after a repose.`
    : "Saving draws the sheet again at the new height. The sheet's pose changes only after a repose."];
  if (miss.includes("mannequin")) return ["warn", "Not ready for videos yet: choose a body (Adult is fine), then Save."];
  if (c.status !== "ready") return ["", "The body is set. The character is ready for videos after its sheet is finished."];
  return ["ok", "✓ Ready for videos. Every Blender video puts this body on the template's dancer."];
}

// Repose (AnyPose): the front posed like this body, then the views and the sheet made again from it (AGENTS.md 4.5).
// For a ready character with a body; a saved body only (the pose source is the saved body).
function mqPoseCard(c) {
  const u = c.urls || {};
  if (c.status !== "ready" || c.source === "sheet" || !u.front || (c.missing || []).includes("mannequin")) return "";
  const busy = (c.jobs || []).some((j) => j.status === "queued" || j.status === "running");
  const why = mqDirty() ? "Save the body first: the pose uses the saved body." : busy ? "Wait until the character's job is done." : "";
  return `<div class="mq-card"><div class="mq-card-head"><b>Pose</b><span class="muted">AnyPose</span></div>
      <div class="sub small">The sheet keeps its old pose until the front is posed like this body.${c.reposed ? " The front was posed before: undo that pose first, so the new pose starts from the original front." : ""}</div>
      <button type="button" id="mq-repose" ${why ? "disabled" : ""} title="${esc(why || "Pose the front like this mannequin, then make the right, back and left views and the sheet again (GPU, about an hour)")}">Pose the front like this body</button></div>`;
}

// the editor in the side panel (the Mannequin view)
function mqEditor(c) {
  const mq = mqEnsure(c), d = mq.draft, p = mq.preview;
  const [kind, text] = mqStatus(c);
  const presets = (state.mqSchema && state.mqSchema.presets) || [];
  const on = presets.find((x) => mqSame(x.body, d));
  const lim = (state.mqSchema && state.mqSchema.limits) || [0.3, 2.0];
  const rows = mqControls().map((f) => {
    const v = d[f.key] ?? 1;
    return `<div class="mq-row" data-key="${f.key}" title="${esc(f.help || "")}">
      <label for="mq-${f.key}">${esc(f.label)}</label>
      <input type="range" id="mq-${f.key}" min="${Math.min(f.min, v)}" max="${Math.max(f.max, v)}" step="${f.step}" value="${v}">
      <input type="number" class="mq-num" min="${lim[0]}" max="${lim[1]}" step="${f.step}" value="${v}" aria-label="${esc(f.label)} value">
      <button type="button" class="mq-reset" title="Back to the template's value (1.0)" aria-label="Reset ${esc(f.label)}">↺</button></div>`;
  }).join("");
  const parts = partsSummary(c.proxy || {});
  const hr = p ? `${p.height_m.toFixed(2)} m · ${Math.round(p.ratio * 100)}%` : "…";
  return `<div class="block mq-editor"><h2>Mannequin <span class="mq-dirty ${mqDirty() ? "" : "hidden"}">● unsaved</span></h2>
      <div class="mq-status ${kind}">${esc(text)}</div>
      <div class="mq-presets" role="group" aria-label="Body presets">${presets.map((x) =>
        `<button type="button" class="chipbtn ${on === x ? "on" : ""}" data-preset="${esc(x.id)}" title="${esc(x.note)}">${esc(x.label)}</button>`).join("")}
        <span class="chipbtn static ${on ? "" : "on"}" title="The sliders' own values">Custom</span></div>
      <div class="mq-card"><div class="mq-card-head"><b>Body</b><span id="mq-height" title="The height of this body (Blender), and the part of the template mannequin's 1.81 m">${hr}</span></div>
        ${rows}</div>
      <div class="mq-card"><div class="mq-card-head"><b>Stand-in parts</b><span class="muted">set by an AI agent</span></div>
        ${parts.length ? `<ul class="mq-parts">${parts.map(([n, t, s]) => `<li>${s}<b>${n}</b> ${esc(t)}</li>`).join("")}</ul>`
          : `<div class="sub">None: the videos use the plain mannequin.</div>`}
        <div class="sub small">Stand-in parts give the layout the character's outline (a mask, hair, a cape). An AI agent makes them from the sheet with the MCP tool <code>set_proxy_kit</code> and checks them with <code>preview_mannequin</code>.</div></div>
      ${mqPoseCard(c)}
      ${c.sheet_stale ? `<button type="button" class="primary mq-redraw" id="mq-redraw" title="Draw the sheet again at this body's height (no GPU; a few seconds)">Draw the sheet again</button>` : ""}
      <div class="mq-footer"><button type="button" id="mq-revert" ${mqDirty() ? "" : "disabled"}>Revert</button>
        <button type="button" class="primary grow" id="mq-save" ${mqDirty() || (c.missing || []).includes("mannequin") ? "" : "disabled"}>Save changes</button></div>
    </div>`;
}

// the canvas of the Mannequin view: the character's front and its mannequin at one scale
function mqStageHtml() {
  const mq = state.mq || {}, p = mq.preview;
  const fig = (url, label, extra = "") => `<figure class="mq-fig ${extra}">${url ? `<img src="${url}" alt="${esc(label)}">` : `<div class="mq-empty"></div>`}<figcaption>${esc(label)}</figcaption></figure>`;
  const note = mq.error ? `<div class="mq-note err">${esc(mq.error)} <button type="button" class="linkbtn" id="mq-retry">Retry</button></div>`
    : mq.loading ? `<div class="mq-note"><span class="spinner"></span> Drawing the mannequin in Blender…</div>`
    : p ? `<div class="mq-note ok">Blender preview: ${p.height_m.toFixed(2)} m, ${Math.round(p.ratio * 100)}% of the template (the teal line: the template's 1.81 m)${mqDirty() ? " · unsaved" : ""}</div>` : "";
  const u = (p && p.urls) || {};
  return `${note}<div class="mq-figs ${mq.loading ? "busy" : ""}">${fig(u.character, "The character (its front)", "char")}
    ${fig(u.front, "Mannequin: front")}${fig(u.right, "Side")}${fig(u.back, "Back")}</div>`;
}
function drawMqStage() {
  const el = $("#mq-stage");
  if (!el) return;
  el.innerHTML = mqStageHtml();
  const r = $("#mq-retry", el);
  if (r) r.addEventListener("click", () => mqPreview(true));
  const hd = $("#mq-height");
  const p = state.mq && state.mq.preview;
  if (hd && p) hd.textContent = `${p.height_m.toFixed(2)} m · ${Math.round(p.ratio * 100)}%`;
}
function mqPreview(now = false) {                   // ~0.6 s after the last change; the newest request wins
  const mq = state.mq;
  if (!mq) return;
  clearTimeout(mq.timer);
  mq.timer = setTimeout(async () => {
    const seq = ++mq.seq, name = mq.name;
    mq.loading = true; mq.error = ""; drawMqStage();
    try {
      const r = await postJSON(`/api/characters/${encodeURIComponent(name)}/mannequin-preview`, { body: JSON.stringify(mqNorm(mq.draft)) });
      if (state.mq !== mq || seq !== mq.seq || r.superseded) return;
      mq.preview = r;
    } catch (e) {
      if (state.mq !== mq || seq !== mq.seq) return;
      mq.error = `The Blender preview failed: ${e.message}`;
    }
    mq.loading = false;
    drawMqStage();
    if (state.char && state.view === "mannequin") { const s = $(".mq-status"); if (s) { const [k, t] = mqStatus(state.char); s.className = `mq-status ${k}`; s.textContent = t; } }
  }, now ? 0 : 600);
}
function mqSyncChrome(c) {                           // the unsaved mark, the buttons and the preset chips, without a redraw
  const dirty = mqDirty();
  const dm = $(".mq-dirty"); if (dm) dm.classList.toggle("hidden", !dirty);
  const rv = $("#mq-revert"); if (rv) rv.disabled = !dirty;
  const sv = $("#mq-save"); if (sv) sv.disabled = !(dirty || (c.missing || []).includes("mannequin"));
  const rp = $("#mq-repose");
  if (rp) {
    const busy = (c.jobs || []).some((j) => j.status === "queued" || j.status === "running");
    rp.disabled = dirty || busy;
    rp.title = dirty ? "Save the body first: the pose uses the saved body." : busy ? "Wait until the character's job is done."
      : "Pose the front like this mannequin, then make the right, back and left views and the sheet again (GPU, about an hour)";
  }
  const presets = (state.mqSchema && state.mqSchema.presets) || [];
  const on = presets.find((x) => mqSame(x.body, state.mq.draft));
  $$(".mq-presets [data-preset]").forEach((b) => b.classList.toggle("on", !!on && b.dataset.preset === on.id));
  const cu = $(".mq-presets .static"); if (cu) cu.classList.toggle("on", !on);
  const s = $(".mq-status"); if (s) { const [k, t] = mqStatus(c); s.className = `mq-status ${k}`; s.textContent = t; }
}
function bindMq(c) {
  if (!$(".mq-editor")) return;
  const mq = state.mq;
  const setVal = (key, v, from) => {
    const row = $(`.mq-row[data-key="${key}"]`);
    if (!Number.isFinite(v)) return;
    if (Math.abs(v - 1) < 1e-9) delete mq.draft[key]; else mq.draft[key] = v;
    if (from !== "range") $("input[type=range]", row).value = v;
    if (from !== "num") $(".mq-num", row).value = v;
    mqSyncChrome(c);
    mqPreview();
  };
  $$(".mq-row").forEach((row) => {
    const key = row.dataset.key;
    $("input[type=range]", row).addEventListener("input", (ev) => setVal(key, +ev.target.value, "range"));
    $(".mq-num", row).addEventListener("change", (ev) => {
      const lim = (state.mqSchema && state.mqSchema.limits) || [0.3, 2.0];
      const v = Math.max(lim[0], Math.min(lim[1], +ev.target.value));
      ev.target.value = v; setVal(key, v, "num");
    });
    $(".mq-reset", row).addEventListener("click", () => setVal(key, 1, ""));
    $("label", row).addEventListener("dblclick", () => setVal(key, 1, ""));
  });
  $$(".mq-presets [data-preset]").forEach((b) => b.addEventListener("click", () => {
    const x = ((state.mqSchema && state.mqSchema.presets) || []).find((p) => p.id === b.dataset.preset);
    if (!x) return;
    mq.draft = { ...x.body };
    renderChar();
    mqPreview(true);
  }));
  $("#mq-revert").addEventListener("click", () => { mq.draft = { ...mq.saved }; renderChar(); mqPreview(true); });
  const rp = $("#mq-repose");
  if (rp) rp.addEventListener("click", () => {
    const again = c.reposed ? "\n\nThe front was posed before. A new pose should start from the ORIGINAL front: undo the old pose of the front first (Front view > Undo) if you have not." : "";
    if (!confirm(`Pose ${shownName(c)} like this body?\n\nAnyPose poses the front like the mannequin (Blender, then one Qwen edit, about 12 minutes). Then the right, back and left views and the sheet are made again from it (three more Qwen jobs, about 45 minutes). It waits in the GPU queue.\n\nThe old front and views stay in the history: Undo brings them back.${again}`)) return;
    startJob(`/api/characters/${encodeURIComponent(c.name)}/repose`, new FormData());
  });
  const rd = $("#mq-redraw");
  if (rd) rd.addEventListener("click", async () => {
    rd.disabled = true; rd.textContent = "Drawing the sheet…";
    try {
      const r = await postJSON(`/api/characters/${encodeURIComponent(c.name)}/redraw-sheet`);
      state.char = r.character;
      const ch = r.character;
      toast(ch.sheet_stale ? "The sheet is still out of date." : ch.status === "ready" ? "The sheet was drawn again."
        : "The sheet was drawn again. Press Use as is (Sheet view) to make the character ready.", !!ch.sheet_stale);
      renderChar(); loadChars();
    } catch (e) { fail(e); rd.disabled = false; rd.textContent = "Draw the sheet again"; }
  });
  $("#mq-save").addEventListener("click", async () => {
    const p = mq.preview, old = c.height && c.height.ratio;
    if (p && old && Math.abs(p.ratio - old) / old > 0.05 &&
        !confirm(`Save the new body of ${shownName(c)}?\n\nThe sheet is drawn again at ${p.height_m.toFixed(2)} m (was ${(old * (p.ref_m || 1.806)).toFixed(2)} m).`)) return;
    const btn = $("#mq-save");
    btn.disabled = true; btn.textContent = "Saving and drawing the sheet…";
    try {
      const r = await postJSON(`/api/characters/${encodeURIComponent(c.name)}/stand-in`, { body: JSON.stringify(mqNorm(mq.draft)) });
      state.char = r.character;
      mq.saved = mqNorm(r.character.mannequin || {}); mq.draft = { ...mq.saved };
      const h = r.character.height;
      toast(h && h.m ? `Saved. The sheet now shows ${h.m.toFixed(2)} m.` : "Saved.");
      renderChar(); loadChars();
    } catch (e) { fail(e); btn.disabled = false; btn.textContent = "Save changes"; }
  });
}

// ------------------------------------------------------------------ region edits: a box on the image (AGENTS.md 4.3)
// Small fixes should always use a box: it is cut out, enlarged, edited and pasted back with soft edges, so nothing
// outside it can change. The user drags the box on the shown image; the form sends it in the image's own pixels.
const regionOf = (c, target) => (state.region && state.region.name === c.name && state.region.target === target ? state.region.box : null);
function regionBlock(c, target) {
  const box = regionOf(c, target), q = state.q;
  const img = $(`#char-canvas img[data-target="${target}"]`);
  const size = img && img.naturalWidth ? ` of ${img.naturalWidth}×${img.naturalHeight}` : "";
  return `<div class="region-row">
      <button type="button" id="btn-region" class="ghost ${state.regionMode ? "on" : ""}"
        title="Drag a box on the image: only that box is edited (best for small fixes: a hand, a strap, a mark). Nothing outside the box can change. Describe the box's close-up in the text.">${state.regionMode ? "Drag on the image…" : box ? "Draw the box again" : "Mark a box"}</button>
      <span class="muted small" id="region-text" title="x0,y0,x1,y1 in the image's pixels">${box ? `box ${box.join(", ")}${size}` : "the whole image"}</span>
      ${box ? `<button type="button" class="linkbtn" id="btn-region-clear">Clear</button>` : ""}
      <input type="hidden" name="region" value="${box ? box.join(",") : ""}">
    </div>
    <label class="denoise-l" title="1.0 rebuilds (can add limbs); 0.85 is a good start for a box; 0.8 changes the pose of a hand; 0.5 refines a part painted in code; 0.6 and below change little."><span>Denoise <i>${box ? "0.85 is a good start for a box" : "empty = 1.0 (a full rebuild)"}</i></span>
      <input name="denoise" type="number" step="0.05" min="0.1" max="1" value="${esc(q.denoise ?? "")}" placeholder="${box ? "0.85" : "1.0"}"></label>`;
}
function regionGeom(img) {                           // the picture inside the element (object-fit: contain)
  const b = img.getBoundingClientRect(), nw = img.naturalWidth, nh = img.naturalHeight;
  const s = Math.min(b.width / nw, b.height / nh);
  return { b, s, nw, nh, dx: (b.width - nw * s) / 2, dy: (b.height - nh * s) / 2 };
}
function drawRegionBox(c) {
  const img = state.sel && $(`#char-canvas img[data-target="${state.sel}"]`);
  if (!img) return;
  const host = img.parentElement;
  let el = $(".region-box", host);
  const box = regionOf(c, state.sel);
  if (!box || !img.naturalWidth) { if (el) el.remove(); return; }
  if (!el) { el = document.createElement("div"); el.className = "region-box"; host.append(el); }
  const g = regionGeom(img), hb = host.getBoundingClientRect();
  const left = g.b.left - hb.left + g.dx, top = g.b.top - hb.top + g.dy;
  Object.assign(el.style, { left: `${left + box[0] * g.s}px`, top: `${top + box[1] * g.s}px`,
                            width: `${(box[2] - box[0]) * g.s}px`, height: `${(box[3] - box[1]) * g.s}px` });
}
function bindRegion(c) {
  const f = $("#form-edit"), img = state.sel && $(`#char-canvas img[data-target="${state.sel}"]`);
  if (!f || !img) return;
  const target = state.sel;
  const den = f.denoise;
  if (den) den.addEventListener("input", () => { state.q.denoise = den.value; });
  const show = () => {                                 // the panel's row follows the box without a redraw (the text stays)
    const box = regionOf(c, target);
    f.region.value = box ? box.join(",") : "";
    $("#region-text").textContent = box ? `box ${box.join(", ")} of ${img.naturalWidth}×${img.naturalHeight}` : "the whole image";
    $("#btn-region").textContent = state.regionMode ? "Drag on the image…" : box ? "Draw the box again" : "Mark a box";
    $("#btn-region").classList.toggle("on", !!state.regionMode);
    img.classList.toggle("region-mode", !!state.regionMode);
    if (den) den.placeholder = box ? "0.85" : "1.0";
    drawRegionBox(c);
  };
  $("#btn-region").addEventListener("click", () => { state.regionMode = !state.regionMode; show(); });
  const clr = $("#btn-region-clear");
  if (clr) clr.addEventListener("click", () => { state.region = null; state.regionMode = false; castRender(); });   // a character or a prop (props.js)
  img.draggable = false;
  const toImg = (ev) => {
    const g = regionGeom(img);
    const x = (ev.clientX - g.b.left - g.dx) / g.s, y = (ev.clientY - g.b.top - g.dy) / g.s;
    return [Math.max(0, Math.min(g.nw, Math.round(x))), Math.max(0, Math.min(g.nh, Math.round(y)))];
  };
  let start = null;
  const boxOf = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
  img.addEventListener("pointerdown", (ev) => {
    if (!state.regionMode) return;
    ev.preventDefault();
    start = toImg(ev);
    img.setPointerCapture(ev.pointerId);
  });
  img.addEventListener("pointermove", (ev) => {
    if (!start) return;
    state.region = { name: c.name, target, box: boxOf(start, toImg(ev)) };
    drawRegionBox(c);
  });
  const end = (ev) => {
    if (!start) return;
    const box = boxOf(start, toImg(ev));
    start = null;
    state.regionUp = Date.now();                      // the click that follows the drag must not close the image
    if (box[2] - box[0] < 16 || box[3] - box[1] < 16) {
      state.region = null;
      toast("The box must be at least 16 × 16 pixels of the image.", true);
    } else {
      state.region = { name: c.name, target, box };
      state.regionMode = false;
    }
    show();
  };
  img.addEventListener("pointerup", end);
  img.addEventListener("pointercancel", () => { start = null; });
  if (!img.complete) img.addEventListener("load", show, { once: true });
  show();
}
window.addEventListener("resize", () => { const c = state.char || state.prop; if (c && state.sel) drawRegionBox(c); });

// ------------------------------------------------------------------ quality settings of edits and redos
const qDefaults = () => {
  const o = state.imgOpts || { modes: { quality: { steps: 40, cfg: 3 } },
                               defaults: { sampler: "euler", scheduler: "simple", shift: 3.1, denoise: 1, angles_strength: 0.9 } };
  return { ...o.defaults, ...(o.modes[state.q.mode] || o.modes.quality) };
};
function etaText() {
  const d = qDefaults(), steps = +state.q.steps || d.steps, cfg = state.q.cfg !== undefined && state.q.cfg !== "" ? +state.q.cfg : d.cfg;
  const sec = 30 + steps * (cfg > 1 ? 18 : 9);           // measured on an RTX 3070: ~18 s per step with cfg above 1
  return sec < 90 ? `~${Math.round(sec)} s` : `~${Math.round(sec / 60)} min`;
}
function qualityBlock(isView) {
  const q = state.q, d = qDefaults(), o = state.imgOpts || {};
  const num = (k, label, step, title) => `<label title="${esc(title)}"><span>${label}</span>
      <input name="${k}" type="number" step="${step}" value="${esc(q[k] ?? "")}" placeholder="${esc(d[k] ?? "")}"></label>`;
  const sel = (k, label, list, title) => `<label title="${esc(title)}"><span>${label}</span><select name="${k}">
      <option value="">${esc(d[k])}</option>${(list || []).filter((x) => x !== d[k]).map((x) =>
        `<option ${q[k] === x ? "selected" : ""}>${esc(x)}</option>`).join("")}</select></label>`;
  const changed = ["steps", "cfg", "seed", "sampler", "scheduler", "shift", "negative", "angles_strength"]   // (denoise is
    .filter((k) => q[k] !== undefined && q[k] !== "").length;                                               // shown with the box)
  // the full model only: the Lightning "fast" mode (a distilled shortcut) is shown only when the server offers it
  const seg = (o.modes || {}).fast ? `<div class="seg" role="radiogroup" title="Fast: Lightning LoRA, 4 steps, about 10 times faster, less detail">
      <label><input type="radio" name="mode" value="quality" ${q.mode !== "fast" ? "checked" : ""}><span>Quality</span></label>
      <label><input type="radio" name="mode" value="fast" ${q.mode === "fast" ? "checked" : ""}><span>Fast</span></label></div>`
    : `<input type="hidden" name="mode" value="quality">`;
  return `${seg}
    <details class="adv" ${state.q.open ? "open" : ""}><summary>Settings${changed ? ` · ${changed} changed` : ""}</summary>
      <div class="adv-grid">
        ${num("steps", "Steps", 1, "More steps: more detail, more time")}
        ${num("cfg", "CFG", 0.1, "How strictly the model follows the text")}
        <label title="Empty: a new random seed each time. The history shows the seed of each change."><span>Seed</span>
          <input name="seed" type="number" value="${esc(q.seed ?? "")}" placeholder="random"></label>
        ${num("shift", "Shift", 0.1, "Lower: more fine detail; higher: more stable overall shape")}
        ${sel("sampler", "Sampler", o.samplers, "The method of each sampling step")}
        ${sel("scheduler", "Scheduler", o.schedulers, "How the noise level falls from step to step")}
        ${isView ? num("angles_strength", "Angle", 0.05, "Redo only: how strongly the camera-angle LoRA turns the character") : ""}
        <label class="wide" title="What the model must not draw (works when CFG is above 1)"><span>Negative</span>
          <input name="negative" value="${esc(q.negative ?? "")}" placeholder="what must not be drawn"></label>
      </div>
      ${changed ? `<button type="button" class="linkbtn" id="q-reset">Back to defaults</button>` : ""}
    </details>`;
}
const modeOf = (f) => (f.querySelector("input[name=mode]:checked") || f.querySelector("input[name=mode]") || {}).value || "quality";
function bindQuality() {
  const f = $("#form-edit");
  if (!f) return;
  const save = () => {
    ["steps", "cfg", "seed", "sampler", "scheduler", "shift", "denoise", "negative", "angles_strength"].forEach((k) => {
      if (f[k]) state.q[k] = f[k].value;
    });
    state.q.mode = modeOf(f);
    $("#q-eta").textContent = etaText();
  };
  f.querySelectorAll(".adv input, .adv select").forEach((x) => x.addEventListener("input", save));
  const placeholders = () => {                        // the defaults of the current mode, shown in grey
    const d = qDefaults();
    ["steps", "cfg", "shift", "denoise", "angles_strength"].forEach((k) => { if (f[k]) f[k].placeholder = d[k] ?? ""; });
    $("#q-eta").textContent = etaText();
  };
  f.querySelectorAll("input[name=mode]").forEach((r) => r.addEventListener("change", () => {
    save();
    delete state.q.steps; delete state.q.cfg;           // a new mode brings its own steps and CFG
    f.steps.value = ""; f.cfg.value = "";
    placeholders();
  }));
  const adv = $(".adv", f);
  adv.addEventListener("toggle", () => { state.q.open = adv.open; });
  const reset = $("#q-reset");
  if (reset) reset.addEventListener("click", () => {
    state.q = { mode: state.q.mode, open: true };
    ["steps", "cfg", "seed", "shift", "denoise", "negative", "angles_strength", "sampler", "scheduler"]
      .forEach((k) => { if (f[k]) f[k].value = ""; });
    reset.remove();
    $("summary", adv).textContent = "Settings";
    placeholders();
  });
}
// the form fields of a redo: the same settings, without the text and the edit-only values
function redoForm() {
  const f = $("#form-edit"), out = new FormData();
  out.append("view", state.sel);
  ["mode", "seed", "steps", "cfg", "sampler", "scheduler", "shift", "negative", "angles_strength"].forEach((k) => {
    const v = k === "mode" ? modeOf(f) : (f[k] ? f[k].value : "");
    if (v !== "") out.append(k, v);
  });
  return out;
}
async function loadImageOptions() {
  try { state.imgOpts = await api("/api/image-settings"); } catch { /* defaults stay */ }
}

// the versions of one image: the original, then the result of each change (undone ones dim)
function settingsText(s) {
  if (!s) return "";
  const parts = [`${s.steps} steps`, `cfg ${s.cfg}`, s.seed !== undefined ? `seed ${s.seed}` : null,
    s.mode === "fast" ? "fast" : null, s.sampler && s.sampler !== "euler" ? s.sampler : null,
    s.scheduler && s.scheduler !== "simple" ? s.scheduler : null, s.denoise && +s.denoise !== 1 ? `denoise ${s.denoise}` : null];
  return "\n" + parts.filter(Boolean).join(" · ");
}

function historyBlock(c, target) {
  const list = (c.edits || []).filter((e) => e.target === target);
  if (!list.length) return "";
  const undone = list.filter((e) => e.undone).length;
  const toggle = undone ? `<button type="button" class="linkbtn" id="btn-undone">${state.showUndone ? "Hide undone" : `Show ${undone} undone`}</button>` : "";
  const items = [{ url: list[0].before_url, title: "Original", current: !list.some((e) => e.current) },
    ...list.map((e, i) => ({ url: e.result_url, title: `${i + 1}. ${e.instruction}${e.undone ? " (undone)" : ""}${settingsText(e.settings)}`,
                             undone: !!e.undone, current: !!e.current }))];
  return `<div class="block hist-block"><h2>History${toggle}</h2>
    <div class="thumbs hist">${items.filter((x) => x.url && (state.showUndone || !x.undone)).map((x) =>
      `<img src="${x.url}" class="${x.undone ? "undone" : ""} ${x.current ? "current" : ""}" alt="" title="${esc(x.title)}"
         data-full="${x.url}" data-title="${esc(x.title)}">`).join("")}</div></div>`;
}

function showImage(url, title) {
  const box = $("#player");
  box.innerHTML = `<div class="player"><img src="${url}" alt=""><div class="bar"><span class="grow" title="${esc(title)}">${esc(title)}</span>
    <button class="close" id="btn-close-img" title="Close" aria-label="Close">${ICON.close}</button></div></div>`;
  box.classList.remove("hidden");
  $("#btn-close-img").addEventListener("click", () => { box.classList.add("hidden"); box.innerHTML = ""; });
}

function bindHistory(c) {
  $$("#char-panel .hist img").forEach((im) => im.addEventListener("click", () => showImage(im.dataset.full, im.dataset.title)));
  const tu = $("#btn-undone");
  if (tu) tu.addEventListener("click", () => {
    state.showUndone = !state.showUndone;
    const block = $("#char-panel .hist-block");
    if (block) { block.outerHTML = historyBlock(c, state.sel); bindHistory(c); }   // the edit text stays
  });
}

function select(target) {
  state.sel = state.sel === target ? null : target;
  state.regionMode = false;
  renderChar();
  const t = $("#form-edit textarea");
  if (t) t.focus();
}

function bindPanel(c) {
  const name = encodeURIComponent(c.name), u = c.urls || {};
  $$("#char-canvas img.pickable").forEach((im) => im.addEventListener("click", () => {
    if (im.dataset.target === state.sel && (state.regionMode || Date.now() - (state.regionUp || 0) < 400)) return;   // a box drag
    select(im.dataset.target);
  }));

  bindHistory(c);
  bindRegion(c);
  bindMq(c);
  const oc = $("#btn-cands");
  if (oc) oc.addEventListener("click", () => { state.view = "candidates"; state.sel = null; renderChar(); });
  const desel = $("#btn-desel");
  if (desel) desel.addEventListener("click", () => select(state.sel));
  const fe = $("#form-edit");
  if (fe) {
    fe.instruction.addEventListener("input", () => setDraft(c.name, state.sel, fe.instruction.value));
    fe.addEventListener("submit", (ev) => {
      ev.preventDefault();
      if (fe.target.value === "front" && u.right && !confirm("The views were made from this front view.\nAfter this edit, the views must be made again.")) return;
      const target = state.sel;
      const fd = new FormData(fe);
      if (!fd.get("region")) fd.delete("region");
      startJob(`/api/characters/${name}/edit`, fd).then((ok) => {
        if (!ok) return;
        setDraft(c.name, target, "");
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
  const form = (fields) => { const f = new FormData(); Object.entries(fields).forEach(([k, v]) => f.append(k, v)); return f; };
  const redo = $("#btn-redo");
  if (redo) redo.addEventListener("click", () => startJob(`/api/characters/${name}/redo`, redoForm()));
  bindQuality();
  const undo = $("#btn-undo");
  if (undo) undo.addEventListener("click", () => startJob(`/api/characters/${name}/undo`, form({ target: state.sel })));
  const views = $("#btn-views");
  if (views) views.addEventListener("click", () => startJob(`/api/characters/${name}/views`, new FormData()));
  const fs = $("#form-style");
  if (fs) fs.addEventListener("submit", (ev) => {
    ev.preventDefault();
    if (!fs.style_lora.value) return toast("Choose the style LoRA of your videos first.", true);
    startJob(`/api/characters/${name}/style`, new FormData(fs));
  });
  const fin = $("#btn-finish");
  if (fin) fin.addEventListener("click", async () => {
    try { state.char = (await postJSON(`/api/characters/${name}/finish`)).character; renderChar(); loadChars(); toast("Ready for videos."); }
    catch (e) { fail(e); }
  });
}

async function startJob(url, form) {
  try {
    const d = await api(url, { method: "POST", body: form });
    if (d.job) { watchJob(d.job.id, state.char ? state.char.name : d.job.character); queued(d.job.action); }
    return true;
  } catch (e) { fail(e); return false; }
}

function watchJob(id, name) {
  const tick = async () => {
    let j;
    try { j = await api(`/api/jobs/${id}`); } catch (e) { stopPoll("job"); return fail(e); }
    const bar = $("#jobbar");
    if (state.char && state.char.name === name) {
      const p = j.progress;
      const frac = p && p.max ? ((p.item || 1) - 1 + p.value / p.max) / (p.items || 1) : 0;
      const elapsed = j.started ? Date.now() / 1000 - j.started : 0;
      const part = p && p.items > 1 ? `<span class="ptext">${p.item} of ${p.items}</span>` : "";
      bar.innerHTML = j.status === "queued" || j.status === "running"
        ? `<div class="jobbar"><span class="spinner"></span><b>${esc(j.action === "new_skin" ? "New skin" : j.action)}</b>${part}
             ${j.status === "running" ? progressBar(frac, elapsed)
               : `<span class="ptext">${j.queue_position ? `waiting · ${j.queue_position} ahead` : "waiting"}</span>`}</div>`
        : j.status === "failed" ? `<div class="jobbar err">${esc(j.action)} failed: ${esc(j.error)}</div>` : "";
    }
    if (j.status !== "queued" && j.status !== "running") {
      stopPoll("job");
      if (j.status === "done") toast(j.action === "new_skin" ? SKIN_DONE : `${j.action}: done`);
      else toast(`${j.action}: ${j.status}${j.error ? ` — ${j.error}` : ""}`, true);
      await loadChars();
      const skin = j.status === "done" && j.result && j.result.skin;    // a new skin: open it (the skin menu lists it)
      if (state.char && state.char.name === name) openChar(skin || name);
    }
  };
  tick();
  poll("job", tick, 4000);
}

$("#btn-new-char").addEventListener("click", () => {
  clearChar();
  $("#btn-new-char").classList.add("active");
  $$("#char-list .tile").forEach((t) => t.classList.remove("active"));
  $("#char-head").innerHTML = `<span class="name">New character</span>`;
  const canvas = $("#char-canvas");
  canvas.innerHTML = "";
  canvas.append($("#tpl-new-char").content.cloneNode(true));
  const f = $("#form-new-char");
  const sync = () => {
    const src = f.source.value;
    $$(".only-text", f).forEach((e) => e.classList.toggle("hidden", src !== "text"));
    $$(".only-images", f).forEach((e) => e.classList.toggle("hidden", src !== "images"));
    $$(".only-sheet", f).forEach((e) => e.classList.toggle("hidden", src !== "sheet"));
    $$(".not-sheet", f).forEach((e) => e.classList.toggle("hidden", src === "sheet"));   // a sheet has its views
    // images and sheets carry their own look: no style text; the description is optional there (the video prompt
    // and the look images use it when it is given)
    f.description.required = src === "text";
    f.style.required = src === "text";
    f.description.placeholder = src === "text"
      ? "What the character looks like: a slim young wanderer with a white kitsune mask, a teal hooded cape, a long red scarf…"
      : "Optional: the costume in exact words (colours, garment lengths). The video prompt uses it to keep small details.";
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
    if (src === "images" && !f.front.files.length) return toast("The front image is needed.", true);
    if (src === "sheet" && !f.image.files.length) return toast("Choose the sheet image.", true);
    const form = new FormData(f);
    for (const [k, v] of [...form.entries()]) if (v instanceof File && !v.size) form.delete(k);   // empty slots
    if (src !== "text") form.set("style", "");                     // the images give the look
    try {
      const d = await api("/api/characters", { method: "POST", body: form });
      const name = d.character ? d.character.name : d.job.character;   // the ID the server made from the name
      if (d.character) { await loadChars(name); toast("Character added."); return; }
      queued("New character");
      // the character folder appears when the job starts; wait for it, then show the progress
      const wait = setInterval(async () => {
        let list, job;
        try { [list, job] = await Promise.all([api("/api/characters"), api(`/api/jobs/${d.job.id}`)]); }
        catch { return; }
        if (list.some((c) => c.name === name)) { clearInterval(wait); await loadChars(name); watchJob(d.job.id, name); }
        else if (!["queued", "running"].includes(job.status)) {
          clearInterval(wait);
          toast(`New character: ${job.status}${job.error ? ` — ${job.error}` : ""}`, job.status !== "done");
        }
      }, 2000);
    } catch (e) { fail(e); }
  });
  f.name.focus();
});
