// cinestudio web app - The Video tab: the run gallery, run details and media, the New video form (with its Background block).
// Plain scripts (no modules), loaded in order by index.html; they share their top-level names.
"use strict";

// ------------------------------------------------------------------ video
async function loadVideoTab() {
  try {
    const [shots, chars] = await Promise.all([api("/api/shots"), api("/api/characters")]);
    const sel = $("#shot-select"), cur = sel.value;
    const count = (s) => { const n = (s.mannequins || []).length; return n ? ` · ${n} character${n === 1 ? "" : "s"}${s.slots_scanned === false ? "?" : ""}` : ""; };
    const errs = (s) => (s.check && s.check.errors ? ` · ⚠ ${s.check.errors} check error${s.check.errors > 1 ? "s" : ""}` : "");
    const opts = (list) => list.map((s) => `<option value="${esc(s.blend)}">${esc(s.name)}${s.video ? " (from text)" : ""}${count(s)}${errs(s)}</option>`).join("");
    const tpls = shots.filter((s) => s.template && !s.builtin), hand = shots.filter((s) => !s.template);
    const built = shots.filter((s) => s.builtin);
    sel.innerHTML = (built.length ? `<optgroup label="Built-in templates">${opts(built)}</optgroup>` : "")
      + (tpls.length ? `<optgroup label="My templates">${opts(tpls)}</optgroup>` : "")
      + (hand.length ? `<optgroup label="Hand-made shots">${opts(hand)}</optgroup>` : "")
      + `<option value="__own__">Choose my own…</option>`;
    if (cur) sel.value = cur;
    const own = () => {                                   // the path box only for "Choose my own"
      const box = $("#form-video").blend_path;
      box.classList.toggle("hidden", sel.value !== "__own__");
      if (sel.value === "__own__") box.focus();
    };
    state.shots = shots;                                  // tplDefaultBg: a template's default background
    const checkMark = () => {                           // a template with check errors (shot check): a chip by the legend
      const s = shots.find((x) => x.blend === sel.value);
      $("#shot-check").innerHTML = s && s.check && s.check.errors ? checkChip(s.check, true) : "";
    };
    sel.onchange = () => {
      own(); showShotPreview(); checkMark();
      const s = shots.find((x) => x.blend === sel.value), v = $("#form-video").video;
      if (s && s.suggest && !v.value.trim()) v.value = s.suggest;   // a text template: its scene and action, no camera words
    };
    $("#form-video").blend_path.onchange = showShotPreview;
    own();
    showShotPreview();
    checkMark();
    loadBackgrounds();
    loadSongChoices();
    state.readyChars = chars.filter((c) => c.status === "ready");
    fillCharSelect();
    fillClipRows.key = null; fillClipRows();          // the template's clips (one prompt per clip)
  } catch (e) { fail(e); }
  loadLoras();
  loadRuns();
  poll("runs", loadRuns, 6000);
}

// busy = its worker is alive (the list says so); a "new" or "running" run whose worker is gone is stopped and can go on
const busyRun = (r) => (r.worker_alive !== undefined ? !!r.worker_alive
  : r.status === "running" || r.status === "new" || r.status === "queued");
// a stopped run that can go on (a paused clip gauntlet, a failed or interrupted step, a worker that never started)
const canContinue = (r) => !busyRun(r) && !r.retired && !["done", "canceled", "unreadable"].includes(r.status);
const continueLabel = (r) => (r.status === "paused" ? "Next clip" : r.status === "unresolved" ? "Submit again" : "Continue");
async function continueRun(id, status, body = {}) {
  if (status === "unresolved" && body.through_clip === undefined) {
    if (!confirm("The last step's ComfyUI job could not be confirmed (the engine may have restarted).\n\nSubmit it again? If the old job still exists, it can run twice.")) return false;
    body = { ...body, new_attempt: true };
  }
  try {
    const r = await postJSON(`/api/runs/${encodeURIComponent(id)}/continue`, body);
    if (r.start.ok) queued("Video"); else toast(`The worker did not start: ${r.start.error}`, true);
    loadRuns.last = null; loadRuns();
    return true;
  } catch (e) { fail(e); return false; }
}
const runDate = (r) => {
  const d = r.created ? new Date(r.created) : null;
  const made = d && !isNaN(d) ? d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : r.id;
  const u = r.updated ? new Date(r.updated) : null;          // the video changed later (a retake was put in)
  if (!u || isNaN(u) || !d || isNaN(d) || u - d < 10 * 60 * 1000) return made;
  const same = u.toDateString() === d.toDateString();
  return `${made} · video ${u.toLocaleString([], same ? { hour: "2-digit", minute: "2-digit" }
    : { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`;
};


const runPager = makePager("videos", (force) => { if (force) loadRuns.last = null; loadRuns(); });

// the Sound block's song: a finished song from the Music tab as the video's sound (Blender shots only)
async function loadSongChoices() {
  const sel = $("#song-select");
  if (!sel) return;
  const cur = sel.value;
  let list = [];
  try { list = ((await api("/api/songs")).songs || []).filter((s) => s.status === "done"); } catch { /* none */ }
  sel.innerHTML = `<option value="">No song</option>` + list.map((s) => {
    const b = s.beats || {};
    const facts = [b.bpm ? `${Math.round(b.bpm)} BPM` : null, b.seconds ? `${Math.round(b.seconds)} s` : null].filter(Boolean).join(", ");
    return `<option value="${esc(s.name)}" data-seconds="${esc(b.seconds || "")}">${esc(s.name)}${facts ? ` (${facts})` : ""}</option>`;
  }).join("");
  if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
  songFormState();
}
function songFormState() {
  const sel = $("#song-select");
  if (!sel) return;
  const opt = sel.selectedOptions[0], on = !!sel.value;
  $("#song-start-field").classList.toggle("hidden", !on);
  const secs = opt && +opt.dataset.seconds;
  $("#form-video").song_start.max = secs ? Math.max(0, secs - 0.1).toFixed(1) : "";
  $("#song-hint").textContent = on ? "The song replaces the template's sound and LTX's sound." : "";
}
$("#song-select").addEventListener("change", songFormState);

// the Background block of the New video form: a 360 place from the text, a saved one, ONE picture of the place for a
// still camera (the backplate: sharper, 2026-10-23), or the template's own set
async function loadBackgrounds() {
  const sel = $("#bg-select"), cur = sel.value;
  let list = [];
  try { list = await api("/api/backgrounds?flat=1"); } catch { /* none yet */ }
  state.bgKinds = Object.fromEntries(list.map((b) => [`saved:${b.name}`, b.kind || "pano"]));
  const group = (label, rows) => rows.length ? `<optgroup label="${esc(label)}">${rows.map((b) =>
    `<option value="saved:${esc(b.name)}" title="${esc(b.camera || "")}">${esc((b.scene || b.name).slice(0, 60))}</option>`).join("")}</optgroup>` : "";
  sel.innerHTML = `<option value="">The template's own set</option><option value="text">A 360° place from the text</option>`
    + `<option value="plate">One picture of the place (still camera, sharper)</option>`
    + `<option value="look-scene" title="No panorama or backplate: a second picture edit puts the place into the look image (the grey Blender room becomes the place). Keyframe looks are off.">No background picture: the place in the look image</option>`
    + group("Saved 360° places", list.filter((b) => b.kind !== "plate"))
    + group("Saved backplates (still camera)", list.filter((b) => b.kind === "plate"));
  if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
  tplDefaultBg();
  bgFormState();
}
// a template with a default background (shot.json "background": a built-in template's backplate drawn over its set):
// chosen while the Background choice is still at its default (the template's own set) or at the default that an
// earlier template chose; a choice of the user's own stays (a new template resets only a choice of the own set).
// The run records it as usual.
function tplDefaultBg(shotChanged = false) {
  const sel = $("#bg-select"), f = $("#form-video");
  if (shotChanged && sel.value === "") delete sel.dataset.user;
  if (sel.dataset.user) return;
  const s = (state.shots || []).find((x) => x.blend === f.blend.value);
  const want = s && s.background && [...sel.options].some((o) => o.value === `saved:${s.background}`) ? `saved:${s.background}` : "";
  const auto = sel.dataset.auto || "";
  if (sel.value === "" || sel.value === auto) {
    sel.value = want;
    sel.dataset.auto = want;
  }
}
function bgFormState() {
  const f = $("#form-video"), sel = $("#bg-select");
  const video = (f.blend.value || "").startsWith("template:");        // a text template keeps its own scenery
  sel.disabled = video;
  if (video) sel.value = "";
  const words = ["text", "plate", "look-scene"].includes(sel.value);
  $("#bg-text").classList.toggle("hidden", !words);
  $("#bg-text").placeholder = sel.value === "look-scene"
    ? "The place for the look image, e.g. a sunny school gym with a ball cart at the far wall (empty = the video text)"
    : "The place, e.g. a neon-lit rooftop in the rain at night (empty = the video text)";
  const plate = sel.value === "plate" || (state.bgKinds || {})[sel.value] === "plate";
  $("#bg-ground-field").classList.toggle("hidden", !sel.value || plate || sel.value === "look-scene");   // only a 360 place has one
  // a template video has no Blender scene: no stand-in parts on a mannequin
  $$("#form-video [data-blend-only]").forEach((el) => el.classList.toggle("hidden", video));
}
// the chosen characters: slot 1's fills the Style text (its own style words) unless the user changed the text; each
// slot says whether its character has stand-in parts (proxy kit)
const FALLBACK_STYLE = "cinematic fantasy anime cel-shaded illustration with hand-drawn linework and painted shading";
// ONE character select per figure of the template (user, 2026-10-25: "if a video only has 1 character there should
// only be one character option, if it has three then three options"): the slots come from the template's Mannequin*
// layers (/api/shots "mannequins"); "Choose my own" and a template without a layer scan have one slot.
function formShot() {
  const f = $("#form-video");
  return (state.shots || []).find((x) => x.blend === f.blend.value) || null;
}
function formSlots() {
  const s = formShot();
  return s && s.mannequins && s.mannequins.length ? s.mannequins
    : [{ index: 1, collection: "Mannequin", label: "Character 1", side: "centre" }];
}
function slotTitle(slot, n, video) {
  if (n === 1) return video ? "Character (replaces the stand-in)" : "Character";
  return `${slot.label || `Character ${slot.index}`}${slot.side ? ` (${slot.side})` : ""}`;
}
// a character needs its own mannequin body and stand-in parts before a BLENDER video (user, 2026-10-24); a video
// template has no mannequin, so there every ready character can be chosen (Codex review). The others show, but cannot
// be chosen, with what they still need.
function fillCharSelect() {
  const ready = state.readyChars || [];
  const video = ($("#form-video").blend.value || "").startsWith("template:");
  const missing = (c) => (c.missing || []).filter((k) => !video || k === "sheet");
  const usable = ready.filter((c) => !missing(c).length);
  state.videoChars = usable;
  const slots = formSlots(), n = slots.length, list = $("#slot-list");
  const cur = Object.fromEntries($$(".slot-char", list).map((s) => [s.dataset.slot, s.value]));
  $$(".slot", list).slice(1).forEach((row) => row.remove());
  for (const slot of slots.slice(1)) {                      // rows 2..n (row 1 is in the page: name="character")
    const row = document.createElement("label");
    row.className = "fl slot";
    row.dataset.slot = slot.index;
    row.innerHTML = `<span class="slot-head"><b class="slot-num">${esc(slot.index)}</b><span class="slot-label"></span></span>
      <select class="slot-char" data-slot="${esc(slot.index)}"></select><i class="hint slot-kit"></i>`;
    list.append(row);
  }
  const taken = [];
  $$(".slot", list).forEach((row, i) => {
    const slot = slots[i], sel = $(".slot-char", row);
    $(".slot-label", row).textContent = slotTitle(slot, n, video);
    row.title = n > 1 ? `The character that plays ${slot.collection || "this figure"}${slot.side ? `, the ${slot.side} figure in the picture` : ""}`
      : "The character that plays the template's figure (only ready characters: front, views, sheet and mannequin body)";
    sel.dataset.collection = slot.collection || "";
    sel.innerHTML = (i > 0 ? `<option value="">Choose a character…</option>` : "") + (ready.length ? ready.map((c) => missing(c).length
      ? `<option value="${esc(c.name)}" disabled>${esc(charLabel(c))} (${esc(readyText(c))})</option>`
      : `<option value="${esc(c.name)}">${esc(charLabel(c))}</option>`).join("") : (i ? "" : `<option value="">No character is ready</option>`));
    const was = cur[String(slot.index)];
    const pick = was && usable.some((c) => c.name === was) ? was
      : (usable.find((c) => !taken.includes(c.name)) || (i ? { name: "" } : usable[0]) || { name: "" }).name;
    sel.value = pick || "";
    taken.push(sel.value);
  });
  $("#slot-chip").innerHTML = slotChip({ mannequins: slots, slots_scanned: (formShot() || {}).slots_scanned });
  slotPoster();
  fillPropRows();
  charFormState();
}
// a small poster of the template with a numbered marker over each figure (the scan's projected head; templates
// scanned before 2026-10-25 have no positions: the poster shows without markers, nothing is guessed)
function slotPoster() {
  const box = $("#slot-poster"), s = formShot(), slots = formSlots();
  const show = !!(s && s.poster_url && slots.length > 1);
  box.classList.toggle("hidden", !show);
  $("#slots").classList.toggle("with-poster", show);
  if (!show) { box.innerHTML = ""; return; }
  const marks = s.markers ? slots.map((x) => {
    const [u, v] = x.screen.head;
    return `<span class="slot-mark" style="left:${(u * 100).toFixed(1)}%;top:max(22px, ${(v * 100 - 3).toFixed(1)}%)" title="${esc(slotTitle(x, slots.length))}">${esc(x.index)}</span>`;
  }).join("") : "";
  box.innerHTML = `<div class="slot-frame"><img src="${s.poster_url}" alt="The template's first frame">${marks}</div>
    <figcaption>${s.markers ? "The numbers show who plays which figure (first frame)." : "The figures' places were not recorded: the slots follow the template's mannequins in order (Mannequin = 1). Templates tab > Layers > Read the .blend again records them."}</figcaption>`;
}
// "Objects in this template": the template's prop groups (its Props layer) with their words; the user's words
// replace the template's in the clip texts (empty = the template's words). Typed words stay per template.
function fillPropRows() {
  const s = formShot(), groups = (s && !s.video && s.props) || [];
  const box = $("#prop-slots"), rows = $("#prop-rows");
  state.propWords = state.propWords || {};
  $$(".prop-words", rows).forEach((x) => { (state.propWords[x.dataset.blend] ||= {})[x.dataset.pattern] = x.value; });
  box.classList.toggle("hidden", !groups.length);
  const kept = (s && state.propWords[s.blend]) || {};
  const pics = propPics(s), full = Object.keys(pics).length >= MAX_PROP_PICTURES;
  const refs = propRefMap(s);
  rows.innerHTML = groups.map((g) => {
    const pic = pics[g.pattern], lib = propLibItem(refs[g.pattern]);
    const picHtml = lib && !pic
      ? `<img class="prop-thumb" src="${esc(lib.thumb_url || "")}" alt="" title="${esc(`${propRefLabel(lib)}: its picture goes into a prop panel of this run's sheet`)}">`
      : pic
      ? `<img class="prop-thumb" src="${esc(pic.url)}" alt="" title="${esc(`${pic.name}: drawn into a prop panel of this run's sheet`)}">
         <button type="button" class="prop-pic-del" data-pattern="${esc(g.pattern)}" title="Remove the picture" aria-label="Remove the picture of ${esc(g.label)}">×</button>`
      : `<button type="button" class="prop-pic-add" data-pattern="${esc(g.pattern)}" ${full ? "disabled" : ""}
           title="${esc(full ? `A run takes at most ${MAX_PROP_PICTURES} prop pictures` : "A picture of this object (best: the object alone on black, in the characters' style). It goes into a prop panel of the run's sheet; not A/B-tested yet.")}">Picture…</button>`;
    return `<div class="prop-row" title="${esc(`${g.count} object${g.count === 1 ? "" : "s"} (${g.objects.slice(0, 8).join(", ")}${g.objects.length > 8 ? " …" : ""}) · pattern ${g.pattern}`)}">
      <span class="prop-name">${esc(g.label)}${g.count > 1 ? ` <i>×${esc(g.count)}</i>` : ""}</span>
      ${propRefSelect(g, s, lib)}
      <input class="prop-words" data-pattern="${esc(g.pattern)}" data-blend="${esc(s.blend)}" maxlength="300" aria-label="Words for ${esc(g.label)}"
        placeholder="${esc(lib ? `own words (empty = the prop's: ${lib.words || ""})` : g.words || "no words yet: describe them, e.g. red rubber balls")}" value="${esc(kept[g.pattern] || "")}">
      <span class="prop-pic">${picHtml}</span></div>`;
  }).join("");
}
// each object row can CHOOSE a prop from the library (the Cast tab's Props, 2026-10-25): the run then takes the prop's
// picture (a prop panel of the run's sheet) and its words (the server resolves prop_refs {pattern: prop ID}); the row's
// own words or own picture still win. Choices are kept per template.
function propRefMap(s) {
  state.propRefs = state.propRefs || {};
  return (s && (state.propRefs[s.blend] ||= {})) || {};
}
const propLibItem = (name) => (name && (state.videoProps || []).find((x) => x.name === name)) || null;
function propRefLabel(x) {                          // "Dodgeball · Anime noir 3D" when the prop has more than one skin
  const base = x.skin_of || x.name, fam = (state.videoProps || []).filter((y) => (y.skin_of || y.name) === base);
  const main = fam.find((y) => !y.skin_of) || x, shown = main.display_name || main.name;
  if (fam.length < 2) return shown;
  const st = x.style_ref && typeof styleByName === "function" ? styleByName(x.style_ref) : null;
  return `${shown} · ${st ? st.label || st.name : x.style_ref || "Original"}`;
}
function propRefSelect(g, s, lib) {
  const list = (state.videoProps || []).filter((x) => x.status === "ready");
  if (state.videoProps === undefined) loadVideoProps();
  return `<select class="prop-ref" data-pattern="${esc(g.pattern)}" data-blend="${esc(s.blend)}" aria-label="A library prop for ${esc(g.label)}"
      title="A prop from the library (Cast tab > Props): its picture and words go into the run">
      <option value="">${list.length ? "No library prop" : "No props yet"}</option>
      ${list.map((x) => `<option value="${esc(x.name)}" ${lib && lib.name === x.name ? "selected" : ""}>${esc(propRefLabel(x))}</option>`).join("")}</select>`;
}
async function loadVideoProps() {
  state.videoProps = state.videoProps || null;      // null: loading (one request)
  if (state.videoPropsLoading) return;
  state.videoPropsLoading = true;
  try { state.videoProps = await api("/api/props"); } catch { state.videoProps = []; }
  state.videoPropsLoading = false;
  if ($("#prop-rows .prop-row")) fillPropRows();
}
function propRefs() {
  const s = formShot(), refs = propRefMap(s), out = {};
  $$("#prop-rows .prop-words").forEach((x) => { if (refs[x.dataset.pattern]) out[x.dataset.pattern] = refs[x.dataset.pattern]; });
  return out;
}
$("#prop-rows").addEventListener("change", (ev) => {
  const sel = ev.target.closest(".prop-ref");
  if (!sel) return;
  const refs = propRefMap(formShot());
  if (sel.value) refs[sel.dataset.pattern] = sel.value; else delete refs[sel.dataset.pattern];
  fillPropRows();
});
function propWords() {
  const out = {};
  $$("#prop-rows .prop-words").forEach((x) => { if (x.value.trim()) out[x.dataset.pattern] = x.value.trim(); });
  return out;
}
// each object row's picture (2026-10-25): uploaded at once (POST /api/prop-images), kept per template; the run copies
// it and draws it into a prop panel of its sheet. The thumbnail is the browser's own copy of the file.
const MAX_PROP_PICTURES = 2;
function propPics(s) {
  state.propPics = state.propPics || {};
  return (s && (state.propPics[s.blend] ||= {})) || {};
}
function propPictures() {
  const s = formShot(), pics = propPics(s), out = {};
  $$("#prop-rows .prop-words").forEach((x) => { if (pics[x.dataset.pattern]) out[x.dataset.pattern] = pics[x.dataset.pattern].path; });
  return out;
}
$("#prop-rows").addEventListener("click", (ev) => {
  const add = ev.target.closest(".prop-pic-add"), del = ev.target.closest(".prop-pic-del");
  if (del) {
    const pics = propPics(formShot());
    if (pics[del.dataset.pattern]) URL.revokeObjectURL(pics[del.dataset.pattern].url);
    delete pics[del.dataset.pattern];
    fillPropRows();
  } else if (add && !add.disabled) {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = "image/png,image/jpeg,image/webp";
    inp.addEventListener("change", () => inp.files[0] && uploadPropPicture(add.dataset.pattern, inp.files[0]));
    inp.click();
  }
});
async function uploadPropPicture(pattern, file) {
  const s = formShot();
  if (!s) return;
  const fd = new FormData();
  fd.append("image", file);
  toast("Uploading the picture…");
  try {
    const r = await api("/api/prop-images", { method: "POST", body: fd });
    propPics(s)[pattern] = { path: r.path, url: URL.createObjectURL(file), name: file.name };
    toast("Picture added: it goes into a prop panel of the run's sheet.");
    if ((formShot() || {}).blend === s.blend) fillPropRows();   // the form may show another template by now
  } catch (e) { fail(e); }
}
// each slot's stand-in parts note, the slot messages and the Generate button (more figures than a run supports, the
// same character twice, a slot without a character)
function slotState() {
  const f = $("#form-video"), video = (f.blend.value || "").startsWith("template:");
  const slots = formSlots(), sels = $$("#slot-list .slot-char"), off = f.proxy.value === "off";
  sels.forEach((sel) => {
    const c = (state.videoChars || []).find((x) => x.name === sel.value), kit = $(".slot-kit", sel.closest(".slot"));
    kit.textContent = video || !c ? "" : off ? "stand-in parts off" : c.proxy ? "stand-in parts ✓" : "no stand-in parts yet (bare mannequin)";
    kit.classList.toggle("ok", !!(c && c.proxy && !off && !video));
  });
  const msgs = [], names = sels.map((s) => s.value);
  let block = false;
  if (slots.length > MAX_VIDEO_CHARACTERS) {
    msgs.push(`This template has ${slots.length} characters; videos support up to ${MAX_VIDEO_CHARACTERS} for now.`);
    block = true;
  }
  const dup = names.findIndex((x, i) => x && names.indexOf(x) !== i);
  if (dup > 0)                                             // allowed (twins, a clone): a note, not a block
    msgs.push(`${slotTitle(slots[names.indexOf(names[dup])], slots.length)} and ${slotTitle(slots[dup], slots.length)} are the same character: it appears twice in the video.`);
  if (slots.length > 2 && !block)
    msgs.push(`${slots.length} characters: one look edit per figure (about 4 minutes each) and a smaller sheet (${slots.length === 3 ? "front and back" : "front"} of each).`);
  const empty = sels.findIndex((s, i) => i > 0 && !s.value);
  if (empty > 0 && !block) { msgs.push(`Choose a character for ${slotTitle(slots[empty], slots.length)}.`); block = true; }
  const shot = formShot();
  if (shot && shot.slots_scanned === false) msgs.push("This template's layers were not read yet: one character is assumed (Templates tab > Layers reads them).");
  const box = $("#slot-msg");
  box.innerHTML = msgs.map((m) => `<div>${esc(m)}</div>`).join("");
  box.classList.toggle("hidden", !msgs.length);
  box.classList.toggle("warn", block);
  const go = $("#form-video button[type=submit]");
  go.disabled = block;
  go.title = block ? msgs[0] : "";
}
function charFormState() {
  const f = $("#form-video"), c = (state.videoChars || []).find((x) => x.name === f.character.value);
  const want = (c && c.style) || FALLBACK_STYLE;
  if (!f.style.value.trim() || f.style.value === f.style.dataset.auto) { f.style.value = want; f.style.dataset.auto = want; }
  slotState();
}
$("#slot-list").addEventListener("change", (ev) => {
  if (!ev.target.classList.contains("slot-char")) return;
  if (ev.target.dataset.slot === "1") charFormState(); else slotState();
});
$("#form-video").proxy.addEventListener("change", slotState);
$("#bg-select").addEventListener("change", () => { $("#bg-select").dataset.user = "1"; bgFormState(); });
// Layout-To-Render draws no skeleton and does not start from the render: the skeleton and start-noise fields only
// matter for video-to-video, so they show only in that mode.
function renderModeState() {
  const f = $("#form-video");
  const v2v = (f.blend.value || "").startsWith("template:");   // only template videos run video-to-video
  $$("#form-video [data-v2v-only]").forEach((el) => el.classList.toggle("hidden", !v2v));
  $$("#form-video [data-layout-only]").forEach((el) => el.classList.toggle("hidden", v2v));
  // Full-model options do not apply to the supported distilled video model.
  const full = false;
  $$("#form-video [data-full-model]").forEach((el) => el.classList.toggle("hidden", !full));
}
renderModeState();
$("#shot-select").addEventListener("change", () => { tplDefaultBg(true); bgFormState(); renderModeState(); fillCharSelect(); fillClipRows(); });
$("#video-model-select").addEventListener("change", renderModeState);

// "Text per clip": ONE PROMPT PER CLIP (user, 2026-10-25: "whatever happened in clip 1 should not be described in
// clip 3"; AGENTS.md 7.2). Each clip's prompt = the text above (the SHARED context: who, place, light, look) + the
// template's clip context + that clip's ACTIONS below. A library template lists its clips (/api/shots "clips": frames,
// seconds and the template's own text); the rows are prefilled with the template's texts, a row emptied shows the
// template's text as placeholder and uses it. A clip with no action text at all is flagged before Generate.
// A shot of your own (no clip list): rows added by hand (clip k = frames 96(k-1)+1 .. +121 at 24 fps).
function clipSpan(k) { const a = 4 * (Math.max(1, k) - 1); return `${a}-${a + 5} s`; }
const CT_PH = "Only what this clip's frames show, e.g. the character turns to the camera and waves";
function clipPlan() { const s = formShot(); return (s && s.clips) || []; }
function clipRowEl(k, text, tpl, fixed, span) {
  const row = document.createElement("div");
  row.className = "ct-row" + (fixed ? " fixed" : "");
  row.dataset.tpl = tpl || "";
  row.innerHTML = fixed
    ? `<div class="ct-num" title="Clip ${esc(k)} of the shot"><span>Clip</span><b>${esc(k)}</b>
        <input class="ct-k" type="hidden" value="${esc(k)}"></div>
      <span class="ct-span sub">${esc(span)}${tpl ? " · the template's action text (edit it, or empty = the template's)" : " · no template text: write what happens"}</span>
      <textarea class="ct-text" rows="${Math.min(6, Math.max(2, Math.ceil(Math.max((text || "").length, (tpl || "").length) / 62)))}" maxlength="1500" placeholder="${esc(tpl || CT_PH)}">${esc(text)}</textarea>`
    : `<label class="ct-num" title="The clip number (clip 1 starts the shot)"><span>Clip</span>
        <input class="ct-k" type="number" min="1" max="40" step="1" value="${esc(k)}"></label>
      <span class="ct-span sub">${esc(clipSpan(k))}</span>
      <textarea class="ct-text" rows="2" maxlength="1500" placeholder="${esc(CT_PH)}">${esc(text)}</textarea>
      <button type="button" class="ct-del" title="Remove this clip's text" aria-label="Remove">×</button>`;
  const num = $(".ct-k", row);
  if (!fixed) {
    num.addEventListener("input", () => { $(".ct-span", row).textContent = clipSpan(+num.value || 1); });
    $(".ct-del", row).addEventListener("click", () => { row.remove(); clipWarnState(); });
  }
  $(".ct-text", row).addEventListener("input", () => {
    const d = (state.clipDrafts = state.clipDrafts || {});         // typed texts stay per template
    const key = $("#form-video").blend.value;
    d[key] = { ...(d[key] || {}), [num.value]: $(".ct-text", row).value };
    clipWarnState();
  });
  return row;
}
function addClipTextRow(k, text = "") {
  const box = $("#clip-text-rows");
  const used = $$(".ct-k", box).map((x) => +x.value);
  if (!k) { k = 2; while (used.includes(k)) k++; }
  const row = clipRowEl(k, text, "", false);
  box.append(row);
  $(".ct-text", row).focus();
  clipWarnState();
}
$("#btn-add-clip-text").addEventListener("click", () => addClipTextRow());
// the template's clips as fixed rows (a new template: its own texts; what was typed for it before stays)
function fillClipRows() {
  const box = $("#clip-text-rows"), plan = clipPlan(), s = formShot();
  const key = $("#form-video").blend.value;
  if (fillClipRows.key === key && box.children.length) { clipWarnState(); return; }
  fillClipRows.key = key;
  const drafts = (state.clipDrafts || {})[key] || {};
  box.innerHTML = "";
  for (const c of plan) {
    const typed = drafts[String(c.clip)];
    const span = `frames ${c.first}-${c.last} · ${(+c.start_s).toFixed(1)}-${(+c.end_s).toFixed(1)} s`;
    box.append(clipRowEl(c.clip, typed !== undefined ? typed : (c.text || ""), c.text || "", true, span));
  }
  $("#btn-add-clip-text").classList.toggle("hidden", plan.length > 0);
  const ctx = (s && s.clip_context) || "";
  const slot = plan.some((c) => (c.text || "").includes("{character}")) || ctx.includes("{character}");
  $("#clip-context").textContent = [ctx ? `Every clip also gets the template's context: "${ctx.replace(/\{character\}/g, "the character")}"` : "",
    slot ? "{character} in a text becomes \"the character\"." : ""].filter(Boolean).join(" ");
  $("#clip-context").classList.toggle("hidden", !ctx && !slot);
  $("#clip-texts").open = plan.length > 1 || (!plan.length && box.children.length > 0);
  const many = plan.length > 1;                  // the main text is the shared context of every clip
  $("#video-text-hint").textContent = many ? "the place, light and look (each clip's action is below)" : "";
  $("#form-video").video.placeholder = many
    ? "The place, the light and the look that hold in EVERY clip, e.g. a dim school gym at night lit by violet light. Each clip's action goes into Text per clip below."
    : "What happens: the character slowly raises its right hand and waves, then lowers it. Write the place, the light and the action; a template already brings the camera move.";
  clipWarnState();
}
// the clips that would get no action text of their own (a multi-clip template): a warning in the section
function clipsWithoutActions() {
  const plan = clipPlan();
  if (plan.length < 2) return [];
  return $$("#clip-text-rows .ct-row").filter((r) => !$(".ct-text", r).value.trim() && !r.dataset.tpl)
    .map((r) => +$(".ct-k", r).value);
}
function clipWarnState() {
  const w = $("#clip-text-warn"), none = clipsWithoutActions(), n = clipPlan().length;
  w.textContent = !none.length ? "" : none.length === n
    ? `No clip has an action text: all ${n} clips would get only the text above. Write what each clip's frames show.`
    : `Clip ${none.join(", ")} ${none.length > 1 ? "have" : "has"} no action text: ${none.length > 1 ? "they get" : "it gets"} only the text above.`;
  w.classList.toggle("hidden", !none.length);
}
function clipTexts() {
  const out = {};
  for (const row of $$("#clip-text-rows .ct-row")) {
    const k = parseInt($(".ct-k", row).value, 10), t = $(".ct-text", row).value.trim();
    if (!t || t === (row.dataset.tpl || "").trim()) continue;   // empty or the template's own text: the run takes it
    if (!(k >= 1)) throw new Error("Text per clip: give each text a clip number (1, 2, 3 ...).");
    if (out[k]) throw new Error(`Text per clip: clip ${k} has two texts.`);
    out[k] = t;
  }
  return out;
}

async function loadRuns() {
  const grid = $("#run-grid");
  const p = runPager;
  p.measure(grid); p.watch(grid);
  let res;
  try { res = await api(`/api/runs?offset=${p.offset()}&limit=${p.per()}`); } catch { return; }
  state.runs = res.items;
  const sig = JSON.stringify([res, p.mode, p.rows, p.cols, p.page]);
  if (sig === loadRuns.last) return;              // nothing changed: keep the page (and a playing preview) as it is
  loadRuns.last = sig;
  if (!drawPaging(p, res.total, res.items.length, "video", $("#run-bar"), $("#run-pager"), grid)) { loadRuns.last = null; return loadRuns(); }
  const plus = `<button type="button" class="gtile new-tile" id="btn-new-video" title="New video" aria-label="New video">
      <span class="plus-box"><svg viewBox="0 0 24 24" width="30" height="30" aria-hidden="true"><path fill="currentColor" d="M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6z"/></svg></span>
      <span class="plus-label">New video</span></button>`;
  grid.innerHTML = plus + (state.runs.length ? state.runs.map((r) => `<div class="gtile" data-id="${esc(r.id)}" title="${esc(r.id)}">
      ${r.final_url ? `<video disablepictureinpicture src="${r.final_url}" ${r.contact_url ? `poster="${r.contact_url}"` : ""} muted loop preload="none"></video>`
        : r.contact_url ? `<img src="${r.contact_url}" alt="" loading="lazy">` : ""}
      ${busyRun(r) && !r.contact_url ? `<div class="busy"><span class="spinner"></span></div>` : ""}
      <div class="meta">${r.status === "done" ? "" : `<span class="chip ${esc(r.status)}">${esc(r.status)}</span>`}
        <span class="grow">${esc(runDate(r))}</span>${r.quality === "preview" ? `<span class="chip">preview</span>` : ""}</div>
      ${canContinue(r) ? `<button type="button" class="tile-go" data-go="${esc(r.id)}" data-status="${esc(r.status)}"
        title="${r.status === "paused" ? "Make the next clip (it waits in the queue)" : "Start this video again where it stopped (it waits in the queue)"}">${esc(continueLabel(r))}</button>` : ""}
      ${busyRun(r) ? "" : trashButton("run", r.id)}</div>`).join("")
    : `<div class="empty">No videos yet: press + to make one</div>`);
  $("#btn-new-video").addEventListener("click", openNewVideo);
  $$(".tile-go", grid).forEach((b) => b.addEventListener("click", (ev) => {
    ev.stopPropagation();
    b.disabled = true;
    continueRun(b.dataset.go, b.dataset.status).finally(() => { b.disabled = false; });
  }));
  $$(".gtile[data-id]", grid).forEach((t) => {
    t.addEventListener("click", () => openRun(t.dataset.id));
    const v = $("video", t);
    if (v) { t.addEventListener("mouseenter", () => v.play().catch(() => {})); t.addEventListener("mouseleave", () => { v.pause(); v.currentTime = 0; }); }
  });
  bindDeletes(grid, (id) => { if (state.run === id) closeRun(); loadRuns.last = null; loadRuns(); });
}

function closeRun() {
  stopPoll("run");
  state.run = null;
  const p = $("#player");
  const v = $("video", p); if (v) v.pause();
  p.classList.add("hidden"); p.innerHTML = "";
}

// a small Workbench preview of the shot chosen in the Video form (made once by the app, then kept)
async function showShotPreview() {
  const box = $("#shot-preview"), f = $("#form-video");
  const blend = f.blend.value === "__own__" ? f.blend_path.value.trim() : f.blend.value;
  clearTimeout(showShotPreview.timer);
  showShotPreview.want = blend;
  if (!blend) { box.innerHTML = ""; return; }
  let r;
  try { r = await postJSON("/api/shots/preview", { blend }); }
  catch (e) { box.innerHTML = `<div class="sub">${esc(e.message)}</div>`; return; }
  if (showShotPreview.want !== blend) return;                    // another shot was chosen meanwhile
  if (r.status === "ready") {
    if (box.dataset.url !== r.url) {
      box.dataset.url = r.url;
      box.innerHTML = `<video disablepictureinpicture src="${r.url}" muted loop autoplay playsinline controls title="Blender preview (Workbench)"></video>
        <div class="sub">${r.video_template ? "The template video: its grey stand-in becomes your character"
          : "Blender preview of this shot (quick Workbench render, not the final light)"}</div>`;
    }
  } else if (r.status === "making" || r.status === "queued") {
    box.dataset.url = "";
    box.innerHTML = `<div class="sub"><span class="spinner"></span> ${r.status === "queued" ? "Waiting for Blender…" : "Blender makes a preview of this shot…"}</div>`;
    showShotPreview.timer = setTimeout(showShotPreview, 3000);
  } else {
    box.dataset.url = "";
    box.innerHTML = `<div class="sub">No preview: ${esc(r.error || r.status)}</div>`;
  }
}

// the Blender render and the control that a run really used (web copies, made on request)
async function loadRunMedia(id) {
  let r;
  try { r = await postJSON(`/api/runs/${id}/media`); } catch (e) { return fail(e); }
  state.media[id] = r;
  const box = $("#run-media");
  if (box && state.run === id) { box.outerHTML = runMediaHtml(id); bindMedia(id); }
  if ((r.status === "making") && state.run === id) setTimeout(() => loadRunMedia(id), 3000);
}
function bindMedia(id) {
  const b = $("#run-media [data-media]");
  if (b) b.addEventListener("click", () => loadRunMedia(id));
}
function runMediaHtml(id) {
  const r = state.media[id];
  const layout = state.runLayout && state.runLayout[id];       // a layout run has no control video
  if (!r) return `<div id="run-media"><button type="button" class="linkbtn" data-media="${esc(id)}">${layout ? "Show the Blender render" : "Show the Blender render and the control it used"}</button></div>`;
  const one = (m, label) => {
    if (!m) return "";
    if (m.status === "ready" && m.url) return `<figure><video disablepictureinpicture src="${m.url}" muted loop autoplay playsinline controls></video><figcaption>${label}</figcaption></figure>`;
    const why = m.status === "making" ? "making the browser copy…" : m.status === "queued" ? "waiting for its turn…"
      : m.status === "unavailable" ? (m.note || "not made yet") : `could not be made: ${m.note || "error"}`;
    return `<figure class="pending"><div class="sub">${m.status === "making" || m.status === "queued" ? `<span class="spinner"></span> ` : ""}${esc(why)}</div><figcaption>${label}</figcaption></figure>`;
  };
  const ok = (m) => m && m.status === "ready";
  const retry = r.status !== "making" && !(ok(r.render) && (!r.control || ok(r.control)));
  return `<div id="run-media" class="imgs media2">${one(r.render, "Blender render (the whole shot)")}${one(r.control, "Control video (the whole shot)")}
    ${retry ? `<button type="button" class="linkbtn" data-media="${esc(id)}">Try again</button>` : ""}</div>`;
}

// the full prompt of each clip (never cut) and the images the model was given
function runDetails(d, open, id) {
  if (!d) return "";
  if (d.error) return `<section class="details"><div class="err">${esc(d.error)}</div></section>`;
  const thumb = (url, label, path) => url
    ? `<a class="thumb" href="${url}" target="_blank" rel="noopener" title="${esc(path || label)}"><img src="${url}" alt="${esc(label)}"><span>${esc(label)}</span></a>` : "";
  const rows = (list) => `<table class="kv">${list.map(([k, v]) =>
    `<tr><th>${esc(k)}</th><td>${esc(v === null || v === undefined ? "-" : String(v))}</td></tr>`).join("")}</table>`;
  const chips = (d.chips || []).map((c) => `<span class="kchip ${c.changed ? "changed" : ""}" title="${esc(c.title || "")}">
      <i>${esc(c.label)}</i>${esc(String(c.value ?? "-"))}</span>`).join("");
  if (id) { state.runLayout = state.runLayout || {}; state.runLayout[id] = !!d.layout; }
  // layout runs: the clip images are the look images (frame -1 of each clip), not first frames
  const firstLabel = (c) => d.layout ? (c.clip === 1 ? "Look image" : `Clip ${c.clip} look`) : (c.clip === 1 ? "First frame" : `Clip ${c.clip} start`);
  const firsts = d.clips.filter((c) => c.first_url).map((c) => thumb(c.first_url, firstLabel(c), c.first_image)).join("");
  // the clips' prompts share a long identical start (sheet text, style, character, shared text): fold it away and
  // highlight what is this clip's own, or the boxes look the same (user, 2026-10-25: "why ... say that the text is
  // still the same between each clips")
  const prompts = d.clips.map((c) => c.prompt || "");
  let common = prompts.length > 1 ? prompts.reduce((a, p) => { let i = 0; while (i < a.length && a[i] === p[i]) i++; return a.slice(0, i); }) : "";
  common = common.slice(0, Math.max(0, common.lastIndexOf(" ") + 1));      // cut at a word boundary
  const promptHtml = (p) => common.length > 200 && p.startsWith(common)
    ? `<details class="shared-part"><summary>the same start in every clip (${common.length} characters): show</summary>${esc(common)}</details><mark class="own-part">${esc(p.slice(common.length))}</mark>`
    : esc(p);
  const clips = d.clips.map((c) => `<div class="clipd"><div class="ph"><b>Clip ${c.clip}${d.clips.length > 1 ? ` of ${d.clips.length}` : ""}</b>
      <span class="src">${c.prompt_source === "recorded" ? "exact text sent" : esc(c.prompt_source)}</span>
      <button type="button" class="copy" data-copy="${esc(c.prompt)}">Copy</button></div>
      ${d.clip_join === "context" ? `<div class="clip-actions">Its own actions (${esc({ run: "written for this run", template: "the template's", auto: "written from the rig", take: "the take in use", none: "none: only the shared text" }[c.actions_source] || c.actions_source || "-")})${c.actions ? `: ${esc(c.actions)}` : ""}</div>` : ""}
      <pre>${promptHtml(c.prompt || "")}</pre></div>`).join("");
  return `<section class="details">
    ${d.retired ? `<div class="note"><div><b>Old run</b> ${esc(d.retired)}</div></div>` : ""}
    ${d.note ? `<div class="note"><div class="clamp" data-clamp="3"><b>Test</b> ${esc(d.note)}</div></div>` : ""}
    ${runStyleWarnings(d)}
    ${(d.prompt_warnings || []).length ? `<div class="note pr-warn"><b>Clip texts</b> ${d.prompt_warnings.map(esc).join("; ")}</div>` : ""}
    <div class="ph"><h3>${d.clip_join === "context" && d.clips.length > 1 ? "Shared text (every clip; each clip adds its own actions)" : "Prompt"}</h3><button type="button" class="copy" data-copy="${esc(d.video_text)}">Copy</button></div>
    <p class="prompt clamp" data-clamp="4">${esc(d.video_text)}</p>
    ${d.style_text ? `<p class="style" title="Style text">${esc(d.style_text)}</p>` : ""}
    <div class="kchips">${chips}</div>
    <div class="thumbs-row">${thumb(d.reference_url || d.sheet_url, "Character sheet", d.reference_image || d.sheet_file)}${firsts}</div>
    ${id ? runMediaHtml(id) : ""}
    ${id && d.clips.length > 1 ? buildOnHtml(d, id, d.live) : ""}
    ${id ? `<button type="button" class="linkbtn wf-adv-only wf-graph-link" data-wf-run="${esc(id)}" title="Advanced: the ComfyUI graphs this video sent, with the nodes added or changed at run time">Show the graphs</button>` : ""}
    <details class="fullp" ${open ? "open" : ""}><summary>Details: full prompt sent, all values, files</summary>
      ${clips}
      ${d.layout && !d.negative_used ? "" : `<div class="lbl">Negative prompt</div><pre>${esc(d.clips[0] ? d.clips[0].negative : "")}</pre>`}
      <div class="lbl">All values</div>${rows([...(d.form || []), ...(d.other || [])])}
    </details>
  </section>`;
}

// make the clips from N on again (the earlier clips stay as they are; the old video is not changed)
function buildOnHtml(d, id, live) {
  const n = d.clips.length;
  if (d.retired) return `<details class="buildon"><summary>Build on this video</summary><div class="sub">${esc(d.retired)}</div></details>`;
  if (live) return `<details class="buildon"><summary>Build on this video</summary><div class="sub">Available when this run has stopped.</div></details>`;
  const dr = (state.forkDrafts || {})[id] || {};
  const v = (k, dflt) => (dr[k] !== undefined ? dr[k] : dflt);
  const opts = Array.from({ length: n - 1 }, (_, i) => i + 2).map((k) => `<option value="${k}" ${String(v("from_clip", 2)) === String(k) ? "selected" : ""}>${k}</option>`).join("");
  // one prompt per clip ("context" runs): the shared text goes into every new clip, each clip adds its own actions
  // (prefilled with the parent's actions, the take in use included; only changed rows are sent)
  const context = d.clip_join === "context";
  const from = +v("from_clip", 2);
  const drafts = dr.clip_videos || {};
  const clipRows = context ? d.clips.filter((c) => c.clip >= 2).map((c) => `<label class="fork-clip" data-clip="${c.clip}" ${c.clip < from ? "hidden" : ""}>
        <span>Clip ${c.clip}: its own actions</span>
        <textarea name="clip_${c.clip}" rows="2" data-parent="${esc(c.actions || "")}" placeholder="none: only the shared text">${esc(drafts[c.clip] !== undefined ? drafts[c.clip] : (c.actions || ""))}</textarea></label>`).join("") : "";
  return `<details class="buildon" ${state.buildOpen ? "open" : ""}><summary>Build on this video</summary>
    <form class="buildon-form">
      <label class="row-l"><span>Make again from clip</span><select name="from_clip">${opts}</select>
        <span class="sub">of ${n} (clips before it are kept; later clips are made again, so there is no jump)</span></label>
      <label><span>${context ? "Shared text (who, place, light, look: it goes into every new clip)" : "Video text for the new clips"}</span><textarea name="video" rows="4">${esc(v("video", d.video_text))}</textarea></label>
      ${clipRows}
      <label><span>Style</span><input name="style" value="${esc(v("style", d.style_text || ""))}"></label>
      ${d.layout ? "" : `<label class="check"><input type="checkbox" name="describe_facing" ${v("describe_facing", (d.chips || []).some((c) => c.label === "Facing text")) ? "checked" : ""}>
        <span>Say in each new clip's prompt which way the character faces</span></label>`}
      <label><span>Note (optional)</span><input name="note" placeholder="what this version changes" value="${esc(v("note", ""))}"></label>
      <button type="submit" class="primary">Make the new version</button>
    </form></details>`;
}
function bindBuildOn(box, id) {
  const det = $(".buildon", box);
  if (!det || det.dataset.bound) return;
  det.dataset.bound = "1";
  det.addEventListener("toggle", () => { state.buildOpen = det.open; });
  const form = $(".buildon-form", det);
  if (!form) return;
  const clipBoxes = () => $$(".fork-clip textarea", form);
  const showClips = () => {                                // only the clips the fork makes again
    const from = +form.from_clip.value;
    $$(".fork-clip", form).forEach((el) => { el.hidden = +el.dataset.clip < from; });
  };
  const save = () => {                                     // a draft per source run: a redraw never loses it
    state.forkDrafts = state.forkDrafts || {};
    const clips = {};
    clipBoxes().forEach((t) => { clips[t.name.slice(5)] = t.value; });
    state.forkDrafts[id] = { from_clip: form.from_clip.value, video: form.video.value, style: form.style.value,
                             describe_facing: !!(form.describe_facing && form.describe_facing.checked), note: form.note.value,
                             clip_videos: clips };
  };
  form.addEventListener("input", save);
  form.addEventListener("change", (ev) => { if (ev.target === form.from_clip) showClips(); save(); });
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const f = ev.target;
    const body = { from_clip: +f.from_clip.value, video: f.video.value, style: f.style.value,
                   describe_facing: !!(f.describe_facing && f.describe_facing.checked), note: f.note.value };
    const clips = {};                                      // the changed actions of the clips made again ("" = none)
    clipBoxes().forEach((t) => {
      const k = +t.name.slice(5);
      if (k >= body.from_clip && t.value.trim() !== (t.dataset.parent || "").trim()) clips[k] = t.value.trim();
    });
    if (Object.keys(clips).length) body.clip_videos = clips;
    try {
      const r = await postJSON(`/api/runs/${id}/fork`, body);
      if (r.start.ok) queued("New version"); else toast(`The worker did not start: ${r.start.error}`, true);
      state.buildOpen = false;
      delete (state.forkDrafts || {})[id];
      await loadRuns();
      openRun(r.id);
    } catch (e) { fail(e); }
  });
}

// long text cut to N lines, with a "Show more" link only when it is really cut
function bindClamps(box) {
  $$(".clamp", box).forEach((el) => {
    if (el.dataset.bound) return;
    el.dataset.bound = "1";
    el.style.setProperty("--lines", el.dataset.clamp);
    requestAnimationFrame(() => {
      if (el.scrollHeight <= el.clientHeight + 2) return;
      const more = document.createElement("button");
      more.type = "button"; more.className = "more-link"; more.textContent = "Show more";
      more.addEventListener("click", () => { const open = el.classList.toggle("open"); more.textContent = open ? "Show less" : "Show more"; });
      el.after(more);
    });
  });
  $$(".copy", box).forEach((b) => {
    if (b.dataset.bound) return;
    b.dataset.bound = "1";
    b.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(b.dataset.copy); b.textContent = "Copied"; }
      catch { b.textContent = "Copy failed"; }
      setTimeout(() => { b.textContent = "Copy"; }, 1400);
    });
  });
}

// ------------------------------------------------------------------ the clip-by-clip loop (AGENTS.md 7.5)
// A long video judged clip by clip: each clip with its takes; "Make this clip again" (a new seed first, then ONE change
// at a time), "Use this take", "Measure" (the picture numbers: seconds on the CPU), and the next clip or the rest.
// The same runner functions as the CLI (run takes / retake / use-take / score / start --through-clip K --detach).
const TAKE_KNOBS = { seed: "seed", keyframe: "keyframe look", ingredients_strength: "sheet guide", adain: "AdaIN",
                     relay_strength: "relay", end_anchor: "end anchor", anchor_strength: "anchor" };
function takeChanges(t) {
  const parts = Object.entries(t.overrides || {}).map(([k, v]) => k === "video" ? "own text"
    : k === "look_image" ? "own look image" : `${TAKE_KNOBS[k] || k.replace(/_/g, " ")} ${v}`);
  return parts.join(" · ") || (t.original ? "the first version" : "the run's settings");
}
// the pass limits of the gauntlet (AGENTS.md 7.5 step 3); the critic still decides
const CLIP_LIMITS = [
  ["Join", (s) => s.overlap && s.overlap.x_median, (v) => v <= 2.5, "How much the shared frames differ from the clip before, as a multiple of the clip's own frame change (pass: at most 2.5)"],
  ["Sharpness", (s) => s.sharpness_vs_clip1, (v) => v >= 0.5, "Sharpness against clip 1 (pass: at least 0.5)"],
  ["Biggest jump", (s) => s.picture && s.picture.max_jump_x_median, (v) => v <= 2.5, "The largest change from one frame to the next, as a multiple of the usual change (pass: at most 2.5)"]];
function clipScoreHtml(sc) {
  if (!sc) return "";
  if (sc.stale) return `<div class="sub" title="${esc(sc.stale)}">The numbers are out of date: measure again.</div>`;
  const chips = CLIP_LIMITS.map(([label, get, ok, title]) => {
    const v = get(sc);
    return v === undefined || v === null ? "" : `<span class="kchip ${ok(v) ? "pass" : "changed"}" title="${esc(title)}"><i>${label}</i>${esc(String(v))}${ok(v) ? "" : " ✗"}</span>`;
  }).join("");
  return chips ? `<div class="kchips">${chips}</div>` : "";
}
// The clip players are small, so the browser hides its own full-screen button: give each clip its own.
const FULL_BTN = `<button type="button" data-full title="Watch this clip full screen (or double-click the clip)">Full screen</button>`;
function clipFullScreen(video) {
  if (!video) return;
  const go = video.requestFullscreen || video.webkitRequestFullscreen;
  if (go) Promise.resolve(go.call(video)).then(() => video.play()).catch(() => {});
}
function clipVideo(url) {
  return url ? `<video disablepictureinpicture src="${url}" muted loop playsinline preload="metadata" controls></video>` : "";
}
function retakeFormHtml(id, c) {
  const dr = (state.retakeDrafts || {})[`${id}:${c.clip}`] || {};
  const v = (k) => esc(dr[k] ?? "");
  const seed = dr.seed ?? Math.floor(Math.random() * 1e6);
  const kf = dr.keyframe || "";
  return `<form class="retake-form buildon-form" data-clip="${c.clip}">
      <div class="sub">First try 2-3 new seeds. Then change ONE thing at a time, so you know what helped.</div>
      <label class="row-l"><span>Seed</span><input name="seed" type="number" min="0" value="${esc(seed)}">
        <button type="button" class="linkbtn" data-newseed title="Another random seed">New seed</button></label>
      <label class="row-l" title="The clip's keyframe look (a look image in the middle of a turning clip)"><span>Keyframe look</span>
        <select name="keyframe"><option value="" ${kf ? "" : "selected"}>as the run</option><option value="on" ${kf === "on" ? "selected" : ""}>on</option><option value="off" ${kf === "off" ? "selected" : ""}>off</option></select></label>
      <label class="row-l" title="How strongly the clip follows the character sheet (run default 1.0)"><span>Sheet guide</span>
        <input name="ingredients_strength" type="number" step="0.05" min="0" max="3" placeholder="as the run" value="${v("ingredients_strength")}"></label>
      <label class="row-l" title="How strongly the clip's colours are pulled toward clip 1 (run default 0.3; 1.0 only as a one-clip repair)"><span>AdaIN</span>
        <input name="adain" type="number" step="0.05" min="0" max="1" placeholder="as the run" value="${v("adain")}"></label>
      <label><span>Text for this clip <i>${c.context_join ? "this clip's actions only (the shared text stays); empty = its own actions" : "only what this clip's frames show; empty = the run's text"}</i></span>
        <textarea name="video" rows="3" placeholder="${esc(c.actions || "")}">${v("video")}</textarea></label>
      <label title="For example a fresh look of this clip's start frame (it repairs colour drift)"><span>Own look image <i>optional</i></span>
        <input type="file" name="look_image" accept="image/png,image/jpeg,image/webp"></label>
      <label><span>Note <i>optional: why this take</i></span><input name="note" maxlength="500" value="${v("note")}" placeholder="another seed"></label>
      <button type="submit" class="primary">Make the take</button>
    </form>`;
}
function clipHtml(id, d, c) {
  const live = d.worker_alive;
  const can = d.retake && !live && c.made;
  const why = !d.retake ? d.why_not : live ? "Available when the run has stopped (it is working now)." : !c.made ? "This clip is not made yet." : "";
  const open = !!(state.retakeOpen || {})[`${id}:${c.clip}`];
  const takes = (c.takes || []).map((t) => {
    const done = t.status === "done";
    const useBtn = done && !t.used ? `<button type="button" data-use="${c.clip}:${t.n}" ${can ? "" : "disabled"}
        title="${esc(can ? "Put this take into the video and join the clips again (the original stays as take 0; the old video is kept)" : why)}">Use this take</button>` : "";
    return `<figure class="take ${t.used ? "used" : ""}">${done ? clipVideo(t.url) : `<div class="take-wait"><span class="chip ${esc(t.status)}">${esc(t.status)}</span></div>`}
      <figcaption><b>${t.original ? "Original" : `Take ${t.n}`}</b>${t.used ? ` <span class="chip done">in the video</span>` : ""}
        <span class="sub">${esc(takeChanges(t))}</span>${t.note ? `<span class="sub">${esc(t.note)}</span>` : ""}
        ${t.error ? `<span class="err">${esc(t.error)}</span>` : ""}</figcaption>
      <div class="take-acts">${useBtn}${done ? FULL_BTN : ""}${done ? `<button type="button" class="wf-adv-only" data-wf-run="${esc(id)}" data-wf-job="${esc(t.original ? (c.clip === 1 ? "clip" : `clip${c.clip}`) : `take${c.clip}_${t.n}`)}" title="Advanced: the ComfyUI graph this take sent">Graph</button>` : ""}${done ? `<button type="button" data-score="${c.clip}:${t.n}" title="The picture numbers of this take (a few seconds)">Measure</button>` : ""}</div>
      ${clipScoreHtml(t.score)}</figure>`;
  }).join("");
  return `<div class="clipbox" data-clip="${c.clip}">
    <div class="clip-head"><b>Clip ${c.clip}</b>${c.made ? "" : ` <span class="chip">not made yet</span>`}
      ${c.made ? `<button type="button" class="linkbtn wf-adv-only wf-graph-link" data-wf-run="${esc(id)}" data-wf-job="${c.clip === 1 ? "clip" : `clip${c.clip}`}" title="Advanced: the ComfyUI graph of this clip's original (each take has its own Graph button)">Graph</button>` : ""}
      ${c.used !== undefined && c.used !== null ? `<span class="sub">uses take ${esc(c.used)}</span>` : ""}</div>
    ${c.made ? `<div class="takes-grid"><figure class="take cur">${clipVideo(c.url)}<figcaption><b>In the video now</b></figcaption>
      <div class="take-acts">${FULL_BTN}<button type="button" data-score="${c.clip}" title="The picture numbers of this clip (a few seconds)">Measure</button></div>
      ${clipScoreHtml(c.score)}</figure>${takes}</div>` : ""}
    ${c.made && d.retake ? `<details class="retake" data-clip="${c.clip}" ${open && can ? "open" : ""}><summary>Make this clip again</summary>
      ${can ? retakeFormHtml(id, c) : `<div class="sub">${esc(why)}</div>`}</details>` : ""}
  </div>`;
}
function clipsHtml(id, d) {
  if (!d || !d.retake || !d.clips_total) return "";
  const n = d.clips_total, made = d.clips.filter((c) => c.made).length;
  const live = d.worker_alive, next = made + 1;
  let cont = "";
  if (!live && made < n && d.status !== "done") {
    cont = next < n
      ? `<button type="button" class="primary" data-cont="${next}" title="Make clip ${next}, then stop again so you can check it">Make clip ${next}</button>
         <button type="button" data-cont="0" title="Make every clip that is left and join the video">Make the rest</button>`
      : `<button type="button" class="primary" data-cont="0" title="Make the last clip and join the video">Make the last clip and the video</button>`;
  }
  return `<section class="details clips-sec">
    <div class="ph"><h3>Clips</h3><span class="sub">${made} of ${n} made${d.clip_limit ? ` · stops after clip ${esc(d.clip_limit)}` : ""}${live ? " · working" : ""}</span></div>
    ${d.clips.map((c) => clipHtml(id, d, c)).join("")}
    ${cont ? `<div class="clip-cont">${cont}</div>` : ""}
  </section>`;
}
function reopenRun(id) { stopPoll("run"); state.clipsShown = null; openRun(id); }
async function drawClips(id) {
  let d;
  try { d = await api(`/api/runs/${encodeURIComponent(id)}/takes`); } catch { return; }
  const box = $("#player .info-clips");
  if (!box || state.run !== id) return;
  const sig = JSON.stringify(d);
  if (state.clipsShown && state.clipsShown.id === id && state.clipsShown.sig === sig) return;   // keep what is typed
  state.clipsShown = { id, sig };
  box.innerHTML = clipsHtml(id, d);
  bindClips(id, box, d);
}
function bindClips(id, box, d) {
  const url = (what) => `/api/runs/${encodeURIComponent(id)}/${what}`;
  $$("details.retake", box).forEach((det) => det.addEventListener("toggle", () => {
    (state.retakeOpen = state.retakeOpen || {})[`${id}:${det.dataset.clip}`] = det.open;
  }));
  $$(".retake-form", box).forEach((form) => {
    const key = `${id}:${form.dataset.clip}`;
    const save = () => {
      (state.retakeDrafts = state.retakeDrafts || {})[key] = { seed: form.seed.value, keyframe: form.keyframe.value,
        ingredients_strength: form.ingredients_strength.value, adain: form.adain.value, video: form.video.value, note: form.note.value };
    };
    save();
    form.addEventListener("input", save);
    form.addEventListener("change", save);
    $("[data-newseed]", form).addEventListener("click", () => { form.seed.value = Math.floor(Math.random() * 1e6); save(); });
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const changed = ["keyframe", "ingredients_strength", "adain", "video"].filter((k) => form[k].value.trim() !== "").length
        + (form.look_image.files.length ? 1 : 0);
      if (changed > 1 && !confirm(`You changed ${changed} things at once. If the take is better, you will not know which change helped.\n\nMake the take anyway?`)) return;
      const fd = new FormData(form);
      fd.append("clip", form.dataset.clip);
      for (const [k, v] of [...fd.entries()]) if ((v instanceof File && !v.size) || v === "") fd.delete(k);
      const btn = $("button[type=submit]", form);
      btn.disabled = true;
      try {
        const r = await api(url("retake"), { method: "POST", body: fd });
        delete (state.retakeDrafts || {})[key];
        (state.retakeOpen = state.retakeOpen || {})[key] = false;
        if (r.start && r.start.ok) queued(`Clip ${form.dataset.clip}, take ${r.take}`);
        else toast(`The take is recorded, but the worker did not start: ${(r.start || {}).error || "?"}. Press Continue.`, true);
        reopenRun(id);
      } catch (e) { fail(e); btn.disabled = false; }
    });
  });
  $$("[data-use]", box).forEach((b) => b.addEventListener("click", async () => {
    const [clip, take] = b.dataset.use.split(":").map(Number);
    if (!confirm(`Put take ${take} into the video for clip ${clip}?\n\nThe clips are joined again. The original stays as take 0, and the old video is kept as a copy.`)) return;
    b.disabled = true; b.textContent = "Joining the clips…";
    try { await postJSON(url("use-take"), { clip, take }); toast(`Clip ${clip} now uses take ${take}.`); reopenRun(id); }
    catch (e) { fail(e); b.disabled = false; b.textContent = "Use this take"; }
  }));
  $$("[data-score]", box).forEach((b) => b.addEventListener("click", async () => {
    const [clip, take] = b.dataset.score.split(":");
    b.disabled = true; b.textContent = "Measuring…";
    try {
      await postJSON(url("score"), { clip: +clip, take: take === undefined ? null : +take });
      state.clipsShown = null; drawClips(id);
    } catch (e) { fail(e); b.disabled = false; b.textContent = "Measure"; }
  }));
  $$("[data-full]", box).forEach((b) => b.addEventListener("click", () => clipFullScreen($("video", b.closest("figure")))));
  $$(".take video", box).forEach((v) => v.addEventListener("dblclick", () => clipFullScreen(v)));
  $$("[data-cont]", box).forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true;
    if (await continueRun(id, d.status, { through_clip: +b.dataset.cont })) reopenRun(id); else b.disabled = false;
  }));
}

async function openRun(id) {
  state.run = id;
  const box = $("#player");
  box.classList.remove("hidden");
  let shown = null;          // the media shown now (so a poll does not restart the video)
  const tick = async () => {
    let r;
    try { r = await api(`/api/runs/${id}`); } catch (e) { stopPoll("run"); fail(e); return false; }
    if (state.run !== id) return false;
    const p = r.progress, live = r.worker_alive;
    const waiting = r.status === "queued";
    const media = r.final_url ? `<video disablepictureinpicture controls autoplay loop src="${r.final_url}"></video>`
      : r.contact_url ? `<img src="${r.contact_url}" alt="">` : `<div class="wait">${live ? `<span class="spinner"></span>` : ""}</div>`;
    const key = r.final_url || r.contact_url || "wait";
    if (shown !== key) {
      box.innerHTML = `<div class="player"><div class="media">${media}</div><div class="info"><div class="info-status"></div><div class="info-clips"></div><div class="info-details"></div></div></div>`;
      $(".player", box).classList.toggle("split", !!r.details && !r.details.error);
      shown = key;
      shownDetails = null;
      state.clipsShown = null;                             // the Clips section is drawn again too
    }
    const prompt = (r.manifest && r.manifest.prompt && r.manifest.prompt.video) || "";
    const openDet = $(".fullp", box) ? $(".fullp", box).open : (state.detailsOpen ?? false);  // closed at first; kept across polls
    $(".info-status", box).innerHTML = `
      ${live && waiting ? `<div class="jobbar"><span class="spinner"></span><span class="ptext">waiting in the queue${r.queue_position ? ` · ${r.queue_position} ahead` : ""}</span></div>` : ""}
      ${live && !waiting ? `<div class="jobbar">${p ? `<span class="ptext">${esc(p.step)}</span>`
          + progressBar(p.value / p.max, r.step_started_at ? r.now - r.step_started_at : 0) : `<span class="spinner"></span><span class="ptext">starting</span>`}</div>` : ""}
      ${r.error ? `<div class="err">${esc(r.error)}</div>` : ""}
      <div class="bar"><span class="chip ${esc(r.status)}">${esc(r.status)}</span>
        <span class="grow meta-line">${esc(runDate({ created: r.manifest && r.manifest.created, id }))}<span class="rid" title="Run ID">${esc(id)}</span></span>
        ${canContinue({ ...r, worker_alive: !!live }) ? `<button id="btn-continue-run" class="primary" title="${r.status === "paused"
          ? "Make the next clip, then stop again (the Clips section can also make the rest)" : "Start this video again where it stopped (it waits in the queue)"}">${esc(continueLabel(r))}</button>` : ""}
        ${live ? `<button id="btn-cancel-run">${waiting ? "Remove" : "Stop"}</button>` : trashButton("run", id)}
        <button class="close" id="btn-close-run" title="Close" aria-label="Close">${ICON.close}</button></div>`;
    if (r.details && !r.details.error) { r.details.live = !!live; r.details.retired = r.retired || null; }
    const dkey = JSON.stringify(r.details || null);
    if (dkey !== shownDetails) {                           // redraw the details only when they change
      shownDetails = dkey;
      $(".info-details", box).innerHTML = runDetails(r.details, openDet, id);
      bindClamps(box);
      bindBuildOn(box, id);
      bindMedia(id);
      const det = $(".fullp", box);
      if (det) det.addEventListener("toggle", () => { state.detailsOpen = det.open; });
    }
    drawClips(id);                                         // the clip-by-clip loop (only when it changed)
    const go = $("#btn-continue-run");
    if (go) go.addEventListener("click", async () => {
      go.disabled = true;
      if (await continueRun(id, r.status)) reopenRun(id); else go.disabled = false;
    });
    const cb = $("#btn-cancel-run");
    if (cb) cb.addEventListener("click", async () => {
      if (!confirm(waiting ? "Remove this video from the queue?" : "Stop this video? The work of this step is lost.")) return;
      try { await postJSON(`/api/runs/${id}/cancel`); toast("Cancel requested."); } catch (e) { fail(e); } });
    $("#btn-close-run").addEventListener("click", closeRun);
    bindDeletes(box, () => { closeRun(); loadRuns(); });
    const keep = live || r.status === "new" || r.status === "running";
    if (!keep) stopPoll("run");
    return keep;
  };
  let shownDetails = null;
  const keep = await tick();
  if (keep && state.run === id && !state.polls.run) poll("run", tick, 4000);
}
$("#player").addEventListener("click", (ev) => {
  if (ev.target.id !== "player") return;
  if (state.run) closeRun(); else { ev.target.classList.add("hidden"); ev.target.innerHTML = ""; }
});
document.addEventListener("keydown", (ev) => {
  if (ev.key !== "Escape") return;
  const box = $("#player");
  if (state.run) closeRun(); else if (!box.classList.contains("hidden")) { box.classList.add("hidden"); box.innerHTML = ""; }
});

// the New video dialog (the form keeps what was typed when it closes)
function openNewVideo() {
  const box = $("#new-video");
  box.classList.remove("hidden");
  showShotPreview();
  const t = $("#form-video").video;
  setTimeout(() => t.focus(), 30);
}
function closeNewVideo() {
  const box = $("#new-video");
  if (box.classList.contains("hidden")) return false;
  box.classList.add("hidden");
  const v = $("#shot-preview video"); if (v) v.pause();
  return true;
}
$("#btn-close-new").addEventListener("click", closeNewVideo);
$("#new-video").addEventListener("click", (ev) => { if (ev.target.id === "new-video") closeNewVideo(); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeNewVideo(); });

$("#form-video").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = ev.target;
  const data = Object.fromEntries(new FormData(f).entries());
  if (data.blend === "__own__") {
    data.blend = data.blend_path.trim().replace(/^"(.*)"$/, "$1");     // a path pasted with quotes works too
    if (!data.blend) return toast("Type the path of your .blend file.", true);
  }
  delete data.blend_path;
  // fields hidden for this kind of shot are not sent, so the run's own defaults apply (the hidden Body type select
  // sent "chibi" with every run: audit 2026-10-25)
  $$("#form-video [data-v2v-only].hidden, #form-video [data-layout-only].hidden, #form-video [data-full-model].hidden, #form-video [data-blend-only].hidden, #song-start-field.hidden")
    .forEach((el) => [el, ...$$("[name]", el)].forEach((x) => { if (x.name) delete data[x.name]; }));
  if (!data.character) return toast("Make a character ready in the Cast tab first.", true);
  if ($("#form-video button[type=submit]").disabled) return toast($("#slot-msg").textContent.trim(), true);
  data.seed = data.seed ? +data.seed : null;
  data.style_lora_strength = data.style_lora_strength ? +data.style_lora_strength : null;
  if (f.describe_facing) data.describe_facing = f.describe_facing.checked;   // the checkbox left the form (2026-10-25)
  // one character per template slot (slot order = the run's characters list on the server); the objects' words
  data.characters = $$("#slot-list .slot-char").map((s) => ({ slot: +s.dataset.slot, collection: s.dataset.collection || null, character: s.value }));
  delete data.character;
  const props = propWords();
  if (Object.keys(props).length) data.props = props;
  const propPicsSent = propPictures();
  if (Object.keys(propPicsSent).length) data.prop_images = propPicsSent;
  const propRefsSent = propRefs();                     // library props (the server fills their pictures and words)
  if (Object.keys(propRefsSent).length) data.prop_refs = propRefsSent;
  try { data.clip_videos = clipTexts(); } catch (e) { return toast(e.message, true); }
  const bare = clipsWithoutActions();                      // one prompt per clip: never one action text for all
  if (bare.length && !confirm(`${$("#clip-text-warn").textContent}\n\nWhatever the text above describes then happens in ${bare.length === clipPlan().length ? "every clip" : `clip ${bare.join(", ")}`}. Make the video anyway?`)) {
    $("#clip-texts").open = true;
    return;
  }
  closeNewVideo();                                         // close at once; the run takes a few seconds to make
  toast("Making the video run…");
  try {
    const d = await postJSON("/api/runs", data);
    if (d.start.ok) queued("Video"); else toast(`The worker did not start: ${d.start.error}`, true);
    await loadRuns();
    openRun(d.id);
  } catch (e) { fail(e); openNewVideo(); }                 // the form keeps what was typed
});
