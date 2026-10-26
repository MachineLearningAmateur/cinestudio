// cinestudio web app - The Backgrounds tab: the background library - 360 panoramas (any camera; 360 depth, style redraw)
// and backplates (one picture for a still camera: from text, from a run, or your own picture); delete.
// Plain scripts (no modules), loaded in order by index.html; they share their top-level names.
"use strict";

const bgSig = (b) => [b.kind, b.url, b.builtin, b.set_url, b.depth, b.runs, JSON.stringify(b.busy || []), b.style_ref, b.style_version, b.style_stale,
                      b.style, (b.style_ref && styleByName(b.style_ref) || {}).label].join("|");

// the style a background was made in: a library style (Styles tab; a warning when the style changed since), or its own words
function bgStyleFact(b) {
  if (b.style_ref) {
    const s = styleByName(b.style_ref);
    if (!s) return `<span class="fact warn" title="Made in the style &quot;${esc(b.style_ref)}&quot;, which is not in the library now">style: ${esc(b.style_ref)} (not in the library)</span>`;
    return b.style_stale
      ? `<span class="fact warn" title="Made with version ${esc(b.style_version || 0)} of this style; the style is at version ${esc(s.version)} now. The picture does not change by itself: make the background again to take the new words.">style: ${esc(s.label || s.name)} · older version</span>`
      : `<span class="fact accent2" title="Made in the style &quot;${esc(s.label || s.name)}&quot; from the Styles tab (version ${esc(b.style_version || s.version)})">style: ${esc(s.label || s.name)}</span>`;
  }
  return b.style ? `<span class="fact muted" title="Its own style words (no library style):
${esc(b.style)}">style: own words</span>` : "";
}
const bgPager = makePager("backgrounds", () => loadBackgroundsTab());

// "drawn over the set of <template>": a link that opens the template's card in the Templates tab
function bgSetFact(b) {
  const shot = b.shot || "a template";
  const tip = `Drawn over the Blender set of ${shot}: its objects stay where the layout has them`;
  return b.shot ? `<button type="button" class="fact accent2 fact-link" data-bg-shot="${esc(b.shot)}" title="${esc(tip)} · open the template">drawn over the set of ${esc(shot)}</button>`
    : `<span class="fact accent2" title="${esc(tip)}">drawn over the set of ${esc(shot)}</span>`;
}

function bgFacts(b) {
  const f = [];
  if (b.builtin) f.push(`<span class="fact builtin" title="Ships with cinestudio (read-only): use it in a video, or redraw it as a new background of your own">Built-in</span>`);
  if (b.kind === "plate") {                    // a backplate: one picture for a still camera
    f.push(`<span class="fact accent" title="One picture at the frame's shape for a still camera (sharper than a panorama)">backplate</span>`);
    f.push(b.source === "set"                  // drawn over a template's own Blender set (background from-set)
      ? bgSetFact(b)
      : `<span class="fact" title="Where it comes from">${esc({ import: "your picture", run: "from a video run" }[b.source] || "from text")}</span>`);
    if (b.camera && b.source !== "set") f.push(`<span class="fact muted" title="The camera it was made for: a template with another camera can put the horizon elsewhere">${esc(b.shot ? `for ${b.shot}` : "camera")}</span>`);
    f.push(bgStyleFact(b));
    f.push(`<span class="fact" title="Video runs that used this background">${b.runs} run${b.runs === 1 ? "" : "s"}</span>`);
    if (b.seed != null) f.push(`<span class="fact muted" title="Seed">seed ${esc(b.seed)}</span>`);
    return f.join("");
  }
  f.push(`<span class="fact" title="A 360 degree place: every camera, also a moving one">360°</span>`);
  f.push(b.source === "redraw"
    ? `<span class="fact" title="Redrawn from ${esc(b.styled_from || "")} with ${esc(b.style_lora || "")}, keep ${esc(b.keep)}">redraw of ${esc(b.styled_from || "?")}</span>`
    : `<span class="fact" title="Made from the text">from text</span>`);
  f.push(bgStyleFact(b));
  f.push(b.depth ? `<span class="fact ok" title="The 360 depth is ready: the 3D ground of the videos uses it">depth ready</span>`
    : `<span class="fact muted" title="No 360 depth yet: a video run makes it when it needs it">no depth</span>`);
  f.push(`<span class="fact" title="Video runs that used this background">${b.runs} run${b.runs === 1 ? "" : "s"}</span>`);
  if (b.seed != null) f.push(`<span class="fact muted" title="Seed">seed ${esc(b.seed)}</span>`);
  return f.join("");
}

function bgBusy(b) {
  return (b.busy || []).map((j) => `<span class="chip ${esc(j.status || "queued")}" title="${esc(j.label)}">${esc({ depth: "depth", style: "redraw" }[j.kind] || j.kind)}: ${esc(j.status || "queued")}</span>`).join(" ");
}

function bgCard(b) {
  const busy = (b.busy || []).length > 0, plate = b.kind === "plate";
  return `<div class="bg-view ${plate ? "plate" : ""}" style="background-image:url('${esc(b.url)}')" role="img"
        title="${plate ? `${esc(b.camera || "One picture for a still camera")} · double-click to view large` : "Drag to look round 360° · double-click to view large"}"
        aria-label="${plate ? "backplate" : "360 degree panorama"} ${esc(b.name)}">${b.set_url
          ? `<a class="bg-set" href="${esc(b.set_url)}" target="_blank" rel="noopener" title="The Blender set it was drawn over (opens the picture)"><img src="${esc(b.set_url)}" alt="the Blender set"></a>` : ""}</div>
    <div class="sc-head"><span class="sc-name" title="${esc(b.name)}">${esc(b.name)}</span>${bgBusy(b)}
      ${busy || b.builtin ? "" : trashButton("background", b.name)}</div>
    <div class="sc-tags clamp" title="Click to show all">${esc(b.scene || b.prompt || "")}</div>
    <div class="sc-facts">${bgFacts(b)}</div>
    <div class="bg-actions">
      <button type="button" data-bg-open="${esc(b.url)}" data-bg-name="${esc(b.name)}" data-bg-flat="${plate ? "1" : ""}" title="${plate ? "Open the picture large" : "Open the panorama large: look round in perspective, or see the whole picture"}">View large</button>
      ${plate || b.depth ? "" : `<button type="button" data-bg-depth="${esc(b.name)}" ${busy ? "disabled" : ""} title="MoGe-2 measures the distance in every direction (about a minute on the GPU)">Make depth</button>`}
      ${plate ? "" : `<button type="button" data-bg-style="${esc(b.name)}" ${busy ? "disabled" : ""} title="LTX-2.5 + a style LoRA redraws it as a new background">Redraw in a style</button>`}
    </div>`;
}

// drag the picture sideways to look round (the panorama wraps round: background-repeat)
function bindBgView(el) {
  let x0 = null, pos = 0;
  el.addEventListener("pointerdown", (ev) => { x0 = ev.clientX; el.setPointerCapture(ev.pointerId); });
  el.addEventListener("pointermove", (ev) => {
    if (x0 === null) return;
    pos += ev.clientX - x0; x0 = ev.clientX;
    el.style.backgroundPositionX = `${pos}px`;
  });
  const end = () => { x0 = null; };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
}

function bindBgCard(card) {
  const t = $(".sc-tags", card);
  if (t) t.addEventListener("click", () => t.classList.toggle("clamp"));
  const view = $(".bg-view", card);
  if (!view.classList.contains("plate")) bindBgView(view);       // a backplate is one flat picture: nothing to turn
  const big = $("[data-bg-open]", card);
  if (big) {
    const open = () => openPanoViewer(big.dataset.bgOpen, big.dataset.bgName, !!big.dataset.bgFlat);
    big.addEventListener("click", open);
    view.addEventListener("dblclick", open);
  }
  bindDeletes(card, () => loadBackgroundsTab());
  $$("[data-bg-shot]", card).forEach((b) => b.addEventListener("click", () => focusCard("templates", b.dataset.bgShot)));
  $$("[data-bg-depth]", card).forEach((b) => b.addEventListener("click", async () => {
    try { await postJSON(`/api/backgrounds/${encodeURIComponent(b.dataset.bgDepth)}/depth`, {}); queued("360 depth"); loadBackgroundsTab(); }
    catch (e) { fail(e); }
  }));
  $$("[data-bg-style]", card).forEach((b) => b.addEventListener("click", () => openBgStyle(b.dataset.bgStyle)));
}

async function loadBackgroundsTab() {
  let d;
  try { d = await api("/api/backgrounds"); } catch { return; }
  try { await loadStyleLib(); } catch { /* the style chips show the bare style name */ }
  const list = d.backgrounds || [], jobs = d.jobs || [], failed = d.failed || [];
  const grid = $("#bg-grid");
  bgPager.measure(grid); bgPager.watch(grid);
  focusPage("backgrounds", bgPager, list.map((b) => b.name));      // a link from a template's card
  const per = bgPager.per();
  if (!drawPaging(bgPager, list.length, Math.min(per || list.length, Math.max(0, list.length - bgPager.offset())),
                  "background", $("#bg-bar"), $("#bg-pager"), grid)) return loadBackgroundsTab();
  const shown = per ? list.slice(bgPager.offset(), bgPager.offset() + per) : list;
  if (!$("#btn-new-bg", grid)) {
    grid.innerHTML = `<button type="button" class="song-card new-tile" id="btn-new-bg" title="New background" aria-label="New background">
        <span class="plus-box"><svg viewBox="0 0 24 24" width="30" height="30" aria-hidden="true"><path fill="currentColor" d="M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6z"/></svg></span>
        <span class="plus-label">New background</span></button>`;
    $("#btn-new-bg").addEventListener("click", openNewBg);
  }
  // backgrounds being made (no picture yet): one card each, right after the New tile
  $$(".song-card[data-bgjob]", grid).forEach((c) => { if (!jobs.some((j) => j.ref === c.dataset.bgjob)) c.remove(); });
  let after = $("#btn-new-bg", grid);
  for (const j of jobs) {
    let card = $$(".song-card[data-bgjob]", grid).find((c) => c.dataset.bgjob === j.ref);
    if (!card) {
      card = document.createElement("div");
      card.className = "song-card bg-card";
      card.dataset.bgjob = j.ref;
      after.after(card);
    }
    card.innerHTML = `<div class="bg-view empty"><span class="spinner"></span></div>
      <div class="sc-head"><span class="sc-name">${esc(j.label)}</span><span class="chip ${esc(j.status || "queued")}">${esc(j.status || "queued")}</span></div>
      <div class="sc-facts"><span class="fact muted">${j.status === "running" ? (j.kind === "plate" ? "making the backplate" : j.kind === "setplate" ? "drawing the backplate over the set" : "making the panorama, then its depth") : `waiting in the queue${j.position ? `: #${j.position + 1}` : ""}`}</span></div>`;
    after = card;
  }
  // failed jobs (the worker's error): a card each until it is dismissed
  $$(".song-card[data-bgfail]", grid).forEach((c) => { if (!failed.some((j) => j.ref === c.dataset.bgfail)) c.remove(); });
  for (const j of failed) {
    let card = $$(".song-card[data-bgfail]", grid).find((c) => c.dataset.bgfail === j.ref);
    if (!card) {
      card = document.createElement("div");
      card.className = "song-card bg-card failed";
      card.dataset.bgfail = j.ref;
      card.innerHTML = `<div class="sc-head"><span class="sc-name" title="${esc(j.label)}">${esc(j.label || j.ref)}</span>
          <span class="chip failed">failed</span></div>
        <div class="sc-tags">${esc(j.error || "the worker stopped with an error")}</div>
        <div class="bg-actions"><button type="button" data-bg-dismiss="${esc(j.ref)}">Dismiss</button></div>`;
      $("[data-bg-dismiss]", card).addEventListener("click", async () => {
        try { await postJSON(`/api/background-jobs/${encodeURIComponent(j.ref)}/dismiss`, {}); loadBackgroundsTab(); }
        catch (e) { fail(e); }
      });
      after.after(card);
    }
    after = card;
  }
  const names = new Set(shown.map((b) => b.name));
  $$(".song-card[data-bg]", grid).forEach((c) => { if (!names.has(c.dataset.bg)) c.remove(); });
  for (const b of shown) {
    let card = $$(".song-card[data-bg]", grid).find((c) => c.dataset.bg === b.name);
    if (!card || card.dataset.sig !== bgSig(b)) {
      const fresh = document.createElement("div");
      fresh.className = "song-card bg-card";
      fresh.dataset.bg = b.name;
      fresh.dataset.sig = bgSig(b);
      fresh.innerHTML = bgCard(b);
      if (card) card.replaceWith(fresh); else after.after(fresh);
      card = fresh;
      bindBgCard(card);
    }
    if (card.previousElementSibling !== after) after.after(card);
    after = card;
  }
  $(".song-empty", grid)?.remove();
  focusFlash("backgrounds", grid, "bg");
  if (!list.length && !jobs.length && !failed.length) grid.insertAdjacentHTML("beforeend", `<div class="empty song-empty">No backgrounds yet: press + to make a 360° panorama or a backplate, or add your own picture</div>`);
}

// the New background dialog: a 360 panorama, a backplate for a template's still camera, or your own picture
const BG_SUBMIT = { pano: ["Make panorama", "Waits for its turn in the queue (GPU)."],
                    plate: ["Make backplate", "Waits for its turn in the queue (GPU)."],
                    setplate: ["Draw over the set", "Waits for its turn in the queue (GPU). The template needs a still camera."],
                    import: ["Add picture", "Saved at once as a backplate."] };
function bgKindState() {
  const f = $("#form-bg"), kind = f.kind.value;
  $$("#form-bg [data-bg-kind]").forEach((el) => el.classList.toggle("hidden", !el.dataset.bgKind.split(" ").includes(kind)));
  $("#bg-scene-label").innerHTML = kind === "import" ? "The place <i>optional · a few words for the library</i>"
    : kind === "setplate" ? "The place <i>optional · what the set stands for; empty = the template's Set words</i>" : "The place";
  $("#bg-shot-label").innerHTML = kind === "setplate"
    ? "Template <i>its Blender set and still camera: the picture is drawn over them</i>"
    : "Template camera <i>the backplate is drawn for this template's still camera (height, angle, horizon)</i>";
  bgSetWords();
  $("#form-bg button[type=submit]").textContent = BG_SUBMIT[kind][0];
  $("#bg-submit-note").textContent = BG_SUBMIT[kind][1];
}
$("#form-bg").image.addEventListener("change", (ev) => {     // a small preview of the chosen picture
  const file = ev.target.files[0], img = $("#bg-file-preview");
  if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
  if (!file) { img.classList.add("hidden"); $("#bg-file-hint").textContent = "Choose a PNG, JPG or WEBP picture · it is fitted to the video's frame"; return; }
  img.dataset.url = img.src = URL.createObjectURL(file);
  img.classList.remove("hidden");
  $("#bg-file-hint").textContent = file.name;
});
// over the set: the template's own Set words are the default place (shown in grey in the empty box)
const BG_SCENE_HINT = "A quiet harbour at dawn with fishing boats, stacked crates and gulls on the poles, soft pink light";
function bgSetWords() {
  const f = $("#form-bg"), opt = $("#bg-shot-select").selectedOptions[0];
  const words = (opt && opt.dataset.set) || "";
  f.scene.placeholder = f.kind.value !== "setplate" ? BG_SCENE_HINT
    : words ? `Empty = the template's Set words: ${words}` : "This template has no Set words: describe the place the set stands for";
}
$("#bg-shot-select").addEventListener("change", bgSetWords);
async function openNewBg() {
  $("#new-bg").classList.remove("hidden");
  const sel = $("#bg-shot-select"), cur = sel.value;
  try {
    const shots = (await api("/api/shots")).filter((s) => s.blend);
    sel.innerHTML = shots.map((s) => `<option value="${esc(s.blend)}" data-set="${esc(s.set_words || "")}">${esc(s.name)}${s.builtin ? " (built-in)" : ""}</option>`).join("");
    if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
  } catch { /* the list stays as it was */ }
  bgKindState();
  setTimeout(() => $("#form-bg").scene.focus(), 30);
}
$$("#form-bg input[name=kind]").forEach((r) => r.addEventListener("change", bgKindState));
$("#btn-close-bg").addEventListener("click", () => closeOverlay("#new-bg"));
$("#new-bg").addEventListener("click", (ev) => { if (ev.target.id === "new-bg") closeOverlay("#new-bg"); });
$("#btn-close-bgstyle").addEventListener("click", () => closeOverlay("#bg-style"));
$("#bg-style").addEventListener("click", (ev) => { if (ev.target.id === "bg-style") closeOverlay("#bg-style"); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") { closeOverlay("#new-bg"); closeOverlay("#bg-style"); } });

$("#form-bg").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = ev.target, kind = f.kind.value;
  if (kind === "import") {                          // instant: the picture is saved as a backplate
    if (!f.image.files.length) return toast("Choose a picture of the place.", true);
    const fd = new FormData();
    fd.append("image", f.image.files[0]);
    fd.append("name", f.name.value.trim());
    fd.append("scene", f.scene.value.trim());
    try {
      await api("/api/backgrounds/import", { method: "POST", body: fd });
      toast("Picture added as a backplate.");
      closeOverlay("#new-bg");
      f.image.value = ""; f.name.value = ""; f.image.dispatchEvent(new Event("change"));
      loadBackgroundsTab();
    } catch (e) { fail(e); }
    return;
  }
  if (kind === "setplate") {                        // drawn over the template's set: the place words are optional
    const opt = $("#bg-shot-select").selectedOptions[0];
    if (!f.shot.value) return toast("Choose the template.", true);
    if (!f.scene.value.trim() && !(opt && opt.dataset.set)) return toast("Describe the place the set stands for (this template has no Set words).", true);
    const body = { shot: f.shot.value, scene: f.scene.value.trim(), style: f.style.value.trim(),
                   style_preset: f.style_preset ? f.style_preset.value : "",       // a library style (styles.js)
                   name: f.name.value.trim() || null, seed: f.seed.value === "" ? null : +f.seed.value };
    closeOverlay("#new-bg");
    try {
      const r = await postJSON("/api/backgrounds/from-set", body);
      queued("Backplate over the set");
      if (r.warning) toast(r.warning, true);
      f.name.value = ""; f.seed.value = "";
      loadBackgroundsTab();
    } catch (e) { fail(e); openNewBg(); }
    return;
  }
  if (!f.scene.value.trim()) return toast("Describe the place.", true);
  const data = { kind, shot: kind === "plate" ? f.shot.value : null, scene: f.scene.value.trim(), style: f.style.value.trim(),
                 style_preset: f.style_preset ? f.style_preset.value : "",       // a library style (styles.js)
                 name: f.name.value.trim() || null, seed: f.seed.value === "" ? null : +f.seed.value, depth: f.depth.checked };
  closeOverlay("#new-bg");
  try {
    const r = await postJSON("/api/backgrounds/new", data);
    queued(kind === "plate" ? "Backplate" : "Background");
    if (r.warning) toast(r.warning, true);
    f.name.value = ""; f.seed.value = "";
    loadBackgroundsTab();
  } catch (e) { fail(e); openNewBg(); }
});

function openBgStyle(name) {
  const sel = $("#bg-style-lora");
  const style = styleLoras();
  sel.innerHTML = style.map((l) => `<option${/3dsrx/i.test(l) ? " selected" : ""}>${esc(l)}</option>`).join("");
  $("#bg-style-name").textContent = name;
  $("#form-bgstyle").dataset.name = name;
  $("#bg-style").classList.remove("hidden");
}
$("#form-bgstyle").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = ev.target, name = f.dataset.name;
  const data = { style_lora: f.style_lora.value, style: f.style.value.trim(), keep: +f.keep.value, strength: +f.strength.value };
  closeOverlay("#bg-style");
  try { await postJSON(`/api/backgrounds/${encodeURIComponent(name)}/style`, data); queued("Style redraw"); loadBackgroundsTab(); }
  catch (e) { fail(e); }
});

// ---------- the large panorama viewer (user, 2026-10-22: "i wish we could expand the panoramas larger")
// "Look round": a perspective view of the equirectangular picture drawn on a canvas (drag = turn and look up/down,
// wheel = zoom); "Whole panorama": the flat picture at the window's size. Esc or the close button closes it.
function openPanoViewer(url, name, flatOnly = false) {
  const ov = document.createElement("div");
  ov.className = "pano-viewer";
  ov.innerHTML = `<div class="pv-bar"><span class="pv-name">${esc(name)}</span>
      <div class="pv-modes"><button type="button" data-pv="look" class="on">Look round</button><button type="button" data-pv="flat">Whole panorama</button></div>
      <span class="pv-hint muted">Drag to look round · wheel to zoom · Esc to close</span>
      <button type="button" class="pv-close" title="Close (Esc)" aria-label="Close">✕</button></div>
    <canvas class="pv-canvas"></canvas>
    <div class="pv-flat hidden"><img alt="360 degree panorama ${esc(name)}"></div>`;
  document.body.append(ov);
  const canvas = $(".pv-canvas", ov), ctx = canvas.getContext("2d"), flat = $(".pv-flat", ov);
  if (flatOnly) {                                   // a backplate: the flat picture only
    $(".pv-modes", ov).classList.add("hidden");
    $(".pv-hint", ov).textContent = "Esc to close";
    canvas.classList.add("hidden");
    flat.classList.remove("hidden");
  }
  $("img", flat).src = url;
  const view = { yaw: 0, pitch: 0, fov: 80 };
  let src = null, sw = 0, sh = 0, raf = 0, low = false;
  const img = new Image();
  img.onload = () => {
    const c = document.createElement("canvas");
    sw = c.width = img.naturalWidth; sh = c.height = img.naturalHeight;
    const g = c.getContext("2d"); g.drawImage(img, 0, 0);
    src = g.getImageData(0, 0, sw, sh).data;
    draw();
  };
  img.src = url;
  function draw() {
    if (!src || canvas.classList.contains("hidden")) return;
    const scale = low ? 0.5 : 1, dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.max(64, Math.round(canvas.clientWidth * dpr * scale)), h = Math.max(64, Math.round(canvas.clientHeight * dpr * scale));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const out = ctx.createImageData(w, h), d = out.data;
    const t = Math.tan((view.fov * Math.PI / 180) / 2), aspect = w / h;
    const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw), cp = Math.cos(view.pitch), spt = Math.sin(view.pitch);
    let k = 0;
    for (let j = 0; j < h; j++) {
      const y = (1 - 2 * (j + 0.5) / h) * t;
      for (let i = 0; i < w; i++) {
        const x = (2 * (i + 0.5) / w - 1) * t * aspect;
        // camera ray (x right, y up, z forward), pitched, then turned by the yaw
        const y1 = y * cp + spt, z1 = -y * spt + cp;
        const dx = x * cy + z1 * sy, dz = -x * sy + z1 * cy;
        const lon = Math.atan2(dx, dz), lat = Math.atan2(y1, Math.hypot(dx, dz));
        let u = Math.floor((lon / (2 * Math.PI) + 0.5) * sw); if (u >= sw) u -= sw; if (u < 0) u += sw;
        let v = Math.floor((0.5 - lat / Math.PI) * sh); if (v >= sh) v = sh - 1; if (v < 0) v = 0;
        const p = (v * sw + u) * 4;
        d[k++] = src[p]; d[k++] = src[p + 1]; d[k++] = src[p + 2]; d[k++] = 255;
      }
    }
    ctx.putImageData(out, 0, 0);
  }
  const redraw = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); };
  let drag = null;
  canvas.addEventListener("pointerdown", (ev) => { drag = { x: ev.clientX, y: ev.clientY }; low = true; canvas.setPointerCapture(ev.pointerId); });
  canvas.addEventListener("pointermove", (ev) => {
    if (!drag) return;
    const per = (view.fov * Math.PI / 180) / canvas.clientHeight;
    view.yaw -= (ev.clientX - drag.x) * per;
    view.pitch = Math.max(-1.45, Math.min(1.45, view.pitch + (ev.clientY - drag.y) * per));
    drag = { x: ev.clientX, y: ev.clientY };
    redraw();
  });
  const endDrag = () => { if (drag) { drag = null; low = false; redraw(); } };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    view.fov = Math.max(30, Math.min(110, view.fov * (ev.deltaY > 0 ? 1.08 : 1 / 1.08)));
    redraw();
  }, { passive: false });
  const onResize = () => redraw();
  window.addEventListener("resize", onResize);
  $$("[data-pv]", ov).forEach((b) => b.addEventListener("click", () => {
    $$("[data-pv]", ov).forEach((x) => x.classList.toggle("on", x === b));
    const look = b.dataset.pv === "look";
    canvas.classList.toggle("hidden", !look);
    flat.classList.toggle("hidden", look);
    if (look) redraw();
  }));
  function close() {
    window.removeEventListener("resize", onResize);
    document.removeEventListener("keydown", onKey, true);
    cancelAnimationFrame(raf);
    ov.remove();
  }
  function onKey(ev) { if (ev.key === "Escape") { ev.stopPropagation(); close(); } }
  document.addEventListener("keydown", onKey, true);
  $(".pv-close", ov).addEventListener("click", close);
}
