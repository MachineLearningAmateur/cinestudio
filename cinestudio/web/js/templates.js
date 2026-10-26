// cinestudio web app - The Templates tab: the shot library, New template (from a video, import).
// Plain scripts (no modules), loaded in order by index.html; they share their top-level names.
"use strict";

// ------------------------------------------------------------------ templates (Blender dance shots from a song)
const busyTpl = (t) => t.status === "queued" || t.status === "running";
const tplSig = (t) => [t.status, t.preview_url, t.poster_url, t.error, t.description, JSON.stringify(t.figure || null),
                      JSON.stringify(t.layers || null), JSON.stringify(t.background || null),
                      JSON.stringify(t.people || null), JSON.stringify(t.mannequins || null),
                      JSON.stringify(t.check || null)].join("|");
const tplState = { routines: [], songs: [], picked: ["S1", "S2"] };
const routineLabel = (id) => {
  const r = tplState.routines.find((x) => x.id === id.replace(/~$/, ""));
  return (r ? r.label : id) + (id.endsWith("~") ? " (mirrored)" : "");
};

function tplMedia(t) {
  if (t.preview_url) return `<video disablepictureinpicture src="${t.preview_url}" ${t.poster_url ? `poster="${t.poster_url}"` : ""} muted loop playsinline preload="metadata"></video>`;
  if (t.poster_url) return `<img src="${t.poster_url}" alt="">`;
  if (t.people && t.people.video_url)                      // a people scan: the video with the numbered people
    return `<video disablepictureinpicture src="${t.people.video_url}" muted loop playsinline preload="metadata"></video>`;
  if (busyTpl(t)) return `<div class="busy"><span class="spinner"></span></div>`;
  return `<div class="noprev">No preview</div>`;
}

function tplBody(t) {
  if (t.status === "running" || t.status === "queued") {
    const p = t.progress || {};
    const frac = p.n ? ((p.i || 1) - 1) / p.n : 0;
    return `<div class="sc-prog">${p.n ? progressBar(frac, t.started ? Date.now() / 1000 - t.started : 0) : `<span class="spinner"></span>`}
      <span class="ptext">${esc(p.text || (t.status === "queued" ? (t.source === "text" || t.source === "video" ? "Waiting for its turn on the GPU…" : "Waiting for Blender…") : "Starting…"))}</span></div>`;
  }
  if (t.error) return `<div class="sc-err">${esc(t.error)}</div>`;
  if (t.status === "pick") return `<div class="sub">SAM 3 found ${esc((t.people || {}).seen || 0)} people: choose who becomes a mannequin.</div>`;
  if (t.status === "interrupted") return `<div class="sc-err">The worker stopped before the template was done. Delete it and make it again.</div>`;
  return "";
}

function tplCard(t) {
  const facts = [];
  if (t.builtin) facts.push(`<span class="fact chip builtin" title="Ships with cinestudio (read-only)">Built-in</span>`);
  if (t.handmade || t.kind === "blender") facts.push(`<span class="fact" title="A Blender shot (not made from a song here)">Blender</span>`);
  if (t.source === "text") facts.push(`<span class="fact chip from-text" title="Made from text: a grey stand-in that the Video tab replaces with your character">from text</span>`);
  if (t.source === "video" && t.kind === "blender") facts.push(`<span class="fact chip from-text" title="The mannequin copies the motion of a dance video (SAM 3D Body)">from video</span>`);
  if (t.status === "done" && t.mannequins) facts.push(slotChip(t));   // how many characters the template has
  const ch = (t.people && t.people.chosen) || [];
  if (ch.length > 1 && !t.mannequins) facts.push(`<span class="fact" title="${esc(ch.map((c) => `${c.collection}: person ${c.track} (${c.place[0] >= 0 ? "+" : ""}${c.place[0]} m, depth ${c.place[1]} m)`).join(" · "))}${t.people.trimmed ? ` · trimmed to the time ${ch.length === 2 ? "both are" : "all are"} in the video` : ""}">${ch.length} people</span>`);
  (t.warnings || []).forEach((w) => facts.push(`<span class="fact warn" title="${esc(w)}">note</span>`));
  if (t.private_sound) facts.push(`<span class="fact warn" title="The sound is the video's own recording: keep this template private (not for a contest submission or the built-in library). An original song at the same tempo can replace it.">private sound</span>`);
  if (t.music && t.music.bpm) facts.push(`<span class="fact" title="The tempo measured in the video's sound (for an original song at the same tempo)">${(+t.music.bpm).toFixed(1)} BPM</span>`);
  if (t.camera) facts.push(`<span class="fact" title="Camera">${esc(t.camera_label || t.camera)}</span>`);
  const bm = t.background_motion;
  if (bm && bm.unavailable) facts.push(`<span class="fact warn" title="${esc(bm.error || "The check could not run")}">motion check unavailable</span>`);
  else if (bm && bm.flag) facts.push(`<span class="fact warn" title="Something besides the stand-in moves at frames ${esc((bm.frames || []).slice(0, 12).join(", "))}: it can become a phantom figure in the video">background moves</span>`);
  else if (bm) facts.push(`<span class="fact" title="Checked: nothing besides the stand-in moves">still background</span>`);
  if (t.seconds) facts.push(`<span class="fact">${(+t.seconds).toFixed(1)} s</span>`);
  if (t.beats) facts.push(`<span class="fact" title="Beats in the dance">${esc(t.beats)} beats</span>`);
  if (t.bpm) facts.push(`<span class="fact" title="Tempo of the shot">${(+t.bpm).toFixed(1)} BPM</span>`);
  if (t.loop) facts.push(`<span class="fact" title="The last frame repeats the first">loop</span>`);
  if (t.night) facts.push(`<span class="fact">night</span>`);
  const rts = (t.routines || []).map((r) => `<span class="chip rt" title="${esc(r)}">${esc(routineLabel(r))}</span>`).join("");
  return `<div class="tpl-media">${tplMedia(t)}</div>
    <div class="tpl-info">
      <div class="sc-head"><span class="sc-icon">${ICON.cube}</span><span class="sc-name" title="${esc(t.name)}">${esc(t.name)}</span>
        ${t.status === "done" ? "" : `<span class="chip ${esc(t.status)}">${esc(t.status)}</span>`}
        ${busyTpl(t) || t.builtin ? "" : trashButton("template", t.name)}</div>
      ${t.description ? `<div class="tpl-line tpl-desc" title="${esc(t.description)}">${esc(t.description)}</div>` : ""}
      ${t.song ? `<div class="tpl-line">${ICON.music} ${esc(t.song)}${t.start != null && !t.loop ? ` from ${(+t.start).toFixed(2)} s` : t.start != null ? ` · section from ${(+t.start).toFixed(2)} s` : ""}</div>` : ""}
      ${rts ? `<div class="sc-facts">${rts}</div>` : ""}
      ${facts.length ? `<div class="sc-facts">${facts.join("")}</div>` : ""}
      ${figRow(t)}
      ${layersRow(t)}
      ${checkRow(t)}
      ${tplBgRow(t)}
      ${peopleRow(t)}
      <div class="sc-body">${tplBody(t)}</div></div>`;
}

// "who is who" after a people scan (New template > From a video > People: Pick after the scan): the numbered sheet and
// one checkbox per person found; one to four of them become the mannequins (no second GPU job)
function peopleRow(t) {
  const p = t.people;
  if (!p || t.status !== "pick") return "";
  const seen = (p.tracks || []).filter((x) => x.visible > 0);
  const rows = seen.map((x) => `<label class="ppl-opt" title="Visible in ${Math.round(x.visible * 100)}% of the frames${x.first_s != null ? `, ${x.first_s}–${x.last_s} s` : ""}">
      <input type="checkbox" value="${x.track}"><span class="ppl-sw" style="background:${esc(x.color)}"></span>Person ${x.track}
      <small>${Math.round(x.visible * 100)}%</small></label>`).join("");
  return `<div class="ppl-pick">${p.sheet_url ? `<a href="${p.sheet_url}" target="_blank" rel="noopener"><img class="ppl-sheet" src="${p.sheet_url}" alt="The people found, numbered"></a>` : ""}
    <div class="ppl-opts">${rows || `<span class="sub">Nobody was seen long enough.</span>`}</div>
    <div class="ppl-act"><button type="button" class="btn-ppl" data-ppl="${esc(t.name)}" title="One to ${MAX_VIDEO_CHARACTERS} people: they become Mannequin, Mannequin2, ... left to right (the left one is the run's Character 1)">Make the template</button>
      <span class="sub">one to ${MAX_VIDEO_CHARACTERS} people</span></div></div>`;
}

// the "Select figure" button and its chip (video templates: the stand-in that a run replaces)
function figRow(t) {
  if (t.kind !== "video" || t.status !== "done") return "";
  const f = t.figure || {};
  let chip = "";
  if (f.status === "queued" || f.status === "running") chip = `<span class="chip running">tracking…</span>`;
  else if (f.selected) chip = f.flags ? `<span class="chip fig-warn" title="Frames where the mask jumps or is empty: check them">figure selected · ${f.flags} to check</span>`
    : `<span class="chip fig-ok" title="SAM 2 follows the figure through the whole video">figure selected</span>`;
  else if (f.status === "failed" || f.status === "interrupted") chip = `<span class="chip failed">selection ${esc(f.status)}</span>`;
  return `<div class="fig-row"><button type="button" class="btn-fig" data-fig="${esc(t.name)}" title="Mark the figure to replace; the selection follows it through the video">Select figure</button>${chip}</div>`;
}

// the "Layers" button: the words of the template's Set and Props (shot layers), used by the video text and by a
// backplate drawn over the set
function layersRow(t) {
  const l = t.layers;
  if (!l || t.status !== "done") return "";
  const bits = [l.set ? "set described" : "no set words",
    l.props ? `${l.described ? `${l.described} prop text${l.described > 1 ? "s" : ""}` : "props without words"}` : null].filter(Boolean);
  return `<div class="fig-row"><button type="button" class="btn-layers" data-layers="${esc(t.name)}"
      title="The words of the template's Set (the place) and Props (moving objects): they go into the video text and a backplate drawn over the set">Layers</button>
    <span class="chip ${l.warnings ? "fig-warn" : ""}" title="${l.scanned ? `${l.props} prop object${l.props === 1 ? "" : "s"} in the Props layer${l.warnings ? ` · ${l.warnings} warning${l.warnings > 1 ? "s" : ""}` : ""}` : "Not read from the .blend yet: open Layers"}">${esc(bits.join(" · "))}</span></div>`;
}

// the template check (`shot check NAME`): a chip with the last result and a "Check" button that opens the list
function checkChip(c, mini) {
  if (!c) return "";
  const cls = c.errors ? "chk-err" : c.warnings ? "chk-warn" : "chk-ok";
  const text = c.errors ? `${c.errors} error${c.errors > 1 ? "s" : ""}${c.warnings ? ` · ${c.warnings} warning${c.warnings > 1 ? "s" : ""}` : ""}`
    : c.warnings ? `${c.warnings} warning${c.warnings > 1 ? "s" : ""}` : "✓ checks";
  const when = c.at ? new Date(c.at * 1000).toLocaleString() : "";
  const title = c.stale ? "The .blend changed since this check: check again" : `Template check${when ? ` of ${when}` : ""}: ${c.errors ? "errors break a video" : c.warnings ? "warnings can mislead the video" : "every rule passed"}`;
  return `<span class="chip ${cls}${c.stale ? " chk-stale" : ""}${mini ? " chk-mini" : ""}" title="${esc(title)}">${esc(text)}${c.stale ? " (old)" : ""}</span>`;
}
function checkRow(t) {
  if (t.status !== "done" || !t.blend || t.kind === "video") return "";
  return `<div class="fig-row"><button type="button" class="btn-check" data-check="${esc(t.name)}"
      title="Check the template rules (bone scales, actions across clip joins, end hold, frame 1, camera speed, layers, sizes, clip texts): Blender reads the .blend in the background, a few seconds">Check</button>
    ${t.check ? checkChip(t.check) : `<span class="chip" title="Not checked yet">not checked</span>`}</div>`;
}
const CHECK_ICON = { error: "!", warning: "!", info: "i", ok: "✓" };
function checkFrames(fr) {
  if (!fr || !fr.length) return "";
  const txt = fr.slice(0, 6).map(([a, b]) => (a === b ? `${a}` : `${a}–${b}`)).join(", ") + (fr.length > 6 ? " …" : "");
  return `<div class="cfr">${fr.length === 1 && fr[0][0] === fr[0][1] ? "frame" : "frames"} ${esc(txt)}</div>`;
}
function checkItem(c) {
  return `<li class="sev-${esc(c.severity)}"><span class="ci" aria-label="${esc(c.severity)}">${CHECK_ICON[c.severity] || "?"}</span>
    <div><span class="cid">${esc(c.id)}</span>${esc(c.message)}</div>${checkFrames(c.frames)}${c.fix && c.severity !== "ok" ? `<div class="cfix">${esc(c.fix)}</div>` : ""}</li>`;
}
function drawCheck(d) {
  const shown = d.checks.filter((c) => c.severity !== "ok"), oks = d.checks.filter((c) => c.severity === "ok");
  $("#check-sum").innerHTML = [checkChip({ errors: d.errors, warnings: d.warnings }),
    d.infos ? `<span class="fact">${d.infos} note${d.infos > 1 ? "s" : ""}</span>` : "",
    `<span class="fact">${oks.length} passed</span>`,
    d.clips ? `<span class="fact" title="Frames ${esc(d.frames.join("–"))}">${d.clips} clip${d.clips > 1 ? "s" : ""}</span>` : ""].join("");
  $("#check-list").innerHTML = shown.map(checkItem).join("")
    + (oks.length ? `<li class="oks"><details${shown.length ? "" : " open"}><summary>${oks.length} check${oks.length > 1 ? "s" : ""} passed</summary><ul>${oks.map(checkItem).join("")}</ul></details></li>` : "");
  $("#check-when").textContent = `${d.cached ? "From the cache (same .blend)" : "Read from the .blend"} · ${new Date(d.at * 1000).toLocaleString()}`;
}
const checkUI = { name: null };
async function openCheck(name, fresh) {
  checkUI.name = name;
  $("#tpl-check-name").textContent = name;
  $("#check-sum").innerHTML = `<span class="fact muted"><span class="spinner"></span> Blender reads the .blend…</span>`;
  $("#check-list").innerHTML = ""; $("#check-when").textContent = "";
  $("#tpl-check").classList.remove("hidden");
  $("#btn-check-again").disabled = true;
  try {
    drawCheck(await postJSON(`/api/templates/${encodeURIComponent(name)}/check`, { fresh: !!fresh }));
    loadTemplates();                                       // the card's chip shows the new result
  } catch (e) { closeOverlay("#tpl-check"); fail(e); }
  finally { $("#btn-check-again").disabled = false; }
}
$("#btn-check-again").addEventListener("click", () => openCheck(checkUI.name, true));
$("#btn-close-check").addEventListener("click", () => closeOverlay("#tpl-check"));
$("#tpl-check").addEventListener("click", (ev) => { if (ev.target.id === "tpl-check") closeOverlay("#tpl-check"); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeOverlay("#tpl-check"); });

// the template's default background (shot.json "background", e.g. a built-in template's backplate drawn over its set):
// a small picture and its name; a click opens its card in the Backgrounds tab
function tplBgRow(t) {
  const b = t.background;
  if (!b || t.status !== "done") return "";
  const label = b.kind === "plate" ? "Backplate" : "Background";
  return `<div class="fig-row tpl-bg"><button type="button" class="tpl-bg-link" data-tpl-bg="${esc(b.name)}"
      title="Its ${label.toLowerCase()} (the New video form preselects it): open it in the Backgrounds tab">${b.url ? `<img src="${esc(b.url)}" alt="">` : ""}<span>${label}: ${esc(b.name)}</span></button></div>`;
}

function bindTplCard(card) {
  const v = $("video", card);
  if (v) {
    card.addEventListener("mouseenter", () => v.play().catch(() => {}));
    card.addEventListener("mouseleave", () => { v.pause(); });
    v.addEventListener("click", () => { v.muted = !v.muted; v.play().catch(() => {}); });   // a click turns the sound on
    v.title = "Hover to play · click for sound";
  }
  bindDeletes(card, () => loadTemplates());
  const bl = $("[data-tpl-bg]", card);
  if (bl) bl.addEventListener("click", (ev) => { ev.stopPropagation(); focusCard("backgrounds", bl.dataset.tplBg); });
  const cb = $(".btn-check", card);
  if (cb) cb.addEventListener("click", (ev) => { ev.stopPropagation(); openCheck(cb.dataset.check); });
  const lb = $(".btn-layers", card);
  if (lb) lb.addEventListener("click", (ev) => { ev.stopPropagation(); openLayers(lb.dataset.layers); });
  const fb = $(".btn-fig", card);
  if (fb) fb.addEventListener("click", (ev) => { ev.stopPropagation(); openFigure(fb.dataset.fig); });
  const pb = $(".btn-ppl", card);
  if (pb) pb.addEventListener("click", async (ev) => {
    ev.stopPropagation();
    const tracks = $$(".ppl-opt input:checked", card).map((i) => +i.value);
    if (!tracks.length || tracks.length > MAX_VIDEO_CHARACTERS) { toast(`Choose one to ${MAX_VIDEO_CHARACTERS} people.`, true); return; }
    pb.disabled = true;
    try {
      await postJSON(`/api/templates/${encodeURIComponent(pb.dataset.ppl)}/people`, { tracks });
      toast(`Blender builds the template with ${tracks.length === 1 ? "one mannequin" : `${tracks.length} mannequins`}.`);
      loadTemplates();
    } catch (e) { pb.disabled = false; fail(e); }
  });
}

const tplPager = makePager("templates", () => loadTemplates());
async function loadTemplates() {
  let d;
  try { d = await api("/api/templates"); } catch { return; }
  tplState.routines = d.routines; tplState.songs = d.songs;
  const grid = $("#tpl-grid");
  tplPager.measure(grid); tplPager.watch(grid);
  focusPage("templates", tplPager, d.shots.map((t) => t.name));   // a link from a background's card
  const per = tplPager.per();
  if (!drawPaging(tplPager, d.shots.length, Math.min(per || d.shots.length, Math.max(0, d.shots.length - tplPager.offset())),
                  "template", $("#tpl-bar"), $("#tpl-pager"), grid)) return loadTemplates();
  const tplShown = per ? d.shots.slice(tplPager.offset(), tplPager.offset() + per) : d.shots;
  if (!$("#btn-new-template", grid)) {
    grid.innerHTML = `<button type="button" class="song-card tpl-card new-tile" id="btn-new-template" title="New template" aria-label="New template">
        <span class="plus-box"><svg viewBox="0 0 24 24" width="30" height="30" aria-hidden="true"><path fill="currentColor" d="M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6z"/></svg></span>
        <span class="plus-label">New template</span></button>`;
    $("#btn-new-template").addEventListener("click", openNewTemplate);
  }
  const names = new Set(tplShown.map((t) => t.name));
  $$(".tpl-card[data-tpl]", grid).forEach((c) => { if (!names.has(c.dataset.tpl)) c.remove(); });
  let after = $("#btn-new-template", grid);
  for (const t of tplShown) {
    let card = $$(".tpl-card[data-tpl]", grid).find((c) => c.dataset.tpl === t.name);
    if (!card || card.dataset.sig !== tplSig(t)) {
      const fresh = document.createElement("div");
      fresh.className = "song-card tpl-card";
      fresh.dataset.tpl = t.name;
      fresh.dataset.sig = tplSig(t);
      fresh.innerHTML = tplCard(t);
      if (card) card.replaceWith(fresh); else after.after(fresh);
      card = fresh;
      bindTplCard(card);
    } else if (busyTpl(t)) {
      $(".sc-body", card).innerHTML = tplBody(t);
    }
    if (card.previousElementSibling !== after) after.after(card);
    after = card;
  }
  $(".song-empty", grid)?.remove();
  focusFlash("templates", grid, "tpl");
  if (!d.shots.length) grid.insertAdjacentHTML("beforeend", `<div class="empty song-empty">No templates yet: press + to make one</div>`);
}

function drawPicked() {
  const f = $("#form-template");
  f.routines.value = tplState.picked.join(",");
  const box = $("#tpl-picked");
  box.innerHTML = tplState.picked.length ? tplState.picked.map((r, i) =>
    `<span class="rt-pick"><span class="n">${i + 1}</span>${esc(routineLabel(r))}<button type="button" data-i="${i}" title="Remove" aria-label="Remove">×</button></span>`).join("")
    : `<span class="empty-hint">Click the routines below</span>`;
  $$("button[data-i]", box).forEach((b) => b.addEventListener("click", () => { tplState.picked.splice(+b.dataset.i, 1); drawPicked(); }));
  tplHint();
}

function tplHint() {
  const f = $("#form-template"), hint = $("#tpl-hint");
  const s = tplState.songs.find((x) => x.name === f.song.value);
  const counts = tplState.picked.reduce((n, r) => n + ((tplState.routines.find((x) => x.id === r.replace(/~$/, "")) || {}).counts || 0), 0);
  if (!s) { hint.textContent = tplState.songs.length ? "" : "Make a song in the Music tab first."; return; }
  if (!f.loop.checked) {
    hint.textContent = `${counts} counts at ${(+s.bpm).toFixed(1)} BPM ≈ ${((counts + 2) * 60 / s.bpm).toFixed(1)} s (count 1 on frame 13).`;
    return;
  }
  const want = Math.max(1, +f.seconds.value || 10) * s.bpm / 60;
  const n = [8, 16, 32, 48, 64].reduce((a, b) => (Math.abs(b - want) < Math.abs(a - want) ? b : a));
  const ok = counts && n % counts === 0 && !tplState.picked.some((r) => r.startsWith("F"));
  hint.textContent = `Loop: ${n} beats at ${(+s.bpm).toFixed(1)} BPM = ${(n * 60 / s.bpm).toFixed(1)} s.`
    + (ok ? (n / counts > 1 ? ` The routines play ${n / counts} times.` : "")
      : tplState.picked.some((r) => r.startsWith("F")) ? " The Finale cannot be part of a loop."
      : ` Choose routines that fill ${n} beats (now ${counts}).`);
}

function openNewTemplate() {
  const f = $("#form-template");
  const cur = f.song.value;
  f.song.innerHTML = tplState.songs.length ? tplState.songs.map((s) => `<option value="${esc(s.name)}">${esc(s.name)} · ${(+s.bpm).toFixed(1)} BPM</option>`).join("")
    : `<option value="">No song is ready: make one in the Music tab</option>`;
  if (cur && tplState.songs.some((s) => s.name === cur)) f.song.value = cur;
  const pal = $("#tpl-palette");
  pal.innerHTML = tplState.routines.map((r) => `<button type="button" data-rt="${esc(r.id)}" title="Add ${esc(r.label)}">
      ${esc(r.label)}<small>${esc(r.id)} · ${r.counts} counts</small></button>`).join("");
  $$("button[data-rt]", pal).forEach((b) => b.addEventListener("click", () => {
    tplState.picked.push(b.dataset.rt + ($("#tpl-mirror").checked ? "~" : "")); drawPicked();
  }));
  drawPicked();
  $("#new-template").classList.remove("hidden");
  setTimeout(() => f.name.focus(), 30);
}
function closeNewTemplate() {
  const box = $("#new-template");
  if (box.classList.contains("hidden")) return false;
  box.classList.add("hidden");
  return true;
}
$("#btn-close-template").addEventListener("click", closeNewTemplate);
$("#new-template").addEventListener("click", (ev) => { if (ev.target.id === "new-template") closeNewTemplate(); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeNewTemplate(); });
$("#tpl-undo").addEventListener("click", () => { tplState.picked.pop(); drawPicked(); });
$("#tpl-clear").addEventListener("click", () => { tplState.picked = []; drawPicked(); });
(() => {
  const f = $("#form-template");
  const loopUi = () => {
    $(".tpl-secs", f).classList.toggle("hidden", !f.loop.checked);
    $(".tpl-start", f).classList.toggle("hidden", f.loop.checked);
    tplHint();
  };
  f.loop.addEventListener("change", loopUi);
  f.seconds.addEventListener("input", tplHint);
  f.song.addEventListener("change", tplHint);
  loopUi();
})();

// the kinds of template: a dance on a song, the motion of a dance video (both Blender) or an import
function tplMode() { return $("#form-template").querySelector('input[name="mode"]:checked').value; }
function setTplMode(mode) {
  const f = $("#form-template");
  f.querySelector(`input[name="mode"][value="${mode}"]`).checked = true;
  for (const [sel, on] of [[".tpl-mode-video", mode === "video"], [".tpl-mode-dance", mode === "dance"], [".tpl-mode-import", mode === "import"]]) {
    const fs = $(sel, f);
    fs.disabled = !on;                                     // a disabled fieldset's required fields do not block Generate
    fs.classList.toggle("hidden", !on);
  }
  $("#tpl-subtitle").textContent = { video: "the mannequin copies the motion of a dance video (a Blender shot)",
    dance: "a Blender dance shot on a song's beats", import: "your own Blender shot or stand-in video" }[mode];
  f.name.placeholder = { video: "folk_dance", dance: "shuffle_loop", import: "my_orbit_shot" }[mode];
  try { localStorage.setItem("tplMode", mode); } catch { /* private window */ }
}
$$('#form-template input[name="mode"]').forEach((r) => r.addEventListener("change", () => setTplMode(tplMode())));
// "Dance on a song" left the app (user, 2026-10-25: "we can just remove the dance on a song option"); the CLI keeps
// `shot make-dance`, and existing dance templates still work
try { const m = localStorage.getItem("tplMode"); setTplMode(["video", "import"].includes(m) ? m : "video"); } catch { setTplMode("video"); }

$("#form-template").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = ev.target;
  if (tplMode() === "import") {                            // an upload: the file goes as a multipart form
    const fd = new FormData();
    fd.append("name", f.name.value.trim());
    fd.append("description", f.description.value.trim());
    fd.append("file", f.file.files[0]);
    const big = f.file.files[0] && f.file.files[0].size > 50e6;
    toast(big ? "Uploading the file…" : "Importing…");
    try {
      const r = await api("/api/templates/import", { method: "POST", body: fd });
      closeNewTemplate();
      toast(`Template "${r.name}" imported.`);
      f.name.value = ""; f.file.value = ""; f.description.value = "";
      loadTemplates();
    } catch (e) { fail(e); }
    return;
  }
  if (tplMode() === "video") {                             // a dance video: uploaded, then a GPU job (SAM 3D Body)
    const fd = new FormData();
    fd.append("name", f.name.value.trim());
    fd.append("file", f.vfile.files[0]);
    fd.append("start", f.vstart.value || "0");
    fd.append("seconds", f.vseconds.value || "10");
    fd.append("sound", f.vsound.value);
    fd.append("people", f.vpeople.value);
    fd.append("description", f.vdesc.value.trim());
    toast(f.vfile.files[0] && f.vfile.files[0].size > 50e6 ? "Uploading the video…" : "Starting the template…");
    try {
      const r = await api("/api/templates/from-video", { method: "POST", body: fd });
      closeNewTemplate();
      toast(`Template "${r.name}": waits for its turn on the GPU.`);
      if (r.warning) toast(r.warning, true);
      f.name.value = ""; f.vfile.value = ""; f.vdesc.value = "";
      loadTemplates();
    } catch (e) { fail(e); }
    return;
  }
  const data = { name: f.name.value.trim(), song: f.song.value, routines: f.routines.value, loop: f.loop.checked,
    seconds: +f.seconds.value || 10, start: f.start.value === "" ? null : +f.start.value, night: f.night.checked };
  closeNewTemplate();                                      // close at once; the work goes on in the background
  toast("Starting the template…");
  try {
    const r = await postJSON("/api/templates/new", data);
    toast(`Template "${r.name}": Blender builds it.`);
    if (r.warning) toast(r.warning, true);
    f.name.value = "";
    loadTemplates();
  } catch (e) { fail(e); openNewTemplate(); }              // the form keeps what was chosen
});

// ------------------------------------------------------------------ the Layers dialog (`shot layers NAME --set --prop --rescan`)
const layersUI = { name: null, keys: [], builtin: false };
function layerPropRow(key = "", text = "") {
  const row = document.createElement("div");
  row.className = "lp-row";
  row.innerHTML = `<input class="lp-key" value="${esc(key)}" placeholder="Ball_*" spellcheck="false" title="A prop's object name, or a pattern such as Ball_* for several">
    <input class="lp-text" value="${esc(text)}" maxlength="300" placeholder="plain red rubber dodgeballs" title="The prop's words: the same concrete words in the background and video texts">
    <button type="button" class="lp-del" title="Remove these words" aria-label="Remove">×</button>`;
  $(".lp-del", row).addEventListener("click", () => row.remove());
  $("#layers-props").append(row);
  return row;
}
function drawLayers(d) {
  layersUI.keys = Object.keys(d.props || {});
  layersUI.builtin = d.builtin;
  const f = $("#form-layers"), sc = d.scan || {};
  f.set.value = d.set || "";
  $("#layers-props").innerHTML = "";
  for (const [k, v] of Object.entries(d.props || {})) layerPropRow(k, v);
  const props = sc.props || [];
  // the scanned props: a click adds a line for it (one pattern for numbered copies: Ball_Dodge1, Ball_Kick2 -> Ball*)
  const stemOf = (n) => n.split(/[_.\-\d]/)[0];
  const count = {};
  props.forEach((p) => { const st = stemOf(p.name); if (st) count[st] = (count[st] || 0) + 1; });
  const stems = [...new Set(props.map((p) => { const st = stemOf(p.name); return st && count[st] > 1 ? `${st}*` : p.name; }))];
  $("#layers-prop-names").innerHTML = props.length
    ? `<span class="sub">In the Props layer:</span> ${stems.map((x) => `<button type="button" class="chip lp-add" data-key="${esc(x)}" title="Add a line for ${esc(x)}">${esc(x)}</button>`).join(" ")}
       <button type="button" class="linkbtn lp-add" data-key="">+ another line</button>`
    : `<span class="sub">The Props layer is empty${sc.props ? "" : " (or not read yet)"}.</span>`;
  $$(".lp-add", $("#layers-prop-names")).forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.key && $$("#layers-props .lp-key").some((i) => i.value === b.dataset.key)) return;
    $(".lp-text", layerPropRow(b.dataset.key)).focus();
  }));
  const ch = sc.characters || [];
  $("#layers-scan").innerHTML = d.scan ? [
    `<span class="fact" title="Mannequin collections">${ch.length} character layer${ch.length === 1 ? "" : "s"}</span>`,
    `<span class="fact" title="Objects in the Props collection">${props.length} prop${props.length === 1 ? "" : "s"}</span>`,
    `<span class="fact" title="Objects in the Set collection">${(sc.set || []).length} set object${(sc.set || []).length === 1 ? "" : "s"}</span>`,
    sc.camera_moves ? `<span class="fact warn" title="A backplate needs a still camera">the camera moves</span>` : `<span class="fact" title="A backplate fits a still camera">still camera</span>`].join("")
    : `<span class="fact muted">${d.builtin ? "Built-in template: its layers are set in the repo" : "Not read yet"}</span>`;
  $("#layers-warn").innerHTML = (d.warnings || []).map((w) => `<li>${esc(w)}</li>`).join("");
  $$("#form-layers input, #form-layers textarea, #form-layers button[type=submit], #btn-layers-rescan").forEach((el) => { el.disabled = d.builtin; });
  $("#form-layers button[type=submit]").title = d.builtin ? "A built-in template is read-only: copy it to your library to change its layers" : "";
}
async function openLayers(name) {
  layersUI.name = name;
  $("#tpl-layers-name").textContent = name;
  $("#layers-scan").innerHTML = `<span class="fact muted"><span class="spinner"></span> reading the layers…</span>`;
  $("#layers-props").innerHTML = ""; $("#layers-prop-names").innerHTML = ""; $("#layers-warn").innerHTML = "";
  $("#tpl-layers").classList.remove("hidden");
  try { drawLayers(await api(`/api/templates/${encodeURIComponent(name)}/layers`)); }
  catch (e) { closeOverlay("#tpl-layers"); fail(e); }
}
async function saveLayers(rescan) {
  const f = $("#form-layers"), props = {};
  for (const row of $$("#layers-props .lp-row")) {
    const k = $(".lp-key", row).value.trim(), t = $(".lp-text", row).value.trim();
    if (k) props[k] = t;
  }
  for (const k of layersUI.keys) if (!(k in props)) props[k] = "";          // a removed line removes its words
  const btns = $$("#form-layers button");
  btns.forEach((b) => { b.disabled = true; });
  try {
    const d = await postJSON(`/api/templates/${encodeURIComponent(layersUI.name)}/layers`, { set: f.set.value, props, rescan: !!rescan });
    drawLayers(d);
    toast(rescan ? "The layers were read again." : "Layers saved.");
    if (!rescan) closeOverlay("#tpl-layers");
    loadTemplates();
  } catch (e) { fail(e); }
  finally { btns.forEach((b) => { b.disabled = layersUI.builtin; }); $("#btn-close-layers").disabled = false; }
}
$("#form-layers").addEventListener("submit", (ev) => { ev.preventDefault(); saveLayers(false); });
$("#btn-layers-rescan").addEventListener("click", () => saveLayers(true));
$("#btn-close-layers").addEventListener("click", () => closeOverlay("#tpl-layers"));
$("#tpl-layers").addEventListener("click", (ev) => { if (ev.target.id === "tpl-layers") closeOverlay("#tpl-layers"); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeOverlay("#tpl-layers"); });
