// cinestudio web app - The Music tab: the song library and the New song dialog.
// Plain scripts (no modules), loaded in order by index.html; they share their top-level names.
"use strict";

// ------------------------------------------------------------------ music (ACE-Step songs)
const busySong = (s) => s.status === "queued" || s.status === "running";
const songSig = (s) => [s.status, s.audio_url, s.error].join("|");     // a card is drawn again only when this changes

function songFacts(s) {
  const b = s.beats;
  const f = [];
  f.push(b && b.bpm ? `<span class="fact" title="Measured from the audio (asked: ${esc(s.bpm)})"><b>${esc(b.bpm)}</b> BPM</span>`
    : `<span class="fact" title="Asked tempo">${esc(s.bpm)} BPM</span>`);
  f.push(`<span class="fact">${esc(Math.round(s.seconds || 0))} s</span>`);
  if (s.key) f.push(`<span class="fact">${esc(s.key)}</span>`);
  if (b && b.count) f.push(`<span class="fact" title="Beats found in the audio">${esc(b.count)} beats</span>`);
  if (b && b.first_downbeat != null) f.push(`<span class="fact" title="The first downbeat (count 1)">count 1 at ${(+b.first_downbeat).toFixed(2)} s</span>`);
  f.push(`<span class="fact muted" title="Seed">seed ${esc(s.seed)}</span>`);
  return f.join("");
}

function songBody(s) {
  if (s.status === "done" && s.audio_url) return `<audio controls preload="metadata" src="${s.audio_url}"></audio>`;
  if (s.status === "running") {
    const p = s.progress || {};
    const frac = p.max ? (p.value || 0) / p.max : 0;
    return `<div class="sc-prog">${frac ? progressBar(frac, s.started ? Date.now() / 1000 - s.started : 0)
      : `<span class="spinner"></span><span class="ptext">Loading the model…</span>`}</div>`;
  }
  if (s.status === "queued") {
    const q = (state.queue.items || []).find((x) => x.kind === "song" && x.ref === s.name);
    const pos = q ? q.position : null;
    return `<div class="sc-prog"><span class="spinner"></span><span class="ptext">${pos ? `Waiting in the queue: #${pos} in line` : "Starting…"}</span></div>`;
  }
  if (s.error) return `<div class="sc-err">${esc(s.error)}</div>`;
  if (s.status === "interrupted") return `<div class="sc-err">The worker stopped before the song was done. Delete it and make it again.</div>`;
  return "";
}

function songCard(s) {
  const lyr = (s.lyrics || "").trim();
  const sung = lyr && lyr.toLowerCase() !== "[instrumental]";
  return `<div class="sc-head"><span class="sc-icon">${ICON.music}</span><span class="sc-name" title="${esc(s.name)}">${esc(s.name)}</span>
      ${s.status === "done" ? "" : `<span class="chip ${esc(s.status)}">${esc(s.status)}</span>`}
      ${busySong(s) ? "" : trashButton("song", s.name)}</div>
    <div class="sc-tags clamp" title="Click to show all">${esc(s.tags)}</div>
    <div class="sc-facts">${songFacts(s)}</div>
    <div class="sc-body">${songBody(s)}</div>
    ${sung ? `<details class="sc-lyrics"><summary>Lyrics</summary><pre>${esc(lyr)}</pre></details>` : ""}`;
}

function bindSongCard(card) {
  const t = $(".sc-tags", card);
  if (t) t.addEventListener("click", () => t.classList.toggle("clamp"));
  bindDeletes(card, () => loadSongs());
}

const songPager = makePager("songs", () => loadSongs());
async function loadSongs() {
  let d;
  try { d = await api("/api/songs"); } catch { return; }
  const keySel = $("#song-key");
  if (!keySel.options.length) {
    keySel.innerHTML = d.keys.map((k) => `<option${k === "E minor" ? " selected" : ""}>${esc(k)}</option>`).join("");
  }
  const grid = $("#song-grid");
  songPager.measure(grid); songPager.watch(grid);
  const per = songPager.per();
  if (!drawPaging(songPager, d.songs.length, Math.min(per || d.songs.length, Math.max(0, d.songs.length - songPager.offset())),
                  "song", $("#song-bar"), $("#song-pager"), grid)) return loadSongs();
  const songsShown = per ? d.songs.slice(songPager.offset(), songPager.offset() + per) : d.songs;
  if (!$("#btn-new-song", grid)) {
    grid.innerHTML = `<button type="button" class="song-card new-tile" id="btn-new-song" title="New song" aria-label="New song">
        <span class="plus-box"><svg viewBox="0 0 24 24" width="30" height="30" aria-hidden="true"><path fill="currentColor" d="M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6z"/></svg></span>
        <span class="plus-label">New song</span></button>`;
    $("#btn-new-song").addEventListener("click", openNewSong);
  }
  const names = new Set(songsShown.map((s) => s.name));
  $$(".song-card[data-song]", grid).forEach((c) => { if (!names.has(c.dataset.song)) c.remove(); });
  let after = $("#btn-new-song", grid);
  for (const s of songsShown) {
    let card = $$(".song-card[data-song]", grid).find((c) => c.dataset.song === s.name);
    if (!card || card.dataset.sig !== songSig(s)) {          // new or changed: draw this card again
      const fresh = document.createElement("div");
      fresh.className = "song-card";
      fresh.dataset.song = s.name;
      fresh.dataset.sig = songSig(s);
      fresh.innerHTML = songCard(s);
      if (card) card.replaceWith(fresh); else after.after(fresh);
      card = fresh;
      bindSongCard(card);
    } else if (busySong(s)) {                                // only the progress changes: the audio players keep playing
      $(".sc-body", card).innerHTML = songBody(s);
    }
    if (card.previousElementSibling !== after) after.after(card);
    after = card;
  }
  $(".song-empty", grid)?.remove();
  if (!d.songs.length) grid.insertAdjacentHTML("beforeend", `<div class="empty song-empty">No songs yet: press + to make one</div>`);
}

function openNewSong() {
  $("#new-song").classList.remove("hidden");
  const f = $("#form-song");
  setTimeout(() => (f.name.value ? f.tags : f.name).focus(), 30);
}
function closeNewSong() {
  const box = $("#new-song");
  if (box.classList.contains("hidden")) return false;
  box.classList.add("hidden");
  return true;
}
$("#btn-close-song").addEventListener("click", closeNewSong);
$("#new-song").addEventListener("click", (ev) => { if (ev.target.id === "new-song") closeNewSong(); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeNewSong(); });

$("#form-song").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = ev.target;
  const data = Object.fromEntries(new FormData(f).entries());
  data.bpm = +data.bpm; data.seconds = +data.seconds; data.seed = data.seed === "" ? null : +data.seed;
  closeNewSong();                                          // close at once; the worker takes a moment to start
  toast("Starting the song…");
  try {
    const r = await postJSON("/api/songs/new", data);
    queued("Song");
    if (r.warning) toast(r.warning, true);
    f.name.value = "";                                     // the next song needs a new name; the style and lyrics stay
    loadSongs();
  } catch (e) { fail(e); openNewSong(); }                  // the form keeps what was typed
});
