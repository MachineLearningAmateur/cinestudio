// cinestudio web app - Templates: select the figure to replace (SAM2 tracking view).
// Plain scripts (no modules), loaded in order by index.html; they share their top-level names.
"use strict";

// ------------------------------------------------------------------ templates: select the figure to replace
const fig = { name: null, info: null, clicks: [], seeds: [], mask: null, frame: 0, timer: null, view: "pick" };
const figKind = () => (document.querySelector('input[name="figkind"]:checked') || {}).value || "pos";

async function openFigure(name) {
  let d;
  try { d = await api(`/api/templates/${encodeURIComponent(name)}/mask`); } catch (e) { fail(e); return; }
  Object.assign(fig, { name, info: d.video, clicks: (d.clicks || []).slice(), mask: d.mask, frame: 0 });
  fig.seeds = ((d.mask && d.mask.points) || []).filter((p) => p.seed);
  $("#fig-name").textContent = name;
  const v = $("#fig-video");
  v.src = d.video.url;
  v.onloadeddata = () => figSeek(fig.frame);
  const ov = $("#fig-overlay");
  if (d.overlay_url) {
    ov.src = `${d.overlay_url}${d.overlay_url.includes("?") ? "&" : "?"}t=${(d.mask && d.mask.made) || 0}`;
    ov.onloadeddata = () => figSeek(fig.frame);
  } else ov.removeAttribute("src");
  const sl = $("#fig-slider");
  sl.max = Math.max(0, (d.video.frames || 1) - 1);
  sl.value = 0;
  $("#fig-marks").setAttribute("viewBox", `0 0 ${d.video.width} ${d.video.height}`);
  const has = !!d.overlay_url;
  figSet(has ? "result" : "pick");
  $("#figure-sel").classList.remove("hidden");
  figView(has ? "result" : "pick");
  figStatus(d.status || {});
  figDraw();
  if ((d.status || {}).status === "queued" || (d.status || {}).status === "running") figPoll();
}

function closeFigure() {
  const box = $("#figure-sel");
  if (box.classList.contains("hidden")) return false;
  box.classList.add("hidden");
  clearTimeout(fig.timer);
  $("#fig-video").pause(); $("#fig-overlay").pause();
  fig.name = null;
  return true;
}

function figSet(view) { const r = document.querySelector(`input[name="figview"][value="${view}"]`); if (r) r.checked = true; }

function figView(view) {
  fig.view = view;
  const pick = view === "pick";
  $("#fig-frame").classList.toggle("hidden", !pick);
  $("#fig-overlay").classList.toggle("hidden", pick);
  $(".fig-kind").style.visibility = pick ? "" : "hidden";
  figSeek(fig.frame);
}

function figSeek(frame) {
  if (!fig.info) return;
  const n = Math.max(1, fig.info.frames || 1);
  fig.frame = Math.max(0, Math.min(n - 1, frame | 0));
  $("#fig-slider").value = fig.frame;
  $("#fig-fno").textContent = `frame ${fig.frame} / ${n - 1}`;
  const t = (fig.frame + 0.5) / (fig.info.fps || 24);
  for (const v of [$("#fig-video"), $("#fig-overlay")]) {
    if (v.getAttribute("src") && v.readyState >= 1 && v.paused) { try { v.currentTime = t; } catch { /* not seekable yet */ } }
  }
  figDraw();
}

function figDraw() {
  if (!fig.info) return;
  const r = Math.max(4, fig.info.width / 110);
  const mk = (p) => p.positive === false
    ? `<g><circle cx="${p.x}" cy="${p.y}" r="${r}" fill="#ff4d4d" stroke="#fff" stroke-width="${r / 3}"/><line x1="${p.x - r / 2}" y1="${p.y}" x2="${p.x + r / 2}" y2="${p.y}" stroke="#fff" stroke-width="${r / 3}"/></g>`
    : `<g><circle cx="${p.x}" cy="${p.y}" r="${r}" fill="#36d27a" stroke="#fff" stroke-width="${r / 3}"/></g>`;
  const here = (p) => p.frame === fig.frame;
  $("#fig-marks").innerHTML = fig.seeds.filter(here).map((p) =>
    `<circle cx="${p.x}" cy="${p.y}" r="${r * 0.8}" fill="none" stroke="#d8dde6" stroke-width="${r / 3}" stroke-dasharray="${r / 2}"/>`).join("")
    + fig.clicks.filter(here).map(mk).join("");
  const by = {};
  fig.clicks.forEach((p, i) => { (by[p.frame] = by[p.frame] || []).push([p, i]); });
  const frames = Object.keys(by).map(Number).sort((a, b) => a - b);
  $("#fig-points").innerHTML = frames.length ? frames.map((f) => by[f].map(([p, i]) =>
    `<span class="fp"><span class="go" data-f="${f}" title="Go to frame ${f}">frame ${f}</span>
      <span class="${p.positive === false ? "neg" : "pos"}">${p.positive === false ? "− exclude" : "+ include"}</span>
      <button type="button" data-rm="${i}" title="Remove this click" aria-label="Remove">×</button></span>`).join("")).join("")
    : `<span class="sub">No clicks yet: the pose detector's points (grey rings) are used${fig.seeds.length ? "" : " when you track"}.</span>`;
  $$("#fig-points .go").forEach((b) => b.addEventListener("click", () => { figSet("pick"); figView("pick"); figSeek(+b.dataset.f); }));
  $$("#fig-points button[data-rm]").forEach((b) => b.addEventListener("click", () => { fig.clicks.splice(+b.dataset.rm, 1); figDraw(); }));
  figTimeline();
}

function figTimeline() {
  const c = $("#fig-timeline"), n = Math.max(1, (fig.info && fig.info.frames) || 1);
  const w = c.clientWidth || 600, h = 34, ratio = window.devicePixelRatio || 1;
  c.width = w * ratio; c.height = h * ratio;
  const g = c.getContext("2d");
  g.scale(ratio, ratio);
  g.clearRect(0, 0, w, h);
  const area = (fig.mask && fig.mask.area) || [];
  const top = Math.max(1e-6, ...area);
  g.fillStyle = "rgba(120, 170, 255, .55)";
  area.forEach((a, i) => { const bh = (a / top) * (h - 8); g.fillRect((i / n) * w, h - bh, Math.max(1, w / n - 0.5), bh); });
  g.fillStyle = "#ff5a5a";
  ((fig.mask && fig.mask.flags) || []).forEach((f) => g.fillRect((f.frame / n) * w - 1, 0, 3, h));
  g.fillStyle = "#36d27a";
  fig.clicks.forEach((p) => g.fillRect((p.frame / n) * w - 1, 0, 2, 6));
  g.fillStyle = "#fff";
  g.fillRect(((fig.frame + 0.5) / n) * w - 1, 0, 2, h);
}

function figStatus(st) {
  const box = $("#fig-status"), btn = $("#fig-track");
  const busy = st.status === "queued" || st.status === "running";
  btn.disabled = busy;
  if (busy) box.innerHTML = `<span class="spinner"></span> ${esc(st.text || (st.status === "queued" ? "Waiting for its turn on the GPU…" : "Tracking…"))}`;
  else if (st.status === "failed") box.innerHTML = `<span class="err">${esc(st.error || "The tracking failed")}</span>`;
  else if (st.status === "interrupted") box.textContent = "The worker stopped before the end: press Track again.";
  else if (fig.mask) {
    const fl = (fig.mask.flags || []).length;
    box.textContent = fl ? `Tracked. ${fl} frame${fl > 1 ? "s" : ""} to check (red ticks).` : "Tracked: the selection follows the figure in every frame.";
  } else box.textContent = "";
}

async function figPoll() {
  clearTimeout(fig.timer);
  if (!fig.name) return;
  const name = fig.name;
  let d;
  try { d = await api(`/api/templates/${encodeURIComponent(name)}/mask`); } catch { fig.timer = setTimeout(figPoll, 3000); return; }
  if (fig.name !== name) return;
  const st = d.status || {};
  if (st.status === "queued" || st.status === "running") { figStatus(st); fig.timer = setTimeout(figPoll, 2000); return; }
  fig.mask = d.mask;
  fig.seeds = ((d.mask && d.mask.points) || []).filter((p) => p.seed);
  if (d.overlay_url) {
    const ov = $("#fig-overlay");
    ov.src = `${d.overlay_url}${d.overlay_url.includes("?") ? "&" : "?"}t=${(d.mask && d.mask.made) || Date.now()}`;
    ov.onloadeddata = () => figSeek(fig.frame);
    figSet("result"); figView("result");
  }
  figStatus(st);
  figDraw();
  loadTemplates();
}

$("#fig-frame").addEventListener("click", (ev) => {
  if (!fig.info) return;
  const rect = $("#fig-video").getBoundingClientRect();
  const x = ((ev.clientX - rect.left) / rect.width) * fig.info.width;
  const y = ((ev.clientY - rect.top) / rect.height) * fig.info.height;
  if (x < 0 || y < 0 || x > fig.info.width || y > fig.info.height) return;
  fig.clicks.push({ frame: fig.frame, x: Math.round(x * 10) / 10, y: Math.round(y * 10) / 10,
    positive: !(ev.shiftKey || figKind() === "neg") });
  figDraw();
});
$("#fig-slider").addEventListener("input", (ev) => { $("#fig-video").pause(); $("#fig-overlay").pause(); figSeek(+ev.target.value); });
$("#fig-timeline").addEventListener("click", (ev) => {
  const rect = ev.currentTarget.getBoundingClientRect(), n = (fig.info && fig.info.frames) || 1;
  $("#fig-video").pause(); $("#fig-overlay").pause();
  figSeek(Math.floor(((ev.clientX - rect.left) / rect.width) * n));
});
$$('input[name="figview"]').forEach((r) => r.addEventListener("change", () => figView(r.value)));
$("#fig-play").addEventListener("click", () => {
  const v = fig.view === "pick" ? $("#fig-video") : $("#fig-overlay");
  if (!v.getAttribute("src")) return;
  if (v.paused) { v.loop = true; v.play().catch(() => {}); } else v.pause();
});
for (const id of ["#fig-video", "#fig-overlay"]) {
  $(id).addEventListener("timeupdate", (ev) => {
    const v = ev.target;
    if (v.paused || !fig.info) return;
    fig.frame = Math.min(Math.floor(v.currentTime * (fig.info.fps || 24)), (fig.info.frames || 1) - 1);
    $("#fig-slider").value = fig.frame;
    $("#fig-fno").textContent = `frame ${fig.frame} / ${(fig.info.frames || 1) - 1}`;
    figDraw();
  });
}
$("#fig-clear").addEventListener("click", () => { fig.clicks = []; figDraw(); });
$("#fig-track").addEventListener("click", async () => {
  if (!fig.name) return;
  try {
    await postJSON(`/api/templates/${encodeURIComponent(fig.name)}/mask`, { clicks: fig.clicks });
    toast("Tracking the figure: SAM 2 follows it through the video.");
    figStatus({ status: "queued" });
    fig.timer = setTimeout(figPoll, 1500);
    loadTemplates();
  } catch (e) { fail(e); }
});
$("#btn-close-figure").addEventListener("click", closeFigure);
$("#figure-sel").addEventListener("click", (ev) => { if (ev.target.id === "figure-sel") closeFigure(); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeFigure(); });
window.addEventListener("resize", () => { if (fig.name) figTimeline(); });
