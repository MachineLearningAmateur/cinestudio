// cinestudio web app - The Settings tab: setup checklist, models, access tokens, engine, graphics.
// Plain scripts (no modules), loaded in order by index.html; they share their top-level names.
"use strict";

// Comfy Cloud is hidden for now (user, 2026-10-25: "cloud doesnt fully support our built in workflow"; the 4 LTXVideo
// pack nodes and 2 gated IC-LoRAs are unverified there: docs/comfy_cloud_compat.md). The backend and the CLI keep the
// "comfy" token; set this to true to show the key, the checklist step and the Run on choice again.
const SHOW_COMFY_CLOUD = false;
const shownTokens = (d) => Object.entries(d.tokens).filter(([n]) => SHOW_COMFY_CLOUD || n !== "comfy");

// ------------------------------------------------------------------ settings
const restartNeeded = new Set();          // apps changed in this session: they use the new choice after a restart
async function loadEngine() {
  const body = $("#engine-body");
  let e;
  try { e = await api("/api/engine"); } catch (err) { body.textContent = err.message; return; }
  const state = e.running ? (e.started_by_cinestudio ? "Running in the background" : "Running (started outside cinestudio)") : "Stopped";
  body.className = "engine-row";
  body.innerHTML = `<span class="edot ${e.running ? "on" : ""}"></span><div class="grow"><b>ComfyUI</b> · ${state}
      <div class="path" title="${esc(e.log)}">${esc(e.url)}${e.source ? ` · ${esc(e.source)} install` : ""}</div></div>
    ${!e.running ? `<button id="btn-engine-start" class="primary">Start</button>` : ""}
    ${e.running && e.started_by_cinestudio ? `<button id="btn-engine-stop" title="Only when no work waits or runs">Stop</button>` : ""}`;
  const st = $("#btn-engine-start");
  if (st) st.addEventListener("click", async () => {
    st.disabled = true; st.textContent = "Starting…";
    try { await postJSON("/api/engine/start"); toast("ComfyUI runs."); } catch (err) { fail(err); }
    loadEngine(); loadHealth();
  });
  const sp = $("#btn-engine-stop");
  if (sp) sp.addEventListener("click", async () => {
    if (!confirm("Stop ComfyUI? It starts again by itself when a job needs it.")) return;
    try { await postJSON("/api/engine/stop"); toast("ComfyUI stopped."); } catch (err) { fail(err); }
    loadEngine(); loadHealth();
  });
}

// ------------------------------------------------------------------ settings: setup, models, tokens, engine
// Layout from settings-UX guidance: a side navigation of sections, a short setup checklist (status + one action per
// step), rows with the label and description on the left and the control on the right, radio cards for the tier
// (exactly one choice), one primary action per section, quiet destructive links with a confirmation that names the
// consequence, saved tokens identified only by prefix + last 4 characters.
const gb = (b) => `${(b / 1e9).toFixed(b >= 1e10 ? 0 : 1)} GB`;
const OK_ICON = `<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z"/></svg>`;
const EXT_ICON = `<svg viewBox="0 0 24 24" width="12" height="12" aria-hidden="true"><path fill="currentColor" d="M14 3v2h3.6l-9.8 9.8 1.4 1.4L19 6.4V10h2V3h-7zM5 5h6V3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-6h-2v6H5V5z"/></svg>`;
const setupUI = { editing: null, checks: {}, pick: null, sec: null, data: null };
try { setupUI.sec = localStorage.getItem("settings.sec"); } catch { /* private window: start at Setup */ }

function showSettingsSection(sec) {
  if (sec === "graphics" && setupUI.data && !setupUI.data.machine.windows) sec = "setup";
  setupUI.sec = sec;
  try { localStorage.setItem("settings.sec", sec); } catch { /* not saved: fine */ }
  if (location.hash.startsWith("#settings")) history.replaceState(null, "", `#settings/${sec}`);
  $$(".set-nav button[data-sec]").forEach((b) => { b.classList.toggle("on", b.dataset.sec === sec); b.setAttribute("aria-current", b.dataset.sec === sec ? "page" : "false"); });
  $$(".set-sec").forEach((s) => s.classList.toggle("hidden", s.dataset.sec !== sec));
  $(".set-main").scrollTop = 0;
  if (sec === "engine") loadEngine();
  if (sec === "graphics") loadGraphics();
}
$$(".set-nav button[data-sec]").forEach((b) => b.addEventListener("click", () => showSettingsSection(b.dataset.sec)));

function tokenState(t) {
  if (!t.set) return `<span class="muted">Not set</span>`;
  const who = t.source === "environment" ? `from ${esc(t.env)}` : t.updated ? `saved ${new Date(t.updated * 1000).toLocaleDateString()}` : "saved";
  return `<code class="mask">${esc(t.hint || "•••• set")}</code> <span class="muted">· ${who}</span>`;
}

function renderChecklist(d) {
  const tier = d.tiers.find((t) => t.tier === d.settings.model_tier) || d.tiers[0];
  const dl = d.download || {};
  const downloading = ["running", "starting"].includes(dl.state) && dl.tier === tier.tier;
  const steps = [
    { key: "engine", required: true, done: d.engine.found,
      title: "ComfyUI engine", detail: d.engine.running ? `Running${d.engine.source ? ` · ${esc(d.engine.source)} install` : ""}`
        : d.engine.found ? "Found. It starts by itself when a job needs it." : `Not found. ${esc(d.engine.error || "Install ComfyUI (setup/install.sh).")}`,
      action: `<button data-go="engine">${d.engine.found ? "View" : "Fix"}</button>` },
    { key: "models", required: true, done: tier.ready,
      title: `Models · ${esc(tier.label)}`,
      detail: tier.ready ? `Downloaded · ${gb(tier.total_bytes)}`
        : downloading ? `Downloading · ${gb(dl.have_bytes || 0)} of ${gb(dl.total_bytes || tier.total_bytes)}`
        : `${gb(tier.missing_bytes)} to download (${tier.missing.length} files)`,
      action: tier.ready ? `<button data-go="models">Change</button>`
        : downloading ? `<button data-go="models">View</button>`
        : `<button class="primary" data-dl="${tier.tier}">Download</button>` },
    { key: "hf", required: false, recommended: true, done: d.tokens.hf.set, title: "Hugging Face token",
      detail: d.tokens.hf.set ? tokenState(d.tokens.hf) : "Faster model downloads",
      action: `<button data-go="tokens">${d.tokens.hf.set ? "Manage" : "Add"}</button>` },
    ...(SHOW_COMFY_CLOUD ? [{ key: "comfy", required: false, done: d.tokens.comfy.set, title: "Comfy Cloud API key",
      detail: d.tokens.comfy.set ? tokenState(d.tokens.comfy) : "Only to run on Comfy Cloud",
      action: `<button data-go="tokens">${d.tokens.comfy.set ? "Manage" : "Add"}</button>` }] : []),
    ...(d.tokens.civitai ? [{ key: "civitai", required: false, recommended: true, done: d.tokens.civitai.set, title: "Civitai API key",
      detail: d.tokens.civitai.set ? tokenState(d.tokens.civitai) : "Only for Civitai LoRAs that need a login",
      action: `<button data-go="tokens">${d.tokens.civitai.set ? "Manage" : "Add"}</button>` }] : []),
  ];
  const left = steps.filter((s) => s.required && !s.done).length;
  const badge = $("#nav-setup-count");
  badge.textContent = left ? String(left) : "";
  $(".rail .tab[data-tab='settings']").classList.toggle("attn", left > 0);
  const el = $("#setup-checklist");
  el.className = "";
  el.innerHTML = `<div class="check-head ${left ? "" : "done"}">${left ? `<b>${left} required step${left > 1 ? "s" : ""} left</b>`
      : `${OK_ICON}<b>Ready to make videos</b>`}</div>
    <ol class="checklist">${steps.map((s) => `<li class="${s.done ? "done" : ""}">
      <span class="ck">${s.done ? OK_ICON : ""}</span>
      <div class="grow"><div class="ck-title">${s.title} ${s.required ? "" : `<span class="tag-opt">${s.recommended ? "Optional · recommended" : "Optional"}</span>`}</div>
        <div class="ck-detail">${s.detail}</div>
        ${s.key === "models" && downloading ? `<div class="progress small"><div style="width:${Math.round(100 * (dl.have_bytes || 0) / (dl.total_bytes || 1))}%"></div></div>` : ""}</div>
      <div class="ck-act">${s.action}</div></li>`).join("")}</ol>`;
}

function renderModels(d) {
  const dl = d.download || {};
  const running = ["running", "starting"].includes(dl.state);
  const pick = setupUI.pick || d.settings.model_tier;
  const m = d.machine;
  const card = (t) => {
    const mine = running && dl.tier === t.tier;
    const frac = mine && dl.total_bytes ? dl.have_bytes / dl.total_bytes : t.total_bytes ? t.have_bytes / t.total_bytes : 0;
    let eta = "";
    if (mine && dl.started && dl.have_bytes) {
      const start = setupUI.dlStart || (setupUI.dlStart = { t: Date.now(), b: dl.have_bytes });
      const rate = (dl.have_bytes - start.b) / Math.max(1, (Date.now() - start.t) / 1000);
      if (rate > 1e6) eta = ` · about ${fmtTime((dl.total_bytes - dl.have_bytes) / rate)} left`;
    }
    const status = t.error ? `<span class="warn">${esc(t.error)}</span>`
      : t.ready ? `<span class="okc">${OK_ICON} Downloaded</span>`
      : mine ? `Downloading · ${gb(dl.have_bytes || 0)} of ${gb(dl.total_bytes || t.total_bytes)}${eta}`
      : t.have_bytes ? `${gb(t.missing_bytes)} to download · ${gb(t.have_bytes)} already on this computer`
      : `${gb(t.missing_bytes)} to download`;
    const inUse = t.tier === d.settings.model_tier && t.ready;
    return `<label class="tier-opt ${pick === t.tier ? "checked" : ""}">
      <input type="radio" name="tier" value="${t.tier}" ${pick === t.tier ? "checked" : ""}>
      <span class="radio-dot" aria-hidden="true"></span>
      <div class="grow">
        <div class="tier-title"><b>${esc(t.label)}</b>${t.tier === d.recommended ? `<span class="tag-rec">Recommended</span>` : ""}${inUse ? `<span class="tag-use">In use</span>` : ""}</div>
        <div class="tier-spec">Needs ${t.vram_gb} GB VRAM and about ${t.ram_gb} GB RAM · ${gb(t.total_bytes)} of files</div>
        <div class="tier-about">${esc(t.about || "")}</div>
        <div class="tier-status">${status}</div>
        ${(mine || (!t.ready && t.have_bytes)) ? `<div class="progress small"><div style="width:${Math.round(frac * 100)}%"></div></div>` : ""}
      </div></label>`;
  };
  const sel = d.tiers.find((t) => t.tier === pick) || d.tiers[0];
  const selRunning = running && dl.tier === sel.tier;
  let action;
  if (selRunning) action = `<button data-dl-stop>Stop download</button><span class="muted">It continues where it stopped when you start it again.</span>`;
  else if (!sel.ready) action = `<button class="primary" data-dl="${sel.tier}" ${running ? "disabled" : ""} ${sel.disk_ok === false ? "disabled" : ""}>Download ${esc(sel.label)} · ${gb(sel.missing_bytes)}</button>
      <span class="muted">${running ? "Another download runs." : sel.disk_ok === false ? "Not enough free disk space." : "Runs in the background; you can close the app."}</span>`;
  else if (sel.tier !== d.settings.model_tier) action = `<button class="primary" data-use="${sel.tier}">Use ${esc(sel.label)} for new videos</button>`;
  else action = `<span class="okc">${OK_ICON} New videos use this tier.</span>`;
  const failed = dl.state === "failed" ? `<div class="notice warn">The last download stopped: ${esc(dl.error || (dl.failed || []).map((f) => `${f.file}: ${f.error}`).join("; "))}. Start it again to continue.</div>` : "";
  const el = $("#models-body");
  el.className = "";
  el.innerHTML = `
    <div class="set-row"><div class="lbl"><b>This computer</b><span>${esc(m.gpu || "No NVIDIA GPU found")} · ${m.vram_gb} GB VRAM · ${m.ram_gb} GB RAM</span></div></div>
    <div class="tier-group" role="radiogroup" aria-label="Model tier">${d.tiers.map(card).join("")}</div>
    <div class="set-actions">${action}</div>${failed}
    <details class="set-details"><summary>Files of the ${esc(sel.label)} tier (${(sel.files || []).length})</summary>
      <table class="files"><tbody>${(sel.files || []).map((f) => `<tr><td class="mono">${esc(f.file)}</td><td>${gb(f.size)}</td>
        <td class="${f.state === "ok" ? "okc" : "muted"}">${f.state === "ok" ? "here" : esc(f.state)}</td></tr>`).join("")}</tbody></table></details>
    <div class="set-row"><div class="lbl"><b>Models folder</b><span class="mono">${esc(d.models_dir || "not found")}</span></div>
      <div class="ctl muted">${d.disk_free != null ? `${gb(d.disk_free)} free` : ""}</div></div>`;
  $$("input[name=tier]", el).forEach((r) => r.addEventListener("change", async () => {
    setupUI.pick = r.value;
    const t = d.tiers.find((x) => x.tier === r.value);
    if (t && t.ready && r.value !== d.settings.model_tier) {
      try { await postJSON("/api/setup/tier", { tier: r.value }); toast(`New videos use the ${t.label} tier.`); setupUI.pick = null; } catch (e) { fail(e); }
    }
    loadSetup.last = null; loadSetup();
  }));
}

function renderTokens(d) {
  const row = (name, t) => {
    const editing = setupUI.editing === name;
    const chk = setupUI.checks[name];
    const btns = editing ? "" : t.source === "environment" ? `<button data-tok-check="${name}">Test</button>`
      : t.set ? `<button data-tok-edit="${name}">Replace</button><button data-tok-check="${name}">Test</button>`
      : `<button data-tok-edit="${name}">Add</button>`;
    return `<div class="set-row tok">
      <div class="lbl"><b>${esc(t.label)} <span class="tag-opt">${name === "comfy" ? "Optional" : "Optional · recommended"}</span></b>
        <span>${esc(t.help)} <a href="${esc(t.url)}" target="_blank" rel="noopener">Get one ${EXT_ICON}</a></span>
        ${t.source === "environment" ? `<span>Set in the environment variable ${esc(t.env)}: change it there.</span>` : ""}</div>
      <div class="ctl"><div class="tok-state">${tokenState(t)}</div><div class="btns">${btns}</div>
        ${t.saved && !editing ? `<button class="linkbtn danger-text" data-tok-clear="${name}">Remove</button>` : ""}</div>
      ${editing ? `<form class="tok-form" data-tok-form="${name}" autocomplete="off">
          <div class="pw"><input type="password" name="value" placeholder="Paste the ${esc(t.label.toLowerCase())}" autocomplete="new-password" spellcheck="false" required>
            <button type="button" class="ghost eye" data-eye title="Show or hide">Show</button></div>
          <button class="primary">Save</button><button type="button" data-tok-cancel>Cancel</button></form>` : ""}
      ${chk && !editing ? `<div class="tok-check ${chk.ok ? "okc" : "warn"}">${chk.ok ? OK_ICON : "✕"} ${esc(chk.detail)}</div>` : ""}
    </div>`;
  };
  const el = $("#tokens-body");
  el.className = "";
  el.innerHTML = shownTokens(d).map(([n, t]) => row(n, t)).join("") +
    `<p class="set-note">Saved in <code>.cinestudio/secrets.json</code> on this computer (never in the project files).
      The environment variables HF_TOKEN${SHOW_COMFY_CLOUD ? ", COMFY_API_KEY" : ""} and CIVITAI_TOKEN win over saved values.</p>`;
  $$("[data-tok-edit]", el).forEach((b) => b.addEventListener("click", () => { setupUI.editing = b.dataset.tokEdit; delete setupUI.checks[b.dataset.tokEdit]; renderTokens(d); }));
  $$("[data-tok-cancel]", el).forEach((b) => b.addEventListener("click", () => { setupUI.editing = null; renderTokens(d); }));
  $$("[data-eye]", el).forEach((b) => b.addEventListener("click", () => {
    const i = b.parentElement.querySelector("input"); const show = i.type === "password";
    i.type = show ? "text" : "password"; b.textContent = show ? "Hide" : "Show";
  }));
  $$("[data-tok-form]", el).forEach((f) => {
    f.querySelector("input").focus();
    f.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const input = f.querySelector("input"); const value = input.value; input.value = "";
      const name = f.dataset.tokForm;
      try {
        const r = await postJSON("/api/setup/token", { name, value });
        setupUI.checks[name] = r.check; setupUI.editing = null;
        toast(r.check.ok ? "Saved." : "Saved, but the test failed.", !r.check.ok);
      } catch (e) { fail(e); }
      loadSetup.last = null; loadSetup();
    });
  });
  $$("[data-tok-check]", el).forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true; b.textContent = "Testing…";
    try { const r = await postJSON("/api/setup/token/check", { name: b.dataset.tokCheck }); setupUI.checks[b.dataset.tokCheck] = r.check; }
    catch (e) { fail(e); }
    loadSetup.last = null; loadSetup();
  }));
  $$("[data-tok-clear]", el).forEach((b) => b.addEventListener("click", async () => {
    const t = d.tokens[b.dataset.tokClear];
    const what = b.dataset.tokClear === "comfy" ? "Comfy Cloud runs will stop working until you add a key again."
      : b.dataset.tokClear === "civitai" ? "Civitai downloads that need a login will stop working until you add a key again."
      : "Downloads of gated models will need a token again.";
    if (!confirm(`Remove the ${t.label} (${t.hint || "saved"}) from this computer?\n${what}`)) return;
    try { await postJSON("/api/setup/token/clear", { name: b.dataset.tokClear }); delete setupUI.checks[b.dataset.tokClear]; toast("Removed."); } catch (e) { fail(e); }
    loadSetup.last = null; loadSetup();
  }));
}

function renderRunOn(d) {
  if (!SHOW_COMFY_CLOUD) { $("#runon-body").innerHTML = ""; return; }   // one choice only: no Run on control
  const key = d.tokens.comfy.set;
  $("#runon-body").innerHTML = `<div class="tier-group" role="radiogroup" aria-label="Where videos are made">
    <label class="tier-opt checked"><input type="radio" name="runon" value="local" checked><span class="radio-dot" aria-hidden="true"></span>
      <div class="grow"><div class="tier-title"><b>This computer</b></div><div class="tier-spec">Uses your graphics card with ComfyUI. No key needed.</div></div></label>
    <label class="tier-opt disabled"><input type="radio" name="runon" value="cloud" disabled><span class="radio-dot" aria-hidden="true"></span>
      <div class="grow"><div class="tier-title"><b>Comfy Cloud</b><span class="tag-opt">Coming soon</span></div>
        <div class="tier-spec">Runs on Comfy's cloud GPUs. Needs a Comfy Cloud API key (a paid Comfy plan)${key ? " · your key is saved" : ""}.</div></div></label></div>`;
}

const setupBusy = () => setupUI.editing || document.activeElement?.closest?.(".tok-form");
async function loadSetup(quiet = false) {
  if (quiet && setupBusy()) return;                     // never replace a form the user is typing in
  let d;
  try { d = await api("/api/setup"); } catch (e) { if (!quiet) $("#setup-checklist").textContent = e.message; return; }
  const sig = JSON.stringify(d);
  if (quiet && sig === loadSetup.last) return;
  loadSetup.last = sig;
  setupUI.data = d;
  $("#nav-graphics").classList.toggle("hidden", !d.machine.windows);
  renderChecklist(d);
  renderModels(d);
  if (!setupUI.editing) renderTokens(d);
  renderRunOn(d);
  if (!setupUI.sec || !$(`.set-nav button[data-sec="${setupUI.sec}"]`)) setupUI.sec = "setup";
  if (!loadSetup.shown) { loadSetup.shown = true; showSettingsSection(setupUI.sec); }
  // shared actions (the checklist and the Models section use the same buttons)
  $$("#tab-settings [data-go]").forEach((b) => b.onclick = () => showSettingsSection(b.dataset.go));
  $$("#tab-settings [data-dl]").forEach((b) => b.onclick = async () => {
    b.disabled = true;
    try { await postJSON("/api/setup/download", { tier: b.dataset.dl }); toast("The download runs in the background."); setupUI.dlStart = null; }
    catch (e) { fail(e); }
    loadSetup.last = null; loadSetup();
  });
  $$("#tab-settings [data-dl-stop]").forEach((b) => b.onclick = async () => {
    try { await postJSON("/api/setup/download", { stop: true }); toast("The download stops."); } catch (e) { fail(e); }
    setTimeout(() => { loadSetup.last = null; loadSetup(); }, 1500);
  });
  $$("#tab-settings [data-use]").forEach((b) => b.onclick = async () => {
    try { await postJSON("/api/setup/tier", { tier: b.dataset.use }); toast("New videos use this tier."); setupUI.pick = null; } catch (e) { fail(e); }
    loadSetup.last = null; loadSetup();
  });
  const running = ["running", "starting"].includes((d.download || {}).state);
  clearTimeout(loadSetup.timer);
  if (running) loadSetup.timer = setTimeout(() => loadSetup(true), 2000);
}

async function loadSettings(quiet = false) {
  loadSetup(quiet);
  if (setupUI.sec === "engine" && !quiet) loadEngine();
  if (setupUI.sec === "graphics") loadGraphics(quiet);
}

async function loadGraphics(quiet = false) {
  const body = $("#gpu-body");
  let d;
  try { d = await api("/api/settings/gpu"); } catch (e) { if (!quiet) body.textContent = e.message; return; }
  const sig = JSON.stringify(d);
  if (quiet && sig === loadGraphics.last) return;                 // nothing changed: keep the page as it is
  loadGraphics.last = sig;
  const a = d.adapters || {};
  const opt = (r, k, label, title) =>
    `<button class="${r.choice === k ? `on ${k}` : ""}" data-exe="${esc(r.exe)}" data-choice="${k}" title="${esc(title)}">${label}</button>`;
  body.className = "";
  body.innerHTML = `<div class="set-row"><div class="lbl"><b>Graphics on this computer</b>
      <span>Integrated: ${esc(a.integrated || "none found")} · GPU: ${esc(a.gpu || "?")}</span></div></div>
    ${d.apps.map((r) => `<div class="gpu-row"><div class="app"><b>${esc(r.name)}</b>
        ${restartNeeded.has(r.exe) && r.running ? `<span class="tag restart">restart the app</span>` : r.running ? "" : `<span class="tag">not open</span>`}
        <div class="path" title="${esc(r.exe)}">${esc(r.exe)}</div></div>
      <div class="seg">${opt(r, "default", "Default", "Windows decides")}${d.can_switch ? opt(r, "integrated", "Integrated", a.integrated || "") : ""}${opt(r, "gpu", "GPU", a.gpu || "")}</div></div>`).join("")}
    <div class="gpu-foot">
      ${d.managed ? `<button id="btn-gpu-reset" title="Put back every app that cinestudio changed">Reset</button>` : ""}
      ${d.can_switch ? `<button id="btn-gpu-all" title="Browsers, chat and media apps that are open now">All to integrated</button>` : ""}</div>`;
  $$(".gpu-row .seg button", body).forEach((b) => b.addEventListener("click", async () => {
    if (b.classList.contains("on")) return;
    try {
      await postJSON("/api/settings/gpu", { exe: b.dataset.exe, choice: b.dataset.choice });
      restartNeeded.add(b.dataset.exe); loadGraphics();
    } catch (e) { fail(e); }
  }));
  const all = $("#btn-gpu-all");
  if (all) all.addEventListener("click", async () => {
    if (!confirm("Move the open browsers, chat and media apps to the integrated graphics?\nRestart them to use it. Reset puts everything back.")) return;
    try { const r = await postJSON("/api/settings/gpu/all-integrated"); (r.changed || []).forEach((x) => restartNeeded.add(x)); loadGraphics(); }
    catch (e) { fail(e); }
  });
  const reset = $("#btn-gpu-reset");
  if (reset) reset.addEventListener("click", async () => {
    if (!confirm("Put every app that cinestudio changed back to its earlier setting?")) return;
    try { const r = await postJSON("/api/settings/gpu/reset"); (r.restored || []).forEach((x) => restartNeeded.add(x)); loadGraphics(); toast("Restored."); }
    catch (e) { fail(e); }
  });
}
