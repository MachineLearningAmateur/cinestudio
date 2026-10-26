// cinestudio web app - Start: the first loads and the tab from the address (loaded LAST: it calls every tab's loaders).
// Plain scripts (no modules), loaded in order by index.html; they share their top-level names.
"use strict";

// ------------------------------------------------------------------ start
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});   // lets Edge install the app
loadHealth();
loadMeters();
setInterval(loadMeters, 2000);
loadQueue();
setInterval(() => { if (!document.hidden) loadQueue(); }, 3000);
loadLoras();
loadImageOptions();
loadChars().then(() => { if (state.chars.length) openChar(state.chars[0].name); });
const runLink = location.hash.startsWith("#run=") ? decodeURIComponent(location.hash.slice(5)) : null;
if (location.hash.startsWith("#settings/")) setupUI.sec = location.hash.split("/")[1];       // a deep link to a section
showTab(runLink ? "video" : ["#video", "#music", "#templates", "#backgrounds", "#styles", "#settings", "#workflows"].includes(location.hash.split("/")[0]) ? location.hash.split("/")[0].slice(1) : "characters");
if (runLink) openRun(runLink);
