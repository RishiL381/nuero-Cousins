"use strict";

// ---------- Settings ----------
const qs = new URLSearchParams(location.search);
const QUICK = qs.has("quick"); // ?quick -> short durations for testing
const DUR = QUICK
  ? { warmup: 10, round: 25, reset: 20 }
  : { warmup: 30, round: 90, reset: 90 };
// Calibration protocols, in seconds (see docs/methods.md)
const PROTOCOLS = QUICK
  ? { standard: { rest: 15, stress: 25, recovery: 10 }, short: { rest: 10, stress: 15, recovery: 0 } }
  : { standard: { rest: 180, stress: 120, recovery: 120 }, short: { rest: 60, stress: 75, recovery: 0 } };
const PER_PROBLEM_S = 6;  // time limit per arithmetic problem under pressure
const BREATH_HALF_S = 5;  // 5 s in + 5 s out = 6 breaths per minute
const SUB_START = 1022, SUB_STEP = 13; // serial subtraction stressor (Trier Social Stress Test style)
const PHASE_COLORS = { study_relax: "#5fa8c8", study_math_easy: "#c7b26a", study_math_hard: "#d9733c", study_stroop_hard: "#c2456b", calib_rest: "#8a96a3", calib_stress: "#e08a3c", calib_recovery: "#7fb3a0", warmup: "#b9a77e", round1: "#d14b3b", reset: "#1f9d6b", round2: "#2563a6" };
const PHASE_NAMES = {
  idle: "", calib_rest: "Calibration: baseline", calib_survey: "Calibration: questionnaire", calib_stress: "Calibration: stress test",
  calib_recovery: "Calibration: recovery", calibration_check: "Calibration",
  study_relax: "Study: relaxation", study_math_easy: "Study: easy math", study_math_hard: "Study: hard math",
  study_stroop_hard: "Study: hard colour words", study_rating: "Study: rating", study_done: "Study: done",
  ready: "Get ready", warmup: "Warm-up", round1: "Round 1", reset: "Reset", round2: "Round 2", results: "Results",
};
// Score scale: 0 = your calm (rest calibration), 100 = your stress (stress-test calibration)
const STRESS_BINS = [
  { key: "calm", label: "Calm", range: "< 35", lo: -Infinity, hi: 35 },
  { key: "mid", label: "Moderate", range: "35–65", lo: 35, hi: 65 },
  { key: "high", label: "High stress", range: "65+", lo: 65, hi: Infinity },
];
const SIGNAL_NAMES = { eeg: "Brain", hr: "Heart", hrv: "Heart rate variability" };
const FACE_LABELS = {
  face_brow: "Brows lowered", face_press: "Lips pressed", face_squint: "Eyes squinted", face_frown: "Frown",
  face_smile: "Smile", blink_rate: "Blinks / min", head_motion: "Head motion",
};
const FACE_KEYS = Object.keys(FACE_LABELS);
const isFaceKey = (k) => FACE_KEYS.includes(k);
const EEG_SENSORS = ["TP9", "AF7", "AF8", "TP10"], EEG_BANDS = ["theta", "alpha", "beta"];
const EEG_KEYS = EEG_SENSORS.flatMap((ch) => EEG_BANDS.map((b) => `${b}_${ch}`));
const SIGNAL_GROUPS = { Brain: EEG_KEYS, Heart: ["hr", "hrv"], Face: FACE_KEYS };
const groupOf = (k) => Object.keys(SIGNAL_GROUPS).find((g) => SIGNAL_GROUPS[g].includes(k));
// Brain card of the calibration profile: one sensor per row, theta / alpha / beta in the person's IAF bands,
// with the direction each one moved under stress (learned per person), + a top view of the head
function brainCard(sig, cal) {
  const used = (k) => !!(cal && cal.used.includes(k));
  const cell = (k) => {
    const s = sig[k];
    if (!s || s.effect == null) return `<td class="muted">—</td>`;
    const arrow = s.direction > 0 ? "↑" : "↓";
    return `<td class="${used(k) ? "better" : "muted"}" title="${(s.note || "").replace(/"/g, "&quot;")} · Cohen's d ${s.effect.toFixed(2)}">${arrow} ${s.effect.toFixed(1)}${used(k) ? " ✓" : ""}</td>`;
  };
  const shown = EEG_SENSORS.filter((ch) => EEG_BANDS.some((b) => sig[`${b}_${ch}`]));
  const rows = shown.map((ch) => `<tr><td>${ch}</td>${EEG_BANDS.map((b) => cell(`${b}_${ch}`)).join("")}</tr>`).join("");
  // share of windows thrown out because of forehead or jaw muscle tension
  const muscle = shown.map((ch) => [ch, Math.max(0, ...EEG_BANDS.map((b) => ((sig[`${b}_${ch}`] || {}).dropped || {}).muscle || 0))])
    .filter(([, pct]) => pct > 0).map(([ch, pct]) => `${ch} ${pct}%`).join(", ");
  const color = (ch) => {
    const keys = EEG_BANDS.map((b) => `${b}_${ch}`).filter((k) => sig[k] && sig[k].effect != null);
    if (!keys.length) return "var(--border)";
    return keys.some(used) ? "var(--good)" : "var(--fair)";
  };
  const dots = EEG_SENSORS.map((ch) => {
    const [x, y] = SENSOR_POS[ch];
    return `<circle cx="${x}" cy="${y}" r="9" fill="${color(ch)}"/><text x="${x}" y="${y < 60 ? y - 13 : y + 21}"
      text-anchor="middle">${ch}</text>`;
  }).join("");
  const head = `<svg viewBox="0 0 120 135" class="mini-head"><path class="skull" d="M60 12 l-7 9 h14 z"/>
    <ellipse class="skull" cx="11" cy="74" rx="6" ry="12"/><ellipse class="skull" cx="109" cy="74" rx="6" ry="12"/>
    <ellipse class="skull" cx="60" cy="72" rx="46" ry="54"/>${dots}</svg>`;
  const iaf = cal && cal.iaf ? ` · your alpha frequency ${cal.iaf.toFixed(1)} Hz` : "";
  return `<div class="sig-card"><h3>Brain · per sensor and band</h3>
    <div class="muted small">how each band moved under stress (↑ / ↓), Cohen's d${iaf}</div>
    <div class="brain-row">${head}<table class="mini"><tr class="muted"><td></td><td>θ</td><td>α</td><td>β</td></tr>
      ${rows || "<tr><td>not enough clean signal</td></tr>"}</table></div>
    ${muscle ? `<p class="muted small">Windows removed for muscle tension: ${muscle}</p>` : ""}
    <p class="muted small">Green: the sensor feeds the score (✓ bands with d ≥ 0.5) · orange: no band reacted enough</p></div>`;
}

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const state = {
  ws: null, hello: null, latest: null, baseline: null, phase: "idle",
  session: null, history: [], live: [], pillVisible: true, baselineWaiter: null,
  scenes: { preview: null, pip: null }, pip: null, busy: false,
};

// ---------- Server connection ----------
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  state.ws = ws;
  ws.onopen = () => {
    $("conn").classList.add("on");
    // Only the tab running a protocol restores its phase after a reconnect: an idle tab (another window,
    // a reload) must not reset the phase of a calibration running elsewhere.
    if (state.busy) send({ type: "phase", name: state.phase });
  };
  ws.onclose = () => { $("conn").classList.remove("on"); setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (handlers[m.type]) handlers[m.type](m);
  };
}
function send(obj) {
  if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify(obj));
}
function setPhase(name) {
  state.phase = name;
  $("phase-label").textContent = PHASE_NAMES[name] ?? name;
  send({ type: "phase", name });
}

const handlers = {
  hello(m) {
    state.hello = m;
    if (m.baseline) state.baseline = m.baseline;
    $("source-label").textContent = m.source === "mock" ? "Simulated headset" : "Muse (live)";
    if (m.source === "mock" && qs.has("debug")) $("debug").hidden = false;
    $("hs-box").hidden = !m.headset_control;
    if (m.default_name && !$("hs-name").value) $("hs-name").value = m.default_name;
    buildHead(m.eeg_names);
  },
  baseline(m) {
    state.baseline = m.baseline;
    const pm = m.baseline && m.baseline.personal;
    if (state.phase === "study_done" && pm && pm.ok) {  // personal model retrained at the end of the study
      $("study-lead").textContent += ` Your personal model was retrained on ${Math.round(pm.seconds / 60)} min of data.`;
    }
    if (state.baselineWaiter) { state.baselineWaiter(m.baseline); state.baselineWaiter = null; }
  },
  session(m) {
    if (state.session) state.session.dir = m.dir;
  },
  metrics(m) {
    state.latest = m;
    const hr = m.hr && m.hr.bpm != null && m.hr.quality >= 0.5 ? m.hr.bpm : null;
    if (state.session && state.session.recording) {
      state.history.push({ t: m.t, phase: state.phase, stress: m.stress, hr });
    }
    state.live.push({ t: m.t, stress: m.stress, body: m.stress_body, ratio: m.eeg ? m.eeg.ratio_log : null });
    while (state.live.length && state.live[0].t < m.t - 120) state.live.shift();
    updateLive(m, hr);
  },
};

// ---------- Stress colours and labels ----------
function stressHue(v) { return clamp(145 - (v - 15) * 2.2, 0, 145); } // green -> amber -> red
function stressColor(v, l = 46) { return v == null ? "var(--muted)" : `hsl(${stressHue(v)}, 62%, ${l}%)`; }
function stressLabel(v) {
  if (v == null) return state.latest && state.latest.calibrated ? "no clean signal" : "not calibrated";
  return v < 30 ? "calm" : v < 55 ? "neutral" : v < 75 ? "tense" : "very tense";
}

// ---------- Gauge ----------
const ARC = "M 30.7 140 A 80 80 0 1 1 169.3 140"; // 240° arc, centre (100,100), r 80
function buildGauges() {
  document.querySelectorAll(".gauge").forEach((el) => {
    el.innerHTML = `<svg viewBox="0 0 200 165" role="img" aria-label="Stress gauge">
      <path class="track" d="${ARC}" pathLength="100"/>
      <path class="value" d="${ARC}" pathLength="100" stroke-dasharray="0 100"/>
      <text class="num" x="100" y="112">—</text>
      <text class="lbl" x="100" y="138">not calibrated</text></svg>`;
  });
}
function updateGauges(v) {
  document.querySelectorAll(".gauge").forEach((el) => {
    const val = el.querySelector(".value");
    val.setAttribute("stroke-dasharray", v == null ? "0 100" : `${Math.max(1, v)} 100`);
    val.style.stroke = stressColor(v);
    el.querySelector(".num").textContent = v == null ? "—" : v;
    el.querySelector(".lbl").textContent = stressLabel(v);
  });
}

// ---------- Head diagram (top view) ----------
const SENSOR_POS = { AF7: [42, 32], AF8: [78, 32], TP9: [17, 88], TP10: [103, 88] };
function buildHead(names) {
  const n = names && names.length ? names : Object.keys(SENSOR_POS);
  const dots = n.filter((s) => SENSOR_POS[s]).map((s) => {
    const [x, y] = SENSOR_POS[s];
    const ty = y < 60 ? y - 11 : y + 17;
    return `<circle class="sensor" id="sensor-${s}" cx="${x}" cy="${y}" r="7"/><text x="${x}" y="${ty}" text-anchor="middle">${s}</text>`;
  }).join("");
  $("head-diagram").innerHTML = `<svg viewBox="0 0 120 135" role="img" aria-label="Sensor contact, top view of the head">
    <path class="skull" d="M60 12 l-7 9 h14 z"/>
    <ellipse class="skull" cx="11" cy="74" rx="6" ry="12"/><ellipse class="skull" cx="109" cy="74" rx="6" ry="12"/>
    <ellipse class="skull" cx="60" cy="72" rx="46" ry="54"/>${dots}</svg>`;
}

// ---------- Headset connection ----------
function updateHeadset(src) {
  const hs = src && src.headset;
  if (!hs) return;
  let cls, text;
  if (src.connected) {
    cls = "ok";
    text = (hs.external ? "Streaming (muselsl started outside the app)" : `Streaming from ${hs.device || hs.name || "Muse"}`) + (hs.address ? ` · ${hs.address}` : "");
  } else if (["searching", "connecting", "retrying"].includes(hs.state) || (hs.state === "streaming" && hs.managed)) {
    cls = "busy";
    text = hs.state === "streaming" ? "Connected, waiting for data…" : hs.message;
  } else if (["not_found", "failed", "disconnected"].includes(hs.state)) {
    cls = "err"; text = hs.message;
  } else {
    cls = ""; text = "Not connected";
  }
  $("hs-dot").className = `hs-dot ${cls}`;
  $("hs-text").textContent = text;
  $("hs-log").textContent = (hs.log || []).join("\n");
  const active = hs.managed || hs.want;
  $("hs-connect").disabled = (active && cls !== "err") || hs.external;
  $("hs-connect").textContent = cls === "err" || hs.state === "idle" && hs.attempts ? "Reconnect" : "Connect";
  $("hs-disconnect").disabled = !active;
}

// ---------- Watch (Garmin or any Bluetooth heart rate device) ----------
function updateWatch(w, hr) {
  if (!w) return;
  const cls = { streaming: w.bpm != null ? "ok" : "busy", searching: "busy", connecting: "busy", error: "err" }[w.state] || "";
  $("watch-dot").className = `hs-dot ${cls}`;
  $("watch-text").textContent = w.state === "streaming" && w.bpm == null ? `${w.message} · waiting for heart rate…` : w.message;
  $("watch-bpm").textContent = w.bpm != null ? `${w.bpm} bpm` : "—";
  $("watch-hrv").textContent = w.rmssd != null ? `${w.rmssd} ms` : w.state === "streaming" && !w.rr
    ? "not sent by this device" : "—";
  $("watch-used").textContent = hr && hr.source === "watch" ? "✓ watch (replaces the Muse sensor)"
    : hr && hr.source === "muse" ? "Muse forehead sensor" : "—";
  const on = ["searching", "connecting", "streaming"].includes(w.state);  // the server stops after 3 failed tries
  $("watch-connect").disabled = on;
  $("watch-connect").textContent = w.state === "error" ? "Retry" : "Connect watch";
  $("watch-disconnect").disabled = w.state === "off";
}

// ---------- Live widgets ----------
function updateLive(m, hr) {
  updateHeadset(m.source);
  updateWatch(m.watch, m.hr);
  updateSteps(m);
  if (presentOpen()) updatePresent(m);
  const s = m.stress == null ? null : Math.round(m.stress);
  document.querySelectorAll('[data-live="stress"]').forEach((el) => { el.textContent = s ?? "—"; el.style.color = stressColor(s); });
  $("pill-dot").style.background = stressColor(s);
  updateGauges(s);
  const fromWatch = m.hr && m.hr.source === "watch";
  document.querySelectorAll('[data-live="hr"]').forEach((el) => {
    el.textContent = hr != null ? `${fromWatch ? "⌚ " : ""}${Math.round(hr)} bpm` : "— bpm";
    el.title = fromWatch ? "Heart rate from your watch" : "Heart rate from the headset's forehead sensor";
  });
  $("score-source").textContent = m.stress_source ? `Score from: ${m.stress_source}` : "";
  document.querySelectorAll('[data-live="hr-big"]').forEach((el) => { el.textContent = hr != null ? Math.round(hr) : "—"; });
  document.querySelectorAll('[data-live="motion"]').forEach((el) => {
    el.textContent = m.motion.level == null ? "—" : m.motion.flag ? "Moving (signal paused)" : "Still";
  });

  const names = (state.hello && state.hello.eeg_names) || Object.keys(SENSOR_POS);
  const chans = (m.eeg && m.eeg.channels) || {};
  const dots = names.map((n) => {
    const q = chans[n] ? chans[n].q : "";
    const el = document.getElementById(`sensor-${n}`);
    if (el) el.setAttribute("class", `sensor ${q}`);
    return `<div class="sensor"><div class="dot ${q}" title="${n}: ${q || "no data"}"></div>${n}</div>`;
  }).join("");
  document.querySelectorAll('[data-live="contact"]').forEach((el) => { el.innerHTML = dots; });

  // Why each sensor is red/orange, in plain words
  const issues = names.filter((n) => chans[n] && chans[n].q !== "good");
  $("contact-why").innerHTML = !Object.keys(chans).length ? "<li class='muted'>Waiting for EEG data…</li>"
    : !issues.length ? "<li class='ok'>All four sensors in good contact</li>"
    : issues.map((n) => `<li class="${chans[n].q}"><b>${n}</b>${chans[n].why || chans[n].q}</li>`).join("");

  // Which signals are feeding the score right now
  const cal = state.baseline;
  $("signal-chips").innerHTML = !m.calibrated || !cal ? "" : Object.entries(SIGNAL_GROUPS).map(([label, keys]) => {
    const inGroup = (k) => keys.includes(k) || (keys === FACE_KEYS && isFaceKey(k));
    const used = cal.mode === "anchored" ? cal.used.some(inGroup)
      : Object.entries(cal.signals).some(([k, s]) => inGroup(k) && s.rest != null);
    const live = (m.used || []).some(inGroup);
    const cls = !used ? "off" : live ? "on" : "wait";
    const tip = !used ? `${label}: not used (no clear response during calibration)`
      : live ? `${label}: feeding the score now` : `${label}: used, but no clean signal right now`;
    return `<span class="chip ${cls}" title="${tip}">${label} ${!used ? "✗" : live ? "✓" : "…"}</span>`;
  }).join("");
  document.querySelectorAll('[data-live="stress-body"]').forEach((el) => {
    const b = m.stress_body == null ? null : Math.round(m.stress_body);
    el.textContent = b ?? "—"; el.style.color = stressColor(b);
  });
  const pm = $("personal-score"), pr = state.baseline && state.baseline.personal;
  pm.hidden = !(pr && pr.ok);
  if (pr && pr.ok) {
    const v = m.stress_personal == null ? null : Math.round(m.stress_personal);
    const el = pm.querySelector("b");
    el.textContent = v ?? "—"; el.style.color = stressColor(v);
  }
  for (const [id, info, value] of [["ml-score", m.learned, m.stress_ml], ["ml-score-headset", m.learned_headset, m.stress_ml_headset]]) {
    const line = $(id);
    line.hidden = !info;
    if (!info) continue;
    line.title = `Gradient-boosted trees (${info.name}) trained on ${info.trained_on.length} session(s), AUC ${info.cv_auc} on unseen data. Starts after the first 45 s of a calibration.`;
    const v = value == null ? null : Math.round(value);
    const el = line.querySelector("b");
    el.textContent = v ?? "—"; el.style.color = stressColor(v);
  }

  $("calib-note").hidden = !!m.calibrated;
  const err = $("engine-error");
  err.hidden = !m.error;
  err.textContent = m.error || "";
  if (m.source && m.source.true_stress != null) $("mock-true").textContent = `simulated stress: ${m.source.true_stress}`;

  const msg = !m.calibrated ? "Calibrate first" : "Waiting for signal";
  for (const sc of Object.values(state.scenes)) if (sc) sc.setStress(m.stress, msg);
  if (s != null) $("breath").style.setProperty("--glow", `hsl(${stressHue(s)}, 70%, 55%)`);

  if (isActive("setup")) drawSetupChart();
  if (isActive("reset") && state.resetStart != null) {
    const pts = state.history.filter((h) => h.phase === "reset").map((h) => ({ t: h.t, v: h.stress }));
    drawChart($("chart-reset"), pts, { tMin: state.resetStart, tMax: state.resetStart + DUR.reset });
  }
}
const BODY_COLOR = "#e08a3c";
// With headset (solid, accent) and without headset (dashed, orange) over the last two minutes
function drawScoresChart(canvas, seconds = 120, width = 2.5, fullColor = null) {
  const calibrated = state.latest && state.latest.calibrated;
  const now = state.latest ? state.latest.t : 0;
  if (!calibrated) {
    drawChart(canvas, state.live.map((p) => ({ t: p.t, v: p.ratio })),
      { tMin: now - seconds, tMax: now, auto: true, label: "raw EEG alpha/beta (log) until you calibrate" });
    return;
  }
  drawChart(canvas, state.live.map((p) => ({ t: p.t, v: p.stress })), {
    tMin: now - seconds, tMax: now, lineColor: fullColor || cssVar("--accent"), lineWidth: width,
    extra: [{ pts: state.live.map((p) => ({ t: p.t, v: p.body })), color: BODY_COLOR, dash: [6, 5], width }],
  });
}
function drawSetupChart() { drawScoresChart($("chart-setup")); }

// ---------- Home page steps ----------
function updateSteps(m) {
  const src = m.source || {};
  const headset = !!src.connected, camera = !!(window.ClutchFace && ClutchFace.isRunning());
  const step1 = headset && camera;
  $("sn-1").classList.toggle("done", step1);
  $("ss-1").className = "step-state" + (step1 ? " ok" : "");
  $("ss-1").textContent = `${headset ? "✓" : "✗"} headset streaming · ${camera ? "✓" : "✗"} camera on`;
  const cal = state.baseline && state.baseline.ok ? state.baseline : null;
  $("sn-2").classList.toggle("done", !!cal);
  $("ss-2").className = "step-state" + (cal ? " ok" : "");
  const minutes = protocol().key === "short" ? "About 3 minutes" : "About 9 minutes";
  $("ss-2").textContent = cal
    ? `✓ Calibrated · using ${[...new Set(cal.used.map(groupOf).filter(Boolean))].join(", ").toLowerCase() || "baseline distance only"}`
    : `${minutes}: baseline, stress test${protocol().recovery ? ", recovery" : ""}`;
  const live = !!cal && m.stress != null;
  $("sn-3").classList.toggle("done", live);
  $("ss-3").className = "step-state" + (live ? " ok" : "");
  $("ss-3").textContent = !cal ? "Calibrate first to see your scores" : live ? "✓ Live" : "Waiting for a clean signal";
}

// ---------- Presentation mode ----------
function presentOpen() { return !$("present").hidden; }
function togglePresent(show = !presentOpen()) {
  $("present").hidden = !show;
  if (show) {
    state.scenes.present = new BoatScene($("present-sea"));
    state.scenes.present.start();
    if (state.latest) updatePresent(state.latest);
  } else if (state.scenes.present) {
    state.scenes.present.stop();
    state.scenes.present = null;
  }
}
function updatePresent(m) {
  const set = (id, v) => { const r = v == null ? null : Math.round(v); $(id).textContent = r ?? "—"; $(id).style.color = r == null ? "#a9bccb" : stressColor(r, 62); };
  set("p-full", m.stress);
  set("p-body", m.stress_body);
  drawScoresChart($("present-chart"), 180, 4, "#5b9be0");
}

// ---------- Charts ----------
function cssVar(name) { return getComputedStyle(document.body).getPropertyValue(name).trim(); }
function prepCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return null;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px Inter, system-ui, sans-serif";
  return { ctx, w, h };
}
function drawChart(canvas, pts, o = {}) {
  const c = prepCanvas(canvas);
  if (!c) return;
  const { ctx, w, h } = c;
  const padL = 30, padR = 8, padT = 8, padB = 8;
  const vals = pts.map((p) => p.v).filter((v) => v != null);
  let yMin = 0, yMax = 100, ticks = [0, 50, 100];
  if (o.auto) {
    if (vals.length) {
      const lo = Math.min(...vals), hi = Math.max(...vals), pad = Math.max(0.05, (hi - lo) * 0.2);
      yMin = lo - pad; yMax = hi + pad;
    } else { yMin = -1; yMax = 1; }
    ticks = [];
  }
  const tMin = o.tMin ?? (pts[0] ? pts[0].t : 0);
  const tMax = o.tMax ?? (pts.length ? pts[pts.length - 1].t : 1);
  const X = (t) => padL + ((t - tMin) / Math.max(1e-6, tMax - tMin)) * (w - padL - padR);
  const Y = (v) => padT + (1 - (v - yMin) / (yMax - yMin)) * (h - padT - padB);

  for (const b of o.bands || []) {
    ctx.globalAlpha = 0.12; ctx.fillStyle = b.color;
    ctx.fillRect(X(b.t0), padT, X(b.t1) - X(b.t0), h - padT - padB);
    ctx.globalAlpha = 1;
  }
  ctx.strokeStyle = cssVar("--border"); ctx.fillStyle = cssVar("--muted"); ctx.lineWidth = 1;
  for (const v of ticks) {
    ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(w - padR, Y(v)); ctx.stroke();
    ctx.fillText(String(v), 4, Y(v) + 4);
  }
  if (o.label) ctx.fillText(o.label, padL + 4, padT + 12);

  // A line is broken where there is no value. Default colour follows the stress level (green -> red).
  const line = (series, color, dash = [], width = 2.5) => {
    ctx.lineWidth = width; ctx.lineCap = "round"; ctx.setLineDash(dash);
    let prev = null;
    for (const p of series) {
      if (p.v == null || p.t < tMin) { prev = null; continue; }
      const pt = { x: X(p.t), y: Y(clamp(p.v, yMin, yMax)), v: p.v };
      if (prev) {
        ctx.strokeStyle = color || (o.auto ? cssVar("--accent") : stressColor((prev.v + pt.v) / 2));
        ctx.beginPath(); ctx.moveTo(prev.x, prev.y); ctx.lineTo(pt.x, pt.y); ctx.stroke();
      }
      prev = pt;
    }
    ctx.setLineDash([]);
  };
  for (const s of o.extra || []) line(s.pts, s.color, s.dash || [], s.width);
  line(pts, o.lineColor, [], o.lineWidth);
  if (!vals.length) {
    ctx.fillStyle = cssVar("--muted"); ctx.textAlign = "center";
    ctx.fillText("waiting for signal", w / 2, h / 2); ctx.textAlign = "left";
  }
}
function drawBars(canvas, bins) {
  const c = prepCanvas(canvas);
  if (!c) return;
  const { ctx, w, h } = c;
  const padL = 34, padB = 40, padT = 18;
  const plotH = h - padB - padT;
  ctx.strokeStyle = cssVar("--border"); ctx.fillStyle = cssVar("--muted");
  for (const v of [0, 50, 100]) {
    const y = padT + plotH * (1 - v / 100);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - 8, y); ctx.stroke();
    ctx.fillText(`${v}%`, 2, y + 4);
  }
  const slot = (w - padL - 8) / bins.length;
  const bw = Math.min(70, slot * 0.5);
  const mid = [15, 50, 85]; // colour of each bin on the 0-100 scale
  bins.forEach((b, i) => {
    const cx = padL + slot * (i + 0.5);
    ctx.textAlign = "center";
    if (b.n) {
      const bh = plotH * b.accuracy;
      ctx.fillStyle = stressColor(mid[i]);
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(cx - bw / 2, padT + plotH - bh, bw, bh, [6, 6, 0, 0]) : ctx.rect(cx - bw / 2, padT + plotH - bh, bw, bh);
      ctx.fill();
      ctx.fillStyle = cssVar("--text");
      ctx.font = "600 12px Inter, system-ui, sans-serif";
      ctx.fillText(`${Math.round(b.accuracy * 100)}%`, cx, padT + plotH - bh - 6);
    }
    ctx.font = "600 11px Inter, system-ui, sans-serif";
    ctx.fillStyle = cssVar("--text");
    ctx.fillText(`${b.label} (${b.range})`, cx, h - padB + 16);
    ctx.font = "11px Inter, system-ui, sans-serif";
    ctx.fillStyle = cssVar("--muted");
    ctx.fillText(b.n ? `${b.n} answers · ${b.rt != null ? b.rt.toFixed(2) + " s" : "—"}` : "no answers", cx, h - padB + 31);
    ctx.textAlign = "left";
  });
}

// ---------- Sound ----------
let actx = null;
function audio() {
  if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
  return actx;
}
function beep(freq, dur, type = "sine", gain = 0.08) {
  const a = audio();
  const osc = a.createOscillator(), g = a.createGain();
  osc.type = type; osc.frequency.value = freq;
  g.gain.setValueAtTime(gain, a.currentTime);
  g.gain.exponentialRampToValueAtTime(0.0001, a.currentTime + dur);
  osc.connect(g).connect(a.destination);
  osc.start(); osc.stop(a.currentTime + dur);
}
const sfx = {
  tick: () => beep(1200, 0.03, "square", 0.03),
  tickUrgent: () => beep(1600, 0.05, "square", 0.06),
  wrong: () => beep(140, 0.3, "sawtooth", 0.12),
  right: () => beep(880, 0.08, "sine", 0.05),
  done: () => beep(660, 0.4, "sine", 0.08),
};

// ---------- Helpers ----------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
function show(name) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
  $(`screen-${name}`).classList.add("active");
  document.body.classList.toggle("immersive", name === "reset");
  window.scrollTo(0, 0);
}
function isActive(name) { return $(`screen-${name}`).classList.contains("active"); }
function waitClick(el) { return new Promise((r) => { el.onclick = () => { el.onclick = null; r(); }; }); }
function countdown(seconds, onTick) {
  return new Promise((resolve) => {
    const end = performance.now() + seconds * 1000;
    onTick && onTick(seconds, seconds);
    const iv = setInterval(() => {
      const rem = Math.max(0, (end - performance.now()) / 1000);
      onTick && onTick(rem, seconds);
      if (rem <= 0) { clearInterval(iv); resolve(); }
    }, 100);
  });
}
function setPill(visible) { $("live-pill").classList.toggle("hidden", !visible); }
function animateNumber(el, to, decimals = 0, suffix = "") {
  if (to == null || Number.isNaN(to)) { el.textContent = "—"; return; }
  const final = to.toFixed(decimals) + suffix;
  el.textContent = final; // shown as-is if the page is not being drawn (background tab)
  const start = performance.now(), dur = 900;
  const step = (now) => {
    const p = Math.min(1, (now - start) / dur), e = 1 - Math.pow(1 - p, 3);
    el.textContent = p < 1 ? (to * e).toFixed(decimals) + suffix : final;
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ---------- Arithmetic task ----------
function rand(a, b) { return a + Math.floor(Math.random() * (b - a + 1)); }
function makeProblem() {
  // Two-digit addition or subtraction: same difficulty in every round.
  const plus = Math.random() < 0.5;
  let a = rand(12, 89), b = rand(12, 89);
  if (!plus && b >= a) [a, b] = [b + 1, a];
  return { text: `${a} ${plus ? "+" : "−"} ${b}`, answer: plus ? a + b : a - b };
}

const TASKS = {
  warmup: { title: "Warm-up", dur: () => DUR.warmup, pressure: false,
    intro: "Practice round: solve each problem and press Enter. Nothing is scored." },
  round1: { title: "Round 1", dur: () => DUR.round, pressure: true,
    intro: `Solve as many as you can. Each problem has ${PER_PROBLEM_S} seconds. Mistakes and timeouts count against you.` },
  round2: { title: "Round 2", dur: () => DUR.round, pressure: true,
    intro: "Same rules as round 1. Solve as many as you can." },
};

function feedback(ok, timeout, pressure) {
  const fb = $("feedback");
  fb.className = "feedback " + (ok ? "ok" : "ko");
  fb.textContent = ok ? "Correct" : timeout ? "Too slow" : "Wrong";
  if (ok) { sfx.right(); return; }
  if (pressure) {
    sfx.wrong();
    document.body.classList.remove("flash-bad");
    void document.body.offsetWidth;
    document.body.classList.add("flash-bad");
  }
}

async function runTask(kind) {
  const T = TASKS[kind];
  show("task");
  $("task-title").textContent = T.title;
  $("task-instructions").textContent = T.intro;
  $("task-score").textContent = "0";
  $("task-timer").textContent = T.dur();
  $("task-timer").classList.remove("urgent");
  $("problem").textContent = "—";
  $("feedback").textContent = "";
  $("problem-bar-wrap").style.visibility = T.pressure ? "visible" : "hidden";
  const input = $("answer");
  input.value = ""; input.disabled = true;
  $("task-ready").hidden = false;
  $("task-ready-text").textContent = "Press Start when ready.";
  setPhase("ready");
  await waitClick($("task-go"));
  $("task-ready").hidden = true;
  setPhase(kind);

  const res = { attempted: 0, correct: 0, timeouts: 0, rts: [], items: [] };
  let current = null, shownAt = 0;
  const next = () => {
    current = makeProblem(); shownAt = performance.now();
    $("problem").textContent = current.text; input.value = "";
  };
  const record = (given, timeout) => {
    const rt = (performance.now() - shownAt) / 1000;
    const ok = !timeout && Number(given.replace("−", "-")) === current.answer;
    const stress = state.latest ? state.latest.stress : null; // stress at the moment of answering
    res.attempted++;
    if (ok) { res.correct++; res.rts.push(rt); }
    if (timeout) res.timeouts++;
    res.items.push({ correct: ok, rt: +rt.toFixed(3), timeout, stress });
    send({ type: "event", name: "answer", data: {
      round: kind, problem: current.text, answer: current.answer, given: timeout ? null : given,
      correct: ok, rt: +rt.toFixed(3), timeout, stress } });
    feedback(ok, timeout, T.pressure);
    $("task-score").textContent = res.correct;
    next();
  };
  input.disabled = false; input.focus();
  input.onkeydown = (e) => {
    if (e.key === "Enter" && input.value.trim() !== "") record(input.value.trim(), false);
  };
  next();

  let lastWhole = null;
  await countdown(T.dur(), (rem) => {
    const whole = Math.ceil(rem);
    $("task-timer").textContent = whole;
    $("task-timer").classList.toggle("urgent", T.pressure && rem <= 10);
    if (T.pressure && rem > 0 && whole !== lastWhole) {
      lastWhole = whole;
      rem <= 10 ? sfx.tickUrgent() : sfx.tick();
    }
    if (T.pressure && current) {
      const el = (performance.now() - shownAt) / 1000;
      $("problem-bar").style.width = `${Math.max(0, 100 * (1 - el / PER_PROBLEM_S))}%`;
      if (el >= PER_PROBLEM_S) record("", true);
    }
  });
  input.onkeydown = null; input.disabled = true;
  $("problem").textContent = "Time";
  $("feedback").textContent = "";
  sfx.done();

  const stats = {
    attempted: res.attempted, correct: res.correct, timeouts: res.timeouts,
    accuracy: res.attempted ? res.correct / res.attempted : null,
    mean_rt: mean(res.rts), items: res.items,
  };
  state.session.rounds[kind] = stats;
  const { items, ...summary } = stats;
  send({ type: "event", name: "round_end", data: { round: kind, ...summary } });
  await sleep(1200);
}

// ---------- Calibration ----------
function waitBaseline(timeoutMs) {
  return new Promise((resolve) => {
    state.baselineWaiter = resolve;
    setTimeout(() => {
      if (state.baselineWaiter === resolve) { state.baselineWaiter = null; resolve(null); }
    }, timeoutMs);
  });
}
// Calibration protocol (methods and references: docs/methods.md)
function protocol() {
  const key = $("protocol") ? $("protocol").value : "standard";
  return { key, ...PROTOCOLS[key] };
}
function calibSteps(p) {
  return [["rest", "Baseline"], ["stress", "Stress test"], ...(p.recovery ? [["recovery", "Recovery"]] : []), ["report", "Your profile"]];
}
function calibShow(part, p) {
  const steps = calibSteps(p);
  const idx = steps.findIndex(([k]) => k === part);
  $("calib-steps").innerHTML = steps.map(([k, label], i) =>
    `<span class="step${i === idx ? " active" : i < idx ? " done" : ""}">${i + 1} · ${label}</span>`).join("");
  ["rest", "survey", "stress", "recovery", "report", "relax", "rating", "study"].forEach((k) => { $(`calib-${k}`).hidden = k !== part; });
}

// Baseline: low-demand "vanilla" task (Jennings et al., 1992). Press Space when the circle turns blue.
async function vanillaBaseline(seconds) {
  const circle = $("vanilla");
  const COLORS = ["#9aa5b1", "#3fae7a", "#e0a03c", "#8f6ad9"], BLUE = "#2f6fd6";
  let isBlue = false, hits = 0, misses = 0, falseAlarms = 0, nextChange = 0, responded = false;
  const onKey = (e) => {
    if (e.code !== "Space") return;
    e.preventDefault();
    if (isBlue && !responded) { hits++; responded = true; circle.classList.add("hit"); setTimeout(() => circle.classList.remove("hit"), 250); }
    else if (!isBlue) falseAlarms++;
  };
  if (document.activeElement) document.activeElement.blur(); // Space must not press a focused button
  document.addEventListener("keydown", onKey);
  await countdown(seconds, (rem, tot) => {
    $("calib-bar").style.width = `${100 * (1 - rem / tot)}%`;
    $("calib-timer").textContent = `${Math.ceil(rem)} s`;
    const el = tot - rem;
    if (el >= nextChange) {
      if (isBlue && !responded) misses++;
      isBlue = Math.random() < 0.2;
      responded = false;
      circle.style.background = isBlue ? BLUE : COLORS[Math.floor(Math.random() * COLORS.length)];
      nextChange = el + 2 + Math.random() * 1.5;
    }
  });
  document.removeEventListener("keydown", onKey);
  return { hits, misses, false_alarms: falseAlarms };
}

// Self-report: six-item STAI state scale (Marteau & Bekker, 1992). 1-4 per item, positive items reversed,
// sum x 20/6 gives the usual 20-80 range.
const STAI6 = [["calm", "I feel calm", true], ["tense", "I am tense", false], ["upset", "I feel upset", false],
  ["relaxed", "I am relaxed", true], ["content", "I feel content", true], ["worried", "I am worried", false]];
const STAI_OPTS = ["Not at all", "Somewhat", "Moderately", "Very much"];
function staiSurvey(p) {
  calibShow("survey", p);
  $("stai-form").innerHTML = STAI6.map(([k, text]) => `<div class="stai-item"><span class="q">${text}</span><div class="stai-opts">` +
    STAI_OPTS.map((o, i) => `<label><input type="radio" name="${k}" value="${i + 1}">${o}</label>`).join("") + "</div></div>").join("");
  const form = $("stai-form"), btn = $("stai-submit");
  btn.disabled = true;
  form.onchange = () => { btn.disabled = STAI6.some(([k]) => !form.querySelector(`input[name="${k}"]:checked`)); };
  return new Promise((resolve) => {
    btn.onclick = () => {
      const answers = {};
      let sum = 0;
      for (const [k, , positive] of STAI6) {
        const v = Number(form.querySelector(`input[name="${k}"]:checked`).value);
        answers[k] = v;
        sum += positive ? 5 - v : v;
      }
      resolve({ score: Math.round(sum * 20 / 6), answers });
    };
  });
}

// Stressor: serial subtraction from the TSST (Kirschbaum et al., 1993) with MIST-style pressure
// (Dedovic et al., 2005): an answer time limit that tightens after each success, and a comparison with
// an "average participant" who always does better. A mistake or a timeout restarts from 1022.
async function serialSubtraction(seconds) {
  const input = $("sub-answer"), fb = $("sub-feedback");
  let expected = SUB_START - SUB_STEP, streak = 0, best = 0, limit = 8, shownAt = performance.now();
  const recent = [];
  const stats = { attempts: 0, correct: 0, timeouts: 0 };
  const updateCompare = () => {
    const you = recent.length ? Math.round(100 * recent.filter(Boolean).length / recent.length) : 0;
    const avg = Math.min(95, you + 25);
    $("cmp-you").style.width = `${you}%`; $("cmp-you-val").textContent = `${you}%`;
    $("cmp-avg").style.width = `${avg}%`; $("cmp-avg-val").textContent = `${avg}%`;
  };
  const restart = (msg) => {
    streak = 0; expected = SUB_START - SUB_STEP;
    limit = Math.min(10, limit * 1.1);
    fb.className = "feedback ko"; fb.textContent = `${msg} Start again from ${SUB_START}.`;
    $("sub-prompt").textContent = `${SUB_START} − ${SUB_STEP} = ?`;
    sfx.wrong();
    document.body.classList.remove("flash-bad"); void document.body.offsetWidth; document.body.classList.add("flash-bad");
  };
  const record = (ok, given, timeout) => {
    stats.attempts++;
    if (ok) stats.correct++;
    if (timeout) stats.timeouts++;
    recent.push(ok); if (recent.length > 10) recent.shift();
    send({ type: "event", name: "calib_answer", data: { expected, given, correct: ok, timeout,
      rt: +((performance.now() - shownAt) / 1000).toFixed(3), limit: +limit.toFixed(2),
      stress: state.latest ? state.latest.stress : null } });
    shownAt = performance.now();
    updateCompare();
  };
  $("sub-prompt").textContent = `${SUB_START} − ${SUB_STEP} = ?`;
  fb.textContent = ""; input.value = ""; input.disabled = false; input.focus();
  updateCompare();
  input.onkeydown = (e) => {
    if (e.key !== "Enter" || input.value.trim() === "") return;
    const given = Number(input.value.trim());
    const ok = given === expected;
    input.value = "";
    record(ok, given, false);
    if (ok) {
      streak++; best = Math.max(best, streak); expected -= SUB_STEP;
      limit = Math.max(2.5, limit * 0.9);
      fb.className = "feedback ok"; fb.textContent = `✓ ${streak} in a row`;
      $("sub-prompt").textContent = `… − ${SUB_STEP} = ?`; // no running total shown: you must keep track
      sfx.right();
    } else {
      restart("Wrong.");
    }
  };
  let lastWhole = null;
  await countdown(seconds, (rem) => {
    const whole = Math.ceil(rem);
    $("sub-timer").textContent = whole;
    $("sub-timer").classList.toggle("urgent", rem <= 10);
    if (rem > 0 && whole !== lastWhole) { lastWhole = whole; rem <= 10 ? sfx.tickUrgent() : sfx.tick(); }
    const el = (performance.now() - shownAt) / 1000;
    $("sub-limit-bar").style.width = `${Math.max(0, 100 * (1 - el / limit))}%`;
    if (el >= limit && rem > 0) { record(false, null, true); restart("Too slow."); }
  });
  input.onkeydown = null; input.disabled = true;
  return { ...stats, best_streak: best };
}

// Quick math, MIST-inspired (Dedovic et al., 2005): a new problem every time, single-digit answer typed
// with one key, five difficulty levels that rise after 3 correct answers in a row and drop after an error.
function makeMathItem(level) {
  const r = rand(0, 9); // the answer is always a single digit
  let text;
  if (level === 1) {
    const b = rand(1, 9);
    text = r + b <= 18 && Math.random() < 0.5 ? `${r + b} − ${b}` : (() => { const a = rand(0, r); return `${a} + ${r - a}`; })();
  } else if (level === 2) {
    const c = rand(1, 9), a = rand(1, 9), b = r + c - a;
    text = b >= 1 && b <= 9 ? `${a} + ${b} − ${c}` : `${r + c} − ${c}`;
  } else if (level === 3) {
    const big = rand(12, 48);
    text = big - r >= 3 ? `${big} − ${big - r}` : `${big} − ${big}`;
    if (big - r < 3) return { text, answer: 0 };
  } else if (level === 4) {
    const a = rand(2, 9), b = rand(2, 9), c = a * b - r;
    text = c >= 0 ? `${a} × ${b} − ${c}` : `${a} × ${b} − ${a * b}`;
    if (c < 0) return { text, answer: 0 };
  } else {
    const a = rand(3, 9), b = rand(3, 9), total = a * b - r;
    if (total < 2) return makeMathItem(4);
    const c = rand(1, total - 1), d = total - c;
    text = `${a} × ${b} − ${c} − ${d}`;
  }
  return { text, answer: r };
}

// Colour words (Stroop, 1935): name the INK colour, not the word. 75% of items are incongruent.
const STROOP = { R: ["RED", "#d64545"], G: ["GREEN", "#2e9e5b"], B: ["BLUE", "#2f6fd6"], Y: ["YELLOW", "#d4a514"] };
function makeStroopItem() {
  const keys = Object.keys(STROOP);
  const ink = keys[rand(0, 3)];
  let word = ink;
  if (Math.random() < 0.75) while (word === ink) word = keys[rand(0, 3)];
  return { html: `<span class="stroop-word" style="color:${STROOP[ink][1]}">${STROOP[word][0]}</span>`, answer: ink };
}

// Shared engine for the two fast tasks: adaptive answer time limit (tighter after successes, looser after
// failures) and the MIST-style comparison with an "average participant". No restart: just the next item.
// easy: fixed easiest level, 10 s per answer, no buzzer, flash or comparison bar (study protocol's easy block)
async function fastTask(kind, seconds, { easy = false } = {}) {
  const stroop = kind === "stroop";
  const fb = $("sub-feedback"), prompt = $("sub-prompt");
  const [minLim, maxLim] = stroop ? [0.9, 3] : [2, 10];
  let limit = easy ? 10 : stroop ? 2.2 : 6, level = 1, streak = 0, best = 0, cur = null, shownAt = 0;
  const recent = [];
  const stats = { attempts: 0, correct: 0, timeouts: 0, max_level: 1 };
  const updateCompare = () => {
    const you = recent.length ? Math.round(100 * recent.filter(Boolean).length / recent.length) : 0;
    const avg = Math.min(95, you + 25);
    $("cmp-you").style.width = `${you}%`; $("cmp-you-val").textContent = `${you}%`;
    $("cmp-avg").style.width = `${avg}%`; $("cmp-avg-val").textContent = `${avg}%`;
  };
  const next = () => {
    cur = stroop ? makeStroopItem() : makeMathItem(level);
    prompt.innerHTML = stroop ? cur.html : `${cur.text} = ?`;
    shownAt = performance.now();
  };
  const answer = (given, timeout) => {
    const ok = !timeout && given === cur.answer;
    stats.attempts++;
    if (ok) stats.correct++;
    if (timeout) stats.timeouts++;
    recent.push(ok); if (recent.length > 10) recent.shift();
    send({ type: "event", name: "calib_answer", data: { task: kind, item: stroop ? cur.answer : cur.text, given,
      correct: ok, timeout, level, rt: +((performance.now() - shownAt) / 1000).toFixed(3), limit: +limit.toFixed(2),
      stress: state.latest ? state.latest.stress : null } });
    if (ok && easy) {
      streak++; best = Math.max(best, streak);
      fb.className = "feedback ok"; fb.textContent = "✓";
    } else if (easy) {
      streak = 0;
      fb.className = "feedback"; fb.textContent = timeout ? "Next one" : "Not quite";
    } else if (ok) {
      streak++; best = Math.max(best, streak);
      limit = Math.max(minLim, limit * 0.92);
      if (!stroop && streak % 3 === 0) { level = Math.min(5, level + 1); stats.max_level = Math.max(stats.max_level, level); }
      fb.className = "feedback ok"; fb.textContent = "✓";
      sfx.right();
    } else {
      streak = 0;
      limit = Math.min(maxLim, limit * 1.12);
      if (!stroop) level = Math.max(1, level - 1);
      fb.className = "feedback ko"; fb.textContent = timeout ? "Too slow" : "Wrong";
      sfx.wrong();
      document.body.classList.remove("flash-bad"); void document.body.offsetWidth; document.body.classList.add("flash-bad");
    }
    updateCompare();
    next();
  };
  const onKey = (e) => {
    if (stroop) {
      const k = e.key.toUpperCase();
      if (STROOP[k]) { e.preventDefault(); answer(k, false); }
    } else if (/^[0-9]$/.test(e.key)) {
      e.preventDefault(); answer(Number(e.key), false);
    }
  };
  if (document.activeElement) document.activeElement.blur();
  document.addEventListener("keydown", onKey);
  fb.textContent = "";
  updateCompare();
  next();
  let lastWhole = null;
  await countdown(seconds, (rem) => {
    const whole = Math.ceil(rem);
    $("sub-timer").textContent = whole;
    $("sub-timer").classList.toggle("urgent", !easy && rem <= 10);
    if (!easy && rem > 0 && whole !== lastWhole) { lastWhole = whole; rem <= 10 ? sfx.tickUrgent() : sfx.tick(); }
    const el = (performance.now() - shownAt) / 1000;
    $("sub-limit-bar").style.width = `${Math.max(0, 100 * (1 - el / limit))}%`;
    if (el >= limit && rem > 0) answer(null, true);
  });
  document.removeEventListener("keydown", onKey);
  prompt.textContent = "Time";
  return { ...stats, best_streak: best };
}

const STRESSOR_LEADS = {
  math: "Solve each calculation as fast as you can: every answer is a single digit, just press the key. " +
    "It gets harder as you succeed, and the time limit gets shorter.",
  stroop: "Press the key for the <b>ink colour</b> of the word, not the word itself: <kbd>R</kbd> red, " +
    "<kbd>G</kbd> green, <kbd>B</kbd> blue, <kbd>Y</kbd> yellow. Be fast: the time limit gets shorter.",
  serial: "Count backwards from <b>1022</b> in steps of <b>13</b>, as fast as you can. Type each number and press Enter. " +
    "A mistake or running out of time sends you back to 1022.",
};
async function runStressor(kind, seconds, { easy = false, title = "Stress test", lead = null } = {}) {
  $("calib-stress").querySelector("h1").textContent = title;
  $("stress-lead").innerHTML = lead || STRESSOR_LEADS[kind];
  $("sub-answer").hidden = kind !== "serial";
  $("stroop-keys").hidden = kind !== "stroop";
  $("calib-stress").querySelector(".compare").hidden = easy;
  const result = kind === "serial" ? await serialSubtraction(seconds) : await fastTask(kind, seconds, { easy });
  $("calib-stress").querySelector("h1").textContent = "Stress test";
  $("calib-stress").querySelector(".compare").hidden = false;
  return { task: kind, easy, ...result };
}

async function recoveryRest(seconds) {
  await countdown(seconds, (rem, tot) => {
    $("rec-bar").style.width = `${100 * (1 - rem / tot)}%`;
    $("rec-timer").textContent = `${Math.ceil(rem)} s`;
  });
}

function renderCalibrationReport(cal, extra) {
  const sig = (cal && cal.signals) || {};
  const card = (key, title, fmt, unit) => {
    const s = sig[key] || {};
    const used = cal && cal.used.includes(key);
    let vals = "—", change = "";
    if (s.rest != null && s.stress != null) {
      vals = `${fmt(s.rest)} → ${fmt(s.stress)}${unit}`;
      change = key === "eeg"
        ? `Alpha/beta ratio ${s.stress < s.rest ? "dropped" : "rose"} ${Math.abs(Math.round((Math.pow(10, s.stress - s.rest) - 1) * 100))}% under stress`
        : `${s.stress > s.rest ? "+" : ""}${Math.round(s.stress) - Math.round(s.rest)} bpm under stress`;
      // recovery only means something for a signal that clearly reacted to the stressor
      if (s.recovered_pct != null && used) change += ` · recovered ${Math.min(100, Math.max(0, s.recovered_pct))}% toward calm afterwards`;
    } else if (s.rest != null) {
      vals = `${fmt(s.rest)}${unit} at baseline`;
    }
    return `<div class="sig-card"><h3>${title}</h3><div class="muted small">baseline → stress test</div>
      <div class="vals">${vals}</div><div class="muted small">${change}</div>
      <p class="verdict ${used ? "used" : "unused"}">${used ? "✓ Used" : "✗ Not used"}: ${s.note || "not enough clean signal"}
      ${s.effect != null ? `<span class="muted">(Cohen's d = ${s.effect.toFixed(2)})</span>` : ""}</p></div>`;
  };
  // Self-report card: did the test actually feel stressful? (manipulation check)
  const s1 = extra.stai_before, s2 = extra.stai_after;
  let selfCard = "";
  if (s1 && s2) {
    const d = s2.score - s1.score;
    const felt = d >= 5 ? "✓ You felt more anxious: the stressor worked" : d > 0 ? "Slightly more anxious" : "✗ You did not feel more anxious";
    selfCard = `<div class="sig-card"><h3>How you felt · STAI-6</h3><div class="muted small">baseline → stress test (20-80)</div>
      <div class="vals">${s1.score} → ${s2.score}</div><div class="muted small">${d >= 0 ? "+" : ""}${d} points</div>
      <p class="verdict ${d >= 5 ? "used" : "unused"}">${felt}</p></div>`;
  }
  // Face card: every webcam indicator, its change and whether it was kept
  const faceRows = FACE_KEYS.filter((k) => sig[k]).map((k) => {
    const s = sig[k], used = cal && cal.used.includes(k);
    const fmt = (v) => (k === "blink_rate" ? v.toFixed(0) : k === "head_motion" ? v.toFixed(3) : v.toFixed(2));
    const vals = s.rest != null && s.stress != null ? `${fmt(s.rest)} → ${fmt(s.stress)}` : "—";
    return `<tr><td>${FACE_LABELS[k]}</td><td>${vals}</td><td>${s.effect != null ? s.effect.toFixed(2) : "—"}</td>
      <td class="${used ? "better" : "worse"}">${used ? "✓" : "✗"}</td></tr>`;
  }).join("");
  const faceCard = faceRows
    ? `<div class="sig-card"><h3>Face · webcam</h3><div class="muted small">baseline → stress test, Cohen's d</div>
        <table class="mini">${faceRows}</table></div>`
    : `<div class="sig-card"><h3>Face · webcam</h3><p class="muted small">Not recorded. Start the camera before calibrating
        to get the "without headset" score.</p></div>`;
  const noHeadset = cal && cal.ok ? (cal.used_without_headset || []) : [];
  const bodyCard = `<div class="sig-card"><h3>Without headset</h3><div class="muted small">signals usable with a watch + webcam</div>
    <div class="vals">${noHeadset.length ? noHeadset.length : "None"}</div>
    <p class="verdict ${noHeadset.length ? "used" : "unused"}">${noHeadset.length
      ? "✓ " + noHeadset.map((k) => SIGNAL_NAMES[k] || FACE_LABELS[k] || k).join(", ")
      : "✗ Only the EEG reacted: the headset-free score is not available yet"}</p></div>`;
  // Personal model: heart + face trained on this calibration to predict the EEG stress position (clutch/personal.py)
  const pm = cal && cal.personal;
  const personalCard = !pm ? "" : `<div class="sig-card"><h3>Your personal model</h3>
    <div class="muted small">heart + face trained to predict your EEG stress</div>
    <div class="vals">${pm.ok ? `${pm.seconds} s` : "—"}</div>
    ${pm.ok ? `<div class="muted small">${pm.features.map((k) => `${SIGNAL_NAMES[k] || FACE_LABELS[k] || k} ${pm.weights[k] > 0 ? "+" : ""}${pm.weights[k].toFixed(2)}`).join(" · ")}<br>fit on this calibration r = ${pm.fit_r.toFixed(2)}</div>` : ""}
    <p class="verdict ${pm.ok && !pm.weak_target ? "used" : "unused"}">${pm.ok ? (pm.weak_target ? "✓ Trained, but " : "✓ ") + pm.note : "✗ Not trained: " + pm.note}</p></div>`;
  // Flip check: anchors from the first half of rest and stress, tested on the second half (engine.split_half_check)
  const chk = cal && cal.check;
  let checkCard = "";
  if (chk) {
    const ok = chk.passed === true, na = chk.passed == null;
    checkCard = `<div class="sig-card check-card ${ok ? "pass" : na ? "" : "fail"}"><h3>Calibration check</h3>
      <div class="muted small">first half sets your calm and stress, second half tests them</div>
      <div class="vals">${na ? "—" : ok ? "✓ Passed" : "✗ Failed"}</div>
      <div class="muted small">${chk.auc != null ? `sorted ${Math.round(chk.auc * 100)}% of pairs right · gap ${chk.gap.toFixed(2)}` : ""}</div>
      <p class="verdict ${ok ? "used" : "unused"}">${ok ? "The score points the right way for you: good to go."
        : na ? "Calibration too short to check." : `Redo recommended: ${chk.note}.`}</p></div>`;
  }
  $("report-signals").innerHTML = checkCard +
    brainCard(sig, cal) +
    card("hr", "Heart rate", (v) => Math.round(v), " bpm") +
    (sig.hrv ? card("hrv", "Heart rate variability (RMSSD)", (v) => Math.round(v), " ms") : "") +
    faceCard + bodyCard + personalCard + selfCard;

  let lead, summary;
  if (!cal || !cal.ok) {
    lead = "Calibration failed: no clean signal from the headset.";
    summary = "Check that the sensors are green (bar flat on the forehead, rubber tips on the skin behind the ears), then redo the calibration.";
  } else if (cal.confidence === "low") {
    lead = "Neither signal showed at least a medium response to the stress test.";
    summary = "The score will only show distance from your baseline, so treat it as rough. Better sensor contact usually fixes this: redo the calibration if you can.";
  } else {
    const names = [...new Set(cal.used.map(groupOf).filter(Boolean))].map((g) => g.toLowerCase()).join(" and ");
    lead = `Your ${names} reacted to stress: the score is personalised to you.`;
    summary = "From now on, 0 means as calm as at baseline and 100 means as stressed as during the test. Signals without at least a medium response are left out.";
  }
  if (chk && chk.passed === false && cal && cal.ok) {
    summary += " The calibration check failed: redo the calibration before trusting the score (sensor contact is the usual cause).";
  }
  $("calib-retry").classList.toggle("primary", !!(chk && chk.passed === false));
  $("calib-continue").classList.toggle("primary", !(chk && chk.passed === false));
  if (s1 && s2 && s2.score - s1.score < 5 && cal && cal.ok) {
    summary += " Note: you did not report feeling more anxious, so the stress anchor may be weak.";
  }
  $("report-lead").textContent = lead;
  $("report-summary").textContent = summary;
  const att = extra.baseline_task;
  $("report-debrief").textContent =
    "About the test: the \"average participant\" bar was not real data. It always sits above you, as in the Montreal " +
    "Imaging Stress Task, because social comparison makes the stressor work." +
    (att ? ` Baseline attention check: ${att.hits} of ${att.hits + att.misses} blue circles caught.` : "");
  $("calib-continue").disabled = !(cal && cal.ok);
}

// standalone: a calibration started from the home screen, recorded as its own dataset session.
// Its recording stops as soon as the protocol ends (not when the profile screen is closed).
async function calibrate({ standalone = false } = {}) {
  const p = protocol();
  const recording = standalone && $("ds-record").checked;
  for (let attempt = 0; ; attempt++) {
    if (recording && attempt > 0) send({ type: "session_start", meta: datasetMeta("calibration") }); // redo = new session
    show("calibration");
    const extra = { protocol: p.key };
    calibShow("rest", p);
    setPhase("calib_rest");
    extra.baseline_task = await vanillaBaseline(p.rest);
    sfx.done();
    setPhase("calib_survey");
    extra.stai_before = await staiSurvey(p);
    calibShow("stress", p);
    setPhase("calib_stress");
    extra.stressor = await runStressor($("stressor").value, p.stress);
    setPhase("calib_survey");
    extra.stai_after = await staiSurvey(p);
    if (p.recovery) {
      calibShow("recovery", p);
      setPhase("calib_recovery");
      await recoveryRest(p.recovery);
    }
    const waiting = waitBaseline(4000);
    setPhase("calibration_check");
    const cal = await waiting;
    send({ type: "event", name: "calibration", data: { ...(cal || {}), ...extra } });
    if (recording) {
      send({ type: "session_end", summary: { kind: "calibration", calibration: cal, ...extra } });
      $("ds-status").textContent = `Calibration recorded for ${$("ds-participant").value.trim() || "anon"}` +
        ($("ds-consent").checked ? " (with consent: will be exported)." : " (no consent: will NOT be exported).");
    }
    calibShow("report", p);
    renderCalibrationReport(cal, extra);
    sfx.done();
    const choice = await new Promise((r) => {
      $("calib-continue").onclick = () => r("continue");
      $("calib-retry").onclick = () => r("retry");
    });
    if (choice === "continue") return true;
  }
}
// ---------- Dataset recording ----------
function datasetMeta(kind) {
  const p = protocol();
  const src = state.latest && state.latest.source;
  return {
    kind, participant: $("ds-participant").value.trim() || $("participant").value.trim() || "anon",
    consent: $("ds-record").checked && $("ds-consent").checked,
    protocol: p.key, stressor: $("stressor").value, protocol_seconds: { rest: p.rest, stress: p.stress, recovery: p.recovery },
    camera: !!(window.ClutchFace && ClutchFace.isRunning()), headset_streaming: !!(src && src.connected),
    device: src && src.headset ? { name: src.headset.device, address: src.headset.address } : null,
    started_at: new Date().toISOString(),
  };
}
// The camera is required for every calibration, study and session: without face data the headset-free score
// and the personal model have nothing to learn from. Starts it if it is off; false if it still is not running.
async function cameraReady() {
  if (window.ClutchFace && ClutchFace.isRunning()) return true;
  await $("cam-toggle").onclick();
  if (window.ClutchFace && ClutchFace.isRunning()) return true;
  alert("The camera must be on to calibrate: face data is needed for the headset-free score.\n\n" +
    "Click \"Start camera\" in step 1 and allow camera access in the browser, then try again.");
  $("cam-toggle").scrollIntoView({ behavior: "smooth", block: "center" });
  return false;
}

function recordingChecksPass() {
  if (!$("ds-record").checked) return true;
  const missing = [];
  if (!$("ds-participant").value.trim()) missing.push("no participant code");
  const src = state.latest && state.latest.source;
  if (!(src && src.connected)) missing.push("the headset is not streaming: no EEG or heart data");
  if (!$("ds-consent").checked) missing.push("consent is not ticked: this recording will not be exported");
  return !missing.length || confirm("Recording for the dataset, but:\n- " + missing.join("\n- ") + "\n\nContinue anyway?");
}

async function calibrateOnly() {
  if (state.busy || !(await cameraReady()) || !recordingChecksPass()) return;
  state.busy = true;
  audio().resume();
  const recording = $("ds-record").checked;
  if (recording) send({ type: "session_start", meta: datasetMeta("calibration") });
  await calibrate({ standalone: true });
  setPhase("idle");
  show("setup");
  state.busy = false;
}

// ---------- Study protocol (EPIStress-style, for training data) ----------
// Calibration, then four blocks with a 1-5 stress rating after each (EPIStress: Mental stress level 1-5).
// Phases study_relax (calm), study_math_hard and study_stroop_hard (stressed) are labelled by ml/run.py;
// study_math_easy is the in-between block. Orders A and B swap the block order between participants.
const STUDY_BLOCKS = {
  relax: { phase: "study_relax", title: "Relax" },
  math_easy: { phase: "study_math_easy", title: "Easy math", task: "math", easy: true,
    lead: "Easy sums with plenty of time: every answer is a single digit, just press the key. No rush." },
  math_hard: { phase: "study_math_hard", title: "Hard math", task: "math" },
  stroop_hard: { phase: "study_stroop_hard", title: "Hard colour words", task: "stroop" },
};
const STUDY_ORDERS = { A: ["relax", "math_easy", "math_hard", "stroop_hard"], B: ["relax", "stroop_hard", "math_hard", "math_easy"] };
const RATINGS = ["Not at all", "A little", "Moderately", "Very", "Extremely"];

function studyShow(part, blocks, idx) {
  calibShow(part, protocol());
  $("calib-steps").innerHTML = ["Calibration", ...blocks.map((b) => b.title), "Summary"].map((label, i) =>
    `<span class="step${i === idx ? " active" : i < idx ? " done" : ""}">${i + 1} · ${label}</span>`).join("");
}

async function relaxBlock(seconds) {
  const scene = new BoatScene($("relax-sea"), { hud: false });
  scene.start();
  scene.setStress(5);
  await countdown(seconds, (rem, tot) => {
    $("relax-bar").style.width = `${100 * (1 - rem / tot)}%`;
    $("relax-timer").textContent = `${Math.ceil(rem)} s`;
  });
  scene.stop();
}

function askRating(title) {
  $("rating-block").textContent = title;
  $("rating-row").innerHTML = RATINGS.map((label, i) => `<button data-v="${i + 1}"><b>${i + 1}</b><span>${label}</span></button>`).join("");
  return new Promise((resolve) => {
    const done = (v) => { document.removeEventListener("keydown", onKey); resolve(v); };
    const onKey = (e) => { if (/^[1-5]$/.test(e.key)) done(Number(e.key)); };
    document.addEventListener("keydown", onKey);
    $("rating-row").querySelectorAll("button").forEach((b) => { b.onclick = () => done(Number(b.dataset.v)); });
  });
}

async function runStudy() {
  if (state.busy || !(await cameraReady()) || !recordingChecksPass()) return;
  state.busy = true;
  audio().resume();
  const order = $("study-order").value, seconds = QUICK ? 20 : Number($("study-block").value);
  const blocks = STUDY_ORDERS[order].map((k) => ({ key: k, ...STUDY_BLOCKS[k] }));
  const recording = $("ds-record").checked;
  if (recording) send({ type: "session_start", meta: { ...datasetMeta("study"), study_order: order, block_seconds: seconds } });

  await calibrate();   // same session; the report shows the flip check, with Continue or Redo
  const results = [];
  for (const [i, b] of blocks.entries()) {
    const scores = [];
    const sampler = setInterval(() => { if (state.latest && state.latest.stress != null) scores.push(state.latest.stress); }, 500);
    setPhase(b.phase);
    if (b.key === "relax") {
      studyShow("relax", blocks, i + 1);
      await relaxBlock(seconds);
    } else {
      studyShow("stress", blocks, i + 1);
      await runStressor(b.task, seconds, { easy: !!b.easy, title: b.title, lead: b.lead });
    }
    clearInterval(sampler);
    sfx.done();
    setPhase("study_rating");
    studyShow("rating", blocks, i + 1);
    const rating = await askRating(b.title);
    send({ type: "event", name: "study_rating", data: { block: b.phase, rating } });
    results.push({ block: b.phase, title: b.title, rating, score: mean(scores) });
  }
  setPhase("study_done");
  if (recording) send({ type: "session_end", summary: { kind: "study", order, block_seconds: seconds, blocks: results } });

  studyShow("study", blocks, blocks.length + 1);
  $("study-lead").textContent = `Order ${order}, ${Math.round(seconds / 60 * 10) / 10} min per block.` +
    (recording ? ($("ds-consent").checked ? " Recorded with consent: it will be exported." : " Recorded without consent: it will NOT be exported.") : " Not recorded.");
  $("study-table").innerHTML = "<tr><th>Block</th><th>Your rating (1-5)</th><th>Average live score</th></tr>" +
    results.map((r) => `<tr><td>${r.title}</td><td>${r.rating}</td><td>${r.score == null ? "—" : Math.round(r.score)}</td></tr>`).join("");
  await waitClick($("study-done"));
  setPhase("idle");
  show("setup");
  state.busy = false;
}

// ---------- Reset ----------
async function runReset(condition) {
  const breathing = condition === "breathing";
  show("reset");
  $("reset-title").textContent = breathing ? "Breathe" : "Quiet rest";
  $("reset-lead").textContent = breathing
    ? "Follow the circle: in as it grows, out as it shrinks. Keep your eyes open and watch your stress come down."
    : "Sit quietly with your eyes open and rest until the timer ends.";
  $("breath-wrap").hidden = !breathing;
  $("reset-live").hidden = !breathing;
  $("chart-reset").hidden = !breathing;
  if (!breathing) setPill(false); // the control condition gets no feedback
  setPhase("reset");
  state.resetStart = state.latest ? state.latest.t : 0;
  let inhale = false, nextSwitch = 0;
  await countdown(DUR.reset, (rem, tot) => {
    $("reset-timer").textContent = `${Math.ceil(rem)} s`;
    if (breathing && tot - rem >= nextSwitch) {
      inhale = !inhale;
      nextSwitch += BREATH_HALF_S;
      $("breath").classList.toggle("inhale", inhale);
      $("breath-label").textContent = inhale ? "Breathe in" : "Breathe out";
    }
  });
  $("breath").classList.remove("inhale");
  state.resetStart = null;
  setPill(state.pillVisible);
  sfx.done();
}

// ---------- Results ----------
function phaseStats(phase) {
  const rows = state.history.filter((h) => h.phase === phase);
  const s = rows.map((r) => r.stress).filter((v) => v != null);
  const hr = rows.map((r) => r.hr).filter((v) => v != null);
  return { stress: mean(s), stress_peak: s.length ? Math.max(...s) : null, hr: mean(hr) };
}
function stressBins(items) {
  return STRESS_BINS.map((b) => {
    const sel = items.filter((it) => it.stress != null && it.stress >= b.lo && it.stress < b.hi);
    const correct = sel.filter((it) => it.correct);
    return { ...b, n: sel.length, accuracy: sel.length ? correct.length / sel.length : null,
      rt: mean(correct.map((it) => it.rt)) };
  });
}
function perfInsight(bins) {
  const ok = bins.filter((b) => b.n >= 3);
  if (ok.length < 2) {
    return "Not enough answers at different stress levels yet to compare. Longer rounds or more sessions will fill this in.";
  }
  const lo = ok[0], hi = ok[ok.length - 1];
  const accDiff = Math.round((lo.accuracy - hi.accuracy) * 100);
  const rtDiff = lo.rt != null && hi.rt != null ? hi.rt - lo.rt : null;
  const acc = accDiff > 0
    ? `Under ${hi.label.toLowerCase()} your accuracy dropped by ${accDiff} percentage points`
    : accDiff < 0 ? `Under ${hi.label.toLowerCase()} your accuracy rose by ${-accDiff} percentage points` : `Your accuracy held steady under ${hi.label.toLowerCase()}`;
  const rt = rtDiff == null ? "" : Math.abs(rtDiff) < 0.05 ? ", at the same speed"
    : rtDiff > 0 ? ` and ${rtDiff.toFixed(2)} s slower per answer` : ` and ${(-rtDiff).toFixed(2)} s faster per answer`;
  return `${acc}${rt} than when ${lo.label.toLowerCase()}.`;
}
function headline(s, p1, p2, r1, r2) {
  const after = s.condition === "breathing" ? "After the breathing reset" : "After quiet rest";
  const parts = [];
  if (p1.stress != null && p2.stress != null) {
    const d = Math.round(p2.stress - p1.stress);
    parts.push(d < 0 ? `your stress dropped ${-d} points` : d > 0 ? `your stress rose ${d} points` : "your stress held steady");
  }
  if (r1.correct != null && r2.correct != null) {
    const d = r2.correct - r1.correct;
    parts.push(d > 0 ? `you solved ${d} more problem${d > 1 ? "s" : ""}` : d < 0 ? `you solved ${-d} fewer` : "you solved the same number of problems");
  }
  return parts.length ? `${after}, ${parts.join(" and ")}.` : "Session complete.";
}
function statCard(label, a, b, { decimals = 0, suffix = "", lowerIsBetter = false } = {}) {
  const el = document.createElement("div");
  el.className = "stat";
  let delta = "";
  if (a != null && b != null) {
    const d = b - a;
    const better = lowerIsBetter ? d < 0 : d > 0;
    const cls = Math.abs(d) < 1e-9 ? "" : better ? "better" : "worse";
    delta = `<span class="delta ${cls}">${d > 0 ? "+" : ""}${d.toFixed(decimals)}${suffix}</span>`;
  }
  el.innerHTML = `<div class="label">${label}</div><div class="value">—</div>
    <div class="from">Round 1: ${a == null ? "—" : a.toFixed(decimals) + suffix} ${delta}</div>`;
  animateNumber(el.querySelector(".value"), b, decimals, suffix);
  return el;
}
function showResults() {
  show("results");
  const s = state.session;
  const r1 = s.rounds.round1 || {}, r2 = s.rounds.round2 || {};
  const p1 = phaseStats("round1"), p2 = phaseStats("round2"), pr = phaseStats("reset");
  const pct = (v) => (v == null ? null : v * 100);

  $("results-eyebrow").textContent = `${s.participant} · ${s.condition === "breathing" ? "Guided breathing" : "Quiet rest (control)"}`;
  $("results-headline").textContent = headline(s, p1, p2, r1, r2);
  const cards = $("stat-cards");
  cards.innerHTML = "";
  cards.append(
    statCard("Average stress, round 2", p1.stress, p2.stress, { lowerIsBetter: true }),
    statCard("Correct answers, round 2", r1.correct, r2.correct),
    statCard("Accuracy, round 2", pct(r1.accuracy), pct(r2.accuracy), { suffix: "%" }),
    statCard("Response time, round 2", r1.mean_rt, r2.mean_rt, { decimals: 2, suffix: " s", lowerIsBetter: true }),
  );

  const items = [...(r1.items || []), ...(r2.items || [])];
  const bins = stressBins(items);
  $("perf-insight").textContent = perfInsight(bins);
  drawBars($("chart-perf"), bins);

  const pts = state.history.map((h) => ({ t: h.t, v: h.stress }));
  const bands = [];
  for (const h of state.history) {
    if (!PHASE_COLORS[h.phase]) continue;
    const last = bands[bands.length - 1];
    if (last && last.phase === h.phase) last.t1 = h.t;
    else bands.push({ phase: h.phase, t0: h.t, t1: h.t, color: PHASE_COLORS[h.phase] });
  }
  drawChart($("chart-results"), pts, { bands });
  $("results-legend").innerHTML = Object.entries(PHASE_COLORS)
    .map(([p, c]) => `<span><i style="background:${c}"></i>${PHASE_NAMES[p]}</span>`).join("");
  $("results-file").textContent = s.dir ? `Saved to ${s.dir}` : "";

  const strip = (r) => { if (!r) return r; const { items: _, ...rest } = r; return rest; };
  send({ type: "session_end", summary: {
    participant: s.participant, condition: s.condition, started_at: s.startedAt, quick: QUICK,
    rounds: { warmup: strip(s.rounds.warmup), round1: strip(r1), round2: strip(r2) },
    phases: { round1: p1, reset: pr, round2: p2, calib_rest: phaseStats("calib_rest"), calib_stress: phaseStats("calib_stress") },
    stress_vs_performance: bins.map(({ key, n, accuracy, rt }) => ({ bin: key, n, accuracy, rt })),
    baseline: state.baseline } });
  s.recording = false;
}

// ---------- Session ----------
async function runSession() {
  if (state.busy || !(await cameraReady()) || !recordingChecksPass()) return;
  state.busy = true;
  audio().resume();
  const participant = $("participant").value.trim() || $("ds-participant").value.trim() || "anon";
  const condition = $("condition").value;
  state.session = { participant, condition, rounds: {}, startedAt: new Date().toISOString(), recording: true, dir: null };
  state.history = [];
  send({ type: "session_start", meta: { ...datasetMeta("session"), participant, condition, durations: DUR, quick: QUICK,
    reused_calibration: !!(state.baseline && state.baseline.ok) } });
  // Reuse the calibration done from the home screen; otherwise calibrate now.
  if (!(state.baseline && state.baseline.ok)) await calibrate();
  await runTask("warmup");
  await runTask("round1");
  await runReset(condition);
  await runTask("round2");
  setPhase("results");
  showResults();
  state.busy = false;
}

// ---------- Sea widget ----------
async function openWidget() {
  if (state.pip && !state.pip.closed) { state.pip.focus && state.pip.focus(); return; }
  if ("documentPictureInPicture" in window) {
    try {
      // Always-on-top mini window (Chrome / Edge). It shares this page's live data.
      const pip = await window.documentPictureInPicture.requestWindow({ width: 360, height: 240 });
      pip.document.title = "Clutch · Sea";
      pip.document.body.style.cssText = "margin:0;background:#1a2735;overflow:hidden";
      const canvas = pip.document.createElement("canvas");
      canvas.style.cssText = "display:block;width:100vw;height:100vh";
      pip.document.body.append(canvas);
      const scene = new BoatScene(canvas);
      scene.start(pip);
      if (state.latest) scene.setStress(state.latest.stress, state.latest.calibrated ? "Waiting for signal" : "Calibrate first");
      state.scenes.pip = scene;
      state.pip = pip;
      pip.addEventListener("pagehide", () => { scene.stop(); state.scenes.pip = null; state.pip = null; });
      return;
    } catch (e) {
      console.warn("Picture-in-Picture unavailable, opening a popup instead", e);
    }
  }
  state.pip = window.open("widget.html", "clutch-sea", "width=380,height=260");
}

// ---------- Wiring ----------
buildGauges();
buildHead();
state.scenes.preview = new BoatScene($("sea-preview"), { breathing: true, canPrompt: () => !state.busy });
state.scenes.preview.start();

$("btn-start").onclick = () => runSession();
$("btn-present").onclick = () => togglePresent(true);
$("present-close").onclick = () => togglePresent(false);
$("protocol").onchange = () => { if (state.latest) updateSteps(state.latest); };
// Face camera (MediaPipe runs in face-sensor.js; only the numbers are sent to the server)
function renderFaceBars(f) {
  $("face-bars").innerHTML = FACE_KEYS.map((k) => {
    const v = f ? f[k] : null;
    const pct = v == null ? 0 : k === "blink_rate" ? Math.min(100, (v / 40) * 100)
      : k === "head_motion" ? Math.min(100, (v / 0.3) * 100) : Math.min(100, v * 100);
    const txt = v == null ? "—" : k === "blink_rate" ? v.toFixed(0) : k === "head_motion" ? v.toFixed(2) : Math.round(v * 100);
    return `<div class="face-bar"><span>${FACE_LABELS[k]}</span><div class="track"><div class="fill" style="width:${pct}%"></div></div><b>${txt}</b></div>`;
  }).join("");
}
renderFaceBars(null);
$("cam-toggle").onclick = async () => {
  if (!window.ClutchFace) { $("cam-status").textContent = "Face module still loading, try again in a second…"; return; }
  if (ClutchFace.isRunning()) {
    if (state.busy) { $("cam-status").textContent = "Keep the camera on until the calibration or session ends"; return; }
    ClutchFace.stop();
    $("cam-toggle").textContent = "Start camera"; $("cam-off").hidden = false;
    $("cam-status").textContent = "Camera off"; renderFaceBars(null);
    return;
  }
  $("cam-toggle").disabled = true;
  try {
    await ClutchFace.start($("cam-video"), $("cam-overlay"),
      (f) => { if (f) send({ type: "face_features", features: f }); renderFaceBars(f); },
      (s) => { $("cam-status").textContent = s; });
    $("cam-off").hidden = true; $("cam-toggle").textContent = "Stop camera";
  } catch (e) {
    $("cam-status").textContent = `Camera error: ${e.message || e}`;
  }
  $("cam-toggle").disabled = false;
};

$("hs-connect").onclick = () => send({ type: "headset_connect", name: $("hs-name").value.trim() });
$("hs-disconnect").onclick = () => send({ type: "headset_disconnect" });
$("hs-name").onkeydown = (e) => { if (e.key === "Enter") $("hs-connect").click(); };
$("btn-desk").onclick = async () => {
  const r = await fetch("/api/desk-boat", { method: "POST" }), j = await r.json().catch(() => ({}));
  const msg = $("desk-msg");
  msg.hidden = false;
  msg.textContent = j.error || (j.started ? "The boat is on your desktop (bottom of the screen)." : "The boat is already on your desktop.");
};
$("watch-connect").onclick = () => send({ type: "watch_connect", name: $("watch-name").value.trim() });
$("watch-disconnect").onclick = () => send({ type: "watch_disconnect" });
$("watch-name").onkeydown = (e) => { if (e.key === "Enter") $("watch-connect").click(); };
$("btn-calibrate").onclick = () => calibrateOnly();
$("btn-study").onclick = () => runStudy();
$("btn-widget").onclick = () => openWidget();
$("btn-widget-2").onclick = () => openWidget();
$("btn-new").onclick = () => { state.session = null; setPhase("idle"); show("setup"); };
$("durations").textContent =
  `Solve mental math under time pressure, reset, then solve again. Uses your calibration if you already did one ` +
  `(about ${Math.round((DUR.warmup + 2 * DUR.round + DUR.reset) / 60 + 0.5)} min), otherwise calibrates first ` +
  `(+3 to 9 min depending on the protocol).` +
  (QUICK ? " (quick test mode)" : "");

$("mock-stress").oninput = (e) => {
  $("mock-stress-val").textContent = e.target.value;
  send({ type: "mock_stress", value: Number(e.target.value) });
};
$("mock-auto").onclick = () => { $("mock-stress-val").textContent = "auto"; send({ type: "mock_stress", value: null }); };

document.addEventListener("keydown", (e) => {
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName);
  if (e.key === "Escape" && presentOpen()) { togglePresent(false); return; }
  if (!typing && !state.busy && e.key.toLowerCase() === "p") { togglePresent(); return; }
  if (e.key === "Escape" && state.busy) {
    if (confirm("Abort?")) {
      if (state.session && state.session.recording) send({ type: "session_end", summary: { aborted: true } });
      setPhase("idle");
      setTimeout(() => location.reload(), 300);
    }
  } else if (!typing && e.key.toLowerCase() === "s") {
    state.pillVisible = !state.pillVisible; setPill(state.pillVisible);
  } else if (!typing && e.key.toLowerCase() === "d") {
    $("debug").hidden = !$("debug").hidden;
  }
});

window.addEventListener("resize", () => { if (isActive("setup")) drawSetupChart(); });
connect();
