"use strict";

// ---------- Settings ----------
const qs = new URLSearchParams(location.search);
const QUICK = qs.has("quick"); // ?quick -> short durations for testing
const DUR = QUICK
  ? { calRest: 12, calStress: 20, warmup: 10, round: 25, reset: 20 }
  : { calRest: 60, calStress: 90, warmup: 30, round: 90, reset: 90 };
const PER_PROBLEM_S = 6;  // time limit per arithmetic problem under pressure
const BREATH_HALF_S = 5;  // 5 s in + 5 s out = 6 breaths per minute
const SUB_START = 1022, SUB_STEP = 13; // serial subtraction stressor (Trier Social Stress Test style)
const PHASE_COLORS = { calib_rest: "#8a96a3", calib_stress: "#e08a3c", warmup: "#b9a77e", round1: "#d14b3b", reset: "#1f9d6b", round2: "#2563a6" };
const PHASE_NAMES = {
  idle: "", calib_rest: "Calibration: relax", calib_stress: "Calibration: stress test", calibration_check: "Calibration",
  ready: "Get ready", warmup: "Warm-up", round1: "Round 1", reset: "Reset", round2: "Round 2", results: "Results",
};
// Score scale: 0 = your calm (rest calibration), 100 = your stress (stress-test calibration)
const STRESS_BINS = [
  { key: "calm", label: "Calm", range: "< 35", lo: -Infinity, hi: 35 },
  { key: "mid", label: "Moderate", range: "35–65", lo: 35, hi: 65 },
  { key: "high", label: "High stress", range: "65+", lo: 65, hi: Infinity },
];
const SIGNAL_NAMES = { eeg: "Brain", hr: "Heart" };

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
  ws.onopen = () => { $("conn").classList.add("on"); send({ type: "phase", name: state.phase }); };
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
    state.live.push({ t: m.t, stress: m.stress, ratio: m.eeg ? m.eeg.ratio_log : null });
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

// ---------- Live widgets ----------
function updateLive(m, hr) {
  updateHeadset(m.source);
  const s = m.stress == null ? null : Math.round(m.stress);
  document.querySelectorAll('[data-live="stress"]').forEach((el) => { el.textContent = s ?? "—"; el.style.color = stressColor(s); });
  $("pill-dot").style.background = stressColor(s);
  updateGauges(s);
  document.querySelectorAll('[data-live="hr"]').forEach((el) => { el.textContent = hr != null ? `${Math.round(hr)} bpm` : "— bpm"; });
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
  $("signal-chips").innerHTML = !m.calibrated || !cal ? "" : Object.entries(SIGNAL_NAMES).map(([k, label]) => {
    const used = cal.mode === "anchored" ? cal.used.includes(k) : cal.signals[k] && cal.signals[k].rest != null;
    const live = (m.used || []).includes(k);
    const cls = !used ? "off" : live ? "on" : "wait";
    const tip = !used ? `${label}: not used (${cal.signals[k] ? cal.signals[k].note : "no data"} during calibration)`
      : live ? `${label}: feeding the score now` : `${label}: used, but no clean signal right now`;
    return `<span class="chip ${cls}" title="${tip}">${label} ${!used ? "✗" : live ? "✓" : "…"}</span>`;
  }).join("");

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
function drawSetupChart() {
  const calibrated = state.latest && state.latest.calibrated;
  const pts = state.live.map((p) => ({ t: p.t, v: calibrated ? p.stress : p.ratio }));
  const now = state.latest ? state.latest.t : 0;
  drawChart($("chart-setup"), pts, calibrated
    ? { tMin: now - 120, tMax: now }
    : { tMin: now - 120, tMax: now, auto: true, label: "alpha/beta (log), last 2 min" });
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

  // Line coloured by stress level (calm green -> tense red), broken where there is no value
  ctx.lineWidth = 2.5; ctx.lineCap = "round";
  let prev = null;
  for (const p of pts) {
    if (p.v == null || p.t < tMin) { prev = null; continue; }
    const pt = { x: X(p.t), y: Y(clamp(p.v, yMin, yMax)), v: p.v };
    if (prev) {
      ctx.strokeStyle = o.auto ? cssVar("--accent") : stressColor((prev.v + pt.v) / 2);
      ctx.beginPath(); ctx.moveTo(prev.x, prev.y); ctx.lineTo(pt.x, pt.y); ctx.stroke();
    }
    prev = pt;
  }
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
function calibStep(n) {
  [1, 2, 3].forEach((i) => {
    $(`step-${i}`).className = "step" + (i === n ? " active" : i < n ? " done" : "");
  });
  $("calib-rest").hidden = n !== 1;
  $("calib-stress").hidden = n !== 2;
  $("calib-report").hidden = n !== 3;
}

// Step 2: serial subtraction under time pressure. A mistake restarts from 1022.
async function serialSubtraction(seconds) {
  const input = $("sub-answer"), fb = $("sub-feedback");
  let expected = SUB_START - SUB_STEP, streak = 0, best = 0, shownAt = performance.now();
  $("sub-prompt").textContent = `${SUB_START} − ${SUB_STEP} = ?`;
  fb.textContent = ""; input.value = ""; input.disabled = false; input.focus();
  input.onkeydown = (e) => {
    if (e.key !== "Enter" || input.value.trim() === "") return;
    const given = Number(input.value.trim());
    const ok = given === expected;
    send({ type: "event", name: "calib_answer", data: { expected, given, correct: ok,
      rt: +((performance.now() - shownAt) / 1000).toFixed(3), stress: state.latest ? state.latest.stress : null } });
    input.value = ""; shownAt = performance.now();
    if (ok) {
      streak++; best = Math.max(best, streak); expected -= SUB_STEP;
      fb.className = "feedback ok"; fb.textContent = `✓ ${streak} in a row`;
      $("sub-prompt").textContent = `… − ${SUB_STEP} = ?`; // no running total shown: you must keep track
      sfx.right();
    } else {
      streak = 0; expected = SUB_START - SUB_STEP;
      fb.className = "feedback ko"; fb.textContent = `Wrong. Start again from ${SUB_START}.`;
      $("sub-prompt").textContent = `${SUB_START} − ${SUB_STEP} = ?`;
      sfx.wrong();
      document.body.classList.remove("flash-bad"); void document.body.offsetWidth; document.body.classList.add("flash-bad");
    }
  };
  let lastWhole = null;
  await countdown(seconds, (rem) => {
    const whole = Math.ceil(rem);
    $("sub-timer").textContent = whole;
    $("sub-timer").classList.toggle("urgent", rem <= 10);
    if (rem > 0 && whole !== lastWhole) { lastWhole = whole; rem <= 10 ? sfx.tickUrgent() : sfx.tick(); }
  });
  input.onkeydown = null; input.disabled = true;
  return best;
}

function renderCalibrationReport(cal) {
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
    } else if (s.rest != null) {
      vals = `${fmt(s.rest)}${unit} at rest`;
    }
    return `<div class="sig-card"><h3>${title}</h3><div class="muted small">rest → stress test</div>
      <div class="vals">${vals}</div><div class="muted small">${change}</div>
      <p class="verdict ${used ? "used" : "unused"}">${used ? "✓ Used" : "✗ Not used"}: ${s.note || "no data"}
      ${s.effect != null ? `<span class="muted">(effect size ${s.effect.toFixed(2)})</span>` : ""}</p></div>`;
  };
  $("report-signals").innerHTML =
    card("eeg", "Brain · alpha/beta ratio", (v) => Math.pow(10, v).toFixed(2), "") +
    card("hr", "Heart rate", (v) => Math.round(v), " bpm");

  let lead, summary;
  if (!cal || !cal.ok) {
    lead = "Calibration failed: no clean signal from the headset.";
    summary = "Check that the sensors are green (bar flat on the forehead, rubber tips on the skin behind the ears), then redo the calibration.";
  } else if (cal.confidence === "low") {
    lead = "Neither signal reacted clearly to the stress test.";
    summary = "The score will only show how far you are from your calm state, so treat it as rough. Better contact or a harder test usually fixes this: redo the calibration if you can.";
  } else {
    const names = cal.used.map((k) => SIGNAL_NAMES[k].toLowerCase()).join(" and ");
    lead = `Your ${names} reacted to stress: the score is personalised to you.`;
    summary = "From now on, 0 means as calm as you were at rest and 100 means as stressed as during the test. Signals that did not react are left out.";
  }
  $("report-lead").textContent = lead;
  $("report-summary").textContent = summary;
  $("calib-continue").disabled = !(cal && cal.ok);
}

async function calibrate() {
  for (;;) {
    show("calibration");
    calibStep(1);
    setPhase("calib_rest");
    await countdown(DUR.calRest, (rem, tot) => {
      $("calib-bar").style.width = `${100 * (1 - rem / tot)}%`;
      $("calib-timer").textContent = `${Math.ceil(rem)} s`;
    });
    sfx.done();
    calibStep(2);
    setPhase("calib_stress");
    const best = await serialSubtraction(DUR.calStress);
    const waiting = waitBaseline(4000);
    setPhase("calibration_check");
    const cal = await waiting;
    send({ type: "event", name: "calibration", data: { ...(cal || {}), best_streak: best } });
    calibStep(3);
    renderCalibrationReport(cal);
    sfx.done();
    const choice = await new Promise((r) => {
      $("calib-continue").onclick = () => r("continue");
      $("calib-retry").onclick = () => r("retry");
    });
    if (choice === "continue") return true;
  }
}
async function calibrateOnly() {
  if (state.busy) return;
  state.busy = true;
  audio().resume();
  await calibrate();
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
  if (state.busy) return;
  state.busy = true;
  audio().resume();
  const participant = $("participant").value.trim() || "anon";
  const condition = $("condition").value;
  state.session = { participant, condition, rounds: {}, startedAt: new Date().toISOString(), recording: true, dir: null };
  state.history = [];
  send({ type: "session_start", meta: { participant, condition, durations: DUR, quick: QUICK, started_at: state.session.startedAt,
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
state.scenes.preview = new BoatScene($("sea-preview"));
state.scenes.preview.start();

$("btn-start").onclick = () => runSession();
$("hs-connect").onclick = () => send({ type: "headset_connect", name: $("hs-name").value.trim() });
$("hs-disconnect").onclick = () => send({ type: "headset_disconnect" });
$("hs-name").onkeydown = (e) => { if (e.key === "Enter") $("hs-connect").click(); };
$("btn-calibrate").onclick = () => calibrateOnly();
$("btn-widget").onclick = () => openWidget();
$("btn-widget-2").onclick = () => openWidget();
$("btn-new").onclick = () => { state.session = null; setPhase("idle"); show("setup"); };
$("durations").textContent =
  `Solve mental math under time pressure, reset, then solve again. Uses your calibration if you already did one ` +
  `(about ${Math.round((DUR.warmup + 2 * DUR.round + DUR.reset) / 60 + 0.5)} min), otherwise calibrates first ` +
  `(+${Math.round((DUR.calRest + DUR.calStress) / 60 * 2) / 2} min).` +
  (QUICK ? " (quick test mode)" : "");

$("mock-stress").oninput = (e) => {
  $("mock-stress-val").textContent = e.target.value;
  send({ type: "mock_stress", value: Number(e.target.value) });
};
$("mock-auto").onclick = () => { $("mock-stress-val").textContent = "auto"; send({ type: "mock_stress", value: null }); };

document.addEventListener("keydown", (e) => {
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName);
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
