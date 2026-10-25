// Face tracker page: webcam -> MediaPipe Face Landmarker -> facial features (face-features.js),
// drawn live for debugging and sent to the Python server for logging next to the EEG targets.
import { DrawingUtils, FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const TICK_MS = 500;          // same tick as the EEG engine
const SAMPLE_MS = 100;        // chart resolution
const CHART_SPAN_MS = 60000;
const MAX_EVENTS_SHOWN = 100;

const { FaceFeatureExtractor, COMPOSITES, FEATURE_COLUMNS, DEFAULTS, headPose } = window.FaceFeatures;
const REGION_OF = {
  brow_furrow: "brow", brow_inner_up: "brow", brow_outer_up: "brow",
  eye_wide: "eyes", eye_squint: "eyes", nose_wrinkle: "nose", jaw_open: "jaw",
  smile: "mouth", dimple: "mouth", frown: "mouth", chin_raise: "mouth", lip_stretch: "mouth", lip_press: "mouth", lip_suck: "mouth",
};
const REGIONS = [
  { key: "brow", label: "Brows" }, { key: "eyes", label: "Eyes" }, { key: "nose", label: "Nose" },
  { key: "mouth", label: "Mouth" }, { key: "jaw", label: "Jaw" },
];

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => (v === null || v === undefined || !isFinite(v) ? "—" : Number(v).toFixed(d));
const cssVar = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();

const state = {
  landmarker: null, stream: null, running: false, lastVideoTime: -1,
  fx: new FaceFeatureExtractor(), t0: null, info: null, blendNames: null,
  frameTimes: [], samples: [], events: [], lastSample: 0, lastRow: null,
  ws: null, metrics: null, recording: false, frameBuf: [], hover: null, chartDirty: true,
};

// ---------- Server connection (receives EEG metrics, logs face features) ----------
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  state.ws = ws;
  ws.onopen = () => $("conn").classList.add("on");
  ws.onclose = () => {
    $("conn").classList.remove("on");
    setRecording(null);
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "metrics") {
      m._rx = performance.now();
      state.metrics = m;
      $("eeg-stress").textContent = m.stress === null ? (m.calibrated ? "—" : "not calibrated") : fmt(m.stress, 0);
    } else if (m.type === "face_recording") {
      setRecording(m.dir, m.rows);
    }
  };
}
const send = (msg) => state.ws && state.ws.readyState === 1 && state.ws.send(JSON.stringify(msg));

// ---------- Camera and model ----------
async function loadModel() {
  setStatus("Loading model…", "fair");
  // MediaPipe needs WebGL even on its CPU delegate
  if (!document.createElement("canvas").getContext("webgl2")) {
    throw new Error("WebGL is off in this browser, and MediaPipe needs it. In Chrome: Settings → System → turn on "
      + "\"Use graphics acceleration when available\", relaunch, then reload this page (check chrome://gpu shows WebGL: Hardware accelerated)");
  }
  const files = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO", numFaces: 1,
    outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
  });
  try {
    return await FaceLandmarker.createFromOptions(files, opts("GPU"));
  } catch (err) {
    console.warn("GPU delegate failed, using CPU", err);
    return await FaceLandmarker.createFromOptions(files, opts("CPU"));
  }
}

async function startCamera(deviceId) {
  stopCamera();
  $("btn-camera").disabled = true;
  try {
    if (!state.landmarker) state.landmarker = await loadModel();
    setStatus("Waiting for camera…", "fair");
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, ...(deviceId ? { deviceId: { exact: deviceId } } : {}) },
      audio: false,
    });
    const video = $("video");
    video.srcObject = state.stream;
    await video.play();
    await listCameras(state.stream.getVideoTracks()[0].getSettings().deviceId);
    $("stage-empty").hidden = true;
    state.running = true;
    state.lastVideoTime = -1;
    if (state.t0 === null) state.t0 = performance.now();
    $("btn-camera").textContent = "Stop camera";
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    setStatus("Camera error", "bad");
    const camera = err.name === "NotAllowedError" || err.name === "NotReadableError" || err.name === "NotFoundError";
    $("stage-msg").textContent = camera
      ? `Could not start the camera: ${err.message}. Check camera permission and that no other app is using it.`
      : `Could not start: ${err.message || err}.`;
    $("stage-empty").hidden = false;
  } finally {
    $("btn-camera").disabled = false;
  }
}

function stopCamera() {
  state.running = false;
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  $("btn-camera").textContent = "Start camera";
  $("stage-empty").hidden = false;
  $("stage-msg").textContent = "Camera stopped.";
  setStatus("Camera off", null);
  const ov = $("overlay");
  ov.getContext("2d").clearRect(0, 0, ov.width, ov.height);
}

async function listCameras(activeId) {
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const sel = $("camera-select");
  sel.innerHTML = cams.map((c, i) => `<option value="${c.deviceId}">${c.label || "Camera " + (i + 1)}</option>`).join("");
  sel.value = activeId || "";
  sel.hidden = cams.length < 2;
}

function setStatus(text, level) {
  $("face-status").textContent = text;
  $("face-dot").style.background = level ? `var(--${level === "good" ? "good" : level === "fair" ? "fair" : "bad"})` : "";
}

// ---------- Per-frame processing ----------
function loop() {
  if (!state.running) return;
  const video = $("video");
  if (video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    const t = performance.now();
    try {
      onResult(t, state.landmarker.detectForVideo(video, t));
    } catch (err) {
      console.error(err);
      stopCamera();
      setStatus("Tracking error", "bad");
      $("stage-msg").textContent = `Tracking stopped: ${err.message || err}`;
      return;
    }
  }
  requestAnimationFrame(loop);
}

function onResult(t, res) {
  const cats = res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories;
  const matrix = res.facialTransformationMatrixes && res.facialTransformationMatrixes[0] && res.facialTransformationMatrixes[0].data;
  const landmarks = res.faceLandmarks && res.faceLandmarks[0];
  if (cats && !state.blendNames) {
    state.blendNames = cats.map((c) => c.categoryName);
    buildRawGrid();
    $("btn-record").disabled = false;
  }
  const scores = cats ? Object.fromEntries(cats.map((c) => [c.categoryName, c.score])) : null;
  const info = state.fx.addFrame(t, scores, matrix ? Array.from(matrix) : null);
  state.info = info;

  if (state.recording) {
    const p = matrix ? headPose(matrix) : null;
    state.frameBuf.push({
      t: Math.round(t * 10) / 10,
      s: cats ? cats.map((c) => Math.round(c.score * 1e4) / 1e4) : null,
      p: p ? [p.yaw, p.pitch, p.roll, p.tx, p.ty, p.tz].map((v) => Math.round(v * 100) / 100) : null,
    });
  }

  state.frameTimes.push(t);
  while (state.frameTimes[0] < t - 1000) state.frameTimes.shift();
  $("fps").textContent = state.frameTimes.length;
  setStatus(info.present ? "Tracking" : "No face", info.present ? "good" : "bad");

  drawOverlay(landmarks, info);
  updateLive(info, cats);
  for (const ev of info.newEvents) addEvent(ev);

  if (t - state.lastSample >= SAMPLE_MS) {
    state.lastSample = t;
    const m = state.metrics;
    state.samples.push({
      t, present: info.present,
      tension: info.present ? info.tension : null,
      activity: info.present ? info.activity : null,
      headSpeed: info.present ? info.headSpeed : null,
      eyeBlink: info.present ? info.eyeBlink : null,
      stress: m && m.stress !== null && performance.now() - (m._rx || 0) < 3000 ? m.stress : null,
    });
    while (state.samples.length && state.samples[0].t < t - CHART_SPAN_MS) state.samples.shift();
    state.chartDirty = true;
  }
}

// ---------- Overlay ----------
const REGION_CONNECTORS = () => [
  { region: null, sets: [FaceLandmarker.FACE_LANDMARKS_FACE_OVAL] },
  { region: "brow", sets: [FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW] },
  { region: "eyes", sets: [FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
    FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS] },
  { region: "mouth", sets: [FaceLandmarker.FACE_LANDMARKS_LIPS] },
];

function drawOverlay(landmarks, info) {
  const video = $("video"), canvas = $("overlay");
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const mode = $("overlay-mode").value;
  if (!landmarks || mode === "off") return;
  const du = new DrawingUtils(ctx);
  const scale = canvas.width / 640;
  if (mode === "mesh") {
    du.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "rgba(255,255,255,0.28)", lineWidth: 0.6 * scale });
  }
  if (mode === "points") {
    du.drawLandmarks(landmarks, { color: "rgba(255,255,255,0.8)", radius: 0.9 * scale, lineWidth: 0 });
    return;
  }
  const activeRegions = new Set(Object.entries(info.live || {}).filter(([, v]) => v.active).map(([k]) => REGION_OF[k]));
  for (const { region, sets } of REGION_CONNECTORS()) {
    const color = region ? cssVar(`--r-${region}`) : "rgba(255,255,255,0.55)";
    const width = (activeRegions.has(region) ? 3.5 : 1.6) * scale;
    for (const s of sets) du.drawConnectors(landmarks, s, { color, lineWidth: width });
  }
}

// ---------- Live panels ----------
function buildAuList() {
  $("au-list").innerHTML = COMPOSITES.map((c) => `
    <div class="au" id="au-${c.key}" style="color: var(--r-${REGION_OF[c.key]})">
      <span class="au-name"><span class="swatch" style="background: var(--r-${REGION_OF[c.key]})"></span>
        <span style="color: var(--text)">${c.label}</span> <span class="code">${c.au}</span></span>
      <span class="au-track"><span class="au-fill" style="background: var(--r-${REGION_OF[c.key]})"></span>
        <span class="au-base"></span><span class="au-thr"></span></span>
      <span class="au-val">—</span>
    </div>`).join("");
  state.auEls = Object.fromEntries(COMPOSITES.map((c) => {
    const el = $(`au-${c.key}`);
    return [c.key, { el, fill: el.querySelector(".au-fill"), base: el.querySelector(".au-base"), thr: el.querySelector(".au-thr"), val: el.querySelector(".au-val") }];
  }));
  $("event-legend").innerHTML = REGIONS.map((r) => `<span><span class="swatch" style="background: var(--r-${r.key})"></span>${r.label}</span>`).join("")
    + `<span><span class="swatch" style="background: var(--muted)"></span>Blink</span>`;
}

function buildRawGrid() {
  $("raw-grid").innerHTML = state.blendNames.map((n, i) => `
    <div class="raw-item"><span title="${n}">${n}</span>
      <span class="au-track"><span class="au-fill" id="raw-${i}"></span></span><span id="rawv-${i}">—</span></div>`).join("");
  state.rawEls = state.blendNames.map((_, i) => [$(`raw-${i}`), $(`rawv-${i}`)]);
}

const pct = (v) => `${Math.max(0, Math.min(1, v)) * 100}%`;

function updateLive(info, cats) {
  $("warm-note").hidden = !!info.warm;
  if (!info.present) {
    $("pose").textContent = $("head-speed").textContent = $("blink-now").textContent = "—";
    return;
  }
  const p = info.pose;
  $("pose").textContent = p ? `${fmt(p.yaw, 0)}° / ${fmt(p.pitch, 0)}° / ${fmt(p.roll, 0)}°` : "—";
  $("head-speed").textContent = `${fmt(info.headSpeed, 0)} °/s`;
  $("blink-now").textContent = info.blinking ? "closed" : fmt(info.eyeBlink, 2);
  for (const c of COMPOSITES) {
    const l = info.live[c.key], e = state.auEls[c.key];
    e.fill.style.width = pct(l.value);
    e.base.style.left = pct(l.base);
    e.thr.style.left = pct(l.thr);
    e.val.textContent = fmt(l.value, 2);
    e.el.classList.toggle("active", l.active);
  }
  if (cats && state.rawEls && $("raw-grid").closest("details").open) {
    cats.forEach((c, i) => {
      const [bar, val] = state.rawEls[i];
      bar.style.width = pct(c.score);
      val.textContent = c.score.toFixed(2);
    });
  }
}

function addEvent(ev) {
  state.events.unshift(ev);
  if (state.events.length > MAX_EVENTS_SHOWN) state.events.pop();
  const label = COMPOSITES.find((c) => c.key === ev.key);
  const rows = state.events.map((e) => {
    const c = COMPOSITES.find((x) => x.key === e.key);
    return `<tr class="${e.kind}"><td>${fmt((e.t - state.t0) / 1000, 1)} s</td>
      <td><span class="au-name"><span class="swatch" style="background: var(--r-${REGION_OF[e.key]})"></span>${c.label} <span class="code">${c.au}</span></span></td>
      <td>${e.kind}</td><td>${Math.round(e.dur)} ms</td><td>+${fmt(e.peak, 2)}</td></tr>`;
  });
  $("event-rows").innerHTML = rows.join("");
  if (ev.kind === "micro" && label) {
    const el = state.auEls[ev.key].el;
    el.classList.add("flash");
    setTimeout(() => el.classList.remove("flash"), 400);
  }
}

// ---------- Tick: one training row every 0.5 s ----------
function tick() {
  if (!state.running && !state.frameBuf.length) return;
  const row = state.fx.tick(performance.now());
  state.lastRow = row;
  renderRow(row);
  $("k-tension").textContent = fmt(row.tension, 2);
  $("k-micro").textContent = row.micro_rate_min === null ? "…" : fmt(row.micro_rate_min, 1);
  $("k-blink").textContent = row.blink_rate_min === null ? "…" : fmt(row.blink_rate_min, 0);
  $("k-activity").textContent = fmt(row.activity, 2);
  if (state.recording) {
    send({ type: "face", sent: performance.now(), row, frames: state.frameBuf });
    state.frameBuf = [];
  }
}

function renderRow(row) {
  const table = $("row-table");
  if (!state.rowCells) {
    table.innerHTML = FEATURE_COLUMNS.map((c) => `<tr><td>${c}</td><td></td></tr>`).join("");
    state.rowCells = Array.from(table.querySelectorAll("tr td:last-child"));
  }
  FEATURE_COLUMNS.forEach((c, i) => {
    const v = row[c];
    state.rowCells[i].textContent = v === null ? "null" : v;
    state.rowCells[i].classList.toggle("null", v === null);
  });
}

// ---------- Recording ----------
function toggleRecording() {
  if (state.recording) {
    send({ type: "face_stop" });
    return;
  }
  state.frameBuf = [];
  send({
    type: "face_start",
    meta: {
      participant: $("participant").value.trim() || "anon",
      started: new Date().toISOString(),
      tick_ms: TICK_MS, detector: state.fx.opts, model: MODEL_URL, user_agent: navigator.userAgent,
    },
    feature_cols: FEATURE_COLUMNS, blendshape_cols: state.blendNames,
  });
}

function setRecording(dir, rows) {
  state.recording = !!dir;
  if (!dir) state.frameBuf = [];
  $("rec-pill").hidden = !dir;
  $("rec-rows").textContent = rows || 0;
  $("btn-record").textContent = dir ? "Stop recording" : "Start recording";
  $("btn-record").classList.toggle("primary", !dir);
  if (dir) $("rec-path").textContent = `Recording to ${dir} · ${rows || 0} rows`;
}

// ---------- Strip chart: one measure per strip, shared time axis ----------
const STRIPS = [
  { key: "tension", label: "Tension index", unit: "", d: 2, min: 0, max: (v) => Math.max(0.2, v) },
  { key: "activity", label: "Facial activity", unit: "/s", d: 2, min: 0, max: (v) => Math.max(0.5, v) },
  { key: "headSpeed", label: "Head speed", unit: "°/s", d: 0, min: 0, max: (v) => Math.max(30, v) },
  { key: "eyeBlink", label: "Eye closure", unit: "", d: 2, min: 0, max: () => 1 },
  { key: "stress", label: "EEG stress (target)", unit: "", d: 0, min: 0, max: () => 100, ref: 50 },
];
const LAYOUT = { left: 12, right: 64, top: 6, stripH: 56, gap: 12, rasterRow: 15, axisH: 22 };

function drawChart() {
  requestAnimationFrame(drawChart);
  if (!state.chartDirty) return;
  state.chartDirty = false;
  const canvas = $("strip-canvas");
  const dpr = window.devicePixelRatio || 1;
  const L = LAYOUT;
  const rasterH = (REGIONS.length + 1) * L.rasterRow;
  const cssH = L.top + STRIPS.length * (L.stripH + L.gap) + rasterH + L.axisH;
  const cssW = canvas.clientWidth;
  if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
    canvas.style.height = cssH + "px";
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const ink = cssVar("--text"), muted = cssVar("--muted"), grid = cssVar("--grid"), line = cssVar("--line");
  const now = state.samples.length ? state.samples[state.samples.length - 1].t : performance.now();
  const x0 = L.left, x1 = cssW - L.right;
  const xOf = (t) => x1 - ((now - t) / CHART_SPAN_MS) * (x1 - x0);
  ctx.font = "11px Inter, system-ui, sans-serif";

  let y = L.top;
  for (const s of STRIPS) {
    const vals = state.samples.map((p) => p[s.key]).filter((v) => v !== null && v !== undefined);
    const hi = s.max(vals.length ? Math.max(...vals) : 0);
    const yOf = (v) => y + L.stripH - ((v - s.min) / (hi - s.min)) * L.stripH;
    // recessive grid: baseline and top
    ctx.strokeStyle = grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, y + L.stripH + 0.5); ctx.lineTo(x1, y + L.stripH + 0.5);
    ctx.moveTo(x0, y + 0.5); ctx.lineTo(x1, y + 0.5); ctx.stroke();
    if (s.ref !== undefined) {
      ctx.setLineDash([3, 3]); ctx.strokeStyle = muted;
      ctx.beginPath(); ctx.moveTo(x0, yOf(s.ref)); ctx.lineTo(x1, yOf(s.ref)); ctx.stroke(); ctx.setLineDash([]);
    }
    // line, broken where the value is missing
    ctx.strokeStyle = line; ctx.lineWidth = 2; ctx.lineJoin = "round";
    ctx.beginPath();
    let pen = false;
    for (const p of state.samples) {
      const v = p[s.key];
      if (v === null || v === undefined) { pen = false; continue; }
      const px = xOf(p.t), py = yOf(Math.min(v, hi));
      if (pen) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      pen = true;
    }
    ctx.stroke();
    // labels in text ink, value at the right edge
    const last = state.samples.length ? state.samples[state.samples.length - 1][s.key] : null;
    ctx.fillStyle = muted; ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(s.label, x0 + 4, y + 3);
    ctx.textAlign = "left";
    ctx.fillText(fmt(hi, s.d), x1 + 6, y);
    ctx.textBaseline = "bottom";
    ctx.fillText(fmt(s.min, s.d), x1 + 6, y + L.stripH);
    ctx.fillStyle = ink; ctx.textBaseline = "middle"; ctx.font = "600 12px Inter, system-ui, sans-serif";
    ctx.fillText(last === null || last === undefined ? "—" : fmt(last, s.d) + s.unit, x1 + 6, y + L.stripH / 2);
    ctx.font = "11px Inter, system-ui, sans-serif";
    s._hi = hi; s._yOf = yOf;
    y += L.stripH + L.gap;
  }

  // event raster: one row per face region, plus blinks
  const rows = [...REGIONS.map((r) => ({ key: r.key, label: r.label, color: cssVar(`--r-${r.key}`) })), { key: "blink", label: "Blinks", color: muted }];
  rows.forEach((r, i) => {
    const ry = y + i * L.rasterRow;
    ctx.fillStyle = muted; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(r.label, x1 + 6, ry + L.rasterRow / 2);
    ctx.strokeStyle = grid; ctx.beginPath(); ctx.moveTo(x0, ry + L.rasterRow - 0.5); ctx.lineTo(x1, ry + L.rasterRow - 0.5); ctx.stroke();
  });
  const rowY = Object.fromEntries(rows.map((r, i) => [r.key, y + i * L.rasterRow]));
  for (const e of state.fx.events) {
    if (e.end < now - CHART_SPAN_MS) continue;
    const r = REGION_OF[e.key], ry = rowY[r];
    const xa = Math.max(x0, xOf(e.t)), w = Math.max(3, xOf(e.end) - xa);
    ctx.fillStyle = cssVar(`--r-${r}`);
    ctx.globalAlpha = e.kind === "micro" ? 1 : 0.35;
    ctx.fillRect(xa, ry + 2, w, L.rasterRow - 4);
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = muted;
  for (const b of state.fx.blinks) {
    if (b.t < now - CHART_SPAN_MS) continue;
    ctx.fillRect(xOf(b.t), rowY.blink + 2, Math.max(2, (b.dur / CHART_SPAN_MS) * (x1 - x0)), L.rasterRow - 4);
  }
  y += rows.length * L.rasterRow;

  // time axis
  ctx.fillStyle = muted; ctx.textBaseline = "top"; ctx.textAlign = "center";
  for (let s = 0; s <= 60; s += 10) ctx.fillText(s === 0 ? "now" : `−${s} s`, xOf(now - s * 1000), y + 6);

  // crosshair
  if (state.hover !== null && state.samples.length) {
    const t = now - ((x1 - state.hover) / (x1 - x0)) * CHART_SPAN_MS;
    let best = state.samples[0];
    for (const p of state.samples) if (Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
    const hx = xOf(best.t);
    ctx.strokeStyle = muted; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(hx + 0.5, L.top); ctx.lineTo(hx + 0.5, y); ctx.stroke();
    for (const s of STRIPS) {
      const v = best[s.key];
      if (v === null || v === undefined) continue;
      ctx.fillStyle = line; ctx.strokeStyle = cssVar("--surface"); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(hx, s._yOf(Math.min(v, s._hi)), 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
    showTooltip(best, hx, now);
  } else {
    $("tooltip").hidden = true;
  }
}

function showTooltip(p, hx, now) {
  const tip = $("tooltip");
  tip.innerHTML = `<div class="tt-time">${fmt((now - p.t) / 1000, 1)} s ago</div>`
    + STRIPS.map((s) => `<div><span class="muted">${s.label}</span><b>${p[s.key] === null || p[s.key] === undefined ? "—" : fmt(p[s.key], s.d) + s.unit}</b></div>`).join("");
  tip.hidden = false;
  const w = tip.offsetWidth, cw = $("strip-canvas").clientWidth;
  tip.style.left = `${hx + 14 + w > cw ? hx - 14 - w : hx + 14}px`;
  tip.style.top = "8px";
}

// ---------- Wiring ----------
function init() {
  buildAuList();
  connect();
  const fx = state.fx;
  $("btn-camera").onclick = () => (state.running ? stopCamera() : startCamera($("camera-select").value || undefined));
  $("camera-select").onchange = (e) => startCamera(e.target.value);
  $("mirror").onchange = (e) => $("stage").classList.toggle("mirror", e.target.checked);
  $("stage").classList.toggle("mirror", $("mirror").checked);
  $("btn-record").onclick = toggleRecording;
  $("btn-reset").onclick = () => { fx.reset(); state.lastSample = 0; state.chartDirty = true; };

  const slider = (id, valId, key) => {
    const el = $(id);
    el.value = DEFAULTS[key];
    $(valId).textContent = DEFAULTS[key];
    el.oninput = () => { fx.opts[key] = Number(el.value); $(valId).textContent = el.value; };
  };
  slider("k-slider", "k-val", "k");
  slider("amp-slider", "amp-val", "minAmp");

  const canvas = $("strip-canvas");
  canvas.addEventListener("mousemove", (e) => { state.hover = e.offsetX; state.chartDirty = true; });
  canvas.addEventListener("mouseleave", () => { state.hover = null; state.chartDirty = true; });
  window.addEventListener("resize", () => (state.chartDirty = true));

  setInterval(tick, TICK_MS);
  requestAnimationFrame(drawChart);
}

init();
