"use strict";
// Clutch demo: replays data.json (built by tools/build_demo.py from a real recording, through the app's own code).
(async function () {
  const $ = (id) => document.getElementById(id);
  const data = await (await fetch("data.json")).json();
  const T = data.ticks, N = T.length, CH = data.channels, BANDS = ["theta", "alpha", "beta"];
  const PHASES = {
    calib_rest: ["Calibration · rest", "calm", "Rest"], calib_stress: ["Calibration · stress test", "stress", "Stress test"],
    calib_recovery: ["Calibration · recovery", "", "Recovery"], study_relax: ["Relaxation", "calm", "Relax"],
    study_math_easy: ["Easy math (control)", "", "Easy math"], study_math_hard: ["Hard math", "stress", "Hard math"],
    study_stroop_hard: ["Colour words", "stress", "Colour words"],
  };
  const isCalib = (p) => p.startsWith("calib_");
  const START = 20;  // the first seconds of the replay, while the scores' smoothing settles, are not shown
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const stressColor = (s) => (s == null ? css("--muted") : `hsl(${Math.round(150 - 1.45 * Math.max(0, Math.min(100, s)))}, 55%, 42%)`);

  // ---------- Timeline ----------
  const segs = [];
  T.forEach((t, i) => {
    if (!segs.length || segs[segs.length - 1].p !== t.p) segs.push({ p: t.p, start: i, n: 0 });
    segs[segs.length - 1].n++;
  });
  const lastCalib = Math.max(...segs.map((s, k) => (isCalib(s.p) ? k : -1)));
  $("segments").innerHTML = segs.map((s, k) => {
    const [, kind, short] = PHASES[s.p] || [s.p, "", s.p];
    return `<div class="seg ${kind}${k === lastCalib ? " split" : ""}" style="width:${(100 * s.n) / N}%" title="${short}">${short}</div>`;
  }).join("");

  // ---------- Static panels: calibration report and results ----------
  const rep = data.report;
  $("report-lead").textContent = `Alpha frequency ${rep.iaf.toFixed(1)} Hz · calibration check: ${rep.check}. `
    + "For each electrode and band: the direction it moved under stress and the effect size (Cohen's d); ✓ = kept as a teacher signal.";
  $("brain-table").innerHTML = "<tr><th></th><th>θ theta</th><th>α alpha</th><th>β beta</th></tr>" + CH.map((ch) =>
    `<tr><td>${ch}</td>${BANDS.map((b) => {
      const s = rep.brain[`${b}_${ch}`];
      if (!s || s.d == null) return "<td class=\"unused\">—</td>";
      return `<td class="${s.used ? "used" : "unused"}">${s.dir > 0 ? "↑" : "↓"} ${s.d.toFixed(1)}${s.used ? " ✓" : ""}</td>`;
    }).join("")}</tr>`).join("");
  const LABELS = { hr: "Heart rate", face_brow: "Brows lowered", face_press: "Lips pressed", face_squint: "Eyes squinted",
    face_frown: "Frown", face_smile: "Smile", blink_rate: "Blinks / min", head_motion: "Head motion" };
  const w = rep.student.weights || {}, wmax = Math.max(0.01, ...Object.values(w).map(Math.abs));
  $("weights").innerHTML = Object.entries(w).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).map(([k, v]) => {
    const half = (50 * Math.abs(v)) / wmax;
    return `<div class="row"><span>${LABELS[k] || k}</span><span class="wbar"><i style="left:${v >= 0 ? 50 : 50 - half}%;width:${half}%"></i></span><b>${v > 0 ? "+" : ""}${v.toFixed(2)}</b></div>`;
  }).join("") + `<p class="note">Fitted on ${Math.round(rep.student.seconds / 60)} min of calibration. The bigger the bar, the more that signal counts.</p>`;

  const bar = (label, v, color) => `<div class="row"><span>${label}</span><span class="track"><i style="width:${100 * v}%;background:${color}"></i>
    <span class="chance" title="chance"></span></span><b>${v.toFixed(2)}</b></div>`;
  $("per-person").innerHTML = data.results.per_person.map((p) => `<div class="group">${p.who}</div>`
    + bar("Brain (teacher)", p.teacher, "var(--accent)") + bar("Heart + face (student)", p.student, "var(--body)")).join("");
  const sh = data.results.shared || {}, auc = sh.auc || {};
  const secs = sh.seconds ? Math.round(((sh.seconds.rest || 0) + (sh.seconds.stress || 0)) / 60) : null;
  $("shared-lead").textContent = `Trained on one person, tested on the other (${sh.sessions} sessions${secs ? `, ${secs} min labelled` : ""}). `
    + "The dashed line is chance.";
  $("shared").innerHTML = [["label:brain", "Brain", "var(--accent)"], ["label:muscle", "Muscle tension", "var(--muted)"],
    ["label:heart", "Heart rate", "var(--muted)"], ["label:face", "Face", "var(--muted)"], ["label:headset_free", "Heart + face, one model", "var(--muted)"]]
    .filter(([k]) => auc[k] != null).map(([k, l, c]) => bar(l, auc[k], c)).join("");

  // ---------- Live panels ----------
  const POS = { AF7: [42, 32], AF8: [78, 32], TP9: [17, 88], TP10: [103, 88] };
  const STATE = { good: "--good", muscle: "--fair", bad: "--bad" };
  const FACE = ["Brows lowered", "Lips pressed", "Eyes squinted", "Frown", "Smile", "Blinks / min", "Head motion"];
  const faceNorm = [1, 1, 1, 1, 1, 1 / 40, 1 / 0.08];
  $("face").innerHTML = FACE.map((l, k) => `<div class="row"><span>${l}</span><span class="fill"><i id="f${k}"></i></span></div>`).join("");
  $("bands").innerHTML = "<tr><th></th><th>θ</th><th>α</th><th>β</th></tr>" + CH.map((ch) =>
    `<tr><td>${ch}</td>${BANDS.map((b) => `<td><span class="bar" id="b-${ch}-${b}"><i></i></span></td>`).join("")}</tr>`).join("");

  const scene = new BoatScene($("sea"));
  scene.start();

  function gauge(s, label) {
    const r = 70, c = Math.PI * r, v = s == null ? 0 : Math.max(0, Math.min(100, s)) / 100;
    $("gauge").innerHTML = `<svg viewBox="0 0 180 112" role="img" aria-label="Stress ${s == null ? "unknown" : Math.round(s)}">
      <path class="track" d="M20 95 A70 70 0 0 1 160 95"/>
      <path class="value" d="M20 95 A70 70 0 0 1 160 95" stroke="${stressColor(s)}" stroke-dasharray="${c * v} ${c}"/>
      <text class="num" x="90" y="88">${s == null ? "—" : Math.round(s)}</text>
      <text class="lbl" x="90" y="108">${label}</text></svg>`;
  }

  function chart(i) {
    const cv = $("chart"), dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const span = 120, from = Math.max(START, i - span + 1), l = 26, r = W - 6, t = 6, b = H - 16;
    const x = (k) => l + ((r - l) * (k - (i - span + 1))) / (span - 1), y = (v) => b - ((b - t) * v) / 100;
    for (let k = from; k <= i; k++) {  // phase bands
      const kind = (PHASES[T[k].p] || [])[1];
      if (kind) { g.fillStyle = css(kind === "calm" ? "--calm-tint" : "--stress-tint"); g.fillRect(x(k) - 0.5, t, (r - l) / (span - 1) + 1, b - t); }
    }
    g.strokeStyle = css("--border"); g.lineWidth = 1; g.fillStyle = css("--muted"); g.font = "10px Inter, sans-serif";
    for (const v of [0, 50, 100]) { g.beginPath(); g.moveTo(l, y(v)); g.lineTo(r, y(v)); g.stroke(); g.fillText(String(v), 2, y(v) + 3); }
    g.fillText("2 min ago", l, H - 3); g.textAlign = "right"; g.fillText("now", r, H - 3); g.textAlign = "left";
    const line = (key, color, dash) => {
      g.strokeStyle = color; g.lineWidth = 2.2; g.setLineDash(dash); g.beginPath();
      let pen = false;
      for (let k = from; k <= i; k++) {
        const v = T[k][key];
        if (v == null) { pen = false; continue; }
        pen ? g.lineTo(x(k), y(v)) : g.moveTo(x(k), y(v)); pen = true;
      }
      g.stroke(); g.setLineDash([]);
    };
    line("b", css("--accent"), []);
    line("h", css("--body"), [6, 4]);
  }

  function render(i) {
    const t = T[i], [pname] = PHASES[t.p] || [t.p];
    $("clock").textContent = `${Math.floor(i / 60)}:${String(i % 60).padStart(2, "0")} / ${Math.floor(N / 60)}:${String(N % 60).padStart(2, "0")}`;
    $("phase-pill").textContent = `${pname}${isCalib(t.p) ? " · learning" : " · unseen data"}`;
    $("playhead").style.left = `${(100 * i) / N}%`;
    $("timeline").setAttribute("aria-valuenow", String(i));
    $("dots").innerHTML = CH.map((ch) => {
      const [px, py] = POS[ch] || [60, 60], st = (t.e[ch] || {}).s || "bad";
      return `<circle cx="${px}" cy="${py}" r="9" fill="${css(STATE[st])}"><title>${ch}: ${st === "good" ? "clean" : st === "muscle" ? "muscle tension" : "no contact"}</title></circle>
        <text x="${px}" y="${py < 60 ? py - 13 : py + 21}" text-anchor="middle">${ch}</text>`;
    }).join("");
    for (const ch of CH) for (const b of BANDS) {
      const el = $(`b-${ch}-${b}`), e = t.e[ch] || {}, v = e.s === "good" ? e[b] : null;
      el.classList.toggle("off", v == null);
      const half = v == null ? 0 : Math.min(50, Math.abs(v) / 2);
      const bar = el.firstElementChild;
      bar.style.left = `${v != null && v < 0 ? 50 - half : 50}%`; bar.style.width = `${half}%`;
      el.title = v == null ? `${ch} ${b}: ignored this second` : `${ch} ${b}: ${v > 0 ? "+" : ""}${v}% vs rest`;
    }
    let k = i;  // the optical sensor has gaps: show the last reading if it is less than 15 s old
    while (k > 0 && T[k].hr == null && i - k < 15) k--;
    $("hr").textContent = T[k].hr == null ? "—" : `${T[k].hr}`;
    t.f.forEach((v, k) => { $(`f${k}`).style.width = `${v == null ? 0 : Math.min(100, 100 * v * faceNorm[k])}%`; });
    const score = t.b != null ? t.b : t.h;
    gauge(score, t.b != null ? "brain score" : "heart + face");
    $("v-brain").textContent = t.b == null ? "—" : Math.round(t.b);
    $("v-brain").style.color = stressColor(t.b);
    $("v-body").textContent = t.h == null ? "—" : Math.round(t.h);
    $("v-body").style.color = stressColor(t.h);
    chart(i);
    scene.setStress(score, "Waiting for signal");
  }

  // ---------- Playback ----------
  let i = 0, speed = 5, playing = true, acc = 0;
  const playBtn = $("play");
  const setPlaying = (on) => { playing = on; playBtn.textContent = on ? "❚❚ Pause" : "▶ Play"; playBtn.setAttribute("aria-label", on ? "Pause" : "Play"); };
  playBtn.onclick = () => setPlaying(!playing);
  document.querySelectorAll("[data-speed]").forEach((btn) => {
    btn.onclick = () => {
      speed = Number(btn.dataset.speed);
      document.querySelectorAll("[data-speed]").forEach((x) => x.classList.toggle("on", x === btn));
    };
  });
  const seek = (ev) => {
    const rect = $("timeline").getBoundingClientRect();
    i = Math.max(START, Math.min(N - 1, Math.round(((ev.clientX - rect.left) / rect.width) * (N - 1))));
    render(i);
  };
  let dragging = false;
  $("timeline").addEventListener("pointerdown", (ev) => { dragging = true; seek(ev); });
  addEventListener("pointermove", (ev) => { if (dragging) seek(ev); });
  addEventListener("pointerup", () => { dragging = false; });
  $("timeline").addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowRight") { i = Math.min(N - 1, i + 10); render(i); }
    if (ev.key === "ArrowLeft") { i = Math.max(START, i - 10); render(i); }
  });
  $("timeline").setAttribute("aria-valuemin", "0");
  $("timeline").setAttribute("aria-valuemax", String(N - 1));
  addEventListener("resize", () => render(i));

  // A timer rather than animation frames, so the replay keeps time even when the tab is in the background
  let lastT = performance.now();
  setInterval(() => {
    const now = performance.now(), dt = Math.min(1, (now - lastT) / 1000);
    lastT = now;
    if (!playing || dragging) return;
    acc += dt * speed;
    if (acc >= 1) {
      i += Math.floor(acc);
      if (i >= N) i = START;  // loop
      acc %= 1;
      render(i);
    }
  }, 50);
  i = START;
  render(i);
})();
