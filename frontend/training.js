// Step 4 of the home page: build the dataset, train the model, and show how well it works.
// The server runs tools/export_dataset.py and tools/train_model.py; the charts read models/report.json
// and models/history.json through GET /api/training.
(() => {
  const el = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmtTime = (s) => (s >= 90 ? `${(s / 60).toFixed(1)} min` : `${Math.round(s)} s`);
  const fmtAuc = (v) => (v == null ? "—" : v.toFixed(2));
  // Colour follows the model everywhere: label model = series A, EEG-teacher model = series B
  const SERIES = {
    label: { name: "Label model", key: "label:headset_free", color: "var(--series-a)", dash: "" },
    teacher: { name: "Brain-taught model", key: "teacher:headset_free", color: "var(--series-b)", dash: "6 4" },
    headset: { name: "With headset (brain)", key: "label:brain", color: "var(--series-c)", dash: "2 3" },
  };
  const MODEL_ROWS = [
    ["label:headset_free", "Heart + face · label model", "var(--series-a)"],
    ["teacher:headset_free", "Heart + face · taught by brain", "var(--series-b)"],
    ["label:heart", "Heart only", "var(--series-other)"],
    ["label:face", "Face only", "var(--series-other)"],
    ["label:brain", "Brain only (headset)", "var(--series-other)"],
    ["label:muscle", "Muscle only (headset)", "var(--series-other)"],
    ["label:headset", "Brain + muscle (headset)", "var(--series-other)"],
    ["label:full", "All signals (headset)", "var(--series-other)"],
  ];
  const HEADSET_SETS = { brain: "brain only", headset: "brain + muscle", full: "brain + muscle + heart + face", eeg: "EEG" };
  const phaseNames = typeof PHASE_NAMES !== "undefined" ? PHASE_NAMES : {};  // from app.js
  let data = null, timer = null, wasRunning = false;

  // ---------- Data ----------
  async function load() {
    try {
      data = await (await fetch("/api/training")).json();
    } catch {
      return schedule(5000);
    }
    render();
    if (wasRunning && !data.job.running) setTimeout(load, 300); // pick up the new report right away
    wasRunning = data.job.running;
    schedule(data.job.running ? 1000 : 10000);
  }
  function schedule(ms) { clearTimeout(timer); timer = setTimeout(load, ms); }

  async function post(url, msgEl) {
    const r = await fetch(url, { method: "POST" });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      el(msgEl).textContent = j.error || `Error ${r.status}`;
    }
    load();
  }
  el("btn-build").onclick = () => post("/api/dataset/build", "tr-sessions");
  el("btn-train").onclick = () => post("/api/model/train", "tr-model");
  el("tr-session-pick").onchange = () => drawTimeline();

  // ---------- Cards ----------
  function render() {
    const sessions = data.sessions || [], ok = sessions.filter((s) => s.consent);
    const rest = ok.reduce((a, s) => a + s.rest_s, 0), stress = ok.reduce((a, s) => a + s.stress_s, 0);
    el("tr-sessions").textContent = sessions.length
      ? `${sessions.length} session${sessions.length > 1 ? "s" : ""} recorded, ${ok.length} with consent: ` +
        `${fmtTime(rest)} calm and ${fmtTime(stress)} stress labelled.`
      : "No session recorded yet. Run a calibration or the study protocol with “Record for the dataset” ticked.";
    el("tr-session-table").innerHTML = sessions.length
      ? "<tr><td>When</td><td>Code</td><td>Kind</td><td>Calm</td><td>Stress</td><td>Consent</td></tr>" +
        sessions.map((s) => `<tr><td>${esc(when(s.folder))}</td><td>${esc(s.participant)}</td><td>${esc(s.kind)}</td>
          <td>${fmtTime(s.rest_s)}</td><td>${fmtTime(s.stress_s)}</td><td class="${s.consent ? "" : "no"}">${s.consent ? "✓" : "✗"}</td></tr>`).join("")
      : "";

    const ds = data.dataset;
    el("tr-dataset").innerHTML = ds
      ? `Built ${esc(ds.built)}: ${ds.sessions} session${ds.sessions > 1 ? "s" : ""}, ${ds.participants} participant${ds.participants > 1 ? "s" : ""}, ` +
        `${fmtTime(ds.rest_s)} calm, ${fmtTime(ds.stress_s)} stress.` +
        (ds.zip ? `<br>Zip to send: <code>${esc(ds.zip)}</code>` : "") +
        (ok.length > ds.sessions ? "<br><b>New sessions since: build again.</b>" : "")
      : "Gathers every session recorded with consent, anonymised (P01, P02…), into <code>dataset_export/</code> and a zip you can send.";

    const job = data.job, running = job.running;
    const build = el("btn-build"), train = el("btn-train");
    build.disabled = running || data.recording || !ok.length;
    train.disabled = running || !ds;
    build.textContent = running && job.name === "dataset" ? "Building…" : "Build dataset";
    train.textContent = running && job.name === "train" ? "Training…" : "Train model";
    build.title = data.recording ? "A session is being recorded" : !ok.length ? "No session with consent yet" : "";
    train.title = !ds ? "Build the dataset first" : "";
    for (const name of ["dataset", "train"]) {
      const log = el(`tr-log-${name}`);
      log.hidden = job.name !== name;
      if (job.name === name) {
        const head = running ? (name === "train" ? "Training (about a minute per session)…" : "Building…")
          : job.ok ? "Done." : "Failed:";
        log.textContent = [head, ...job.log].join("\n");
        log.classList.toggle("failed", job.ok === false);
        log.scrollTop = log.scrollHeight;
      }
    }

    const rep = data.report;
    el("tr-model").textContent = rep
      ? `Trained ${rep.trained_at || ""} on ${rep.sessions.length} session${rep.sessions.length > 1 ? "s" : ""}. ` +
        `Live with the headset: ${HEADSET_SETS[rep.with_headset] || "brain"} model ` +
        `(AUC ${fmtAuc((rep.results[`label:${rep.with_headset || "brain"}`] || {}).auc)}); without: ` +
        (rep.live_model ? `${rep.live_model === "teacher" ? "brain-taught" : "label"} model (AUC ${fmtAuc(rep.results[SERIES[rep.live_model].key].auc)}).`
          : "no trained heart + face model: none beat chance on unseen data, so the personal model of each calibration is used.") +
        (rep.iaf ? ` Your alpha frequency: ${Object.values(rep.iaf).map((v) => v.toFixed(1)).join(", ")} Hz.` : "")
      : "No model trained yet.";

    const done = !!rep;
    el("sn-4").classList.toggle("done", done);
    el("ss-4").className = "step-state" + (done ? " ok" : "");
    el("ss-4").textContent = done ? `✓ Models trained · AUC on unseen data: ${fmtAuc((rep.results[`label:${rep.with_headset || "brain"}`] || {}).auc)} ` +
        `with headset, ${rep.live_model ? fmtAuc(rep.results[SERIES[rep.live_model].key].auc) : "personal calibration"} without`
      : ds ? "Dataset ready: train the model"
      : ok.length ? `${ok.length} session${ok.length > 1 ? "s" : ""} ready: build the dataset`
      : "Record calibrations or study sessions first";

    el("tr-results").hidden = !rep;
    if (rep) renderResults(rep);
  }

  function when(folder) {
    const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/.exec(folder);
    if (!m) return folder;
    const d = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]);
    return d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
  }

  function renderResults(rep) {
    SERIES.headset.key = `label:${rep.with_headset || "brain"}`;
    const res = (k) => rep.results[SERIES[k].key] || {};
    const secs = rep.seconds || {};
    const kpi = (k, v, s, color) => `<div class="tr-kpi"><div class="k">${color ? `<i style="background:${color}"></i>` : ""}${k}</div>
      <div class="v">${v}</div><div class="s">${s}</div></div>`;
    el("tr-kpis").innerHTML =
      kpi("With headset · live", fmtAuc(res("headset").auc),
        `${HEADSET_SETS[rep.with_headset] || "brain"}, AUC on unseen data`, SERIES.headset.color) +
      kpi("Training data", fmtTime((secs.rest || 0) + (secs.stress || 0)),
        `${rep.sessions.length} session${rep.sessions.length > 1 ? "s" : ""} · tested by ${rep.validation.replace(/ \(.*\)/, "")}`);

    const pick = el("tr-session-pick"), names = Object.keys(rep.timeline || {});
    if ([...pick.options].map((o) => o.value).join() !== names.join()) {
      pick.innerHTML = names.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
      pick.value = names[names.length - 1] || "";
    }
    pick.hidden = names.length < 2;
    drawAll();
  }

  // ---------- Charts (inline SVG, colours from CSS tokens so dark mode follows) ----------
  function drawAll() {
    if (!data || !data.report || el("tr-results").hidden) return;
    drawModels(); drawLearning(); drawTimeline(); drawHistory();
  }
  const width = (id) => Math.max(260, el(id).clientWidth);
  const svg = (w, h, inner) => `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">${inner}</svg>`;
  const legend = (keys) => `<div class="legend chart-legend">${keys.map((k) =>
    `<span><svg width="22" height="8"><line x1="1" y1="4" x2="21" y2="4" stroke="${SERIES[k].color}" stroke-width="2.5"
      stroke-dasharray="${SERIES[k].dash}"/></svg> ${SERIES[k].name}</span>`).join("")}</div>`;
  const barEnd = (x, y, w, h, r = 4) => {  // bar anchored left, rounded at the data end only
    r = Math.min(r, w, h / 2);
    return `M${x},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${h - 2 * r}a${r},${r} 0 0 1 -${r},${r}h-${w - r}z`;
  };
  // y axis for AUC charts: chance (0.5) dashed, gridlines recessive
  function aucAxis(w, y, x0, x1, ticks) {
    return ticks.map((v) => `<line class="${v === 0.5 ? "chance" : "grid-line"}" x1="${x0}" x2="${x1}" y1="${y(v)}" y2="${y(v)}"/>
      <text x="${x0 - 6}" y="${y(v) + 4}" text-anchor="end">${v === 0.5 ? "chance 0.5" : v.toFixed(2)}</text>`).join("");
  }
  function line(points, x, y, s) {  // points: [{x, y}] with y null for gaps
    let d = "", pen = false;
    for (const p of points) {
      if (p.y == null) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`;
      pen = true;
    }
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-dasharray="${s.dash}" stroke-linejoin="round"/>`;
  }
  function marker(cx, cy, s, tip) {
    return `<circle cx="${cx}" cy="${cy}" r="4" fill="${s.color}" stroke="var(--surface)" stroke-width="2"/>
      <circle cx="${cx}" cy="${cy}" r="12" fill="transparent" data-tip="${esc(tip)}"/>`;
  }
  // direct end labels, nudged apart when they would collide
  function endLabels(items, x) {
    items = items.filter((i) => i.y != null).sort((a, b) => a.y - b.y);
    for (let i = 1; i < items.length; i++) if (items[i].y - items[i - 1].y < 14) items[i].y = items[i - 1].y + 14;
    return items.map((i) => `<text class="val" x="${x}" y="${i.y + 4}">${esc(i.text)}</text>`).join("");
  }

  function drawModels() {
    const rep = data.report, W = width("ch-models"), rowH = 30, top = 8;
    const labelW = Math.min(180, W * 0.42), x0 = labelW + 8, x1 = W - 44;
    const x = (v) => x0 + (x1 - x0) * Math.max(0, Math.min(1, v));
    const H = top + MODEL_ROWS.length * rowH + 22;
    let g = [0, 0.25, 0.5, 0.75, 1].map((v) => `<line class="${v === 0.5 ? "chance" : "grid-line"}" x1="${x(v)}" x2="${x(v)}" y1="${top}"
      y2="${H - 20}"/>${v * 4 % 2 ? "" : `<text x="${x(v)}" y="${H - 6}" text-anchor="middle">${v === 0.5 ? "0.5 chance" : v}</text>`}`).join("");
    MODEL_ROWS.forEach(([key, name, color], i) => {
      const r = rep.results[key] || {}, y = top + i * rowH;
      const live = (rep.live_model && key === SERIES[rep.live_model].key) || key === SERIES.headset.key;
      if (key === SERIES.headset.key) color = SERIES.headset.color;
      g += `<text x="${labelW}" y="${y + 19}" text-anchor="end"${live ? ' class="val"' : ""}>${esc(name)}${live ? " · live" : ""}</text>`;
      if (r.auc == null) { g += `<text x="${x0 + 4}" y="${y + 19}">no data</text>`; return; }
      const tip = `${name}\nAUC ${r.auc.toFixed(2)} on unseen data` +
        (r.auc_folds && r.auc_folds.length ? `\nper test fold: ${r.auc_folds.join(", ")}` : "") +
        (r.balanced_acc != null ? `\nbalanced accuracy ${r.balanced_acc.toFixed(2)}` : "") +
        (r.imitation_r != null ? `\nreproduces the brain model: r ${r.imitation_r.toFixed(2)}` : "");
      g += `<path d="${barEnd(x0, y + 8, Math.max(1, x(r.auc) - x0), 14)}" fill="${color}"/>
        <rect x="0" y="${y + 2}" width="${W}" height="${rowH - 4}" fill="transparent" data-tip="${esc(tip)}"/>
        <text class="val" x="${x(r.auc) + 6}" y="${y + 19}">${r.auc.toFixed(2)}</text>`;
    });
    el("ch-models").innerHTML = svg(W, H, g);
  }

  function drawLearning() {
    const lc = data.report.learning_curve;
    if (!lc) { el("ch-learning").innerHTML = ""; return; }
    const W = width("ch-learning"), H = 200, l = 64, r = W - 110, t = 10, b = H - 26;
    const vals = [...lc.label, ...lc.teacher].filter((v) => v != null);
    const lo = Math.min(0.4, Math.floor((Math.min(...vals, 1) - 0.05) * 10) / 10);
    const x = (f) => l + (r - l) * (f - lc.fractions[0]) / (lc.fractions[lc.fractions.length - 1] - lc.fractions[0]);
    const y = (v) => b - (b - t) * (v - lo) / (1 - lo);
    let g = aucAxis(W, y, l, r, [0.5, 0.75, 1].filter((v) => v >= lo));
    g += lc.fractions.map((f) => `<text x="${x(f)}" y="${H - 6}" text-anchor="middle">${Math.round(f * 100)}%</text>`).join("");
    const ends = [];
    for (const k of ["label", "teacher"]) {
      const s = SERIES[k], pts = lc.fractions.map((f, i) => ({ x: f, y: lc[k][i] }));
      g += line(pts, x, y, s);
      g += pts.filter((p) => p.y != null).map((p) => marker(x(p.x), y(p.y), s,
        `${s.name}\ntrained on the first ${Math.round(p.x * 100)}% of each phase\nAUC ${p.y.toFixed(2)} on unseen data`)).join("");
      const last = pts[pts.length - 1];
      if (last.y != null) ends.push({ y: y(last.y), text: `${k === "label" ? "Label" : "Brain-taught"} ${last.y.toFixed(2)}` });
    }
    g += endLabels(ends, r + 10);
    el("ch-learning").innerHTML = svg(W, H, g) + legend(["label", "teacher"]) +
      `<p class="hint" style="margin-top:4px">x: share of each recorded phase used for training</p>`;
    const a = lc.label, gain = a[a.length - 1] != null && a[a.length - 2] != null ? a[a.length - 1] - a[a.length - 2] : null;
    el("ch-learning-note").textContent = gain == null ? ""
      : gain > 0.02 ? "Still rising at 100%: more recordings should make the model better."
      : "Flat at the end: more of the same won't help much; new sessions on other days will.";
  }

  function drawTimeline() {
    const rep = data && data.report, box = el("ch-timeline");
    const rows = rep && rep.timeline && rep.timeline[el("tr-session-pick").value];
    if (!rows || !rows.length) { box.innerHTML = ""; return; }
    const W = width("ch-timeline"), H = 210, l = 34, r = W - 8, t = 18, b = H - 24;
    const tMax = rows[rows.length - 1][0] / 60 || 1;
    const x = (m) => l + (r - l) * m / tMax, y = (v) => b - (b - t) * Math.max(0, Math.min(1, v));
    let g = [0, 0.5, 1].map((v) => `<line class="grid-line" x1="${l}" x2="${r}" y1="${y(v)}" y2="${y(v)}"/>
      <text x="${l - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`).join("");
    // phase bands
    let start = 0;
    for (let i = 1; i <= rows.length; i++) {
      if (i < rows.length && rows[i][1] === rows[start][1]) continue;
      const [t0, phase, label] = rows[start], t1 = rows[i - 1][0] + 2;
      const xa = x(t0 / 60), xb = x(t1 / 60), name = phaseNames[phase] || phase;
      if (label != null) g += `<rect x="${xa}" y="${t}" width="${Math.max(0, xb - xa)}" height="${b - t}"
        fill="${label ? "var(--bad)" : "var(--good)"}" opacity="0.1"/>`;
      if (xb - xa > 70) g += `<text x="${(xa + xb) / 2}" y="${t - 6}" text-anchor="middle">${esc(name.replace(/^(Calibration|Study): /, ""))}</text>`;
      start = i;
    }
    const minutes = Math.ceil(tMax);
    for (let m = 0; m <= minutes; m += Math.max(1, Math.round(minutes / 6)))
      g += `<text x="${x(Math.min(m, tMax))}" y="${H - 6}" text-anchor="middle">${m} min</text>`;
    for (const [k, col] of [["headset", 5], ["label", 3], ["teacher", 4]]) {
      const pts = [];  // a gap in time (skipped questionnaire) breaks the line
      let run = [];  // 10 s moving average (5 points at 2 s), restarted after each gap
      rows.forEach((row, i) => {
        if (i && row[0] - rows[i - 1][0] > 6) { pts.push({ y: null }); run = []; }
        if (row[col] != null) run = [...run.slice(-4), row[col]];
        pts.push({ x: row[0] / 60, y: run.length ? run.reduce((a, v) => a + v, 0) / run.length : null });
      });
      g += line(pts, x, y, SERIES[k]);
    }
    g += `<line id="tl-cross" x1="0" x2="0" y1="${t}" y2="${b}" stroke="var(--muted)" stroke-width="1" visibility="hidden"/>
      <rect id="tl-hit" x="${l}" y="${t}" width="${r - l}" height="${b - t}" fill="transparent"/>`;
    box.innerHTML = svg(W, H, g) + legend(["headset", "label", "teacher"]);
    const hit = box.querySelector("#tl-hit"), cross = box.querySelector("#tl-cross");
    hit.addEventListener("mousemove", (e) => {
      const rect = hit.getBoundingClientRect(), m = (e.clientX - rect.left) / rect.width * tMax;
      let best = 0;
      for (let i = 1; i < rows.length; i++) if (Math.abs(rows[i][0] / 60 - m) < Math.abs(rows[best][0] / 60 - m)) best = i;
      const [tt, phase, label, A, B, H] = rows[best], name = phaseNames[phase] || phase;
      cross.setAttribute("x1", x(tt / 60)); cross.setAttribute("x2", x(tt / 60)); cross.setAttribute("visibility", "visible");
      showTip(e, `${(tt / 60).toFixed(1)} min · ${name}${label == null ? "" : label ? " (stress)" : " (calm)"}\n` +
        `With headset ${H == null ? "—" : H.toFixed(2)}\nLabel model ${A == null ? "—" : A.toFixed(2)}\n` +
        `Brain-taught model ${B == null ? "—" : B.toFixed(2)}`);
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  }

  function drawHistory() {
    // Only trainings tested the same way as the latest one are comparable (within a session, on a new session,
    // on a new person: each is harder than the one before), so the chart shows those.
    const all = data.history || [], box = el("ch-history");
    const kind = (h) => (h.validation || "").replace(/ \(.*\)/, "").replace(/^\d+ time blocks.*/, "time blocks");
    const latest = all.length ? kind(all[all.length - 1]) : null;
    const hist = all.filter((h) => kind(h) === latest);
    const hidden = all.length - hist.length;
    if (!hist.length) { box.innerHTML = ""; return; }
    const W = width("ch-history"), H = 190, l = 64, r = W - 110, t = 10, b = H - 26;
    const get = (h, k) => (h.auc || {})[k === "headset" ? `label:${h.with_headset || "brain"}` : SERIES[k].key];
    const vals = hist.flatMap((h) => [get(h, "headset"), get(h, "label"), get(h, "teacher")]).filter((v) => v != null);
    const lo = Math.min(0.4, Math.floor((Math.min(...vals, 1) - 0.05) * 10) / 10);
    const n = hist.length, x = (i) => (n === 1 ? (l + r) / 2 : l + (r - l) * i / (n - 1));
    const y = (v) => b - (b - t) * (v - lo) / (1 - lo);
    let g = aucAxis(W, y, l, r, [0.5, 0.75, 1].filter((v) => v >= lo));
    const step = Math.max(1, Math.ceil(n / 8));
    hist.forEach((h, i) => { if (i % step === 0 || i === n - 1) g += `<text x="${x(i)}" y="${H - 6}" text-anchor="middle">#${i + 1}</text>`; });
    const ends = [];
    for (const k of ["headset", "label", "teacher"]) {
      const s = SERIES[k], pts = hist.map((h, i) => ({ x: i, y: get(h, k) }));
      g += line(pts, x, y, s);
      g += hist.map((h, i) => {
        const v = get(h, k);
        if (v == null) return "";
        const sec = (h.seconds || {}), tot = (sec.rest || 0) + (sec.stress || 0);
        return marker(x(i), y(v), s, `Training #${i + 1} · ${h.trained_at}\n${h.sessions} session${h.sessions > 1 ? "s" : ""}, ` +
          `${fmtTime(tot)} labelled\n${s.name}: AUC ${v.toFixed(2)}\nvalidation: ${h.validation}`);
      }).join("");
      const last = get(hist[n - 1], k);
      if (last != null) ends.push({ y: y(last), text: `${{ headset: "Headset", label: "Label", teacher: "Brain-taught" }[k]} ${last.toFixed(2)}` });
    }
    g += endLabels(ends, r + 10);
    box.innerHTML = svg(W, H, g) + legend(["headset", "label", "teacher"]);
    el("ch-history-note").textContent = `Trainings tested by ${latest}` +
      (hidden ? ` (${hidden} earlier training${hidden > 1 ? "s" : ""} tested an easier way are not shown).` : ".") +
      (n < 2 ? " Each new training tested the same way adds a point." : "");
  }

  // ---------- Tooltip ----------
  const tip = document.createElement("div");
  tip.className = "chart-tip";
  tip.hidden = true;
  document.body.append(tip);
  function showTip(e, text) {
    tip.textContent = text;
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = `${Math.min(window.innerWidth - w - 8, e.clientX + 14)}px`;
    tip.style.top = `${Math.max(8, e.clientY - h - 10)}px`;
  }
  function hideTip() { tip.hidden = true; }
  document.addEventListener("mousemove", (e) => {
    const target = e.target.closest && e.target.closest(".svg-chart [data-tip]");
    if (target) showTip(e, target.dataset.tip);
    else if (!(e.target.id === "tl-hit")) hideTip();
  });

  let resizeTimer = null;
  new ResizeObserver(() => { clearTimeout(resizeTimer); resizeTimer = setTimeout(drawAll, 150); }).observe(el("tr-results"));
  load();
})();
