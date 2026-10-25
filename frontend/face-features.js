// Facial feature extraction from MediaPipe Face Landmarker output (pure, no DOM).
//
// Per frame:  52 blendshape scores + head transformation matrix
//   -> 14 action-unit-like composites (FACS codes are approximate mappings)
//   -> micro-expression events: a composite rises above its own rolling baseline
//      for 60-500 ms (longer = macro expression)
//   -> blinks, head pose and head speed, facial activity (how fast the face changes)
// Per tick (every 0.5 s, like the EEG engine): one row of training features, FEATURE_COLUMNS.
//
// Loaded as a classic script in the browser (window.FaceFeatures) and with require() in Node tests.

(function (root) {
  "use strict";

  const COMPOSITES = [
    { key: "brow_furrow", au: "AU4", label: "Brow lowerer", src: ["browDownLeft", "browDownRight"] },
    { key: "brow_inner_up", au: "AU1", label: "Inner brow raise", src: ["browInnerUp"] },
    { key: "brow_outer_up", au: "AU2", label: "Outer brow raise", src: ["browOuterUpLeft", "browOuterUpRight"] },
    { key: "eye_wide", au: "AU5", label: "Upper lid raise", src: ["eyeWideLeft", "eyeWideRight"] },
    { key: "eye_squint", au: "AU6/7", label: "Eye squint", src: ["eyeSquintLeft", "eyeSquintRight"] },
    { key: "nose_wrinkle", au: "AU9", label: "Nose wrinkle", src: ["noseSneerLeft", "noseSneerRight"] },
    { key: "smile", au: "AU12", label: "Lip corner pull", src: ["mouthSmileLeft", "mouthSmileRight"] },
    { key: "dimple", au: "AU14", label: "Dimpler", src: ["mouthDimpleLeft", "mouthDimpleRight"] },
    { key: "frown", au: "AU15", label: "Lip corner depress", src: ["mouthFrownLeft", "mouthFrownRight"] },
    { key: "chin_raise", au: "AU17", label: "Chin raise", src: ["mouthShrugLower"] },
    { key: "lip_stretch", au: "AU20", label: "Lip stretch", src: ["mouthStretchLeft", "mouthStretchRight"] },
    { key: "lip_press", au: "AU24", label: "Lip press", src: ["mouthPressLeft", "mouthPressRight"] },
    { key: "lip_suck", au: "AU28", label: "Lip suck", src: ["mouthRollLower", "mouthRollUpper"] },
    { key: "jaw_open", au: "AU26", label: "Jaw drop", src: ["jawOpen"] },
  ];
  // Mean of these composites = tension index (an engineering choice, not a validated scale)
  const TENSION_KEYS = ["brow_furrow", "eye_squint", "nose_wrinkle", "frown", "lip_stretch", "lip_press", "lip_suck"];
  const EYE_KEYS = new Set(["eye_wide", "eye_squint"]);  // blinks move these, so no events mid-blink

  const DEFAULTS = {
    smoothMs: 40,         // light smoothing of each composite against frame jitter
    baselineMs: 8000,     // rolling baseline and spread of each composite
    minSpread: 0.02,
    k: 3,                 // onset when value > baseline + k * spread ...
    minAmp: 0.08,         // ... and at least this far above baseline
    microMinMs: 60,       // shorter = noise
    microMaxMs: 500,      // longer = macro expression
    maxEventMs: 4000,
    warmupMs: 3000,       // let baselines settle before detecting events
    maxHeadDps: 60,       // no events while the head turns faster than this
    blinkOn: 0.5,
    blinkOff: 0.35,
    maxBlinkMs: 500,      // longer eye closures are not counted as blinks
    windowMs: 2000,       // window for the per-tick means
    rateWindowMs: 60000,  // window for blink and micro-expression rates
    fidgetWindowMs: 10000,
  };

  const FEATURE_COLUMNS = [
    "face_present", "fps", "tension", "asymmetry", "activity",
    ...COMPOSITES.map((c) => c.key),
    "blink_rate_min", "blink_dur_ms",
    "head_yaw", "head_pitch", "head_roll", "head_speed_dps", "head_fidget_deg",
    "micro_rate_min", "micro_new", "macro_rate_min",
    ...COMPOSITES.map((c) => "micro_" + c.key),
  ];

  const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : null);
  const std = (a) => {
    const m = mean(a);
    return m === null ? null : Math.sqrt(mean(a.map((v) => (v - m) ** 2)));
  };
  const round = (v, d = 4) => (v === null || v === undefined || !isFinite(v) ? null : Math.round(v * 10 ** d) / 10 ** d);
  const DEG = 180 / Math.PI;

  // Column-major 4x4 matrix from MediaPipe -> yaw, pitch, roll in degrees and translation in cm.
  function headPose(m) {
    const r = (i, j) => m[j * 4 + i];
    return {
      yaw: Math.atan2(r(0, 2), r(2, 2)) * DEG,
      pitch: Math.asin(Math.max(-1, Math.min(1, -r(1, 2)))) * DEG,
      roll: Math.atan2(r(1, 0), r(1, 1)) * DEG,
      tx: m[12], ty: m[13], tz: m[14],
    };
  }

  class FaceFeatureExtractor {
    constructor(opts = {}) {
      this.opts = Object.assign({}, DEFAULTS, opts);
      this.reset();
    }

    reset() {
      this.firstT = null;
      this.lastT = null;
      this.prevRaw = null;
      this.prevPose = null;
      this.headSpeed = 0;
      this.comp = Object.fromEntries(COMPOSITES.map((c) => [c.key, { s: null, base: null, spread: null, onset: null, peak: 0 }]));
      this.blink = { on: false, onset: null };
      this.frames = [];   // recent frame summaries (fidgetWindowMs)
      this.events = [];   // micro and macro expressions (rateWindowMs)
      this.blinks = [];   // { t, dur } (rateWindowMs)
      this.lastTickT = null;
    }

    // scores: { blendshapeName: score } or null when no face; matrix: 16 numbers or null.
    addFrame(t, scores, matrix) {
      const o = this.opts;
      if (this.firstT === null) this.firstT = t;
      const dt = this.lastT === null ? 0 : Math.max(0, Math.min(200, t - this.lastT));
      this.lastT = t;
      const newEvents = [];

      if (!scores) {
        for (const st of Object.values(this.comp)) st.onset = null;  // drop half-seen events
        this.blink.on = false;
        this.prevRaw = this.prevPose = null;
        this._pushFrame({ t, present: false });
        return { t, present: false, newEvents };
      }

      // Facial activity: mean absolute change of all blendshapes, per second
      let activity = null;
      if (this.prevRaw && dt > 0) {
        let sum = 0, n = 0;
        for (const k in scores) {
          if (k in this.prevRaw) { sum += Math.abs(scores[k] - this.prevRaw[k]); n++; }
        }
        activity = n ? (sum / n) * (1000 / dt) : null;
      }
      this.prevRaw = scores;

      // Head pose and angular speed
      const pose = matrix ? headPose(matrix) : null;
      if (pose && this.prevPose && dt > 0) {
        const d = Math.hypot(pose.yaw - this.prevPose.yaw, pose.pitch - this.prevPose.pitch, pose.roll - this.prevPose.roll);
        const a = 1 - Math.exp(-dt / 100);
        this.headSpeed += a * ((d * 1000) / dt - this.headSpeed);
      }
      this.prevPose = pose;

      // Blinks (hysteresis on the mean eyeBlink score)
      const eb = ((scores.eyeBlinkLeft || 0) + (scores.eyeBlinkRight || 0)) / 2;
      if (!this.blink.on && eb > o.blinkOn) {
        this.blink = { on: true, onset: t };
      } else if (this.blink.on && eb < o.blinkOff) {
        const dur = t - this.blink.onset;
        if (dur <= o.maxBlinkMs) this.blinks.push({ t: this.blink.onset, dur });
        this.blink = { on: false, onset: null };
      }

      // Composites, baselines and micro-expression events
      const warm = t - this.firstT >= o.warmupMs;
      const values = {}, live = {};
      let asymSum = 0, asymN = 0;
      for (const c of COMPOSITES) {
        const raw = c.src.reduce((s, k) => s + (scores[k] || 0), 0) / c.src.length;
        if (c.src.length === 2) { asymSum += Math.abs((scores[c.src[0]] || 0) - (scores[c.src[1]] || 0)); asymN++; }
        const st = this.comp[c.key];
        if (st.s === null) {
          st.s = st.base = raw;
          st.spread = o.minSpread;
        } else {
          st.s += (1 - Math.exp(-dt / o.smoothMs)) * (raw - st.s);
        }
        const thr = Math.max(o.k * Math.max(st.spread, o.minSpread), o.minAmp);
        const d = st.s - st.base;

        if (st.onset === null) {
          // baseline follows the face only outside events, so an expression does not erase itself
          const ab = 1 - Math.exp(-dt / o.baselineMs);
          st.base += ab * (st.s - st.base);
          st.spread += ab * (Math.abs(st.s - st.base) - st.spread);
          const blocked = this.headSpeed > o.maxHeadDps || (EYE_KEYS.has(c.key) && this.blink.on);
          if (warm && !blocked && d > thr) { st.onset = t; st.peak = d; }
        } else {
          st.peak = Math.max(st.peak, d);
          // a held expression ends after maxEventMs so the baseline can adapt to the new resting face
          if (d < thr * 0.5 || t - st.onset > o.maxEventMs) {
            const dur = t - st.onset;
            if (dur >= o.microMinMs) {
              const ev = { t: st.onset, end: t, key: c.key, au: c.au, dur, peak: st.peak, kind: dur <= o.microMaxMs ? "micro" : "macro" };
              this.events.push(ev);
              newEvents.push(ev);
            }
            if (t - st.onset > o.maxEventMs) st.base = st.s;  // held: this is the new resting face
            st.onset = null;
          }
        }
        values[c.key] = st.s;
        live[c.key] = { value: st.s, base: st.base, thr: st.base + thr, active: st.onset !== null };
      }

      const tension = mean(TENSION_KEYS.map((k) => values[k]));
      const asymmetry = asymN ? asymSum / asymN : null;
      const f = { t, present: true, values, tension, asymmetry, activity, pose, headSpeed: this.headSpeed };
      this._pushFrame(f);
      return Object.assign({}, f, { live, blinking: this.blink.on, eyeBlink: eb, newEvents, warm });
    }

    _pushFrame(f) {
      this.frames.push(f);
      const o = this.opts;
      while (this.frames.length && this.frames[0].t < f.t - o.fidgetWindowMs) this.frames.shift();
      while (this.events.length && this.events[0].t < f.t - o.rateWindowMs) this.events.shift();
      while (this.blinks.length && this.blinks[0].t < f.t - o.rateWindowMs) this.blinks.shift();
    }

    // One training row: features over the last windowMs, rates over the last rateWindowMs.
    tick(t) {
      const o = this.opts;
      const win = this.frames.filter((f) => f.t > t - o.windowMs);
      const seen = win.filter((f) => f.present);
      const row = Object.fromEntries(FEATURE_COLUMNS.map((c) => [c, null]));
      row.face_present = win.length ? seen.length / win.length : 0;
      row.fps = win.length / (o.windowMs / 1000);

      const pick = (fn) => seen.map(fn).filter((v) => v !== null && v !== undefined);
      if (seen.length) {
        row.tension = mean(pick((f) => f.tension));
        row.asymmetry = mean(pick((f) => f.asymmetry));
        row.activity = mean(pick((f) => f.activity));
        for (const c of COMPOSITES) row[c.key] = mean(pick((f) => f.values[c.key]));
        const posed = seen.filter((f) => f.pose);
        if (posed.length) {
          row.head_yaw = mean(posed.map((f) => f.pose.yaw));
          row.head_pitch = mean(posed.map((f) => f.pose.pitch));
          row.head_roll = mean(posed.map((f) => f.pose.roll));
        }
        row.head_speed_dps = mean(pick((f) => f.headSpeed));
      }
      const longPosed = this.frames.filter((f) => f.present && f.pose);
      if (longPosed.length > 5) {
        row.head_fidget_deg = Math.hypot(std(longPosed.map((f) => f.pose.yaw)), std(longPosed.map((f) => f.pose.pitch)));
      }

      // Rates need at least 10 s of history; scaled to per minute
      const elapsed = this.firstT === null ? 0 : Math.min(o.rateWindowMs, t - this.firstT);
      if (elapsed >= 10000) {
        const perMin = 60000 / elapsed;
        const micro = this.events.filter((e) => e.kind === "micro");
        row.blink_rate_min = this.blinks.length * perMin;
        row.micro_rate_min = micro.length * perMin;
        row.macro_rate_min = (this.events.length - micro.length) * perMin;
        for (const c of COMPOSITES) row["micro_" + c.key] = micro.filter((e) => e.key === c.key).length * perMin;
      }
      row.blink_dur_ms = mean(this.blinks.map((b) => b.dur));
      const since = this.lastTickT === null ? -Infinity : this.lastTickT;
      row.micro_new = this.events.filter((e) => e.kind === "micro" && e.end > since && e.end <= t).length;
      this.lastTickT = t;

      for (const k in row) row[k] = round(row[k]);
      return row;
    }
  }

  const api = { COMPOSITES, TENSION_KEYS, FEATURE_COLUMNS, DEFAULTS, FaceFeatureExtractor, headPose };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FaceFeatures = api;
})(typeof window !== "undefined" ? window : globalThis);
