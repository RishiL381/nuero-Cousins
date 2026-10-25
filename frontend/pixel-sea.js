"use strict";

/*
 * Pixel sea: a cozy pixel-art boat on an animated sea, drawn into a 128 x 128 canvas.
 * Stress (0-1) drives the swell, foam, rain and darkness; focus (0-1) drives day and night.
 * Every visual follows its input along a smooth curve, so a short blip barely registers.
 *
 *   const sea = PixelSea(canvas);
 *   sea.target = 0.7;          // stress: 0 relaxed .. 1 tense
 *   sea.focusTarget = 0.9;     // focus: 0 drifting (night) .. 1 locked in (day)
 *   sea.update(dt);            // every animation frame
 *   sea.draw(dt);              // render; about 30 fps is plenty
 *   sea.BR                     // breathing-ring state, driven by a breathing break UI
 *
 * BoatScene (bottom of the file) wraps it in the same API as the original sailboat scene.
 */
(function () {
  function PixelSea(canvas) {
    const W = 128, H = 128;
    const cv = canvas;
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(W, H), buf = new Uint32Array(img.data.buffer);
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const hex = h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
    const pack = (r, g, b) => (255 << 24) | (b << 16) | (g << 8) | r;
    const P = h => { const c = hex(h); return pack(c[0], c[1], c[2]); };
    const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
    const sstep = (a, b, v) => { const t = clamp((v - a) / (b - a)); return t * t * (3 - 2 * t); };
    const hash = (x, y) => { let n = (Math.imul(x, 374761393) + Math.imul(y, 668265263)) | 0; n = Math.imul(n ^ (n >>> 13), 1274126177); return ((n ^ (n >>> 16)) >>> 0) / 4294967295; };
    const rgbPack = c => pack(c[0] | 0, c[1] | 0, c[2] | 0);
    const darken = (c, k) => pack(((c & 255) * k) | 0, (((c >> 8) & 255) * k) | 0, (((c >> 16) & 255) * k) | 0);
    const blend = (c, to, a) => pack(
      ((c & 255) * (1 - a) + to[0] * a) | 0, (((c >> 8) & 255) * (1 - a) + to[1] * a) | 0, (((c >> 16) & 255) * (1 - a) + to[2] * a) | 0);
    // smooth value noise for clouds
    const vn = (x, y) => {
      const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi, u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
      const a = hash(xi, yi), b = hash(xi + 1, yi), c = hash(xi, yi + 1), d = hash(xi + 1, yi + 1);
      return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
    };
    const fbm = (x, y) => vn(x / 22, y / 22) * 0.55 + vn(x / 11 + 5.3, y / 11 + 1.7) * 0.3 + vn(x / 5.5 + 9.1, y / 5.5 + 3.3) * 0.15;
    // ordered dithering keeps soft edges looking like pixel art
    const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map(v => (v + 0.5) / 16);
    const dither = (x, y, a) => a > BAYER[(y & 3) * 4 + (x & 3)];

    const CAL = { deep: hex('#2a859e'), mid: hex('#38a0b2'), light: hex('#6fcbc9'), foam: hex('#f6fcf6') };
    const STO = { deep: hex('#172f48'), mid: hex('#244e6e'), light: hex('#4b7c9c'), foam: hex('#dde8ee') };

    // ---- boat sprites ----
    const HULL = [[-20, 0], [-14, -5.5], [-5, -8], [12, -8.2], [16, -7], [16, 7], [12, 8.2], [-5, 8], [-14, 5.5]];
    const sc = (p, kx, ky) => p.map(([x, y]) => [-2 + (x + 2) * kx, y * ky]);
    const TRIM = sc(HULL, 0.9, 0.8), DECK = sc(HULL, 0.82, 0.62), RING = sc(HULL, 1.22, 1.5);
    const inPoly = (x, y, p) => { let c = false; for (let i = 0, j = p.length - 1; i < p.length; j = i++) { const xi = p[i][0], yi = p[i][1], xj = p[j][0], yj = p[j][1]; if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c; } return c; };
    const inBox = (x, y, x0, x1, y0, y1) => x >= x0 && x <= x1 && y >= y0 && y <= y1;
    const COL = {
      cream: P('#f7ecd2'), side: P('#d9c8a2'), keel: P('#8e3b3b'), red: P('#c8453b'),
      wood1: P('#d29250'), wood2: P('#bd7b3d'), seam: P('#8d5a2b'), bench: P('#e7b06a'), benchSide: P('#a9672f'),
      wall: P('#b5743a'), wallDark: P('#9a5f2e'), glassF: P('#a6dbe8'), glassS: P('#8ccbdc'),
      frame: P('#8b5a2b'), roof: P('#cf9250'), roofIn: P('#e2ab68'), brass: P('#8a6a2a'), lamp: P('#ffd86b')
    };
    const plank = (x, y) => { const v = (y + 20) / 2.4; return v % 1 < 0.12 ? COL.seam : (Math.floor(v) % 2 ? COL.wood2 : COL.wood1); };
    const LAYERS = [
      { box: [-21, 17, -9, 9], h: [-3, -1], test: (x, y) => inPoly(x, y, HULL), col: (x, y, h) => (h === -3 ? COL.keel : COL.side) },
      { box: [-21, 17, -9, 9], h: [0, 0], test: (x, y) => inPoly(x, y, HULL), col: (x, y) => (!inPoly(x, y, TRIM) ? COL.cream : !inPoly(x, y, DECK) ? COL.red : plank(x, y)) },
      { box: [6, 12.5, -5, 5], h: [1, 2], test: () => true, col: (x, y, h) => (h === 2 ? COL.bench : COL.benchSide) },
      { box: [-6, 4, -4.5, 4.5], h: [1, 7], test: () => true, col: (x, y, h) => {
        const win = h >= 3 && h <= 5;
        if (win && x < -4.6 && Math.abs(y) < 3) return COL.glassF;   // windshield, faces the bow
        if (win && y > 3.1 && x > -4.2 && x < 2.6) return COL.glassS; // side window, faces the viewer
        if (h === 7) return COL.frame;
        return y > 0 || x < -4.6 ? COL.wallDark : COL.wall;
      } },
      { box: [-7, 5, -5.6, 5.6], h: [8, 8], test: () => true, col: (x, y) => (inBox(x, y, -4.5, 2.5, -3, 3) ? COL.roofIn : COL.roof) },
      { box: [-16, -14, -1, 1], h: [1, 3], test: () => true, col: (x, y, h) => (h === 1 ? COL.brass : COL.lamp) }
    ];
    const SW = 100, SH = 112, SCX = 50, SCY = 74, SQ = 0.6, K = 1.3;
    const PHI = -42 * Math.PI / 180, cP = Math.cos(PHI), sP = Math.sin(PHI);
    // plan point (lx, ly) at height h, rolled by r and pitched by p, to a screen offset
    function fwd(lx, ly, h, r, p) {
      const cr = Math.cos(r), sr = Math.sin(r), cp = Math.cos(p), sp = Math.sin(p);
      const x1 = lx * cp + (h / K) * sp, h1 = h * cp - lx * K * sp;
      const y2 = ly * cr - (h1 / K) * sr, h2 = h1 * cr + ly * K * sr;
      return [K * (x1 * cP - y2 * sP), K * (x1 * sP + y2 * cP) * SQ - h2];
    }
    function renderBoat(r, p) {
      const out = new Uint32Array(SW * SH);
      for (const L of LAYERS) {
        const [bx0, bx1, by0, by1] = L.box;
        for (let h = L.h[0]; h <= L.h[1]; h++)
          for (let lx = bx0; lx <= bx1; lx += 0.3) for (let ly = by0; ly <= by1; ly += 0.3) {
            if (!L.test(lx, ly)) continue;
            const c = L.col(lx, ly, h); if (!c) continue;
            const [ox, oy] = fwd(lx, ly, h, r, p), sx = Math.round(SCX + ox), sy = Math.round(SCY + oy);
            if (sx >= 0 && sx < SW && sy >= 0 && sy < SH) out[sy * SW + sx] = c;
          }
      }
      return out;
    }
    const ROLL_STEP = 4 * Math.PI / 180, PITCH_STEP = 3.5 * Math.PI / 180, sprites = new Map();
    const sprite = (ri, pi) => { const k = ri * 10 + pi; if (!sprites.has(k)) sprites.set(k, renderBoat(ri * ROLL_STEP, pi * PITCH_STEP)); return sprites.get(k); };
    sprite(0, 0);

    // ---- state ----
    let target = 0.12, cur = target, sea = target, wx = target, calmVis = 1 - sstep(0.2, 0.5, target);
    let t = 0;
    let fTarget = 0.7, focus = 0.7, dayV = 0.7; // focus: 0 drifting .. 1 locked in; dayV follows it slowly
    const boatMask = new Uint8Array(W * H);
    let hv = 0, hvel = 0, pr = 0, prv = 0, rl = 0, rlv = 0, ri = 0, pi = 0;
    const drops = Array.from({ length: 44 }, (_, i) => ({ x: hash(i, 3) * W, y: hash(i, 9) * H, v: 1.6 + hash(i, 5) * 1.4 }));
    const fish = Array.from({ length: 5 }, (_, i) => ({ x: hash(i, 41) * W, y: hash(i, 43) * H, a: hash(i, 47) * 6.28, sp: 4 + hash(i, 53) * 4, ph: hash(i, 59) * 10 }));
    let ripple = null, nextRipple = 3;
    const BR = { want: 0, on: 0, r: 58, glow: 0, glowT: 0 }; // breathing ring: a breathing break UI sets want, r and glowT
    const spray = [];

    // ---- sea ----
    const WV = [{}, {}, {}, {}];
    const Hf = new Float32Array(W * H);
    // crest lines bend and drift (warp), sets of waves swell and fade (env), crests are pointed and troughs round
    function wave(gx, gz, chop) {
      const warp = 6 * Math.sin(gx * 0.037 + t * 0.11 + 2 * Math.sin(gz * 0.029 - t * 0.07)) + 5 * Math.sin(gz * 0.045 - gx * 0.012 + t * 0.09);
      let h = 0;
      for (let i = 0, n = chop ? 4 : 3; i < n; i++) {
        const w = WV[i], d = w.nx * gx + w.nz * gz + warp * w.wf;
        const env = 1 - w.ev + w.ev * Math.sin(d * 0.021 + gz * 0.008 - t * 0.33 + i * 2.1);
        h += w.a * env * (1.273 - 2 * Math.abs(Math.sin((w.k * d - w.om) * 0.5)));
      }
      return h;
    }
    const spring = (v, vel, tg, k, c, dt) => { vel += (k * (tg - v) - c * vel) * dt; return [v + vel * dt, vel]; };

    function frame(dt) {
      t += dt;
      const s = sea, storm = sstep(0.45, 1, wx);
      const pal = {}; for (const k of ['deep', 'mid', 'light', 'foam']) pal[k] = rgbPack(mix(CAL[k], STO[k], sstep(0.1, 0.95, s)));
      const tks = Math.floor(t * 1.5);

      const A0 = 0.9 + 3.4 * s, a1 = -0.25 + 0.22 * Math.sin(t * 0.05);
      [[0.70, 0.72, 50 - 16 * s, 7 + 3 * s, 0.55, 1, 0.35], [Math.cos(a1), Math.sin(a1), 31 - 10 * s, 9 + 2.5 * s, 0.3, 0.6, 0.4],
       [0.2, 0.98, 19 - 5 * s, 9 + 1.5 * s, 0.18, 0.3, 0.5], [-0.6, 0.8, 9, 11, 0.1 * (0.3 + s), 0, 0.6]].forEach(([nx, nz, lam, c, a, wf, ev], i) => {
        const k = 2 * Math.PI / lam, n = Math.hypot(nx, nz), w = WV[i];
        w.nx = nx / n; w.nz = nz / n; w.k = k; w.om = k * c * t; w.a = A0 * a; w.wf = wf; w.ev = ev;
      });

      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const h0 = wave(x, y / SQ, true);
        Hf[y * W + x] = wave(x, (y + h0) / SQ, true);
      }
      // breathing: water inside the lantern ring and behind each calm ripple goes glassy
      const bcx = 64, bcy = 74 - Math.round(hv) + 2;
      const glass = (x, y) => (BR.on <= 0 ? 1 : 1 - 0.5 * BR.on * (1 - sstep(BR.r - 4, BR.r + 4, Math.hypot(x - bcx, (y - bcy) / SQ))));
      // distance from the breathing wave's crest; the circle wobbles and changes shape as it moves
      const crestAt = (x, y) => {
        const dx = x - bcx, dy = (y - bcy) / SQ, an = Math.atan2(dy, dx);
        const R = BR.r + 2.2 * Math.sin(3 * an + t * 0.7) + 1.4 * Math.sin(5 * an - t * 1.1 + 1.3) + 0.8 * Math.sin(9 * an + t * 1.9);
        const k = clamp(1 - Math.abs(Math.hypot(dx, dy) - R) / 2.8);
        return k * k * (0.75 + 0.25 * Math.sin(4 * an - t * 0.9));
      };
      if (BR.on > 0) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
        Hf[y * W + x] = Hf[y * W + x] * glass(x, y) + (0.9 + 0.6 * A0) * BR.on * crestAt(x, y);
      const hAt = (x, y) => Hf[clamp(y, 0, H - 1) * W + clamp(x, 0, W - 1)];
      const lightK = 0.7 + 0.5 * s, foamCap = sstep(0.12, 0.75, s);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const h = Hf[y * W + x];
        const lit = ((hAt(x + 1, y) - hAt(x - 1, y)) * 0.6 + (hAt(x, y + 1) - hAt(x, y - 1)) * 0.8) * 0.5;
        const hn = h / (A0 * 1.1), q = hn * 0.55 + (lit / A0) * 1.1 * lightK + (hash(x >> 1, y >> 1) - 0.5) * 0.14;
        let c;
        if (hn > 0.8 - 0.2 * s && hash(x + tks, y) > 0.45 && foamCap > hash(x, y + 77)) c = pal.foam;
        else if (q > 0.62 - 0.25 * s) c = s > 0.35 && hash(x + tks, y + 5) > 0.55 ? pal.foam : pal.light;
        else if (q > 0.3 - 0.1 * s) c = pal.light;
        else if (q < -0.5 + 0.12 * s) c = pal.deep;
        else c = pal.mid;
        if (calmVis > 0.35 && hn > 0.2 && hash(x + tks * 13, y + tks * 7) > 0.994) c = pal.foam;
        buf[y * W + x] = c;
      }

      // ---- boat motion: big swells only, slow springs, a gentle pendulum sway ----
      const cx = 64, cy = 74, gz0 = cy / SQ;
      const samp = (lx, ly) => wave(cx + K * (lx * cP - ly * sP), gz0 + K * (lx * sP + ly * cP), true);
      const hc = samp(-2, 0), hb = samp(-11, 0), hs = samp(8, 0), hp = samp(-2, -6), hst = samp(-2, 6);
      const tgH = (2 * hc + hb + hs + hp + hst) / 6 * 1.1;
      const tgP = clamp(Math.atan((hb - hs) / (19 * K)) * 1.05, -0.22, 0.22);
      const tgR = clamp(Math.atan((hst - hp) / (12 * K)) * 0.95 + 0.035 * Math.sin(t * 1.05) * (1 - s * 0.5), -0.22, 0.22);
      const steps = Math.max(1, Math.ceil(dt / 0.016)), sdt = dt / steps, steady = glass(64, 74);
      for (let i = 0; i < steps; i++) {
        [hv, hvel] = spring(hv, hvel, tgH * steady, 12, 4.6, sdt);
        [pr, prv] = spring(pr, prv, tgP * steady, 9, 3.4, sdt);
        [rl, rlv] = spring(rl, rlv, tgR * steady, 7, 2.3, sdt);
      }
      // pick a sprite frame, with a little hysteresis so it never chatters between two frames
      if (Math.abs(rl / ROLL_STEP - ri) > 0.65) ri = clamp(Math.round(rl / ROLL_STEP), -4, 4);
      if (Math.abs(pr / PITCH_STEP - pi) > 0.65) pi = clamp(Math.round(pr / PITCH_STEP), -4, 4);
      const hy = Math.round(hv), by = cy - hy;
      const ground = (x, y) => { const sx = (x - cx) / K, gy = (y - by) / SQ / K; return [sx * cP + gy * sP, -sx * sP + gy * cP]; };

      // ---- foam ring and wake (streaks drift backwards instead of flickering) ----
      const foam = pal.foam;
      for (let y = by - 40; y < by + 38; y++) for (let x = cx - 48; x < cx + 50; x++) {
        if (x < 0 || y < 0 || x >= W || y >= H) continue;
        const [lx, ly] = ground(x, y);
        if (inPoly(lx, ly, HULL)) continue;
        let f = inPoly(lx, ly, RING) && hash(x + tks * 31, y + tks * 17) > 0.5 - 0.15 * s;
        if (!f && lx > 14 && lx < 58) {
          const wdt = 6 + (lx - 14) * 0.42, fade = (lx - 14) / 44, a = Math.abs(ly), st = hash(Math.floor(lx * 0.7 - t * 4), Math.floor(ly * 1.2));
          f = (Math.abs(a - wdt) < 1.3 && st > fade * 0.8) || (a < wdt - 2 && st > 0.7 + fade * 0.25 - 0.15 * s);
        }
        if (f) buf[y * W + x] = foam;
      }

      // ---- fish under calm water ----
      if (calmVis > 0.02) {
        const fc = [18, 72, 92];
        for (const f of fish) {
          f.a += Math.sin(t * 0.4 + f.ph) * 0.5 * dt;
          if (Math.hypot(f.x - cx, (f.y - by) / SQ) < 26) f.a += 1.6 * dt;
          f.x += Math.cos(f.a) * f.sp * dt; f.y += Math.sin(f.a) * f.sp * dt * SQ;
          if (f.x < -8) f.x = W + 6; if (f.x > W + 8) f.x = -6; if (f.y < -8) f.y = H + 6; if (f.y > H + 8) f.y = -6;
          const ca = Math.cos(f.a), sa = Math.sin(f.a) * SQ;
          for (let i = 0; i < 6; i++) {
            const wig = i >= 4 ? Math.sin(t * 9 + f.ph) * 0.8 : 0;
            const pts = [[f.x - ca * i - sa * wig, f.y - sa * i + ca * wig]];
            if (i >= 1 && i <= 3) pts.push([f.x - ca * i + sa, f.y - sa * i - ca]);
            for (const [fx, fy] of pts) {
              const px = Math.round(fx), py = Math.round(fy);
              if (px >= 0 && px < W && py >= 0 && py < H && dither(px, py, calmVis)) buf[py * W + px] = blend(buf[py * W + px], fc, 0.5);
            }
          }
        }
        nextRipple -= dt;
        if (nextRipple <= 0 && calmVis > 0.6) { const f = fish[Math.floor(Math.random() * fish.length)]; ripple = { x: f.x, y: f.y, age: 0 }; nextRipple = 3 + Math.random() * 4; }
      }
      if (ripple) {
        ripple.age += dt;
        const r = 1 + ripple.age * 9, a = (1 - ripple.age / 1.4) * calmVis;
        if (a <= 0) ripple = null;
        else for (let k = 0; k < 48; k++) {
          const px = Math.round(ripple.x + Math.cos(k / 48 * 6.283) * r), py = Math.round(ripple.y + Math.sin(k / 48 * 6.283) * r * SQ);
          if (px >= 0 && px < W && py >= 0 && py < H && dither(px, py, a)) buf[py * W + px] = blend(buf[py * W + px], CAL.foam, 0.6);
        }
      }

      // ---- boat ----
      boatMask.fill(0);
      const spr = sprite(ri, pi), ox = cx - SCX, oy = by - SCY;
      for (let j = 0; j < SH; j++) { const y = oy + j; if (y < 0 || y >= H) continue;
        for (let i = 0; i < SW; i++) { const c = spr[j * SW + i], x = ox + i; if (c && x >= 0 && x < W) { buf[y * W + x] = c; boatMask[y * W + x] = 1; } } }
      // bow spray when the bow drops into a wave
      if (prv < -0.12 && s > 0.3 && dt > 0) {
        const [bx, bY] = fwd(-20, 0, 0, ri * ROLL_STEP, pi * PITCH_STEP);
        for (let k = 0; k < 3; k++) spray.push({ x: cx + bx + (Math.random() - 0.5) * 4, y: by + bY, vx: -6 - Math.random() * 14, vy: -14 - Math.random() * 16, life: 0.7 });
      }
      for (let i = spray.length - 1; i >= 0; i--) {
        const p = spray[i]; p.life -= dt; p.vy += 55 * dt; p.x += p.vx * dt; p.y += p.vy * dt;
        if (p.life <= 0) { spray.splice(i, 1); continue; }
        const px = Math.round(p.x), py = Math.round(p.y);
        if (px >= 0 && px < W && py >= 0 && py < H) buf[py * W + px] = foam;
      }

      // ---- time of day: focus brings the sun up, drifting or resting brings night; a storm mutes both ----
      const dayK = sstep(0.4, 0.75, dayV), nightK = 1 - sstep(0.2, 0.55, dayV), sky = 1 - sstep(0.3, 0.8, wx), gradeK = 1 - 0.6 * storm;
      const nA = 0.62 * nightK * gradeK, dA = 0.24 * clamp(1 - Math.abs(dayV - 0.45) / 0.22) * gradeK, sA = 0.1 * dayK * gradeK;
      if (nA + dA + sA > 0.01) for (let i = 0; i < buf.length; i++) {
        let c = buf[i];
        if (nA > 0.01) c = blend(c, [10, 22, 52], nA);
        if (dA > 0.01) c = blend(c, [255, 168, 112], dA);
        if (sA > 0.01) c = blend(c, [255, 250, 230], sA);
        buf[i] = c;
      }
      // the sun or moon reflected in the water, broken up by the waves
      if (sky > 0.01) {
        const sunC = [255, 236, 190], moonC = [214, 228, 255], sunK = dayK * sky, moonK = nightK * sky;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const i = y * W + x; if (boatMask[i]) continue;
          const h = Hf[i] / (A0 * 1.1), dxp = Math.abs(x - (40 + y * 0.18));
          const sunPath = sunK * clamp(1 - dxp / (13 + y * 0.08)) * (1 - y / 210);
          if (sunPath > 0 && h > 0.15 && hash(x + tks * 7, y) > 1 - sunPath * 0.8) buf[i] = rgbPack(sunC);
          const moonPath = moonK * clamp(1 - dxp / 8) * (1 - y / 170);
          if (moonPath > 0 && h > 0.2 && hash(x + tks * 5, y + 9) > 1 - moonPath * 0.65) buf[i] = rgbPack(moonC);
          const dd = Math.hypot(x - 40, (y - 14) / 0.55);
          if (sunK > 0.01 && dd < 13 && Math.sin(y * 1.7 + t * 2 + x * 0.2) > -0.3) buf[i] = blend(buf[i], sunC, 0.9 * sunK * clamp(1.2 - dd / 13));
          if (moonK > 0.01 && dd < 7.5 && Math.sin(y * 1.9 + t * 1.5) > -0.2) buf[i] = blend(buf[i], moonC, 0.9 * moonK);
          if (moonK * calmVis > 0.05 && hash(x, y + 991) > 0.996 && hash(x + tks, y) > 0.4) buf[i] = blend(buf[i], moonC, 0.8 * moonK);
        }
      }

      // ---- weather: the scene darkens gradually as the storm builds ----
      const tint = [38, 54, 82], ta = storm * 0.22;
      if (ta > 0.01) for (let i = 0; i < buf.length; i++) buf[i] = blend(buf[i], tint, ta);
      // breathing: lantern light pools inside the ring, the sea outside dims, the ring and ripples sit on the water
      if (BR.on > 0) {
        const hi = hex('#d8f6f2');
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const i = y * W + x, k = crestAt(x, y) * BR.on, d = Math.hypot(x - bcx, (y - bcy) / SQ);
          if (k > 0.02) buf[i] = hash(x + tks, y + 11) > 1 - k * 0.4 ? pal.foam : blend(buf[i], hi, 0.3 * k);
          else if (d > BR.r) buf[i] = darken(buf[i], 1 - 0.16 * BR.on * sstep(BR.r + 4, BR.r + 30, d));
        }
      }
      // lantern glow, steady, stronger as it gets darker
      const [lx0, ly0] = fwd(-15, 0, 3, ri * ROLL_STEP, pi * PITCH_STEP), gx = Math.round(cx + lx0), gy = Math.round(by + ly0);
      const ga = 0.22 + 0.3 * storm + 0.05 * Math.sin(t * 1.6) + 0.35 * BR.on * BR.glow + 0.25 * nightK;
      for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
        const d = Math.abs(dx) + Math.abs(dy), x = gx + dx, y = gy + dy;
        if (d === 0 || d > 3 || x < 0 || y < 0 || x >= W || y >= H) continue;
        buf[y * W + x] = blend(buf[y * W + x], [255, 214, 120], ga * (1 - d / 4));
      }
      const nDrops = Math.floor(sstep(0.55, 1, wx) * drops.length), rc = hex('#bcd6e8');
      for (let i = 0; i < nDrops; i++) {
        const d = drops[i]; d.y += d.v * dt * 40; d.x += d.v * dt * 20;
        if (d.y > H) { d.y = -3; d.x = Math.random() * W; } if (d.x > W) d.x = 0;
        for (let k = 0; k < 3; k++) { const px = (d.x - k * 0.5) | 0, py = (d.y - k) | 0; if (px >= 0 && px < W && py >= 0 && py < H) buf[py * W + px] = blend(buf[py * W + px], rc, 0.85 - k * 0.2); }
      }

      ctx.putImageData(img, 0, 0);
    }

    // critically damped follow over roughly T seconds: eases in, glides, settles without overshoot
    const follow = (x, v, tg, T, dt) => { const w = 2 / T; v += (w * w * (tg - x) - 2 * w * v) * dt; return [clamp(x + v * dt), v]; };
    let curV = 0, seaV = 0, wxV = 0, focusV = 0, dayVV = 0;

    // model output reacts in ~2 s; the sea builds over ~6 s; the storm takes ~14 s and the sun ~25 s
    function update(dt) {
      if (reduce) { cur = sea = wx = target; calmVis = 1 - sstep(0.2, 0.5, target); focus = dayV = fTarget; }
      else {
        [cur, curV] = follow(cur, curV, target, 2, dt);
        [sea, seaV] = follow(sea, seaV, cur, 6, dt);
        [wx, wxV] = follow(wx, wxV, cur, 14, dt);
        calmVis += ((1 - sstep(0.2, 0.5, sea)) - calmVis) * Math.min(1, dt * 0.8);
        [focus, focusV] = follow(focus, focusV, fTarget, 3, dt);
        [dayV, dayVV] = follow(dayV, dayVV, focus, 25, dt);
      }
      BR.on = clamp(BR.on + (BR.want ? dt : -dt) / 0.8);   // the breathing ring fades in and out
      BR.glow += (BR.glowT - BR.glow) * Math.min(1, dt * 2);
    }

    frame(0);
    return {
      BR, update, draw: dt => frame(reduce ? 0 : dt), reduce,
      get target() { return target; }, set target(v) { target = clamp(+v); },
      get focusTarget() { return fTarget; }, set focusTarget(v) { fTarget = clamp(+v); },
      get cur() { return cur; }, get dayV() { return dayV; }
    };
  }

  const HIGH_SCORE = 70, HIGH_FOR_S = 20, SNOOZE_MS = 60000;   // breathing break: "stressed for too long"
  const BREATH_PHASES = [["Breathe in", 4], ["Hold", 2], ["Breathe out", 6]], CYCLE = 12, CYCLES = 5;
  const R_MIN = 30, R_MAX = 58;                                     // breathing ring radius, in sea pixels

  // Drop-in for the original sailboat scene (same API), drawn as the pixel sea.
  //   const scene = new BoatScene(canvas); scene.start(win); scene.setStress(score); scene.setFocus(score)
  class BoatScene {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.sea = PixelSea(canvas);
      canvas.style.imageRendering = "pixelated";
      canvas.style.objectFit = "cover";
      this.score = null;
      this.message = "Waiting for signal";
      this.hud = opts.hud === false ? null : this.makeHud(canvas);
      // Breathing break (from the desktop widget): offered when stress stays high, drawn on the water itself.
      // opts.breathing turns it on; opts.canPrompt() can veto the offer (e.g. during a calibration or study).
      this.canPrompt = opts.canPrompt || (() => true);
      this.breath = null; this.highFor = 0; this.snoozeUntil = 0;
      if (opts.breathing && this.wrap) this.makeBreathing();
    }

    makeHud(canvas) {
      const doc = canvas.ownerDocument;
      const wrap = doc.createElement("div");
      wrap.className = "px-wrap";
      wrap.dataset.for = canvas.id;
      wrap.style.cssText = "position:relative;display:block;width:100%;height:100%"; // keeps the canvas's CSS size
      this.wrap = wrap;
      canvas.parentNode.insertBefore(wrap, canvas);
      wrap.append(canvas);
      const hud = doc.createElement("div");
      hud.style.cssText = "position:absolute;left:8px;top:8px;padding:3px 9px;border-radius:6px;pointer-events:none;"
        + "background:#0f2730;border:1px solid #5d8a95;color:#fdf6e3;box-shadow:0 2px 0 #0a1c24;"
        + "font:500 13px 'Pixelify Sans', ui-monospace, Menlo, monospace;white-space:nowrap";
      wrap.append(hud);
      return hud;
    }

    /** score: 0-100 stress (0 = your calm, 100 = your stress test) or null when there is no signal yet. */
    setStress(score, message) {
      this.score = score == null ? null : Math.max(0, Math.min(100, score));
      if (this.score != null) this.sea.target = this.score / 100;
      if (message !== undefined) this.message = message;
      if (!this.hud) return;
      if (this.score == null) { this.hud.textContent = this.message; return; }
      const s = Math.round(this.score);
      const label = s < 30 ? "Calm sea" : s < 55 ? "Gentle swell" : s < 75 ? "Rough sea" : "Storm"; // same bands as the gauge
      this.hud.textContent = `${label} · ${s}`;
    }

    /** focus: 0-100 or null. Focused brings the sun up, resting brings a moonlit night. */
    setFocus(score) {
      if (score != null) this.sea.focusTarget = Math.max(0, Math.min(100, score)) / 100;
    }

    makeBreathing() {
      const doc = this.canvas.ownerDocument, mk = (tag, cls, html = "") => {
        const e = doc.createElement(tag); e.className = cls; e.innerHTML = html; this.wrap.append(e); return e;
      };
      this.blabel = mk("div", "px-blabel");
      this.bstop = mk("button", "px-bstop", "end"); this.bstop.type = "button"; this.bstop.hidden = true;
      this.prompt = mk("div", "px-prompt", "<p>Rough water for a while. Take a minute to breathe with the sea?</p>" +
        '<div class="row"><button type="button" class="yes">Yes, breathe</button><button type="button" class="no">Not now</button></div>');
      this.prompt.hidden = true;
      this.prompt.querySelector(".yes").onclick = () => this.startBreath();
      this.prompt.querySelector(".no").onclick = () => { this.prompt.hidden = true; this.highFor = 0; this.snoozeUntil = performance.now() + SNOOZE_MS; };
      this.bstop.onclick = () => this.endBreath();
    }

    startBreath() {
      this.prompt.hidden = true;
      this.breath = { t: 0, outro: 0 };
      this.sea.BR.r = R_MAX; this.sea.BR.want = 1; this.bstop.hidden = false;
    }

    endBreath() {
      this.breath = null; this.highFor = 0; this.snoozeUntil = performance.now() + SNOOZE_MS;
      const BR = this.sea.BR; BR.want = 0; BR.glowT = 0;
      this.bstop.hidden = true; this.blabel.innerHTML = "";
    }

    // offer a break once the live score has stayed at or above HIGH_SCORE for HIGH_FOR_S seconds
    watchStress(dt) {
      const high = this.score != null && this.score >= HIGH_SCORE;
      this.highFor = high ? this.highFor + dt : Math.max(0, this.highFor - 2 * dt);
      if (!this.canPrompt()) { this.prompt.hidden = true; return; }
      if (this.highFor > HIGH_FOR_S && this.prompt.hidden && performance.now() > this.snoozeUntil) this.prompt.hidden = false;
    }

    // 5 breaths of 4 s in, 2 s hold, 6 s out: breathing in pulls a wave in around the boat and lights the lantern,
    // breathing out pushes it back out. The sea keeps following the live score: that is the feedback.
    breathStep(dt) {
      const b = this.breath, BR = this.sea.BR, ease = (u) => 0.5 - 0.5 * Math.cos(Math.PI * Math.max(0, Math.min(1, u)));
      b.t += dt;
      if (b.t >= CYCLE * CYCLES) {
        b.outro += dt; BR.r = R_MAX; BR.glowT = 0;
        this.blabel.innerHTML = "<b>Nice work</b><small>5 breaths done</small>";
        if (b.outro > 3) this.endBreath();
        return;
      }
      const cyc = Math.floor(b.t / CYCLE), u = b.t % CYCLE;
      const phase = u < 4 ? 0 : u < 6 ? 1 : 2, dur = BREATH_PHASES[phase][1], prog = (u - [0, 4, 6][phase]) / dur;
      const lift = phase === 0 ? 0.07 * ease(prog) : phase === 1 ? 0.07 : 0.07 * (1 - ease(prog));
      if (this.score != null) this.sea.target = Math.min(1, this.score / 100 + lift);
      BR.r = phase === 0 ? R_MAX - (R_MAX - R_MIN) * ease(prog) : phase === 1 ? R_MIN : R_MIN + (R_MAX - R_MIN) * ease(prog);
      BR.glowT = phase === 2 ? 0 : 1;
      this.blabel.innerHTML = `<b>${BREATH_PHASES[phase][0]}</b><span>${Math.ceil(dur - prog * dur)}</span><small>${cyc + 1} of ${CYCLES}</small>`;
    }

    start(win = window) {
      this.win = win;
      let last = null, acc = 0;
      const loop = (ts) => {
        const dt = last == null ? 0 : Math.min(0.1, (ts - last) / 1000);
        last = ts;
        if (this.prompt) { if (this.breath) this.breathStep(dt); else this.watchStress(dt); }
        this.sea.update(dt);
        acc += dt;
        if (acc >= 1 / 30) { this.sea.draw(acc); acc = 0; }
        this.raf = win.requestAnimationFrame(loop);
      };
      this.raf = win.requestAnimationFrame(loop);
    }

    stop() { if (this.win && this.raf) this.win.cancelAnimationFrame(this.raf); }
  }

  window.PixelSea = PixelSea;
  window.BoatScene = BoatScene;
})();
