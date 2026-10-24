"use strict";

/*
 * BoatScene: a sailboat on the sea, drawn on a <canvas>. The sea follows the stress score:
 * calm mind -> flat water, sun and gulls; high stress -> big waves, dark sky, rain, lightning,
 * and the boat bounces and pitches on the swell.
 *
 *   const scene = new BoatScene(canvas);
 *   scene.start();            // or scene.start(otherWindow) to animate in a popup / PiP window
 *   scene.setStress(62);      // 0-100 score: 0 = your calm, 100 = your stress test; null = no signal yet
 */
(function () {
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  function hex(c) { return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)]; }
  function mix(c1, c2, t) {
    const a = hex(c1), b = hex(c2);
    return `rgb(${Math.round(lerp(a[0], b[0], t))},${Math.round(lerp(a[1], b[1], t))},${Math.round(lerp(a[2], b[2], t))})`;
  }

  // Palettes: [calm, storm]
  const SKY_TOP = ["#7fb8e6", "#232b38"];
  const SKY_LOW = ["#fde2c0", "#4d5a6b"];
  const SEA = [["#5fb7c9", "#2f4f66"], ["#3d9bb3", "#22405a"], ["#26809c", "#16304a"]];
  const CLOUD = ["#ffffff", "#5c6573"];

  class BoatScene {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.hud = opts.hud !== false;
      this.score = null;      // latest stress score (0-100) or null
      this.k = 0.33;          // displayed storm intensity 0-1 (smoothed)
      this.target = 0.33;
      this.t = 0;             // sea time (runs faster in a storm)
      this.clock = 0;         // wall time
      this.last = null;
      this.boat = { y: null, vy: 0, angle: 0, va: 0 };
      this.splash = [];
      this.drops = [];
      this.flash = 0;
      this.message = "Waiting for signal";
      this.clouds = Array.from({ length: 6 }, (_, i) => ({ x: i / 6, y: 0.08 + 0.18 * Math.random(), s: 0.7 + Math.random() * 0.6, v: 0.006 + Math.random() * 0.01 }));
    }

    /** score: 0-100 stress (50 = baseline) or null when not calibrated / no signal. */
    setStress(score, message) {
      this.score = score == null ? null : clamp(score, 0, 100);
      if (this.score != null) this.target = clamp((this.score - 5) / 85, 0, 1); // 0 = your calm -> glassy, 90+ -> storm
      if (message !== undefined) this.message = message;
    }

    start(win = window) {
      this.win = win;
      const loop = (ts) => { this.frame(ts); this.raf = win.requestAnimationFrame(loop); };
      this.raf = win.requestAnimationFrame(loop);
    }

    stop() { if (this.win && this.raf) this.win.cancelAnimationFrame(this.raf); }

    resize() {
      const dpr = (this.win || window).devicePixelRatio || 1;
      const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
      if (!w || !h) return false;
      if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
        this.canvas.width = Math.round(w * dpr);
        this.canvas.height = Math.round(h * dpr);
      }
      this.w = w; this.h = h; this.dpr = dpr;
      return true;
    }

    // Height of sea layer `i` at x (px, positive = up).
    wave(x, i) {
      const { w, h, k, t } = this;
      const amp = h * (0.012 + 0.11 * k) * [0.45, 0.7, 1][i];
      const len = w * [0.55, 0.42, 0.34][i] / (1 + 0.6 * k); // shorter, choppier waves in a storm
      const p = x / len * Math.PI * 2;
      const sp = [0.7, 1.0, 1.3][i];
      return amp * (0.62 * Math.sin(p + t * 1.6 * sp + i * 1.7)
        + 0.28 * Math.sin(p * 2.3 - t * 2.1 * sp + i)
        + 0.10 * Math.sin(p * 5.1 + t * 3.3 * sp) * k);
    }
    seaLevel(i) { return this.h * [0.58, 0.64, 0.72][i]; }

    frame(ts) {
      if (!this.resize()) return;
      const dt = this.last == null ? 0.016 : clamp((ts - this.last) / 1000, 0, 0.05); // never negative
      this.last = ts;
      this.clock += dt;
      this.k += (this.target - this.k) * (1 - Math.exp(-dt / 1.2));
      this.t += dt * (0.55 + 1.5 * this.k);

      const ctx = this.ctx;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      this.drawSky(ctx, dt);
      this.drawSea(ctx, 0);
      this.drawSea(ctx, 1);
      this.drawBoat(ctx, dt);
      this.drawSea(ctx, 2);
      this.drawSplash(ctx, dt);
      this.drawRain(ctx, dt);
      this.drawFlash(ctx, dt);
      if (this.hud) this.drawHud(ctx);
    }

    drawSky(ctx, dt) {
      const { w, h } = this;
      const k = this.k * this.k; // colours darken late, so the baseline (k ~ 0.33) still looks like a fair day
      const g = ctx.createLinearGradient(0, 0, 0, h * 0.65);
      g.addColorStop(0, mix(SKY_TOP[0], SKY_TOP[1], k));
      g.addColorStop(1, mix(SKY_LOW[0], SKY_LOW[1], k));
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);

      // Sun, fading out as clouds gather
      const sunA = clamp(1 - k * 1.8, 0, 1);
      if (sunA > 0) {
        const sx = w * 0.78, sy = h * 0.2, r = Math.min(w, h) * 0.08;
        const glow = ctx.createRadialGradient(sx, sy, r * 0.3, sx, sy, r * 3);
        glow.addColorStop(0, `rgba(255,236,190,${0.55 * sunA})`);
        glow.addColorStop(1, "rgba(255,236,190,0)");
        ctx.fillStyle = glow;
        ctx.fillRect(0, 0, w, h);
        ctx.fillStyle = `rgba(255,214,120,${sunA})`;
        ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2); ctx.fill();
      }

      // Clouds: more, darker and faster with stress
      const n = Math.round(2 + k * 4);
      for (let i = 0; i < n; i++) {
        const c = this.clouds[i];
        c.x += c.v * dt * (1 + 3 * k);
        if (c.x > 1.25) c.x = -0.25;
        this.drawCloud(ctx, c.x * w, c.y * h, Math.min(w, h) * 0.07 * c.s * (1 + 0.6 * k), k);
      }

      // Gulls when calm
      const gullA = clamp(1 - k * 2.5, 0, 1);
      if (gullA > 0) {
        ctx.strokeStyle = `rgba(60,60,70,${0.7 * gullA})`;
        ctx.lineWidth = 1.5;
        for (let i = 0; i < 2; i++) {
          const gx = ((this.clock * 0.025 + i * 0.37) % 1.2 - 0.1) * w;
          const gy = h * (0.18 + i * 0.07) + Math.sin(this.clock * 1.3 + i) * h * 0.015;
          const flap = Math.sin(this.clock * 6 + i * 2) * 0.35;
          const s = Math.min(w, h) * 0.022;
          ctx.beginPath();
          ctx.moveTo(gx - s, gy - s * (0.3 + flap));
          ctx.quadraticCurveTo(gx - s * 0.4, gy - s * 0.6, gx, gy);
          ctx.quadraticCurveTo(gx + s * 0.4, gy - s * 0.6, gx + s, gy - s * (0.3 + flap));
          ctx.stroke();
        }
      }
    }

    drawCloud(ctx, x, y, r, k) {
      ctx.fillStyle = mix(CLOUD[0], CLOUD[1], k);
      ctx.globalAlpha = 0.55 + 0.4 * k;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.arc(x + r * 0.9, y + r * 0.2, r * 0.75, 0, Math.PI * 2);
      ctx.arc(x - r * 0.9, y + r * 0.25, r * 0.65, 0, Math.PI * 2);
      ctx.arc(x + r * 0.2, y - r * 0.45, r * 0.7, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    drawSea(ctx, i) {
      const { w, h, k } = this;
      const base = this.seaLevel(i);
      const kc = k * k;
      const g = ctx.createLinearGradient(0, base - h * 0.1, 0, h);
      g.addColorStop(0, mix(SEA[i][0], SEA[i][1], kc));
      g.addColorStop(1, mix(SEA[i][1], "#0b1a2a", 0.4 + 0.4 * kc));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.moveTo(0, h);
      const step = Math.max(4, w / 120);
      for (let x = 0; x <= w + step; x += step) ctx.lineTo(x, base - this.wave(x, i));
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fill();

      // Foam on the crests once the sea gets rough
      if (k > 0.35) {
        ctx.strokeStyle = `rgba(255,255,255,${clamp((k - 0.35) * 1.6, 0, 0.85) * (0.5 + 0.25 * i)})`;
        ctx.lineWidth = 1.5 + i;
        ctx.lineCap = "round";
        for (let x = step; x < w; x += step) {
          const y0 = this.wave(x - step, i), y1 = this.wave(x, i), y2 = this.wave(x + step, i);
          if (y1 > y0 && y1 >= y2 && y1 > h * 0.02) {
            ctx.beginPath();
            ctx.moveTo(x - step * 2, base - this.wave(x - step * 2, i) + 1);
            ctx.lineTo(x, base - y1);
            ctx.lineTo(x + step * 1.5, base - this.wave(x + step * 1.5, i) + 1);
            ctx.stroke();
          }
        }
      }
    }

    drawBoat(ctx, dt) {
      const { w, h, k } = this;
      const b = this.boat;
      const x = w * 0.45 + Math.sin(this.clock * 0.25) * w * 0.04;
      const surface = this.seaLevel(1) - this.wave(x, 1);
      if (b.y == null) b.y = surface;

      // Buoyancy spring: less damping in a storm, so the boat overshoots and hops off crests
      const stiff = 55, damp = lerp(11, 3.2, k);
      b.vy += ((surface - b.y) * stiff - b.vy * damp) * dt;
      b.y += b.vy * dt;
      const prevVy = b.vy;

      const slope = (this.wave(x + 6, 1) - this.wave(x - 6, 1)) / 12;
      const targetA = -Math.atan(slope) * 0.9 + Math.sin(this.clock * 2.7) * 0.12 * k;
      b.va += ((targetA - b.angle) * 30 - b.va * lerp(9, 4, k)) * dt;
      b.angle += b.va * dt;

      // Splash when the hull slaps back down into the water
      if (k > 0.4 && b.y > surface && prevVy > h * 0.25) {
        for (let i = 0; i < 6 + 10 * k; i++) {
          this.splash.push({ x: x + (Math.random() - 0.5) * w * 0.12, y: surface,
            vx: (Math.random() - 0.5) * w * 0.25, vy: -Math.random() * h * (0.3 + 0.5 * k), life: 1 });
        }
      }

      const s = Math.min(w, h) * 0.0042;  // boat scale
      ctx.save();
      ctx.translate(x, b.y + 4 * s);
      ctx.rotate(b.angle);
      ctx.scale(s, s);

      // Mast
      ctx.strokeStyle = "#4a3426"; ctx.lineWidth = 2.2;
      ctx.beginPath(); ctx.moveTo(2, -4); ctx.lineTo(2, -62); ctx.stroke();
      // Flag, flapping harder in the wind
      const flap = Math.sin(this.clock * (6 + 10 * k)) * (2 + 5 * k);
      ctx.fillStyle = "#d9534f";
      ctx.beginPath(); ctx.moveTo(2, -62); ctx.quadraticCurveTo(9, -63 + flap * 0.4, 15, -59 + flap * 0.6);
      ctx.lineTo(2, -55); ctx.closePath(); ctx.fill();
      // Main sail
      const belly = 6 + 6 * k;
      const sail = ctx.createLinearGradient(-30, -55, 0, -8);
      sail.addColorStop(0, "#fffdf7"); sail.addColorStop(1, "#e9e2d0");
      ctx.fillStyle = sail;
      ctx.beginPath(); ctx.moveTo(0, -58); ctx.quadraticCurveTo(-18 - belly, -30, -30, -9); ctx.lineTo(0, -9); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = "rgba(90,80,60,0.35)"; ctx.lineWidth = 0.8; ctx.stroke();
      // Jib
      ctx.fillStyle = "#f3eee0";
      ctx.beginPath(); ctx.moveTo(4, -52); ctx.quadraticCurveTo(14 + belly * 0.6, -30, 26, -10); ctx.lineTo(4, -10); ctx.closePath(); ctx.fill();
      ctx.stroke();
      // Hull
      const hull = ctx.createLinearGradient(0, -10, 0, 10);
      hull.addColorStop(0, "#a4562f"); hull.addColorStop(1, "#6b3219");
      ctx.fillStyle = hull;
      ctx.beginPath();
      ctx.moveTo(-38, -9); ctx.lineTo(36, -9);
      ctx.quadraticCurveTo(30, 6, 18, 9); ctx.lineTo(-24, 9);
      ctx.quadraticCurveTo(-34, 4, -38, -9); ctx.closePath(); ctx.fill();
      // Trim stripe and portholes
      ctx.fillStyle = "#f2e6c9"; ctx.fillRect(-36, -8, 70, 2.4);
      ctx.fillStyle = "#e8d9b0";
      for (const px of [-16, -4, 8]) { ctx.beginPath(); ctx.arc(px, -1, 1.8, 0, Math.PI * 2); ctx.fill(); }
      ctx.restore();
    }

    drawSplash(ctx, dt) {
      const g = this.h * 1.6;
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      this.splash = this.splash.filter((p) => p.life > 0);
      for (const p of this.splash) {
        p.vy += g * dt; p.x += p.vx * dt; p.y += p.vy * dt; p.life -= dt * 1.4;
        ctx.globalAlpha = clamp(p.life, 0, 1);
        ctx.beginPath(); ctx.arc(p.x, p.y, Math.max(1, this.h * 0.006), 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    drawRain(ctx, dt) {
      const { w, h, k } = this;
      const want = k > 0.6 ? Math.round((k - 0.6) * 220) : 0;
      while (this.drops.length < want) this.drops.push({ x: Math.random() * w * 1.2, y: Math.random() * h });
      if (this.drops.length > want) this.drops.length = want;
      ctx.strokeStyle = "rgba(200,215,230,0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (const d of this.drops) {
        d.y += h * 1.4 * dt; d.x -= w * 0.25 * dt;
        if (d.y > h) { d.y = -10; d.x = Math.random() * w * 1.2; }
        ctx.moveTo(d.x, d.y); ctx.lineTo(d.x + h * 0.012, d.y - h * 0.045);
      }
      ctx.stroke();
    }

    drawFlash(ctx, dt) {
      if (this.k > 0.85 && Math.random() < dt * 0.25) this.flash = 1;
      if (this.flash > 0) {
        ctx.fillStyle = `rgba(255,255,255,${this.flash * 0.55})`;
        ctx.fillRect(0, 0, this.w, this.h);
        this.flash = Math.max(0, this.flash - dt * 3);
      }
    }

    drawHud(ctx) {
      const { w, h } = this;
      const fs = clamp(Math.min(w, h) * 0.07, 11, 18);
      ctx.font = `600 ${fs}px Inter, system-ui, sans-serif`;
      let text;
      if (this.score == null) text = this.message;
      else {
        const s = Math.round(this.score);
        const label = s < 30 ? "Calm sea" : s < 55 ? "Gentle swell" : s < 75 ? "Rough sea" : "Storm"; // same bands as the gauge
        text = `${label} · ${s}`;
      }
      const pad = fs * 0.6, tw = ctx.measureText(text).width;
      ctx.fillStyle = "rgba(10,20,35,0.45)";
      const x = pad, y = pad, bw = tw + pad * 2, bh = fs + pad * 1.2, r = bh / 2;
      ctx.beginPath();
      ctx.moveTo(x + r, y); ctx.arcTo(x + bw, y, x + bw, y + bh, r); ctx.arcTo(x + bw, y + bh, x, y + bh, r);
      ctx.arcTo(x, y + bh, x, y, r); ctx.arcTo(x, y, x + bw, y, r); ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.fillText(text, x + pad, y + bh / 2 + fs * 0.35);
    }
  }

  window.BoatScene = BoatScene;
})();
