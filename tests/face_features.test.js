// Checks the facial feature extractor on synthetic MediaPipe output.
// Run: node --test tests/face_features.test.js

const test = require("node:test");
const assert = require("node:assert");
const { FaceFeatureExtractor, FEATURE_COLUMNS, headPose } = require("../frontend/face-features.js");

const FPS = 30;
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -40, 1];

// Neutral face with a little jitter; `bumps` add a value to some blendshapes between two times (ms).
function run(durationMs, bumps = [], opts = {}) {
  const fx = new FaceFeatureExtractor(opts);
  let seed = 7;
  const noise = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 0.02;
  const events = [];
  for (let t = 0; t <= durationMs; t += 1000 / FPS) {
    const scores = { browDownLeft: 0.05, browDownRight: 0.05, eyeBlinkLeft: 0.1, eyeBlinkRight: 0.1, mouthPressLeft: 0.1, mouthPressRight: 0.1 };
    for (const k in scores) scores[k] += noise();
    for (const b of bumps) if (t >= b.from && t < b.to) for (const k of b.keys) scores[k] = b.value;
    events.push(...fx.addFrame(t, scores, IDENTITY).newEvents);
  }
  return { fx, events };
}

test("a short brow furrow is a micro-expression, a long one is macro", () => {
  const brows = ["browDownLeft", "browDownRight"];
  const { events } = run(12000, [
    { from: 5000, to: 5200, keys: brows, value: 0.5 },
    { from: 8000, to: 9500, keys: brows, value: 0.5 },
  ]);
  const furrow = events.filter((e) => e.key === "brow_furrow");
  assert.deepStrictEqual(furrow.map((e) => e.kind), ["micro", "macro"]);
  assert.ok(furrow[0].dur >= 150 && furrow[0].dur <= 300, furrow[0].dur);
  assert.strictEqual(events.length, 2, JSON.stringify(events));  // jitter alone triggers nothing
});

test("a held expression becomes the new baseline instead of repeating", () => {
  const { events } = run(20000, [{ from: 5000, to: 20000, keys: ["browDownLeft", "browDownRight"], value: 0.5 }]);
  assert.deepStrictEqual(events.map((e) => e.kind), ["macro"]);
});

test("no events during warm-up or from jitter", () => {
  const { events } = run(4000, [{ from: 1000, to: 1200, keys: ["browDownLeft", "browDownRight"], value: 0.5 }]);
  assert.strictEqual(events.length, 0);
});

test("blink rate and tick row", () => {
  const blinks = [];
  for (let t = 1000; t < 30000; t += 3000) blinks.push({ from: t, to: t + 150, keys: ["eyeBlinkLeft", "eyeBlinkRight"], value: 0.9 });
  const { fx } = run(30000, blinks);
  const row = fx.tick(30000);
  assert.deepStrictEqual(Object.keys(row), FEATURE_COLUMNS);
  assert.ok(Math.abs(row.blink_rate_min - 20) < 2.5, row.blink_rate_min);  // one blink every 3 s
  assert.ok(row.blink_dur_ms > 100 && row.blink_dur_ms < 220, row.blink_dur_ms);
  assert.strictEqual(row.face_present, 1);
  assert.ok(Math.abs(row.fps - FPS) < 1.5, row.fps);
  assert.ok(Math.abs(row.head_yaw) < 1e-6 && row.head_fidget_deg < 1e-6);
});

test("missing face lowers face_present and leaves features empty", () => {
  const fx = new FaceFeatureExtractor();
  for (let t = 0; t <= 2000; t += 1000 / FPS) fx.addFrame(t, null, null);
  const row = fx.tick(2000);
  assert.strictEqual(row.face_present, 0);
  assert.strictEqual(row.tension, null);
});

test("head pose reads yaw from a rotation about the vertical axis", () => {
  const a = (30 * Math.PI) / 180;
  // Ry(30 deg), column-major
  const m = [Math.cos(a), 0, -Math.sin(a), 0, 0, 1, 0, 0, Math.sin(a), 0, Math.cos(a), 0, 0, 0, -40, 1];
  const p = headPose(m);
  assert.ok(Math.abs(p.yaw - 30) < 1e-6 && Math.abs(p.pitch) < 1e-6 && Math.abs(p.roll) < 1e-6, JSON.stringify(p));
});
