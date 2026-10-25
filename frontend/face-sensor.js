// Face sensor: webcam -> MediaPipe Face Landmarker (runs locally in the browser, no image leaves the page)
// -> stress-related indicators averaged over 1 s, sent to the Clutch server.
//
// Indicators (MediaPipe blendshapes are 0-1 scores of facial movements):
//   face_brow    brows lowered        (browDownLeft/Right)
//   face_press   lips pressed         (mouthPressLeft/Right)
//   face_squint  eyes squinted        (eyeSquintLeft/Right)
//   face_frown   mouth corners down   (mouthFrownLeft/Right)
//   face_smile   smile                (mouthSmileLeft/Right)
//   blink_rate   blinks per minute    (eyeBlink peaks)
//   head_motion  head movement, in inter-eye distances per second (nose tip landmark)
import { FaceLandmarker, FilesetResolver } from "./vendor/mediapipe/vision_bundle.mjs";

const BASE = new URL("vendor/mediapipe/", location.href).href;
const PAIRS = {
  face_brow: ["browDownLeft", "browDownRight"],
  face_press: ["mouthPressLeft", "mouthPressRight"],
  face_squint: ["eyeSquintLeft", "eyeSquintRight"],
  face_frown: ["mouthFrownLeft", "mouthFrownRight"],
  face_smile: ["mouthSmileLeft", "mouthSmileRight"],
};

let landmarker = null, stream = null, running = false, rafId = null;

async function createLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(BASE + "wasm");
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: BASE + "face_landmarker.task", delegate },
    outputFaceBlendshapes: true, runningMode: "VIDEO", numFaces: 1,
  });
  try {
    return await FaceLandmarker.createFromOptions(fileset, opts("GPU"));
  } catch (e) {
    console.warn("MediaPipe GPU delegate unavailable, using CPU", e);
    return await FaceLandmarker.createFromOptions(fileset, opts("CPU"));
  }
}

async function start(video, overlay, onFeatures, onStatus) {
  if (running) return;
  onStatus("Loading the face model…");
  landmarker = landmarker || await createLandmarker();
  onStatus("Asking for the camera…");
  stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
  video.srcObject = stream;
  await video.play();
  running = true;
  onStatus("Looking for a face…");

  const ctx = overlay.getContext("2d");
  let lastVideoTime = -1, prevNose = null, prevT = null, blinkOpen = true, lastBlinkT = 0;
  const blinks = [];                       // timestamps (ms) of detected blinks
  const startedAt = performance.now();
  let acc = null;
  const reset = () => ({ n: 0, total: 0, sums: Object.fromEntries(Object.keys(PAIRS).map((k) => [k, 0])), motion: 0, motionT: 0 });
  acc = reset();
  let windowStart = performance.now();

  const loop = () => {
    if (!running) return;
    const now = performance.now();
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const res = landmarker.detectForVideo(video, now);
      acc.total++;
      overlay.width = video.videoWidth; overlay.height = video.videoHeight;
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      const lm = res.faceLandmarks && res.faceLandmarks[0];
      const bs = res.faceBlendshapes && res.faceBlendshapes[0];
      if (lm && bs) {
        const score = Object.fromEntries(bs.categories.map((c) => [c.categoryName, c.score]));
        for (const [k, [l, r]] of Object.entries(PAIRS)) acc.sums[k] += ((score[l] || 0) + (score[r] || 0)) / 2;
        acc.n++;
        // blink: eyes closing past 0.5, at most one every 150 ms
        const blink = ((score.eyeBlinkLeft || 0) + (score.eyeBlinkRight || 0)) / 2;
        if (blinkOpen && blink > 0.5 && now - lastBlinkT > 150) { blinks.push(now); lastBlinkT = now; blinkOpen = false; }
        if (blink < 0.35) blinkOpen = true;
        // head motion: nose tip displacement, normalised by the distance between the outer eye corners
        const eyeDist = Math.hypot(lm[33].x - lm[263].x, lm[33].y - lm[263].y) || 1;
        if (prevNose && prevT) {
          acc.motion += Math.hypot(lm[1].x - prevNose.x, lm[1].y - prevNose.y) / eyeDist;
          acc.motionT += (now - prevT) / 1000;
        }
        prevNose = { x: lm[1].x, y: lm[1].y }; prevT = now;
        // draw the tracked points (mirrored like the video)
        ctx.fillStyle = "rgba(91, 155, 224, 0.75)";
        for (let i = 0; i < lm.length; i += 3) {
          ctx.fillRect((1 - lm[i].x) * overlay.width - 1, lm[i].y * overlay.height - 1, 2, 2);
        }
      } else {
        prevNose = null; prevT = null;
      }
    }
    // once per second: average, send, reset
    if (now - windowStart >= 1000) {
      while (blinks.length && now - blinks[0] > 60000) blinks.shift();
      const minutes = Math.max(10, Math.min(60, (now - startedAt) / 1000)) / 60;
      if (acc.n >= 3 && acc.n / Math.max(1, acc.total) > 0.5) {
        const features = Object.fromEntries(Object.entries(acc.sums).map(([k, v]) => [k, +(v / acc.n).toFixed(4)]));
        features.blink_rate = +(blinks.length / minutes).toFixed(1);
        features.head_motion = +(acc.motionT > 0 ? acc.motion / acc.motionT : 0).toFixed(4);
        onFeatures(features);
        onStatus("Tracking your face");
      } else {
        onFeatures(null);
        onStatus("No face in view: sit facing the camera, with light on your face");
      }
      acc = reset();
      windowStart = now;
    }
    rafId = requestAnimationFrame(loop);
  };
  rafId = requestAnimationFrame(loop);
}

function stop() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
}

window.ClutchFace = { start, stop, isRunning: () => running };
window.dispatchEvent(new Event("clutch-face-ready"));
