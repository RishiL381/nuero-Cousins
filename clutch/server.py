"""FastAPI server: streams metrics to the web app over a WebSocket and serves the frontend.

Run:  python -m clutch --name Muse-XXXX          (starts muselsl and reads its LSL streams)
      python -m clutch --source mock             (simulated headset)
Then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import Config, Engine
from .face import FaceLogger
from .learned import LiveScorer
from .training import DATASET, MODELS, Jobs, dataset_info, model_report, recorded_sessions
from .sources import MUSE_BOARDS, LslSource, MockSource, MuseSource

ROOT = Path(__file__).resolve().parent.parent
# MediaPipe (frontend/vendor) needs these exact types; Windows' registry can map them wrongly.
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")


def create_app(engine: Engine) -> FastAPI:
    clients: set[WebSocket] = set()
    face_log = FaceLogger(engine.logger.root)
    face_owner: list = [None]  # the face tracker tab that is recording

    async def broadcast_loop():
        last_seq = 0
        while True:
            await asyncio.sleep(0.1)
            msgs = []
            if engine.pending_baseline is not None:
                msgs.append({"type": "baseline", "baseline": engine.pending_baseline})
                engine.pending_baseline = None
            m = engine.latest
            if m is not None and m["seq"] != last_seq:
                last_seq = m["seq"]
                msgs.append(dict(m, error=engine.error))
            for msg in msgs:
                text = json.dumps(msg)
                for ws in list(clients):
                    try:
                        await ws.send_text(text)
                    except Exception:
                        clients.discard(ws)

    @asynccontextmanager
    async def lifespan(app):
        engine.start()
        task = asyncio.create_task(broadcast_loop())
        yield
        task.cancel()
        face_log.stop()
        engine.stop()

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def no_stale_frontend(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"  # browsers revalidate, so edits show up on reload
        return response

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        clients.add(ws)
        src = engine.source
        await ws.send_text(json.dumps({
            "type": "hello", "source": src.kind, "eeg_names": list(src.eeg_names),
            "has_ppg": bool(src.ppg_fs), "calibrated": engine.model.calibration is not None,
            "headset_control": hasattr(src, "headset"), "default_name": getattr(getattr(src, "headset", None), "name", ""),
            "baseline": engine.model.calibration,
        }))
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                kind = msg.get("type")
                if kind == "phase":
                    engine.set_phase(msg["name"])
                elif kind == "session_start":
                    path = engine.logger.start(msg.get("meta", {}))
                    await ws.send_text(json.dumps({"type": "session", "dir": str(path)}))
                elif kind == "event":
                    engine.logger.event(engine.now(), engine.phase, msg.get("name", "event"), msg.get("data", {}))
                elif kind == "session_end":
                    engine.logger.finish(msg.get("summary", {}))
                elif kind == "face_features":  # home-page face sensor -> stress model
                    engine.set_face(msg.get("features") or {})
                elif kind == "headset_connect" and hasattr(src, "headset"):
                    src.headset.connect(msg.get("name"))
                elif kind == "headset_disconnect" and hasattr(src, "headset"):
                    src.headset.disconnect()
                elif kind == "watch_connect":
                    engine.watch.connect(msg.get("name"))  # on demand only: the watch serves one computer
                elif kind == "watch_disconnect":
                    engine.watch.disconnect()
                elif kind == "mock_stress" and isinstance(src, MockSource):
                    src.set_stress(msg.get("value"))
                elif kind == "face_start":
                    path = face_log.start(engine.logger.dir, msg.get("meta", {}),
                                          msg.get("feature_cols", []), msg.get("blendshape_cols", []))
                    face_owner[0] = ws
                    await ws.send_text(json.dumps({"type": "face_recording", "dir": str(path), "rows": 0}))
                elif kind == "face" and face_owner[0] is ws:
                    face_log.write(engine.now(), engine.phase, engine.latest, msg)
                    if msg.get("row") is not None:
                        await ws.send_text(json.dumps({"type": "face_recording", "dir": str(face_log.dir), "rows": face_log.rows}))
                elif kind == "face_stop" and face_owner[0] is ws:
                    face_log.stop()
                    face_owner[0] = None
                    await ws.send_text(json.dumps({"type": "face_recording", "dir": None}))
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(ws)
            if face_owner[0] is ws:
                face_log.stop()
                face_owner[0] = None

    # ---------- Dataset and model (step 4 of the home page) ----------
    jobs = Jobs()

    @app.get("/api/training")
    async def training_status():
        report, history = model_report()
        return {"sessions": recorded_sessions(engine.logger.root), "dataset": dataset_info(), "job": jobs.snapshot(),
                "recording": engine.logger.active, "report": report, "history": history,
                "live_model": engine.learned.info if engine.learned else None,
                "live_model_headset": engine.learned_headset.info if engine.learned_headset else None}

    @app.post("/api/dataset/build")
    async def build_dataset():
        if engine.logger.active:
            return JSONResponse({"error": "A session is being recorded: finish it first."}, status_code=409)
        if not jobs.start("dataset", ["tools/export_dataset.py"]):
            return JSONResponse({"error": "Another step is still running."}, status_code=409)
        return {"started": True}

    @app.post("/api/model/train")
    async def train_model():
        if dataset_info() is None:
            return JSONResponse({"error": "Build the dataset first."}, status_code=409)

        def reload_model():  # the app switches to the new models as soon as training succeeds
            engine.learned = LiveScorer.load(MODELS / "headset_free.joblib")
            engine.learned_headset = LiveScorer.load(MODELS / "with_headset.joblib")

        args = ["tools/train_model.py", "--data", str(DATASET), "--save", str(MODELS), "--notch", str(engine.cfg.notch_hz)]
        if not jobs.start("train", args, on_done=reload_model):
            return JSONResponse({"error": "Another step is still running."}, status_code=409)
        return {"started": True}

    app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    return app


def main(argv=None):
    p = argparse.ArgumentParser(description="Clutch: stress-aware performance trainer")
    p.add_argument("--source", choices=["muselsl", "lsl", "brainflow", "mock"], default="muselsl",
                   help="muselsl: connect the Muse at launch (default); "
                        "lsl: connect from the web app (Connect button) or use a muselsl started elsewhere; "
                        "brainflow: connect with BrainFlow instead; mock: simulated headset")
    p.add_argument("--name", "--serial-number", dest="name", default="",
                   help="Muse name shown in Bluetooth, e.g. Muse-4A2B (picks your headset in a busy room)")
    p.add_argument("--board", choices=list(MUSE_BOARDS), default="muse2", help="brainflow only")
    p.add_argument("--mac-address", default="", help="brainflow only")
    p.add_argument("--ppg-preset", default="p50", help="brainflow only: config string that enables PPG")
    p.add_argument("--notch", type=float, default=60.0, help="mains frequency: 60 (US) or 50 (EU)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if args.source in ("muselsl", "lsl"):
        source = LslSource(args.name, autoconnect=args.source == "muselsl")
    elif args.source == "brainflow":
        source = MuseSource(args.board, args.name, args.mac_address, args.ppg_preset)
    else:
        source = MockSource()
    learned = LiveScorer.load(MODELS / "headset_free.joblib")  # written by tools/train_model.py --save models
    engine = Engine(source, Config(notch_hz=args.notch), ROOT / "sessions", learned,
                    LiveScorer.load(MODELS / "with_headset.joblib"))

    import uvicorn
    print(f"Clutch running with {args.source} source -> http://{args.host}:{args.port}")
    if learned:
        print(f"Learned model loaded (heart + face, trained on {len(learned.info['trained_on'])} sessions, "
              f"validation AUC {learned.info['cv_auc']})")
    # Open browser tabs keep WebSockets alive: without a timeout, Ctrl-C would wait for them forever.
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=3)
