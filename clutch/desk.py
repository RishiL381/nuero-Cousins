"""Desktop companion: Rishi's pixel boat in a small always-on-top window that drifts along the bottom of the
screen while you work, so your stress stays in the corner of your eye.

Run it while the app is running (the Live step also has a "Desktop boat" button):
    .venv\\Scripts\\python -m clutch.desk
The calmer you are, the slower it drifts and the flatter it rides; under stress it moves faster and pitches more.
Drag it anywhere: it stays put for a while, then drifts on. Close it with the x in its corner, or Esc.
"""
from __future__ import annotations

import argparse
import math
import threading
import time

SIZE = 176            # window side, px
MARGIN = 16           # distance from the screen edges, px
BOTTOM_GAP = 56       # keeps it above the Windows taskbar
HOLD_AFTER_DRAG_S = 20
FPS = 30


class Api:
    """Called from the page (widget.html?desk) through window.pywebview.api. Attributes are private: pywebview
    exposes every public attribute to the page and would walk the whole window object."""

    def __init__(self):
        self._stress = None    # 0-100, None before calibration
        self._window = None

    def set_stress(self, value):
        self._stress = None if value is None else max(0.0, min(100.0, float(value)))

    def close(self):
        if self._window:
            self._window.destroy()


def roam(window, api: Api, stop: threading.Event):
    import webview
    screen = webview.screens[0]
    left, right = MARGIN, screen.width - SIZE - MARGIN
    base_y = screen.height - SIZE - BOTTOM_GAP
    x, direction, t = float(right), -1, 0.0
    hold_until = 0.0
    last = (int(x), base_y)
    window.move(*last)
    while not stop.is_set():
        time.sleep(1 / FPS)
        t += 1 / FPS
        try:
            cur = (window.x, window.y)
        except Exception:  # window closed
            break
        if cur[0] is None:
            break
        if abs(cur[0] - last[0]) > 4 or abs(cur[1] - last[1]) > 4:  # the user dragged it: stay there a while,
            x, base_y = float(cur[0]), cur[1]                          # then drift on at that height
            hold_until, last = time.time() + HOLD_AFTER_DRAG_S, cur
            continue
        if time.time() < hold_until:
            continue
        s = (api._stress if api._stress is not None else 30.0) / 100
        x += direction * (10 + 50 * s) / FPS                     # 10 px/s when calm, 60 px/s at full stress
        if x <= left or x >= right:
            direction, x = -direction, min(max(x, left), right)
        bob = (3 + 12 * s) * math.sin(t * (0.8 + 1.6 * s))          # rides higher waves under stress
        new = (int(x), int(base_y + bob))
        if new != last:
            window.move(*new)
            last = new


def main(argv=None):
    p = argparse.ArgumentParser(description="Clutch desktop boat")
    p.add_argument("--url", default="http://127.0.0.1:8000/widget.html?desk")
    args = p.parse_args(argv)
    import webview
    api, stop = Api(), threading.Event()
    window = webview.create_window("Clutch", args.url, width=SIZE, height=SIZE, frameless=True, easy_drag=True,
                                   on_top=True, resizable=False, js_api=api, background_color="#12303b")
    api._window = window
    window.events.closed += stop.set
    webview.start(roam, (window, api, stop))


if __name__ == "__main__":
    main()
