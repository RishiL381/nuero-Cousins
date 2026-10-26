# Clutch live demo

A static page that replays a real Clutch recording second by second, as the app showed it live: electrodes, band
power, heart rate, face indicators, the brain (teacher) and heart + face (student) scores, the pixel sea, what the
calibration learned and the model results. It needs no headset and no server: open `index.html` through any static
web server (GitHub Pages works as is).

`data.json` is generated from a recording by the app's own code:

```bash
.venv\Scripts\python tools\build_demo.py
```

The recording shown is a team member's (P01, study protocol), shared with consent. Other participants appear only
as aggregate results. Wellness prototype, not a medical device.

Run it locally:

```bash
.venv\Scripts\python -m http.server 8002 --directory demo
```
