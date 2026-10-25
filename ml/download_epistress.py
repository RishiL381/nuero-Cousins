"""Download the part of EPIStress the pipeline uses (about 0.6 GB of the 7 GB zip, 53 GB unpacked).

EPIStress: Zenodo 10.5281/zenodo.16407549, CC BY 4.0. Only the lab tasks are fetched: per-task Muse EEG and
accelerometer, Empatica heart rate and pulse wave, and the self-report labels. Files are read straight out of
the remote zip with HTTP range requests, 3 at a time, backing off when Zenodo rate-limits (HTTP 429).
Already downloaded files are skipped, so it can be re-run after an interruption.

Run:  python ml/download_epistress.py           -> data/epistress/   (takes 15-30 min)
      python ml/epistress.py                    -> data/epistress_windows.csv (about 1 min)
"""

import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from remotezip import RemoteZip

URL = "https://zenodo.org/records/16407549/files/EPIStress.zip?download=1"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "epistress"
TASKS = "relaxation_video|arithmetix_easy|arithmetix_hard|stroop_easy|stroop_hard|n_back_easy|n_back_hard|sudoku_easy|sudoku_hard"
KEEP = re.compile(
    rf"ES\d+/Lab/Labeled/({TASKS})/(Muse__RAW_(TP9|AF7|AF8|TP10)|Muse__ACC_[XYZ]|E4__HR|E4__BVP)\.pickle$"
    r"|ES\d+/Lab/(Task_Labels|Panas_Labels)\.csv$"
    r"|ES\d+/Notes\.pdf$")
local = threading.local()
done = [0]


def fetch(name, total):
    dest = OUT / Path(name).relative_to("EPIStress")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(8):
        try:
            if not hasattr(local, "z"):
                local.z = RemoteZip(URL)
            tmp.write_bytes(local.z.read(name))
            break
        except Exception as e:  # 429 rate limit or dropped connection: back off and retry
            wait = 5 * 2 ** attempt
            print(f"retry {name} in {wait}s ({str(e)[:50]})", flush=True)
            time.sleep(wait)
            local.z = RemoteZip(URL)
    else:
        raise RuntimeError(f"gave up on {name}")
    tmp.rename(dest)
    done[0] += 1
    if done[0] % 50 == 0 or done[0] == total:
        print(f"{done[0]}/{total}", flush=True)


def main():
    with RemoteZip(URL) as z:
        todo = [i.filename for i in z.infolist() if KEEP.search(i.filename)]
    todo = [f for f in todo if not (OUT / Path(f).relative_to("EPIStress")).exists()]
    print(f"{len(todo)} files to fetch into {OUT}", flush=True)
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda f: fetch(f, len(todo)), todo))
    print("done: now run python ml/epistress.py")


if __name__ == "__main__":
    main()
