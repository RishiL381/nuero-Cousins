"""Run `muselsl` with a fix for a Bleak race condition on Windows.

Bleak's WinRT backend (bleak 3.x) wraps Windows async operations in `FutureLike`. The result is
stored by a completion callback on a WinRT thread, but the asyncio side can see the operation as
COMPLETED before that callback has run, and then fails on `assert self._result` while reading the
Muse's GATT services. We make `result()` wait briefly for the callback, then fall back to reading
the result directly.

Usage: python -m clutch.muselsl_runner stream --ppg --acc [--name Muse-XXXX]
"""

import sys
import time


def patch_bleak():
    if sys.platform != "win32":
        return
    try:
        from bleak.backends.winrt import client as winrt_client
        from winrt.windows.foundation import AsyncStatus
    except Exception:
        return
    future_like = getattr(winrt_client, "FutureLike", None)
    if future_like is None or getattr(future_like, "_clutch_patched", False):
        return
    original = future_like.result

    def result(self):
        if (self._op.status == AsyncStatus.COMPLETED and self._result is None
                and not self._cancel_requested):
            deadline = time.monotonic() + 2.0
            while self._result is None and time.monotonic() < deadline:
                time.sleep(0.005)  # lets the WinRT completion callback store the result
            if self._result is None:
                self._result = self._op.get_results()
        return original(self)

    future_like.result = result
    future_like._clutch_patched = True


def relax_auto_disconnect(seconds=30):
    """muselsl quits after only 3 s without a packet; in a crowded Bluetooth room short gaps are common."""
    import muselsl.stream
    muselsl.stream.AUTO_DISCONNECT_DELAY = seconds


def main():
    patch_bleak()
    relax_auto_disconnect()
    from muselsl.__main__ import main as muselsl_main
    sys.argv = ["muselsl"] + sys.argv[1:]
    muselsl_main()


if __name__ == "__main__":
    main()
