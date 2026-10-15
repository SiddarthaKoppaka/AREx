"""Poll a running child's RSS via `ps`, the memory backstop where
RLIMIT_AS cannot be set (confirmed unsettable on macOS; Linux generally
supports it directly in `sandbox_worker.py` instead). Best-effort: polling
has latency, so a very fast allocation can exceed the limit briefly before
this notices and kills the process.
"""

import os
import signal
import subprocess
import threading
import time
from contextlib import suppress

POLL_INTERVAL_SECONDS = 0.05


def _rss_bytes(pid: int) -> int | None:
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    text = result.stdout.strip()
    return int(text) * 1024 if text.isdigit() else None


def watch(
    pid: int, memory_bytes: int, exceeded: threading.Event, stop: threading.Event
) -> None:
    while not stop.is_set():
        rss = _rss_bytes(pid)
        if rss is not None and rss > memory_bytes:
            exceeded.set()
            with suppress(OSError):
                os.kill(pid, signal.SIGKILL)
            return
        time.sleep(POLL_INTERVAL_SECONDS)
