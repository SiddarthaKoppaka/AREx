"""Persistent Python REPL for model-written code, with output and time limits."""

import contextlib
import io
import signal
import threading
import traceback
from typing import Any


class Repl:
    def __init__(self, timeout: float = 10.0, max_output: int = 3000) -> None:
        self.namespace: dict[str, Any] = {}
        self.timeout, self.max_output = timeout, max_output

    def _alarm(self, *_: Any) -> None:
        raise TimeoutError(f"code exceeded {self.timeout}s")

    def run(self, code: str) -> str:
        out = io.StringIO()
        timed = threading.current_thread() is threading.main_thread()
        if timed:
            signal.signal(signal.SIGALRM, self._alarm)
            signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:
            with contextlib.redirect_stdout(out):
                exec(compile(code, "<repl>", "exec"), self.namespace)  # noqa: S102
        except BaseException:  # model code may raise anything, even SystemExit
            out.write(traceback.format_exc(limit=-3))
        finally:
            if timed:
                signal.setitimer(signal.ITIMER_REAL, 0)
        text = out.getvalue().rstrip()
        if len(text) > self.max_output:
            text = text[: self.max_output] + f"\n...[truncated {len(text)} chars]"
        return text or "(no output)"
