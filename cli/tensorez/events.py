"""JSONL event stream (CONTRACT.md §2).

One JSON object per line on stdout; every event carries ``event`` and ``t``
(monotonic seconds since run start).  With ``--pretty`` the stdout side is
rendered for humans instead, but the raw JSONL is always mirrored verbatim
to ``log.txt`` in the run directory once that directory exists (events
emitted before then are buffered and flushed into the log when it opens).

``progress`` events are throttled to ~10/s per stage; the final
(current == total) progress event is always emitted.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, IO

PROGRESS_MIN_INTERVAL = 0.1  # seconds -> <= ~10/s


class EventEmitter:
    def __init__(self, pretty: bool = False, stream: IO[str] | None = None):
        self.pretty = pretty
        self.stream = stream if stream is not None else sys.stdout
        self.start = time.monotonic()
        self._log_file: IO[str] | None = None
        self._pending_log_lines: list[str] = []
        self._last_progress: dict[str, float] = {}

    def now(self) -> float:
        return time.monotonic() - self.start

    def open_log(self, path: Path) -> None:
        self._log_file = open(path, "w", buffering=1)
        for line in self._pending_log_lines:
            self._log_file.write(line + "\n")
        self._pending_log_lines.clear()

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def emit(self, event: str, **fields: Any) -> None:
        payload: dict[str, Any] = {"event": event, "t": round(self.now(), 3)}
        payload.update({k: v for k, v in fields.items() if v is not None})
        line = json.dumps(payload)

        if self._log_file is not None:
            self._log_file.write(line + "\n")
        else:
            self._pending_log_lines.append(line)

        if self.pretty:
            rendered = self._render_pretty(payload)
            if rendered is not None:
                print(rendered, file=self.stream, flush=True)
        else:
            print(line, file=self.stream, flush=True)

    # -- typed helpers ------------------------------------------------------

    def progress(self, stage: str, current: int, total: int, message: str | None = None) -> None:
        now = time.monotonic()
        last = self._last_progress.get(stage, 0.0)
        if current < total and now - last < PROGRESS_MIN_INTERVAL:
            return
        self._last_progress[stage] = now
        self.emit("progress", stage=stage, current=current, total=total, message=message)

    def log(self, message: str, level: str = "info") -> None:
        self.emit("log", level=level, message=message)

    def error(self, message: str, stage: str | None = None, traceback: str | None = None) -> None:
        self.emit("error", message=message, stage=stage, traceback=traceback)

    # -- pretty rendering ---------------------------------------------------

    def _render_pretty(self, e: dict[str, Any]) -> str | None:
        t = f"[{e['t']:8.2f}s]"
        kind = e["event"]
        if kind == "run_start":
            return f"{t} run: {e['recipe_path']} -> {e['run_dir']} ({e['frame_count']} frames)"
        if kind == "stage_start":
            cached = " (cached)" if e.get("cached") else ""
            return f"{t} stage {e['stage']}{cached}"
        if kind == "progress":
            msg = f"  {e['message']}" if "message" in e else ""
            return f"{t}   {e['stage']}: {e['current']}/{e['total']}{msg}"
        if kind == "artifact":
            return f"{t}   {e['stage']}: wrote {e['path']}"
        if kind == "stage_end":
            return f"{t} stage {e['stage']} done in {e['seconds']:.2f}s"
        if kind == "log":
            return f"{t} {e['level']}: {e['message']}"
        if kind == "error":
            tb = ("\n" + e["traceback"]) if "traceback" in e else ""
            return f"{t} ERROR: {e['message']}{tb}"
        if kind == "done":
            return f"{t} done in {e['seconds']:.2f}s -> {e['final']}"
        return json.dumps(e)
