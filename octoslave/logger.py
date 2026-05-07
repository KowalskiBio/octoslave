"""Structured session logger for octoslave.

Writes one JSON Lines file per session under ~/.octoslave/logs/
with a rotating cap (keeps the 20 most-recent sessions by mtime).
Each line is a flat JSON object with at minimum::

    {
      "ts": "2026-05-07T14:32:01.123456",
      "lvl": "info",
      "evt": "turn_start",
      "iter": 3,
      ...payload
    }

Trace-style invariants:
  • one chronological stream per session
  • tool calls + results are paired with call_id
  • model tokens are captured as summaries (not full content, to keep files sane)
  • errors always include a stack / exception string when available
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_DIR = Path.home() / ".octoslave" / "logs"
_MAX_SESSIONS = 20          # keep N most-recent log files
_MAX_JSON_LEN = 2_000_000   # hard cap on a single JSON line (truncate payload)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_file: Any | None = None     # current open file handle
_session_id: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rotate() -> None:
    """Delete oldest log files when we exceed _MAX_SESSIONS."""
    try:
        files = sorted(_LOG_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        while len(files) > _MAX_SESSIONS:
            files.pop(0).unlink(missing_ok=True)
    except Exception:
        pass


def _init_file() -> Any:
    global _file, _session_id
    if _file is not None:
        return _file
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _rotate()
    _session_id = _now().replace(":", "-").replace(".", "-")
    path = _LOG_DIR / f"{_session_id}.jsonl"
    _file = open(path, "a", encoding="utf-8", buffering=1)  # line-buffered
    return _file


def _write(payload: dict) -> None:
    with _lock:
        fh = _init_file()
        line = json.dumps(payload, ensure_ascii=False, default=str)
        if len(line) > _MAX_JSON_LEN:
            line = json.dumps(
                {**payload, "_truncated": True, "_orig_len": len(line)},
                ensure_ascii=False,
                default=str,
            )
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_session_start(
    *,
    model: str,
    working_dir: str,
    backend: str,
    prompt_profile: str,
    permission_mode: str,
    enable_plan: bool,
    enable_verify: bool,
    enable_memory: bool,
) -> None:
    _write({
        "ts": _now(),
        "lvl": "info",
        "evt": "session_start",
        "model": model,
        "working_dir": working_dir,
        "backend": backend,
        "prompt_profile": prompt_profile,
        "permission_mode": permission_mode,
        "enable_plan": enable_plan,
        "enable_verify": enable_verify,
        "enable_memory": enable_memory,
    })


def log_turn(iteration: int, *, content_preview: str = "", finish_reason: str = "", tool_count: int = 0) -> None:
    _write({
        "ts": _now(),
        "lvl": "info",
        "evt": "turn",
        "iter": iteration,
        "content_preview": content_preview,
        "finish_reason": finish_reason,
        "tool_count": tool_count,
    })


def log_tool_call(call_id: str, name: str, args: dict) -> None:
    _write({
        "ts": _now(),
        "lvl": "info",
        "evt": "tool_call",
        "call_id": call_id,
        "name": name,
        "args": args,
    })


def log_tool_result(call_id: str, name: str, ok: bool, preview: str = "") -> None:
    _write({
        "ts": _now(),
        "lvl": "info" if ok else "error",
        "evt": "tool_result",
        "call_id": call_id,
        "name": name,
        "ok": ok,
        "preview": preview,
    })


def log_info(message: str, **extra) -> None:
    _write({
        "ts": _now(),
        "lvl": "info",
        "evt": "info",
        "msg": message,
        **extra,
    })


def log_error(message: str, exc: Exception | None = None, **extra) -> None:
    payload = {
        "ts": _now(),
        "lvl": "error",
        "evt": "error",
        "msg": message,
        **extra,
    }
    if exc is not None:
        payload["exc_type"] = type(exc).__name__
        payload["exc_str"] = str(exc)
    _write(payload)


def log_plan(plan_text: str) -> None:
    _write({
        "ts": _now(),
        "lvl": "info",
        "evt": "plan",
        "plan": plan_text,
    })


def log_verify(verdict: str) -> None:
    _write({
        "ts": _now(),
        "lvl": "info",
        "evt": "verify",
        "verdict": verdict,
    })


def log_session_end(iterations: int, reason: str = "completed") -> None:
    """reason: completed | interrupted | max_iterations | error"""
    _write({
        "ts": _now(),
        "lvl": "info",
        "evt": "session_end",
        "iterations": iterations,
        "reason": reason,
    })
    close()


def flush() -> None:
    with _lock:
        if _file is not None:
            _file.flush()


def close() -> None:
    global _file
    with _lock:
        if _file is not None:
            try:
                _file.close()
            except Exception:
                pass
            _file = None
