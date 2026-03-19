"""Simple JSONL event logger for skills. Writes to configured log_path.
This is intentionally minimal and sync; heavy workloads should replace with
an async queue or external sink.
"""
from pathlib import Path
import json
from datetime import datetime
from . import config


def _ensure_log_dir():
    path = Path(config().get("log_path", "omega/skills/logs/"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_event(event: dict):
    path = _ensure_log_dir()
    ts = datetime.utcnow().isoformat() + "Z"
    event.setdefault("ts", ts)
    fname = path / "events.jsonl"
    with fname.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return fname
