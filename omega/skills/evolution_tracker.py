"""MVP evolution-tracker: reads events.jsonl and emits a simple summary.
This is intentionally synchronous and file-based for initial integration.
"""

import json
from pathlib import Path

from . import is_enabled, register


@register("evolution-tracker")
class EvolutionTracker:
    def __init__(self):
        self.name = "evolution-tracker"

    def summarize(self, log_path: str = None) -> dict:
        if not is_enabled(self.name):
            return {}
        lp = log_path or "omega/skills/logs/events.jsonl"
        p = Path(lp)
        if not p.exists():
            return {"summary": "no events"}
        events = [
            json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        # basic summary: count by skill, count of actions
        by_skill: dict[str, int] = {}
        total = 0
        for e in events:
            total += 1
            sk = e.get("skill") or e.get("event", {}).get("skill") or "unknown"
            by_skill[sk] = by_skill.get(sk, 0) + 1
        return {"total_events": total, "by_skill": by_skill}
