# Omega Run Recipe - Phase 6h

Omega now runs local VM / MCP-first. The retired lite bridge and standalone sandbox artifact are not part of the runtime.

## Preferred Path

```bash
pip install -e .[mcp]
python -m omega.mcp.server
```

Use the typed MCP analyze tools with explicit `session_id` and `bankroll`.

## Direct Python Path

Use this only when no MCP client is available:

```python
import hashlib
from omega.core.contracts.service import analyze

prompt = "Boston Celtics vs Indiana Pacers"
date = "2026-05-18"
seed = int.from_bytes(hashlib.sha256(f"{prompt}|{date}".encode("utf-8")).digest()[:4], "big")

result = analyze(
    {
        "home_team": "Boston Celtics",
        "away_team": "Indiana Pacers",
        "league": "NBA",
        "seed": seed,
        "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
        "away_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
        "odds": {"moneyline_home": -160, "moneyline_away": 140},
    },
    session_id="sess-20260518-a1b2",
    bankroll=1000.0,
)
```

The result is a trace envelope with `trace_id`, `session_id`, `bankroll`, `input_snapshot`, `result`, and `downgrades`.

## Downgrade Discipline

Before rendering a Bet Card, confirm:

- critical inputs are present;
- aggregate data quality is at least `0.7`;
- the engine status is not `skipped` or `error`;
- the trace ID came from Python execution.

If those checks fail, repair missing pre-decision inputs and rerun. If repair fails, produce qualitative research only.

## Verification

```bash
python -m pytest tests/ -v
```
