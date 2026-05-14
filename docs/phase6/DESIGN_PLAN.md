# Phase 6 Design Plan — Trace Persistence, Backtest Integration, Calibration Learning

## Prerequisites completed

- [x] Calibration drift bug fixed (`apply_calibration()` shared policy)
- [x] Parity test added (`test_calibration_parity_service_and_backtest`)
- [x] Red-team analysis of two-plane architecture (see `RED_TEAM.md`)
- [x] Skills infrastructure in place (trace-recorder writes JSONL)

---

## Part 1: Trace Persistence

### Current state
- `ExecutionTrace` is fully populated across 7 pipeline stages
- `trace_recorder.py` writes traces as JSONL (schema_version: 1)
- No retrieval API, no outcome attachment, no SQLite

### Design

#### Package: `omega/trace/`

```
omega/trace/
    __init__.py
    store.py          # TraceStore — SQLite persistence + retrieval
    models.py         # FrozenTrace, OutcomeRecord (typed Pydantic models)
    schema.py         # DDL strings, migration helpers, version tracking
```

#### SQLite schema (version 1)

```sql
-- traces table: one row per ExecutionTrace
CREATE TABLE traces (
    trace_id         TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL,
    timestamp        TEXT NOT NULL,          -- ISO 8601
    prompt           TEXT NOT NULL,
    league           TEXT,
    matchup          TEXT,                   -- "Away @ Home"
    execution_mode   TEXT,
    simulation_seed  INTEGER,
    aggregate_quality REAL,
    predictions      TEXT,                   -- JSON blob
    recommendations  TEXT,                   -- JSON blob
    odds_snapshot    TEXT,                   -- JSON blob
    downgrades       TEXT,                   -- JSON blob (list)
    full_trace       TEXT NOT NULL,          -- complete trace dict as JSON
    schema_version   INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_traces_league ON traces(league);
CREATE INDEX idx_traces_timestamp ON traces(timestamp);
CREATE INDEX idx_traces_matchup ON traces(matchup);

-- outcomes table: attached after initial trace persistence
CREATE TABLE outcomes (
    outcome_id       TEXT PRIMARY KEY,
    trace_id         TEXT NOT NULL REFERENCES traces(trace_id),
    home_score       INTEGER NOT NULL,
    away_score       INTEGER NOT NULL,
    result           TEXT NOT NULL,          -- "home_win", "away_win", "draw"
    attached_at      TEXT NOT NULL DEFAULT (datetime('now')),
    source           TEXT NOT NULL DEFAULT 'manual'  -- "manual", "api", "backtest"
);

CREATE INDEX idx_outcomes_trace_id ON outcomes(trace_id);

-- schema_versions table: tracks migrations
CREATE TABLE schema_versions (
    version          INTEGER PRIMARY KEY,
    applied_at       TEXT NOT NULL DEFAULT (datetime('now')),
    description      TEXT
);
```

#### TraceStore API

```python
class TraceStore:
    def __init__(self, db_path: str = "omega_traces.db"):
        """Initialize SQLite connection and ensure schema exists."""

    def persist(self, trace: Dict[str, Any]) -> str:
        """Write a trace. Returns trace_id. Idempotent on trace_id."""

    def attach_outcome(self, trace_id: str, home_score: int, away_score: int,
                       source: str = "manual") -> str:
        """Attach outcome to a persisted trace. Returns outcome_id."""

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full trace by ID."""

    def query_traces(self, league: str = None, start: str = None,
                     end: str = None, has_outcome: bool = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """Query traces with filters. Returns list of trace dicts."""

    def get_graded_traces(self, league: str = None) -> List[Dict[str, Any]]:
        """Return traces with attached outcomes (for calibration fitting)."""

    def schema_version(self) -> int:
        """Return current schema version."""
```

#### trace-recorder update

When `omega/trace/store.py` exists, update `trace_recorder.py._write_trace()`:

```python
def _write_trace(record):
    try:
        from omega.trace.store import TraceStore
        store = TraceStore()
        store.persist(record)
        return ""
    except ImportError:
        # Fallback to JSONL if omega/trace not yet available
        return _write_jsonl(record)
```

#### Design decisions

1. **SQLite, not Postgres.** Single-file, zero-config, embedded. Omega is not a multi-user server. SQLite handles the expected volume (~10K traces) without any deployment complexity.

2. **Full trace stored as JSON blob.** The `full_trace` column contains the entire trace dict. The denormalized columns (`league`, `matchup`, etc.) exist for querying only. This avoids schema coupling — the trace schema can evolve without SQLite migrations for every new field.

3. **Outcomes in a separate table.** Per CLAUDE.md: "outcome attachment must happen after initial trace persistence, not by mutating source records ad hoc." The `outcomes` table links to `traces` via `trace_id`. No mutation of the trace record.

4. **Idempotent persist.** `INSERT OR IGNORE` on `trace_id`. Safe to call multiple times with the same trace.

---

## Part 2: Backtest Integration — Frozen Artifacts

### Current state
- `HistoricalGame` is a bare dict subclass (no validation, no schema version)
- Backtest engine consumes hand-constructed dicts
- No connection between persisted traces and backtest inputs
- Backtest engine does not call `validate_sim_context()`

### Design

#### FrozenArtifact model

```python
class FrozenArtifact(BaseModel):
    """A historically valid, typed input for the backtest engine.

    Derived from a persisted ExecutionTrace + attached outcome.
    Every field is decision-time data; no post-outcome contamination.
    """
    # Identity
    artifact_id: str                      # deterministic hash of event identity
    schema_version: int = 1
    source_trace_id: Optional[str]        # links back to ExecutionTrace

    # Event
    home_team: str
    away_team: str
    league: str
    date: str                             # YYYY-MM-DD

    # Contexts (as used by the sim at decision time)
    home_context: Dict[str, Any]
    away_context: Dict[str, Any]

    # Odds (decision-time snapshot)
    odds: Dict[str, Any]                  # {moneyline_home, moneyline_away, spread_home, over_under}

    # Deterministic seed (as used by the orchestrator)
    simulation_seed: int

    # Calibration policy reference
    calibration_policy: str = "static_v1"  # Phase 6 profiles will use profile IDs

    # Outcome (attached only at grading time, NOT during simulation)
    outcome: Optional[Dict[str, Any]] = None  # {home_score, away_score}
    closing_odds: Optional[Dict[str, Any]] = None  # for CLV
```

#### Artifact converter: trace → frozen artifact

```python
def trace_to_artifact(trace: Dict[str, Any], outcome: Dict[str, Any]) -> FrozenArtifact:
    """Convert a graded trace into a frozen backtest artifact.

    Extracts decision-time data from the trace. Attaches outcome separately.
    Derives artifact_id from event identity (team+league+date).
    """
```

This function lives in `omega/strategy/artifacts.py`. It:
1. Extracts `home_context` and `away_context` from `trace["execution_result"]`
2. Extracts `odds` from `trace["odds_snapshot"]`
3. Uses `trace["simulation_seed"]` as the seed
4. Computes `artifact_id` as `sha256(home_team + away_team + league + date)`
5. Attaches outcome from the `outcomes` table

#### BacktestEngine update

Replace `HistoricalGame` consumption with `FrozenArtifact`:

```python
class BacktestEngine:
    def run(self, strategy: StrategyEntry,
            artifacts: List[FrozenArtifact]) -> BacktestResult:
        ...

    def _process_artifact(self, strategy: StrategyEntry,
                          artifact: FrozenArtifact) -> List[dict]:
        ...
```

The engine continues to accept the old `HistoricalGame` dict format for backward compatibility via a shim:

```python
def _compat_dict_to_artifact(game: dict) -> FrozenArtifact:
    """Convert legacy HistoricalGame dict to FrozenArtifact."""
```

#### BacktestResult update

Add trace linkage fields:

```python
class BacktestResult(BaseModel):
    # ... existing fields ...

    # Phase 6 additions
    artifact_schema_version: int = 1
    calibration_policy: str = "static_v1"
    trace_ids: List[str] = Field(default_factory=list)  # source trace IDs
```

#### Design decisions

1. **Artifacts derived from traces, not from a separate pipeline.** This closes the "context assembly" blind spot identified in the red team. What the orchestrator actually used is what the backtest evaluates.

2. **Backward-compatible shim.** Existing tests that use `HistoricalGame` dicts continue to work. New tests use `FrozenArtifact`.

3. **artifact_id is deterministic.** Same event always produces the same artifact_id. Prevents duplicate artifacts from multiple trace replays of the same game.

4. **Seed from trace, not re-derived.** The backtest uses the exact seed from the original trace. No risk of seed derivation drift.

---

## Part 3: Calibration Learning

### Current state
- Three methods: shrinkage, cap, isotonic (+ combined)
- Static parameters hardcoded
- `apply_calibration()` is the shared policy (just created)
- No profile storage, no fitting, no selection

### Design

#### Package additions in `omega/core/calibration/`

```
omega/core/calibration/
    __init__.py          # (unchanged)
    probability.py       # (unchanged — raw methods)
    profiles.py          # NEW: CalibrationProfile model
    fitter.py            # NEW: fit profiles from graded traces
    registry.py          # NEW: profile storage + selection policy
```

#### CalibrationProfile model

```python
class CalibrationProfile(BaseModel):
    """A versioned, attributable calibration configuration."""
    profile_id: str              # e.g. "iso_nba_v3"
    version: int
    method: str                  # "shrinkage", "cap", "isotonic", "combined"
    league: str                  # profile is league-specific
    status: str                  # "candidate", "staging", "production", "archived"

    # Method parameters
    params: Dict[str, Any]       # method-specific: {shrink_factor, cap_max, ...}
                                 # or {calibration_map: {...}} for isotonic

    # Training provenance
    training_window: str         # e.g. "2024-01-01 to 2024-12-31"
    sample_size: int
    dataset_hash: str            # hash of the artifact IDs used for fitting

    # Quality metrics (measured on held-out set)
    metrics: Dict[str, float]    # {brier_score, log_loss, calibration_error, ...}

    created_at: str              # ISO 8601
```

#### Fitter

```python
class CalibrationFitter:
    """Fit calibration profiles from graded traces."""

    def fit_isotonic(self, predictions: List[float],
                     outcomes: List[bool],
                     league: str) -> CalibrationProfile:
        """Fit an isotonic calibration map from historical predictions vs outcomes.

        Uses sklearn.isotonic.IsotonicRegression or a pure-Python equivalent.
        Bins raw probabilities and maps to observed win rates.
        """

    def fit_shrinkage(self, predictions: List[float],
                      outcomes: List[bool],
                      league: str) -> CalibrationProfile:
        """Optimize shrink_factor to minimize Brier score."""

    def evaluate(self, profile: CalibrationProfile,
                 predictions: List[float],
                 outcomes: List[bool]) -> Dict[str, float]:
        """Evaluate a profile on held-out data. Returns metrics dict."""
```

#### Registry (profile storage + selection policy)

```python
class CalibrationRegistry:
    """Stores versioned calibration profiles with promotion workflow.

    Storage: JSON file at omega/core/calibration/profiles.json
    (Phase 7+ could move to SQLite if volume warrants it.)
    """

    def register(self, profile: CalibrationProfile) -> None:
    def get_production(self, league: str) -> Optional[CalibrationProfile]:
    def promote(self, profile_id: str) -> None:
    def reject(self, profile_id: str, reason: str) -> None
    def list_profiles(self, league: str = None) -> List[CalibrationProfile]:
```

#### Selection policy update

Replace the hardcoded `apply_calibration()` with profile-aware selection:

```python
def apply_calibration(raw_prob: float, league: str = None) -> float:
    """Apply calibration using the active profile for the given league.

    Falls back to static policy if no production profile exists.
    """
    from omega.core.calibration.registry import CalibrationRegistry
    registry = CalibrationRegistry()
    profile = registry.get_production(league) if league else None

    if profile is None:
        # Static fallback (current behavior)
        if not should_apply_calibration(raw_prob, strict_cap=False):
            return raw_prob
        result = calibrate_probability(raw_prob, method="combined",
                                       shrink_factor=0.7, cap_max=0.90, cap_min=0.10)
        return result["calibrated"]

    # Profile-driven calibration
    result = calibrate_probability(raw_prob, method=profile.method, **profile.params)
    return result["calibrated"]
```

#### Promotion criteria

A candidate profile is promoted to production when:
1. Brier score improves over incumbent on held-out evaluation set
2. No degradation > 2% on broader ROI in backtest re-run
3. Sample size >= 100 graded predictions
4. Explicit comparison against incumbent is recorded

#### Design decisions

1. **League-specific profiles.** NBA and NFL have different probability distributions. A single global profile would be worse than no calibration for some sports.

2. **JSON file storage (not SQLite).** The profile registry will have ~10-50 entries. JSON is readable, diffable, and version-controllable in git. SQLite would be overkill here.

3. **Backward-compatible `apply_calibration()`.** The function signature adds `league` as an optional parameter. Existing calls without `league` get the static fallback. No breaking changes.

4. **No sklearn dependency for MVP.** The isotonic fitter can use a pure-Python implementation (binned observed rates). Add sklearn as optional enhancement later if calibration quality demands it.

5. **Profile fitting requires graded traces.** The fitter consumes `(prediction, outcome)` pairs from `TraceStore.get_graded_traces()`. No fitting until traces are persisted AND outcomes are attached. This is the correct dependency chain: Part 1 → Part 2 → Part 3.

---

## Files to create or modify

### New files

| File | Package | Role |
|---|---|---|
| `omega/trace/__init__.py` | trace | Package init |
| `omega/trace/store.py` | trace | TraceStore (SQLite CRUD) |
| `omega/trace/models.py` | trace | FrozenTrace type aliases |
| `omega/trace/schema.py` | trace | DDL, migrations, version tracking |
| `omega/strategy/artifacts.py` | strategy | FrozenArtifact model + trace-to-artifact converter |
| `omega/core/calibration/profiles.py` | calibration | CalibrationProfile model |
| `omega/core/calibration/fitter.py` | calibration | Fit profiles from graded data |
| `omega/core/calibration/registry.py` | calibration | Profile storage + selection |
| `tests/trace/test_trace_store.py` | tests | Persistence round-trip, outcome attachment |
| `tests/strategy/test_artifacts.py` | tests | Artifact conversion, deterministic ID |
| `tests/core/test_calibration_profiles.py` | tests | Profile fitting, selection, promotion |
| `docs/phase6/RED_TEAM.md` | docs | Red-team analysis (this session) |
| `docs/phase6/DESIGN_PLAN.md` | docs | This document |

### Modified files

| File | Change |
|---|---|
| `omega/skills/trace_recorder.py` | Update `_write_trace()` to call `TraceStore.persist()` with JSONL fallback |
| `omega/strategy/backtest/engine.py` | Accept `FrozenArtifact` alongside legacy dict; add `_compat_dict_to_artifact()` |
| `omega/strategy/models.py` | Add `artifact_schema_version`, `calibration_policy`, `trace_ids` to `BacktestResult` |
| `omega/core/calibration/probability.py` | Update `apply_calibration()` to check registry for league profile |
| `omega/core/models.py` | No changes needed — ExecutionTrace is already complete |

---

## Implementation order

### Phase 6a: Trace persistence (can ship independently)
1. Create `omega/trace/schema.py` — DDL strings + version table
2. Create `omega/trace/store.py` — TraceStore with persist, query, attach_outcome
3. Create `omega/trace/models.py` — type aliases
4. Update `omega/skills/trace_recorder.py` — call TraceStore with JSONL fallback
5. Write `tests/trace/test_trace_store.py` — round-trip, idempotent persist, outcome attachment
6. Migrate any existing JSONL traces into SQLite

### Phase 6b: Backtest integration (depends on 6a)
1. Create `omega/strategy/artifacts.py` — FrozenArtifact model + converter
2. Update `omega/strategy/backtest/engine.py` — accept FrozenArtifact, add compat shim
3. Update `omega/strategy/models.py` — add trace linkage fields to BacktestResult
4. Write `tests/strategy/test_artifacts.py` — conversion, deterministic ID, round-trip
5. Write a parity test: same game through orchestrator trace → artifact → backtest must produce identical sim output

### Phase 6c: Calibration learning (depends on 6a + 6b)
1. Create `omega/core/calibration/profiles.py` — CalibrationProfile model
2. Create `omega/core/calibration/registry.py` — JSON storage + selection
3. Create `omega/core/calibration/fitter.py` — fit from graded traces
4. Update `omega/core/calibration/probability.py` — profile-aware `apply_calibration()`
5. Write `tests/core/test_calibration_profiles.py` — fitting, selection, promotion, parity
6. Validate: backtest with learned profile vs static profile; compare Brier scores

---

## Risks and failure modes

| Risk | Phase | Mitigation |
|---|---|---|
| SQLite file locking under concurrent writes | 6a | Use WAL mode; trace writes are low-frequency (~1/sec max) |
| Outcome attachment to wrong trace | 6a | Match on (matchup, league, date) + manual confirmation |
| Artifact conversion loses fields | 6b | Round-trip test: trace → artifact → sim must match original sim output |
| Calibration profile overfits to small sample | 6c | Minimum sample_size=100; held-out evaluation mandatory |
| `apply_calibration(league=None)` breaks existing callers | 6c | Default `league=None` falls back to static policy; no breaking change |
| Profile registry grows stale | 6c | Archival workflow; promotion demotes previous production profile |

---

## Verification plan

### Phase 6a
- `test_trace_store.py`: persist → retrieve matches input
- `test_trace_store.py`: attach_outcome → get_graded_traces returns joined data
- `test_trace_store.py`: idempotent persist (same trace_id twice → no error, one row)
- `test_trace_store.py`: schema_version query works
- Manual: run orchestrator query → trace appears in SQLite → JSONL still written as fallback

### Phase 6b
- `test_artifacts.py`: trace_to_artifact produces valid FrozenArtifact
- `test_artifacts.py`: artifact_id is deterministic (same inputs → same ID)
- `test_artifacts.py`: backtest with FrozenArtifact produces identical result to legacy dict
- `test_artifacts.py`: legacy HistoricalGame dict still works via shim

### Phase 6c
- `test_calibration_profiles.py`: fit_isotonic produces a valid CalibrationProfile
- `test_calibration_profiles.py`: registry.get_production returns promoted profile
- `test_calibration_profiles.py`: apply_calibration with profile matches manual calculation
- `test_calibration_profiles.py`: parity — apply_calibration(league=None) matches static behavior
- `test_calibration_profiles.py`: promotion requires improvement on held-out set

---

## Rollback plan

- **6a:** Delete `omega/trace/` package + revert `trace_recorder.py` to JSONL-only. Traces already in SQLite can be exported to JSONL if needed. No data loss.
- **6b:** Delete `omega/strategy/artifacts.py` + revert engine to dict-only. Existing backtest tests use dicts and will continue to pass.
- **6c:** Revert `apply_calibration()` to static-only (remove registry lookup). Delete profiles.json. All behavior returns to pre-Phase-6c state.

Each phase is independently rollback-safe because they extend existing behavior rather than replacing it.
