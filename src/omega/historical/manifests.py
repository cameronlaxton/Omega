"""On-disk persistence/registry for dataset and replay-run manifests.

Keeps IO concerns out of ``dataset_manifest.py`` (pure contract + hashing) and
``contracts.py`` (pure shapes). Artifacts live under ``var/historical/`` — the
canonical runtime root — unless an explicit root is supplied (tests use tmp
dirs).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from omega.historical.contracts import (
    BacktestReport,
    HistoricalEvent,
    HistoricalOutcome,
    HistoricalPropMarket,
    OddsObservation,
    ReplayCandidateSelection,
    ReplayTraceManifest,
)
from omega.historical.dataset_manifest import DatasetManifest
from omega.historical.snapshots import TeamGameRow

_DEFAULT_ROOT = Path("var/historical")


def _root(root: str | Path | None) -> Path:
    return Path(root) if root is not None else _DEFAULT_ROOT


def datasets_dir(root: str | Path | None = None) -> Path:
    return _root(root) / "datasets"


def replays_dir(root: str | Path | None = None) -> Path:
    return _root(root) / "replays"


# ---------------------------------------------------------------------------
# Dataset manifests
# ---------------------------------------------------------------------------


def save_dataset_manifest(manifest: DatasetManifest, root: str | Path | None = None) -> Path:
    """Persist a dataset manifest under ``<root>/datasets/<manifest_id>/manifest.json``."""
    out = datasets_dir(root) / manifest.manifest_id / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
    return out


def load_dataset_manifest(manifest_id: str, root: str | Path | None = None) -> DatasetManifest:
    path = datasets_dir(root) / manifest_id / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No dataset manifest at {path}")
    return DatasetManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def list_dataset_manifests(root: str | Path | None = None) -> list[str]:
    base = datasets_dir(root)
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "manifest.json").exists())


# ---------------------------------------------------------------------------
# Replay-run manifests
# ---------------------------------------------------------------------------


def save_replay_manifest(manifest: ReplayTraceManifest, root: str | Path | None = None) -> Path:
    out = replays_dir(root) / manifest.replay_id / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
    return out


def load_replay_manifest(replay_id: str, root: str | Path | None = None) -> ReplayTraceManifest:
    path = replays_dir(root) / replay_id / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No replay manifest at {path}")
    return ReplayTraceManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def list_replay_manifests(root: str | Path | None = None) -> list[str]:
    base = replays_dir(root)
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "manifest.json").exists())


def save_selections(
    replay_id: str,
    selections: list[ReplayCandidateSelection],
    root: str | Path | None = None,
) -> Path:
    out = replays_dir(root) / replay_id / "selections.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([s.model_dump(mode="json") for s in selections], indent=2), encoding="utf-8"
    )
    return out


def load_selections(
    replay_id: str, root: str | Path | None = None
) -> list[ReplayCandidateSelection]:
    path = replays_dir(root) / replay_id / "selections.json"
    if not path.exists():
        return []
    return [
        ReplayCandidateSelection.model_validate(d)
        for d in json.loads(path.read_text(encoding="utf-8"))
    ]


def save_replay_summary(replay_id: str, summary: dict, root: str | Path | None = None) -> Path:
    """Persist the machine-readable replay summary beside the replay manifest."""
    out = replays_dir(root) / replay_id / "replay_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def save_run_audit(replay_id: str, text: str, root: str | Path | None = None) -> Path:
    """Persist the human-readable RUN_AUDIT.md beside the replay manifest."""
    out = replays_dir(root) / replay_id / "RUN_AUDIT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def save_backtest_report(report: BacktestReport, root: str | Path | None = None) -> Path:
    out = replays_dir(root) / report.replay_id / "backtest_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    return out


def load_backtest_report(replay_id: str, root: str | Path | None = None) -> BacktestReport:
    path = replays_dir(root) / replay_id / "backtest_report.json"
    if not path.exists():
        raise FileNotFoundError(f"No backtest report at {path}")
    return BacktestReport.model_validate(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Normalized dataset persistence (ingest → replay handoff)
# ---------------------------------------------------------------------------


def save_normalized_dataset(
    manifest_id: str,
    *,
    events: list[HistoricalEvent],
    outcomes: list[HistoricalOutcome],
    odds: list[OddsObservation] | None = None,
    extra_context: dict[str, dict] | None = None,
    history_override: dict[str, list[TeamGameRow]] | None = None,
    prop_markets: dict[str, list[HistoricalPropMarket]] | None = None,
    prop_context: dict[str, dict] | None = None,
    root: str | Path | None = None,
) -> Path:
    """Persist the normalized, identity-resolved dataset for a manifest.

    Odds are stored as a flat list (each observation carries its event_key) and
    regrouped on load. ``history_override`` (tennis serve/return rows) is stored
    as JSON-encoded ``TeamGameRow`` dicts.
    """
    base = datasets_dir(root) / manifest_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "events.json").write_text(
        json.dumps([e.model_dump(mode="json") for e in events], indent=2), encoding="utf-8"
    )
    (base / "outcomes.json").write_text(
        json.dumps([o.model_dump(mode="json") for o in outcomes], indent=2), encoding="utf-8"
    )
    (base / "odds.json").write_text(
        json.dumps([o.model_dump(mode="json") for o in (odds or [])], indent=2), encoding="utf-8"
    )
    if extra_context:
        (base / "extra_context.json").write_text(
            json.dumps(extra_context, indent=2), encoding="utf-8"
        )
    if history_override:
        (base / "history_override.json").write_text(
            json.dumps(
                {k: [asdict(r) for r in rows] for k, rows in history_override.items()}, indent=2
            ),
            encoding="utf-8",
        )
    if prop_markets:
        flat = [m.model_dump(mode="json") for ms in prop_markets.values() for m in ms]
        (base / "prop_markets.json").write_text(json.dumps(flat, indent=2), encoding="utf-8")
    if prop_context:
        (base / "prop_context.json").write_text(json.dumps(prop_context, indent=2), encoding="utf-8")
    return base


def load_normalized_dataset(manifest_id: str, root: str | Path | None = None) -> dict:
    """Load a normalized dataset into the pieces ``ReplayDataset`` consumes."""
    base = datasets_dir(root) / manifest_id
    if not (base / "events.json").exists():
        raise FileNotFoundError(f"No normalized dataset at {base}")

    events = [
        HistoricalEvent.model_validate(d)
        for d in json.loads((base / "events.json").read_text(encoding="utf-8"))
    ]
    outcome_list = [
        HistoricalOutcome.model_validate(d)
        for d in json.loads((base / "outcomes.json").read_text(encoding="utf-8"))
    ]
    outcomes = {o.event_id: o for o in outcome_list}

    odds: dict[str, list[OddsObservation]] = {}
    odds_path = base / "odds.json"
    if odds_path.exists():
        for d in json.loads(odds_path.read_text(encoding="utf-8")):
            obs = OddsObservation.model_validate(d)
            odds.setdefault(obs.event_key, []).append(obs)

    extra_path = base / "extra_context.json"
    extra_context = (
        json.loads(extra_path.read_text(encoding="utf-8")) if extra_path.exists() else {}
    )

    history_override: dict[str, list[TeamGameRow]] | None = None
    hist_path = base / "history_override.json"
    if hist_path.exists():
        raw = json.loads(hist_path.read_text(encoding="utf-8"))
        history_override = {k: [TeamGameRow(**r) for r in rows] for k, rows in raw.items()}

    prop_markets: dict[str, list[HistoricalPropMarket]] = {}
    pm_path = base / "prop_markets.json"
    if pm_path.exists():
        for d in json.loads(pm_path.read_text(encoding="utf-8")):
            m = HistoricalPropMarket.model_validate(d)
            prop_markets.setdefault(m.event_key, []).append(m)

    pc_path = base / "prop_context.json"
    prop_context = (
        json.loads(pc_path.read_text(encoding="utf-8")) if pc_path.exists() else {}
    )

    return {
        "events": events,
        "outcomes": outcomes,
        "odds": odds,
        "extra_context": extra_context,
        "history_override": history_override,
        "prop_markets": prop_markets,
        "prop_context": prop_context,
    }
