"""Replay engine: historical events → normal TraceStore traces.

For each historical event the engine builds an as-of feature snapshot and an
as-of odds snapshot, runs the leakage guard, calls the **normal** ``analyze()``
path, persists a normal trace (with ``historical_replay=true`` + replay metadata)
into the *backtest* TraceStore, attaches the outcome and closing line, and —
when staking is enabled — records a ``historical_replay`` ledger bet plus a
``ReplayCandidateSelection`` audit row. Autolog is suppressed via the explicit
context manager so replay never pollutes the ledger with engine_auto rows.

Determinism: each event's simulation seed is
``sha256(manifest_id | config_hash | code_version | event_id)``.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

from omega.core.contracts.schemas import GameAnalysisRequest, OddsInput
from omega.core.contracts.service import analyze
from omega.historical.contracts import (
    HistoricalEvent,
    HistoricalOutcome,
    OddsObservation,
    ReplayCandidateSelection,
    ReplayConfig,
    ReplayEventRecord,
    ReplayTraceManifest,
)
from omega.historical.leakage import evaluate_leakage
from omega.historical.odds_snapshots import build_odds_snapshot
from omega.historical.snapshots import (
    MatchupHistory,
    TeamGameRow,
    build_feature_snapshot,
)
from omega.historical.staking import size_historical_bet
from omega.trace.persistable import PersistableTrace
from omega.trace.store import TraceStore


class LeakageError(RuntimeError):
    """Raised when an event fails the leakage guard under ``policy='fail'``."""


@dataclass
class ReplayDataset:
    """All inputs a replay needs, keyed by ``event_id`` (the canonical event_key)."""

    events: list[HistoricalEvent]
    outcomes: dict[str, HistoricalOutcome] = field(default_factory=dict)
    odds: dict[str, list[OddsObservation]] = field(default_factory=dict)
    extra_context: dict[str, dict] = field(default_factory=dict)
    # Tennis (and other non-team-score sports) supply per-participant rows here;
    # team-score sports leave this None and the engine derives history from
    # events + outcomes.
    history_override: dict[str, list[TeamGameRow]] | None = None

    @staticmethod
    def group_odds(observations: list[OddsObservation]) -> dict[str, list[OddsObservation]]:
        grouped: dict[str, list[OddsObservation]] = defaultdict(list)
        for obs in observations:
            grouped[obs.event_key].append(obs)
        return dict(grouped)


@dataclass
class ReplayResult:
    manifest: ReplayTraceManifest
    selections: list[ReplayCandidateSelection]
    n_persisted: int
    n_skipped: int


def derive_seed(config: ReplayConfig, event_id: str) -> int:
    """Deterministic per-event simulation seed (32-bit)."""
    raw = "|".join(
        [config.dataset_manifest_id, config.config_hash(), config.code_version, event_id]
    )
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def build_team_histories(
    events: list[HistoricalEvent], outcomes: dict[str, HistoricalOutcome]
) -> dict[str, list[TeamGameRow]]:
    hist: dict[str, list[TeamGameRow]] = defaultdict(list)
    for ev in events:
        oc = outcomes.get(ev.event_id)
        if not oc or oc.home_score is None or oc.away_score is None:
            continue
        hist[ev.home_team].append(
            TeamGameRow(
                date=ev.start_time,
                points_for=oc.home_score,
                points_against=oc.away_score,
                was_home=True,
                opponent=ev.away_team,
            )
        )
        hist[ev.away_team].append(
            TeamGameRow(
                date=ev.start_time,
                points_for=oc.away_score,
                points_against=oc.home_score,
                was_home=False,
                opponent=ev.home_team,
            )
        )
    return dict(hist)


def _odds_input_from_snapshot(snapshot) -> OddsInput | None:
    """Map decision quotes to the legacy flat OddsInput fields analyze() reads."""
    if snapshot.missing_odds:
        return None
    kw: dict = {}
    for q in snapshot.decision:
        market, sel = q.market, q.selection_descriptor
        first = sel.split("_", 1)[0]
        if market in ("moneyline", "home_draw_away"):
            if first == "home":
                kw["moneyline_home"] = q.odds
            elif first == "away":
                kw["moneyline_away"] = q.odds
            elif first == "draw":
                kw["moneyline_draw"] = q.odds
        elif market in ("spread", "puck_line", "run_line"):
            # puck line (hockey) and run line (baseball) are spread-type primitives.
            if first == "home":
                kw["spread_home"] = q.line
                kw["spread_home_price"] = q.odds
            elif first == "away":
                kw["spread_away_price"] = q.odds
        elif market == "total":
            if first == "over":
                kw["over_under"] = q.line
                kw["total_over_price"] = q.odds
            elif first == "under":
                kw["total_under_price"] = q.odds
    return OddsInput(**kw) if kw else None


class ReplayEngine:
    """Drives a deterministic replay of a historical dataset into a backtest DB."""

    def __init__(self, store: TraceStore, config: ReplayConfig) -> None:
        self.store = store
        self.config = config

    def run(self, dataset: ReplayDataset, *, replay_id: str, league: str) -> ReplayResult:
        events = sorted(dataset.events, key=lambda e: e.start_time)
        team_hist = dataset.history_override or build_team_histories(events, dataset.outcomes)

        manifest = ReplayTraceManifest(
            replay_id=replay_id,
            dataset_manifest_id=self.config.dataset_manifest_id,
            league=league.upper(),
            code_version=self.config.code_version,
            config_hash=self.config.config_hash(),
        )
        selections: list[ReplayCandidateSelection] = []
        n_persisted = 0
        n_skipped = 0

        for ev in events:
            record, event_sels, persisted = self._replay_event(ev, dataset, team_hist, replay_id)
            manifest.records.append(record)
            selections.extend(event_sels)
            if persisted:
                n_persisted += 1
            else:
                n_skipped += 1

        return ReplayResult(manifest, selections, n_persisted, n_skipped)

    def _replay_event(self, ev, dataset, team_hist, replay_id):
        decision_time = ev.start_time
        history = MatchupHistory(
            home_rows=team_hist.get(ev.home_team, []),
            away_rows=team_hist.get(ev.away_team, []),
        )
        extra = dataset.extra_context.get(ev.event_id, {})
        snapshot = build_feature_snapshot(
            ev, history, decision_time, extra_game_context=extra
        )
        obs = dataset.odds.get(ev.event_id, [])
        odds_snapshot = build_odds_snapshot(
            ev.event_id,
            obs,
            decision_time,
            event_start=ev.start_time,
            policy=self.config.decision_odds_policy,
        )

        leak = evaluate_leakage(ev, snapshot, odds_snapshot, policy=self.config.leakage_policy)
        if not leak.is_clean:
            if leak.status == "failed":
                raise LeakageError(f"{ev.event_id}: {', '.join(leak.reasons)}")
            return (
                ReplayEventRecord(
                    event_id=ev.event_id,
                    trace_id=None,
                    decision_time=decision_time,
                    feature_snapshot_hash=snapshot.feature_snapshot_hash,
                    odds_snapshot_hash=odds_snapshot.odds_snapshot_hash,
                    leakage_status=leak.status,
                    leakage_reasons=leak.reasons,
                    identity_status=ev.identity_status,
                    context_source=snapshot.context_source,
                    is_stale=snapshot.is_stale,
                    missing_odds=odds_snapshot.missing_odds,
                ),
                [],
                False,
            )

        seed = derive_seed(self.config, ev.event_id)
        request = GameAnalysisRequest(
            home_team=ev.home_team,
            away_team=ev.away_team,
            league=ev.league,
            odds=_odds_input_from_snapshot(odds_snapshot),
            n_iterations=self.config.n_iterations,
            simulation_backend=self.config.simulation_backend,
            allow_baseline=True,
            home_context=snapshot.home_context,
            away_context=snapshot.away_context,
            game_context=snapshot.game_context,
            seed=seed,
            evidence=[],
        )
        envelope = analyze(
            request, session_id=self.config.session_id, bankroll=self.config.bankroll
        )
        record = PersistableTrace.from_analyze_output(envelope).to_store_record()
        record.update(
            {
                "historical_replay": True,
                "replay_id": replay_id,
                "event_id": ev.event_id,
                "decision_time": decision_time,
                "dataset_manifest_id": self.config.dataset_manifest_id,
                "feature_snapshot_hash": snapshot.feature_snapshot_hash,
                "odds_snapshot_hash": odds_snapshot.odds_snapshot_hash,
                "leakage_status": leak.status,
            }
        )

        with self.store.autolog_suppressed():
            trace_id = self.store.persist(record)

        self._attach_outcome(trace_id, dataset.outcomes.get(ev.event_id))
        self._attach_closing(trace_id, odds_snapshot, decision_time)

        event_sels: list[ReplayCandidateSelection] = []
        ledger_ids: list[str] = []
        if self.config.enable_staking:
            result = size_historical_bet(
                record,
                replay_id=replay_id,
                event_id=ev.event_id,
                decision_time=decision_time,
                bankroll=self.config.bankroll,
            )
            if result.bet is not None and result.selection is not None:
                ledger_id = self.store.record_ledger_bet(result.bet)
                result.selection.ledger_id = ledger_id
                event_sels.append(result.selection)
                ledger_ids.append(ledger_id)

        record_out = ReplayEventRecord(
            event_id=ev.event_id,
            trace_id=trace_id,
            decision_time=decision_time,
            feature_snapshot_hash=snapshot.feature_snapshot_hash,
            odds_snapshot_hash=odds_snapshot.odds_snapshot_hash,
            leakage_status=leak.status,
            leakage_reasons=leak.reasons,
            identity_status=ev.identity_status,
            context_source=snapshot.context_source,
            is_stale=snapshot.is_stale,
            missing_odds=odds_snapshot.missing_odds,
            ledger_ids=ledger_ids,
        )
        return record_out, event_sels, True

    def _attach_outcome(self, trace_id: str, outcome: HistoricalOutcome | None) -> None:
        if outcome is None or outcome.home_score is None or outcome.away_score is None:
            return
        try:
            self.store.attach_outcome(
                trace_id, outcome.home_score, outcome.away_score, source="historical_replay"
            )
        except ValueError:
            # idempotent: an outcome is already attached for this trace
            pass

    def _attach_closing(self, trace_id: str, odds_snapshot, decision_time: str) -> None:
        for q in odds_snapshot.closing:
            self.store.attach_closing_line(
                trace_id,
                market=q.market,
                selection_descriptor=q.selection_descriptor,
                closing_odds=q.odds,
                closing_line=q.line,
                closing_timestamp=q.timestamp or decision_time,
                source="historical_replay",
            )
