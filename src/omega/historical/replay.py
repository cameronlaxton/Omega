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

from omega.core.contracts.schemas import GameAnalysisRequest, OddsInput, PlayerPropRequest
from omega.core.contracts.service import analyze
from omega.historical.contracts import (
    HistoricalEvent,
    HistoricalOutcome,
    HistoricalPropMarket,
    OddsObservation,
    ReplayCandidateSelection,
    ReplayConfig,
    ReplayEventRecord,
    ReplayTraceManifest,
)
from omega.historical.leakage import evaluate_leakage
from omega.historical.odds_snapshots import build_odds_snapshot
from omega.historical.odds_timing import is_selection_safe
from omega.historical.snapshots import (
    MatchupHistory,
    TeamGameRow,
    build_feature_snapshot,
)
from omega.historical.staking import size_historical_bet
from omega.trace.persistable import PersistableTrace
from omega.trace.store import TraceStore

# Schema version for the replay record provenance block. Bump when the shape of
# the persisted replay metadata (source_provenance, hashes) changes.
ARTIFACT_SCHEMA_VERSION = 1


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
    # Player-prop replay inputs (optional). prop_markets carries decision-time
    # lines/prices by event_key; prop_context carries as-of player context keyed
    # by "<event_key>|<player>|<stat_type>". Prop OUTCOMES live on the matching
    # HistoricalOutcome.prop_outcomes (post-game stat values).
    prop_markets: dict[str, list[HistoricalPropMarket]] = field(default_factory=dict)
    prop_context: dict[str, dict] = field(default_factory=dict)

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


def build_league_rating_index(
    events: list[HistoricalEvent], outcomes: dict[str, HistoricalOutcome]
) -> tuple[list[str], list[float]]:
    """Build a date-sorted index of every team-game score for as-of league means.

    Returns ``(dates, prefix_sums)`` where ``dates`` are the ISO start times of
    each team-game score (two per game — home and away) sorted ascending, and
    ``prefix_sums[i]`` is the cumulative points over the first ``i`` entries. Used
    by :func:`as_of_league_mean` to compute the league average score *strictly
    before* a decision time (no leakage) in O(log n).
    """
    entries: list[tuple[str, float]] = []
    for ev in events:
        oc = outcomes.get(ev.event_id)
        if not oc or oc.home_score is None or oc.away_score is None:
            continue
        entries.append((ev.start_time, float(oc.home_score)))
        entries.append((ev.start_time, float(oc.away_score)))
    entries.sort(key=lambda e: e[0])
    dates = [e[0] for e in entries]
    prefix = [0.0]
    for _, pts in entries:
        prefix.append(prefix[-1] + pts)
    return dates, prefix


def as_of_league_mean(index: tuple[list[str], list[float]], decision_time: str) -> float | None:
    """League average team-score over games strictly before ``decision_time``.

    Leak-safe: uses ``bisect_left`` so same-timestamp games are excluded (the
    conservative choice). Returns None when no prior games exist (early events),
    so the snapshot falls back to the raw rolling mean for those.
    """
    import bisect

    dates, prefix = index
    i = bisect.bisect_left(dates, decision_time)
    if i <= 0:
        return None
    return prefix[i] / i


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
        # As-of league-mean index for empirical-Bayes rating shrinkage (team-score
        # sports). Built from the same events+outcomes; queried leak-safely per event.
        self._league_index = build_league_rating_index(events, dataset.outcomes)

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
        league_mean = as_of_league_mean(
            getattr(self, "_league_index", ([], [0.0])), decision_time
        )
        league_baseline = (
            {"off_rating": league_mean, "def_rating": league_mean}
            if league_mean is not None
            else None
        )
        history = MatchupHistory(
            home_rows=team_hist.get(ev.home_team, []),
            away_rows=team_hist.get(ev.away_team, []),
            league_baseline=league_baseline,
        )
        extra = dataset.extra_context.get(ev.event_id, {})
        # Empirical-Bayes shrink team off/def toward the as-of league mean: tempers
        # over-dispersed small-sample rolling means (game-plane tail overconfidence).
        # No-op for tennis/non-team-score families (their context ignores baseline).
        snapshot = build_feature_snapshot(
            ev, history, decision_time, extra_game_context=extra, shrink_ratings=True
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
                # Queryable selection tag the calibration fitter uses to
                # include/exclude replayed traces. Overrides the default
                # "sandbox_<kind>"; the game/prop plane is still carried by
                # record["kind"] and the predictions payload. context_source is
                # intentionally left untouched ("provided") — eligibility depends
                # on it, so replay provenance lives here, not there.
                "execution_mode": "historical_replay",
                "historical_replay": True,
                "replay_id": replay_id,
                "event_id": ev.event_id,
                "decision_time": decision_time,
                "dataset_manifest_id": self.config.dataset_manifest_id,
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "feature_snapshot_hash": snapshot.feature_snapshot_hash,
                "odds_snapshot_hash": odds_snapshot.odds_snapshot_hash,
                "leakage_status": leak.status,
                "odds_timing_class": self.config.odds_timing_class,
                "source_provenance": {
                    "source_name": ev.source_name,
                    "dataset_manifest_id": self.config.dataset_manifest_id,
                    "source_row_ref": ev.source_row_ref,
                },
            }
        )

        with self.store.autolog_suppressed():
            trace_id = self.store.persist(record)

        self._attach_outcome(trace_id, dataset.outcomes.get(ev.event_id))
        self._attach_closing(trace_id, odds_snapshot, decision_time)
        self._replay_props(ev, dataset, replay_id, decision_time, snapshot.game_context)

        event_sels: list[ReplayCandidateSelection] = []
        ledger_ids: list[str] = []
        # Odds-timing guard: only decision_time_safe sources may drive staking.
        if self.config.enable_staking and is_selection_safe(self.config.odds_timing_class):
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

    def _replay_props(self, ev, dataset, replay_id, decision_time, game_context) -> None:
        """Replay each decision-time prop market for an event as a kind='prop' trace.

        Props stay league-scoped player-stat markets (no standalone props league).
        The line/prices are decision-time inputs; the realized stat_value is
        attached as the outcome only after the prediction is persisted.
        """
        markets = dataset.prop_markets.get(ev.event_id, [])
        if not markets:
            return
        outcome = dataset.outcomes.get(ev.event_id)
        po_by_key = {}
        if outcome is not None:
            po_by_key = {(po.player_name, po.stat_type): po for po in outcome.prop_outcomes}

        for m in markets:
            ctx = dict(dataset.prop_context.get(f"{ev.event_id}|{m.player_name}|{m.stat_type}", {}))
            seed = derive_seed(self.config, f"{ev.event_id}|{m.player_name}|{m.stat_type}")
            request = PlayerPropRequest(
                player_name=m.player_name,
                league=ev.league,
                home_team=ev.home_team,
                away_team=ev.away_team,
                game_date=decision_time[:10],
                prop_type=m.stat_type,
                line=m.line,
                odds_over=m.over_price,
                odds_under=m.under_price,
                player_context=ctx,
                game_context=game_context,
                seed=seed,
                evidence=[],
            )
            envelope = analyze(
                request, session_id=self.config.session_id, bankroll=self.config.bankroll
            )
            record = PersistableTrace.from_analyze_output(envelope).to_store_record()
            record.update(
                {
                    "execution_mode": "historical_replay",
                    "historical_replay": True,
                    "replay_id": replay_id,
                    "event_id": ev.event_id,
                    "decision_time": decision_time,
                    "dataset_manifest_id": self.config.dataset_manifest_id,
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "odds_timing_class": self.config.odds_timing_class,
                    "source_provenance": {
                        "source_name": ev.source_name,
                        "dataset_manifest_id": self.config.dataset_manifest_id,
                        "source_row_ref": ev.source_row_ref,
                    },
                }
            )
            with self.store.autolog_suppressed():
                trace_id = self.store.persist(record)

            po = po_by_key.get((m.player_name, m.stat_type))
            if po is not None:
                try:
                    self.store.attach_prop_outcome(
                        trace_id,
                        player_name=m.player_name,
                        stat_type=m.stat_type,
                        # stat_value is ignored by the store when void=True, but it
                        # still float()s the arg — pass 0.0 rather than None.
                        stat_value=po.stat_value if po.stat_value is not None else 0.0,
                        line=m.line,
                        side="over",
                        source="historical_replay",
                        void=po.void,
                    )
                except ValueError:
                    # idempotent: outcome already attached for this (trace, player, stat)
                    pass
