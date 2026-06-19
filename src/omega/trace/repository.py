"""Postgres implementation behind the public ``TraceStore`` API."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, exists, func, select, text, update
from sqlalchemy import insert as sa_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from omega.trace.bet_record import BetRecord
from omega.trace.bet_settlement import extract_recommended_bet
from omega.trace.db import create_postgres_engine, create_session_factory
from omega.trace.ledger_bet import (
    DEFAULT_BANKROLL,
    BetProvenance,
    LedgerBet,
    LedgerStatus,
)
from omega.trace.market_snapshot import (
    EarlyMarketSnapshot,
    MarketMovement,
    MarketSnapshot,
)
from omega.trace.models import (
    BetLedgerRow,
    ClosingLineRow,
    EarlyMarketSnapshotRow,
    EvidenceSignalRow,
    MarketSnapshotRow,
    OutcomeRow,
    PropOutcomeRow,
    SchemaVersionRow,
    SignalPerformanceRow,
    SimulationDistributionRow,
    TraceQaVerdictRow,
    TraceRow,
)
from omega.trace.prop_outcome import derive_prop_outcome_result, normalize_prop_side
from omega.trace.schema import CURRENT_VERSION

UTC = timezone.utc
logger = logging.getLogger("omega.trace.repository")


def _row_dict(row: Any) -> dict[str, Any]:
    return dict(row._mapping if hasattr(row, "_mapping") else row)


def _json_or_none(value: Any) -> str | None:
    return json.dumps(value, default=str) if value else None


class PostgresRepository:
    """SQLAlchemy repository with one session/transaction per public call."""

    _LEDGER_COLUMNS = (
        "ledger_id",
        "trace_id",
        "bet_date",
        "league",
        "sport",
        "matchup",
        "market",
        "bookmaker",
        "selection",
        "selection_descriptor",
        "line",
        "odds",
        "stake_amount",
        "payout_amount",
        "net_pnl",
        "bankroll_at_open",
        "status",
        "provenance",
        "decision_timestamp",
        "graded_at",
        "session_id",
        "created_at",
        "staking_policy_id",
        "staking_policy_version",
        "exposure_limits_version",
        "sizing_reasons",
        "correlation_group",
    )

    def __init__(self, url: str, *, read_only: bool = False) -> None:
        self.url = url
        self.read_only = bool(read_only)
        self.engine = create_postgres_engine(url)
        self.Session = create_session_factory(self.engine)

    def _ensure_writeable(self) -> None:
        if self.read_only:
            raise RuntimeError("TraceStore opened read_only=True; writes are disabled")

    def persist(self, trace: dict[str, Any] | Any) -> str:
        self._ensure_writeable()
        if hasattr(trace, "to_store_record"):
            trace = trace.to_store_record()
        elif hasattr(trace, "model_dump"):
            trace = trace.model_dump(mode="json")

        trace_id = str(trace.get("trace_id", ""))
        run_id = str(trace.get("run_id", ""))
        timestamp = str(trace.get("timestamp", ""))
        if not trace_id or not run_id or not timestamp:
            raise ValueError(
                f"Trace missing required fields: trace_id={trace_id!r}, "
                f"run_id={run_id!r}, timestamp={timestamp!r}"
            )

        session_id = trace.get("session_id")
        session_id = str(session_id) or None if session_id is not None else None

        # V20 (Phase 8): mirror the SQLite parameter_profile_ref provenance column
        # so a Postgres persist does not silently drop backend-parameter provenance.
        from omega.trace.parameter_profiles import extract_parameter_profile_ref

        values = {
            "trace_id": trace_id,
            "run_id": run_id,
            "timestamp": timestamp,
            "prompt": trace.get("prompt", ""),
            "league": trace.get("league"),
            "matchup": trace.get("matchup"),
            "execution_mode": trace.get("execution_mode"),
            "simulation_seed": trace.get("simulation_seed"),
            "aggregate_quality": trace.get("aggregate_quality", 0.0),
            "predictions": _json_or_none(trace.get("predictions")),
            "recommendations": _json_or_none(trace.get("recommendations")),
            "odds_snapshot": _json_or_none(trace.get("odds_snapshot")),
            "downgrades": json.dumps(trace.get("downgrades", []), default=str),
            "full_trace": json.dumps(trace, default=str),
            "schema_version": CURRENT_VERSION,
            "session_id": session_id,
            "parameter_profile_ref": _json_or_none(extract_parameter_profile_ref(trace)),
        }

        with self.Session() as session:
            with session.begin():
                stmt = (
                    pg_insert(TraceRow.__table__)
                    .values(values)
                    .on_conflict_do_nothing(index_elements=["trace_id"])
                    .returning(TraceRow.trace_id)
                )
                result = session.execute(stmt)
                inserted = result.first() is not None
                if inserted:
                    self._write_evidence_signals(session, trace_id, trace)
                    self._write_simulation_distributions(session, trace_id, trace)
                    self._maybe_autolog_ledger_bet(session, trace_id, trace)
        return trace_id

    def _maybe_autolog_ledger_bet(self, session, trace_id: str, trace: dict[str, Any]) -> None:
        import os

        if os.environ.get("OMEGA_BET_LEDGER_AUTOLOG", "1") not in ("1", "true", "True"):
            return
        try:
            result = extract_recommended_bet(trace, provenance=BetProvenance.ENGINE_AUTO)
            if result.bet is not None:
                self._record_ledger_bet(session, result.bet)
        except Exception as exc:  # noqa: BLE001
            logger.warning("bet_ledger autolog skipped for %s: %s", trace_id, exc)

    def _write_simulation_distributions(self, session, trace_id: str, trace: dict[str, Any]) -> int:
        rows_in = trace.get("simulation_distributions")
        if not isinstance(rows_in, list) or not rows_in:
            result = trace.get("result") or {}
            rows_in = result.get("simulation_distributions") or []
        if not isinstance(rows_in, list) or not rows_in:
            return 0

        rows: list[dict[str, Any]] = []
        for item in rows_in:
            if not isinstance(item, dict):
                continue
            dist_type = item.get("distribution_type")
            target = item.get("target")
            if not dist_type or not target:
                continue
            rows.append(
                {
                    "trace_id": trace_id,
                    "kind": trace.get("kind"),
                    "league": trace.get("league"),
                    "target": target,
                    "market": item.get("market"),
                    "stat_key": item.get("stat_key"),
                    "distribution_type": dist_type,
                    "distribution_params": json.dumps(
                        item.get("distribution_params") or {}, default=str, sort_keys=True
                    ),
                    "params_schema_version": int(item.get("params_schema_version") or 1),
                    "sample_mean": item.get("sample_mean"),
                    "sample_std": item.get("sample_std"),
                    "p10": item.get("p10"),
                    "p50": item.get("p50"),
                    "p90": item.get("p90"),
                    "n_iterations": item.get("n_iterations"),
                    "seed": item.get("seed", trace.get("simulation_seed")),
                    "context_hash": item.get("context_hash"),
                    "component_version": item.get("component_version") or trace.get("model_version"),
                }
            )
        if rows:
            session.execute(sa_insert(SimulationDistributionRow.__table__), rows)
        return len(rows)

    def get_simulation_distributions(self, trace_id: str) -> list[dict[str, Any]]:
        table = SimulationDistributionRow.__table__
        with self.Session() as session:
            rows = session.execute(
                select(table)
                .where(table.c.trace_id == trace_id)
                .order_by(table.c.distribution_id)
            ).mappings()
            result = []
            for row in rows:
                data = dict(row)
                try:
                    data["distribution_params"] = json.loads(data["distribution_params"])
                except (TypeError, json.JSONDecodeError):
                    data["distribution_params"] = {}
                result.append(data)
            return result

    def _write_evidence_signals(self, session, trace_id: str, trace: dict[str, Any]) -> int:
        input_snap = trace.get("input_snapshot") or {}
        evidence = input_snap.get("evidence") or []
        if not isinstance(evidence, list) or not evidence:
            return 0

        league = trace.get("league")
        application = trace.get("evidence_application")
        if not isinstance(application, list):
            application = []
        trace_evidence_mode = trace.get("evidence_mode")

        rows: list[dict[str, Any]] = []
        for idx, sig in enumerate(evidence):
            if not isinstance(sig, dict):
                continue
            app = (
                application[idx]
                if idx < len(application) and isinstance(application[idx], dict)
                else {}
            )
            rows.append(
                {
                    "trace_id": trace_id,
                    "signal_type": sig.get("signal_type"),
                    "category": sig.get("category"),
                    "plane": sig.get("plane"),
                    "source": sig.get("source"),
                    "confidence": sig.get("confidence"),
                    "obs_window": sig.get("window"),
                    "direction": sig.get("direction"),
                    "stat_key": sig.get("stat_key"),
                    "league": league,
                    "value_json": json.dumps(sig.get("value"), default=str),
                    "applied": 1 if app.get("applied") else 0,
                    "applied_factor": app.get("factor"),
                    "policy_version": app.get("policy_version"),
                    "evidence_mode": app.get("evidence_mode") or trace_evidence_mode,
                }
            )
        if rows:
            session.execute(sa_insert(EvidenceSignalRow.__table__), rows)
        return len(rows)

    def get_evidence_signals(self, trace_id: str) -> list[dict[str, Any]]:
        table = EvidenceSignalRow.__table__
        with self.Session() as session:
            rows = session.execute(
                select(table).where(table.c.trace_id == trace_id).order_by(table.c.id)
            ).mappings()
            return [dict(row) for row in rows]

    def write_qa_verdict(
        self,
        trace_id: str,
        verdict: Any,
        *,
        session_id: str | None = None,
        ran_at: str | None = None,
    ) -> None:
        self._ensure_writeable()
        table = TraceQaVerdictRow.__table__
        values = {
            "trace_id": trace_id,
            "session_id": session_id,
            "verdict": verdict.verdict,
            "scope": verdict.scope,
            "gate_name": verdict.gate_name,
            "reason": verdict.reason,
            "event_id": verdict.event_id,
            "matched_trace_id": verdict.matched_trace_id,
            "ran_at": ran_at,
        }
        with self.Session() as session:
            with session.begin():
                stmt = pg_insert(table).values(values)
                excluded = stmt.excluded
                session.execute(
                    stmt.on_conflict_do_update(
                        index_elements=["trace_id"],
                        set_={
                            "session_id": excluded.session_id,
                            "verdict": excluded.verdict,
                            "scope": excluded.scope,
                            "gate_name": excluded.gate_name,
                            "reason": excluded.reason,
                            "event_id": excluded.event_id,
                            "matched_trace_id": excluded.matched_trace_id,
                            "ran_at": excluded.ran_at,
                        },
                    )
                )

    def get_qa_verdict(self, trace_id: str) -> dict[str, Any] | None:
        table = TraceQaVerdictRow.__table__
        with self.Session() as session:
            row = session.execute(select(table).where(table.c.trace_id == trace_id)).mappings().first()
            return dict(row) if row else None

    def upsert_signal_performance(self, rows: list[Any], dataset_hash: str) -> int:
        self._ensure_writeable()
        scored_at = datetime.now(UTC).isoformat()
        payload = [
            {
                "signal_type": r.signal_type,
                "source": r.source,
                "obs_window": r.obs_window,
                "league": r.league,
                "sample_size": int(r.sample_size),
                "direction_correct": int(r.direction_correct),
                "direction_accuracy": float(r.direction_accuracy),
                "mean_confidence": float(r.mean_confidence),
                "realized_hit_rate": float(r.realized_hit_rate),
                "calibration_gap": float(r.calibration_gap),
                "brier": float(r.brier),
                "dataset_hash": dataset_hash,
                "scored_at": scored_at,
            }
            for r in rows
        ]
        if not payload:
            return 0

        table = SignalPerformanceRow.__table__
        with self.Session() as session:
            with session.begin():
                for row in payload:
                    stmt = pg_insert(table).values(row)
                    excluded = stmt.excluded
                    session.execute(
                        stmt.on_conflict_do_update(
                            index_elements=[
                                "signal_type",
                                "source",
                                "obs_window",
                                "league",
                                "dataset_hash",
                            ],
                            set_={
                                "sample_size": excluded.sample_size,
                                "direction_correct": excluded.direction_correct,
                                "direction_accuracy": excluded.direction_accuracy,
                                "mean_confidence": excluded.mean_confidence,
                                "realized_hit_rate": excluded.realized_hit_rate,
                                "calibration_gap": excluded.calibration_gap,
                                "brier": excluded.brier,
                                "scored_at": excluded.scored_at,
                            },
                        )
                    )
        return len(payload)

    def get_signal_performance(
        self, league: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        table = SignalPerformanceRow.__table__
        with self.Session() as session:
            latest_stmt = select(table.c.scored_at)
            if league:
                latest_stmt = latest_stmt.where(table.c.league == league)
            latest = session.execute(
                latest_stmt.order_by(table.c.scored_at.desc()).limit(1)
            ).scalar_one_or_none()
            if latest is None:
                return []
            stmt = select(table).where(table.c.scored_at == latest)
            if league:
                stmt = stmt.where(table.c.league == league)
            rows = session.execute(
                stmt.order_by(table.c.sample_size.desc(), table.c.signal_type).limit(limit)
            ).mappings()
            return [dict(row) for row in rows]

    def get_outcome(self, trace_id: str) -> dict[str, Any] | None:
        table = OutcomeRow.__table__
        with self.Session() as session:
            row = session.execute(
                select(table).where(table.c.trace_id == trace_id).order_by(table.c.attached_at).limit(1)
            ).mappings().first()
            return dict(row) if row else None

    def attach_outcome(
        self,
        trace_id: str,
        home_score: int,
        away_score: int,
        source: str = "manual",
        result_override: str | None = None,
    ) -> str:
        self._ensure_writeable()
        with self.Session() as session:
            with session.begin():
                if not session.execute(
                    select(TraceRow.trace_id).where(TraceRow.trace_id == trace_id)
                ).first():
                    raise ValueError(f"No trace found with trace_id={trace_id!r}")
                existing = session.execute(
                    select(OutcomeRow.outcome_id).where(OutcomeRow.trace_id == trace_id)
                ).scalar_one_or_none()
                if existing:
                    raise ValueError(
                        f"Outcome already attached for trace_id={trace_id!r}; "
                        "delete the existing outcome explicitly before re-grading"
                    )
                result = result_override or (
                    "home_win" if home_score > away_score else "away_win" if away_score > home_score else "draw"
                )
                outcome_id = uuid.uuid4().hex[:12]
                session.execute(
                    sa_insert(OutcomeRow.__table__).values(
                        outcome_id=outcome_id,
                        trace_id=trace_id,
                        home_score=home_score,
                        away_score=away_score,
                        result=result,
                        source=source,
                    )
                )
                return outcome_id

    def attach_prop_outcome(
        self,
        trace_id: str,
        player_name: str,
        stat_type: str,
        stat_value: float,
        line: float,
        side: str,
        source: str = "manual",
        void: bool = False,
    ) -> str:
        self._ensure_writeable()
        side_norm = normalize_prop_side(side)

        with self.Session() as session:
            with session.begin():
                if not session.execute(
                    select(TraceRow.trace_id).where(TraceRow.trace_id == trace_id)
                ).first():
                    raise ValueError(f"No trace found with trace_id={trace_id!r}")
                existing = session.execute(
                    select(PropOutcomeRow.prop_outcome_id).where(
                        PropOutcomeRow.trace_id == trace_id,
                        PropOutcomeRow.player_name == player_name,
                        PropOutcomeRow.stat_type == stat_type,
                    )
                ).scalar_one_or_none()
                if existing:
                    return existing

                result, side_norm = derive_prop_outcome_result(
                    stat_value=stat_value,
                    line=line,
                    side=side_norm,
                    void=void,
                )

                prop_outcome_id = uuid.uuid4().hex[:12]
                session.execute(
                    sa_insert(PropOutcomeRow.__table__).values(
                        prop_outcome_id=prop_outcome_id,
                        trace_id=trace_id,
                        player_name=player_name,
                        stat_type=stat_type,
                        stat_value=float(stat_value),
                        line=float(line),
                        side=side_norm,
                        result=result,
                        source=source,
                    )
                )
                return prop_outcome_id

    def get_prop_outcomes(self, trace_id: str) -> list[dict[str, Any]]:
        table = PropOutcomeRow.__table__
        with self.Session() as session:
            rows = session.execute(
                select(table).where(table.c.trace_id == trace_id).order_by(table.c.attached_at)
            ).mappings()
            return [dict(row) for row in rows]

    def record_bet(self, bet: BetRecord) -> str:
        stake = round((bet.stake_units or 0.0) * (DEFAULT_BANKROLL / 100.0), 2) or 25.0
        ts = bet.decision_timestamp or ""
        ledger = LedgerBet(
            ledger_id=bet.bet_id,
            trace_id=bet.trace_id,
            bet_date=ts[:10] if len(ts) >= 10 else None,
            league=None,
            sport=None,
            matchup="",
            market=bet.market,
            bookmaker=bet.book,
            selection=bet.selection,
            selection_descriptor=bet.selection_descriptor,
            line=bet.line_taken,
            odds=bet.odds_taken,
            stake_amount=stake,
            bankroll_at_open=DEFAULT_BANKROLL,
            status=LedgerStatus(bet.status.value)
            if bet.status.value in {s.value for s in LedgerStatus}
            else LedgerStatus.PENDING,
            provenance=BetProvenance.USER_CONFIRMED,
            decision_timestamp=bet.decision_timestamp,
        )
        return self.record_ledger_bet(ledger)

    def get_bet_records(self, trace_id: str) -> list[dict[str, Any]]:
        table = BetLedgerRow.__table__
        with self.Session() as session:
            rows = session.execute(
                select(table)
                .where(table.c.trace_id == trace_id, table.c.provenance == "user_confirmed")
                .order_by(table.c.created_at)
            ).mappings()
            out: list[dict[str, Any]] = []
            for r in rows:
                bankroll = r["bankroll_at_open"] or DEFAULT_BANKROLL
                units = round(r["stake_amount"] / (bankroll / 100.0), 4) if bankroll else None
                out.append(
                    {
                        "bet_id": r["ledger_id"],
                        "trace_id": r["trace_id"],
                        "book": r["bookmaker"],
                        "market": r["market"],
                        "selection": r["selection"],
                        "selection_descriptor": r["selection_descriptor"],
                        "line_taken": r["line"],
                        "odds_taken": r["odds"],
                        "stake_units": units,
                        "decision_timestamp": r["decision_timestamp"],
                        "status": r["status"],
                        "recorded_at": r["created_at"],
                        "session_id": r["session_id"],
                    }
                )
            return out

    def query_ungraded_prop_bet_traces(
        self,
        league: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        clauses = [
            "b.status = 'pending'",
            "b.market LIKE 'player_prop:%'",
            "b.provenance = 'user_confirmed'",
            "NOT EXISTS (SELECT 1 FROM prop_outcomes p WHERE p.trace_id = b.trace_id)",
        ]
        params: dict[str, Any] = {"limit": limit}
        if league:
            clauses.append("t.league = :league")
            params["league"] = league
        if start:
            clauses.append("t.timestamp >= :start")
            params["start"] = start
        if end:
            clauses.append("t.timestamp <= :end")
            params["end"] = end
        sql = text(
            f"""
            SELECT DISTINCT t.trace_id, t.full_trace
            FROM bet_ledger b
            JOIN traces t ON t.trace_id = b.trace_id
            WHERE {" AND ".join(clauses)}
            ORDER BY t.timestamp DESC
            LIMIT :limit
            """
        )
        with self.Session() as session:
            rows = session.execute(sql, params).mappings()
            return [json.loads(row["full_trace"]) for row in rows]

    def update_bet_status(self, bet_id: str, status: str) -> None:
        self._ensure_writeable()
        with self.Session() as session:
            with session.begin():
                session.execute(
                    update(BetLedgerRow.__table__)
                    .where(BetLedgerRow.ledger_id == bet_id)
                    .values(status=status)
                )

    def _record_ledger_bet(self, session, bet: LedgerBet) -> str:
        if not session.execute(select(TraceRow.trace_id).where(TraceRow.trace_id == bet.trace_id)).first():
            raise ValueError(f"No trace found with trace_id={bet.trace_id!r}")

        table = BetLedgerRow.__table__
        trace_session_id = (
            select(TraceRow.session_id).where(TraceRow.trace_id == bet.trace_id).scalar_subquery()
        )
        stmt = pg_insert(table).values(
            ledger_id=bet.ledger_id,
            trace_id=bet.trace_id,
            bet_date=bet.bet_date,
            league=bet.league,
            sport=bet.sport,
            matchup=bet.matchup,
            market=bet.market,
            bookmaker=bet.bookmaker,
            selection=bet.selection,
            selection_descriptor=bet.selection_descriptor,
            line=bet.line,
            odds=bet.odds,
            stake_amount=bet.stake_amount,
            payout_amount=bet.payout_amount,
            net_pnl=bet.net_pnl,
            bankroll_at_open=bet.bankroll_at_open,
            status=bet.status.value,
            provenance=bet.provenance.value,
            decision_timestamp=bet.decision_timestamp,
            graded_at=bet.graded_at,
            staking_policy_id=bet.staking_policy_id,
            staking_policy_version=bet.staking_policy_version,
            exposure_limits_version=bet.exposure_limits_version,
            sizing_reasons=(
                json.dumps(bet.sizing_reasons) if bet.sizing_reasons is not None else None
            ),
            correlation_group=bet.correlation_group,
            session_id=trace_session_id,
        )
        excluded = stmt.excluded
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=["trace_id", "market", "selection_descriptor"],
                set_={
                    "provenance": excluded.provenance,
                    "bookmaker": excluded.bookmaker,
                    "selection": excluded.selection,
                    "line": excluded.line,
                    "odds": excluded.odds,
                    "stake_amount": excluded.stake_amount,
                    "bankroll_at_open": excluded.bankroll_at_open,
                    "bet_date": excluded.bet_date,
                    "decision_timestamp": excluded.decision_timestamp,
                    "staking_policy_id": excluded.staking_policy_id,
                    "staking_policy_version": excluded.staking_policy_version,
                    "exposure_limits_version": excluded.exposure_limits_version,
                    "sizing_reasons": excluded.sizing_reasons,
                    "correlation_group": excluded.correlation_group,
                },
                where=and_(
                    excluded.provenance == "user_confirmed",
                    table.c.provenance != "user_confirmed",
                    table.c.status == "pending",
                ),
            )
        )
        stored = session.execute(
            select(table.c.ledger_id).where(
                table.c.trace_id == bet.trace_id,
                table.c.market == bet.market,
                table.c.selection_descriptor == bet.selection_descriptor,
            )
        ).scalar_one_or_none()
        return stored or bet.ledger_id

    def record_ledger_bet(self, bet: LedgerBet) -> str:
        self._ensure_writeable()
        with self.Session() as session:
            with session.begin():
                return self._record_ledger_bet(session, bet)

    def grade_ledger_bet(
        self,
        ledger_id: str,
        status: LedgerStatus | str,
        payout_amount: float | None,
        net_pnl: float | None,
    ) -> None:
        self._ensure_writeable()
        status_val = status.value if isinstance(status, LedgerStatus) else str(status)
        with self.Session() as session:
            with session.begin():
                session.execute(
                    update(BetLedgerRow.__table__)
                    .where(BetLedgerRow.ledger_id == ledger_id)
                    .values(
                        status=status_val,
                        payout_amount=payout_amount,
                        net_pnl=net_pnl,
                        graded_at=datetime.now(UTC).isoformat(),
                    )
                )

    def get_ledger_bets(self, trace_id: str) -> list[dict[str, Any]]:
        table = BetLedgerRow.__table__
        with self.Session() as session:
            rows = session.execute(
                select(*(table.c[name] for name in self._LEDGER_COLUMNS))
                .where(table.c.trace_id == trace_id)
                .order_by(table.c.created_at)
            ).mappings()
            return [self._decode_ledger_row(row) for row in rows]

    def query_ledger(
        self,
        league: str | None = None,
        sport: str | None = None,
        status: str | None = None,
        provenance: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        table = BetLedgerRow.__table__
        clauses = []
        if league:
            clauses.append(table.c.league == league)
        if sport:
            clauses.append(table.c.sport == sport)
        if status:
            clauses.append(table.c.status == status)
        if provenance:
            clauses.append(table.c.provenance == provenance)
        if start:
            clauses.append(table.c.decision_timestamp >= start)
        if end:
            clauses.append(table.c.decision_timestamp <= end)
        stmt = select(*(table.c[name] for name in self._LEDGER_COLUMNS))
        if clauses:
            stmt = stmt.where(*clauses)
        stmt = stmt.order_by(table.c.decision_timestamp.desc()).limit(limit)
        with self.Session() as session:
            rows = session.execute(stmt).mappings()
            return [self._decode_ledger_row(row) for row in rows]

    @staticmethod
    def _decode_ledger_row(row: Any) -> dict[str, Any]:
        data = dict(row)
        if data.get("sizing_reasons") is not None and isinstance(data["sizing_reasons"], str):
            data["sizing_reasons"] = json.loads(data["sizing_reasons"])
        return data

    def attach_closing_line(
        self,
        trace_id: str,
        market: str,
        selection_descriptor: str,
        closing_odds: float,
        closing_line: float | None,
        closing_timestamp: str,
        source: str,
    ) -> str:
        self._ensure_writeable()
        table = ClosingLineRow.__table__
        with self.Session() as session:
            with session.begin():
                if not session.execute(select(TraceRow.trace_id).where(TraceRow.trace_id == trace_id)).first():
                    raise ValueError(f"No trace found with trace_id={trace_id!r}")
                existing = session.execute(
                    select(table.c.closing_id).where(
                        table.c.trace_id == trace_id,
                        table.c.market == market,
                        table.c.selection_descriptor == selection_descriptor,
                    )
                ).scalar_one_or_none()
                if existing:
                    return existing
                closing_id = uuid.uuid4().hex[:12]
                session.execute(
                    sa_insert(table).values(
                        closing_id=closing_id,
                        trace_id=trace_id,
                        market=market,
                        selection_descriptor=selection_descriptor,
                        closing_line=closing_line,
                        closing_odds=closing_odds,
                        closing_timestamp=closing_timestamp,
                        source=source,
                    )
                )
                return closing_id

    def get_closing_lines(self, trace_id: str) -> list[dict[str, Any]]:
        table = ClosingLineRow.__table__
        with self.Session() as session:
            rows = session.execute(
                select(table).where(table.c.trace_id == trace_id).order_by(table.c.captured_at)
            ).mappings()
            return [dict(row) for row in rows]

    def record_early_market_snapshot(self, snapshot: EarlyMarketSnapshot) -> str:
        self._ensure_writeable()
        early_id = snapshot.stable_id()
        table = EarlyMarketSnapshotRow.__table__
        with self.Session() as session:
            with session.begin():
                stmt = pg_insert(table).values(
                    early_id=early_id,
                    trace_id=snapshot.trace_id,
                    league=snapshot.league.upper(),
                    market=snapshot.market,
                    selection_descriptor=snapshot.selection_descriptor,
                    early_line=snapshot.early_line,
                    early_odds=snapshot.early_odds,
                    liquidity_profile=snapshot.liquidity_profile,
                    captured_at=snapshot.captured_at,
                    source=snapshot.source,
                )
                session.execute(stmt.on_conflict_do_nothing())
        return early_id

    def get_early_market_snapshots(self, trace_id: str) -> list[dict[str, Any]]:
        table = EarlyMarketSnapshotRow.__table__
        with self.Session() as session:
            rows = session.execute(
                select(table).where(table.c.trace_id == trace_id).order_by(table.c.captured_at)
            ).mappings()
            return [dict(row) for row in rows]

    def record_market_snapshot(self, snapshot: MarketSnapshot) -> str:
        self._ensure_writeable()
        snapshot_id = snapshot.stable_id()
        table = MarketSnapshotRow.__table__
        with self.Session() as session:
            with session.begin():
                stmt = pg_insert(table).values(
                    snapshot_id=snapshot_id,
                    league=snapshot.league.upper(),
                    provider=snapshot.provider,
                    provider_event_id=snapshot.provider_event_id,
                    home_team=snapshot.home_team,
                    away_team=snapshot.away_team,
                    commence_time=snapshot.commence_time,
                    bookmaker=snapshot.bookmaker,
                    market=snapshot.market,
                    selection=snapshot.selection,
                    player=snapshot.player,
                    point=snapshot.point,
                    price=snapshot.price,
                    snapshot_timestamp=snapshot.snapshot_timestamp,
                    provider_last_update=snapshot.provider_last_update,
                    source=snapshot.source,
                    schema_version=snapshot.schema_version,
                )
                session.execute(stmt.on_conflict_do_nothing())
        return snapshot_id

    def get_market_snapshots(
        self,
        provider_event_id: str,
        market: str | None = None,
        bookmaker: str | None = None,
        selection: str | None = None,
    ) -> list[dict[str, Any]]:
        table = MarketSnapshotRow.__table__
        clauses = [table.c.provider_event_id == provider_event_id]
        if market:
            clauses.append(table.c.market == market)
        if bookmaker:
            clauses.append(table.c.bookmaker == bookmaker)
        if selection:
            clauses.append(table.c.selection == selection)
        with self.Session() as session:
            rows = session.execute(
                select(table).where(*clauses).order_by(table.c.snapshot_timestamp)
            ).mappings()
            return [dict(row) for row in rows]

    def compute_market_movement(
        self,
        provider_event_id: str,
        market: str,
        selection: str,
        bookmaker: str,
    ) -> dict[str, Any] | None:
        rows = self.get_market_snapshots(
            provider_event_id=provider_event_id,
            market=market,
            bookmaker=bookmaker,
            selection=selection,
        )
        if len(rows) < 2:
            return None
        first = rows[0]
        last = rows[-1]
        point_delta = None
        if first["point"] is not None and last["point"] is not None:
            point_delta = float(last["point"]) - float(first["point"])
        movement = MarketMovement(
            market=market,
            selection=selection,
            bookmaker=bookmaker,
            first_timestamp=first["snapshot_timestamp"],
            last_timestamp=last["snapshot_timestamp"],
            first_point=first["point"],
            last_point=last["point"],
            first_price=float(first["price"]),
            last_price=float(last["price"]),
            point_delta=point_delta,
            price_delta=float(last["price"]) - float(first["price"]),
        )
        return movement.model_dump()

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self.Session() as session:
            full_trace = session.execute(
                select(TraceRow.full_trace).where(TraceRow.trace_id == trace_id)
            ).scalar_one_or_none()
            return json.loads(full_trace) if full_trace is not None else None

    def query_traces(
        self,
        league: str | None = None,
        start: str | None = None,
        end: str | None = None,
        has_outcome: bool | None = None,
        execution_mode: str | None = None,
        limit: int = 100,
        calibration_eligible_only: bool = False,
    ) -> list[dict[str, Any]]:
        t = TraceRow.__table__.alias("t")
        o = OutcomeRow.__table__.alias("o")
        p = PropOutcomeRow.__table__
        stmt = (
            select(
                t.c.trace_id,
                t.c.full_trace,
                o.c.outcome_id,
                o.c.home_score,
                o.c.away_score,
                o.c.result,
            )
            .select_from(t.outerjoin(o, t.c.trace_id == o.c.trace_id))
            .order_by(t.c.timestamp.desc())
            .limit(limit)
        )
        clauses = []
        if league:
            clauses.append(t.c.league == league)
        if start:
            clauses.append(t.c.timestamp >= start)
        if end:
            clauses.append(t.c.timestamp <= end)
        if execution_mode:
            clauses.append(t.c.execution_mode == execution_mode)

        any_outcome = exists(select(1).select_from(OutcomeRow.__table__).where(OutcomeRow.trace_id == t.c.trace_id)) | exists(
            select(1).select_from(p).where(p.c.trace_id == t.c.trace_id)
        )
        if has_outcome is True:
            clauses.append(any_outcome)
        elif has_outcome is False:
            clauses.append(~any_outcome)

        if calibration_eligible_only:
            clauses.append(t.c.predictions.is_not(None))
            clauses.append(text("(t.full_trace::jsonb #>> '{trace_quality,calibration_eligible}') = 'true'"))
        if clauses:
            stmt = stmt.where(*clauses)

        with self.Session() as session:
            rows = list(session.execute(stmt).mappings())
            trace_ids = [row["trace_id"] for row in rows]
            # Batch-load the 1:N side-tables for every returned trace in two
            # queries instead of 2N per-row round-trips. Grouped in Python with
            # the same intra-trace ordering the per-trace getters use.
            props_by_trace = self._prop_outcomes_by_trace(session, trace_ids)
            dists_by_trace = self._distributions_by_trace(session, trace_ids)

        results: list[dict[str, Any]] = []
        for row in rows:
            trace = json.loads(row["full_trace"])
            if row["outcome_id"]:
                trace["_outcome"] = {
                    "outcome_id": row["outcome_id"],
                    "home_score": row["home_score"],
                    "away_score": row["away_score"],
                    "result": row["result"],
                }
            # Attach only when non-empty (parity with the SQLite store path).
            prop_rows = props_by_trace.get(row["trace_id"])
            if prop_rows:
                trace["_prop_outcomes"] = prop_rows
            distribution_rows = dists_by_trace.get(row["trace_id"])
            if distribution_rows:
                trace["_simulation_distributions"] = distribution_rows
            results.append(trace)
        return results

    def _prop_outcomes_by_trace(
        self, session, trace_ids: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        if not trace_ids:
            return grouped
        table = PropOutcomeRow.__table__
        rows = session.execute(
            select(table)
            .where(table.c.trace_id.in_(trace_ids))
            .order_by(table.c.trace_id, table.c.attached_at)
        ).mappings()
        for row in rows:
            grouped.setdefault(row["trace_id"], []).append(dict(row))
        return grouped

    def _distributions_by_trace(
        self, session, trace_ids: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        if not trace_ids:
            return grouped
        table = SimulationDistributionRow.__table__
        rows = session.execute(
            select(table)
            .where(table.c.trace_id.in_(trace_ids))
            .order_by(table.c.trace_id, table.c.distribution_id)
        ).mappings()
        for row in rows:
            data = dict(row)
            try:
                data["distribution_params"] = json.loads(data["distribution_params"])
            except (TypeError, json.JSONDecodeError):
                data["distribution_params"] = {}
            grouped.setdefault(row["trace_id"], []).append(data)
        return grouped

    def get_graded_traces(
        self, league: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        return self.query_traces(
            league=league,
            has_outcome=True,
            calibration_eligible_only=True,
            limit=limit,
        )

    def query_by_session(self, session_id: str) -> list[dict[str, Any]]:
        t = TraceRow.__table__
        o = OutcomeRow.__table__
        stmt = (
            select(
                t.c.trace_id,
                t.c.timestamp,
                t.c.league,
                t.c.matchup,
                t.c.execution_mode,
                t.c.aggregate_quality,
                t.c.full_trace,
                o.c.outcome_id,
                o.c.home_score,
                o.c.away_score,
                o.c.result,
            )
            .select_from(t.outerjoin(o, t.c.trace_id == o.c.trace_id))
            .where(t.c.session_id == session_id)
            .order_by(t.c.timestamp)
        )
        with self.Session() as session:
            rows = list(session.execute(stmt).mappings())

        results: list[dict[str, Any]] = []
        for row in rows:
            trace = json.loads(row["full_trace"])
            trace["_row"] = {
                "trace_id": row["trace_id"],
                "timestamp": row["timestamp"],
                "kind": trace.get("kind"),
                "league": row["league"],
                "matchup": row["matchup"],
                "execution_mode": row["execution_mode"],
                "aggregate_quality": row["aggregate_quality"],
            }
            if row["outcome_id"]:
                trace["_outcome"] = {
                    "outcome_id": row["outcome_id"],
                    "home_score": row["home_score"],
                    "away_score": row["away_score"],
                    "result": row["result"],
                }
            prop_rows = self.get_prop_outcomes(row["trace_id"])
            if prop_rows:
                trace["_prop_outcomes"] = prop_rows
            bet_rows = self.get_bet_records(row["trace_id"])
            if bet_rows:
                trace["_bet_records"] = bet_rows
            results.append(trace)
        return results

    def get_session_summary(
        self, league: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        clauses = ["t.session_id IS NOT NULL"]
        params: dict[str, Any] = {"limit": limit}
        if league:
            clauses.append("t.league = :league")
            params["league"] = league
        sql = text(
            f"""
            SELECT t.session_id,
                   COUNT(*) AS trace_count,
                   SUM(
                       CASE WHEN
                           EXISTS (SELECT 1 FROM outcomes o WHERE o.trace_id = t.trace_id)
                           OR EXISTS (SELECT 1 FROM prop_outcomes p WHERE p.trace_id = t.trace_id)
                       THEN 1 ELSE 0 END
                   ) AS graded_count,
                   MIN(t.timestamp) AS first_ts,
                   MAX(t.timestamp) AS last_ts
            FROM traces t
            WHERE {" AND ".join(clauses)}
            GROUP BY t.session_id
            ORDER BY last_ts DESC
            LIMIT :limit
            """
        )
        with self.Session() as session:
            rows = session.execute(sql, params).mappings()
            return [dict(row) for row in rows]

    def schema_version(self) -> int:
        with self.Session() as session:
            version = session.execute(select(func.max(SchemaVersionRow.version))).scalar_one_or_none()
            return int(version or 0)

    def count(self) -> int:
        with self.Session() as session:
            return int(session.execute(select(func.count()).select_from(TraceRow.__table__)).scalar_one())

    def close(self) -> None:
        self.engine.dispose()
