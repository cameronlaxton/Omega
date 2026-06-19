"""SQLAlchemy table models for the Postgres trace backend.

The shape mirrors SQLite schema V14. JSON blobs and timestamps intentionally
remain TEXT for byte-for-byte behavioral parity with the existing store.
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Index, Integer, Text, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, mapped_column

# Timestamp columns are TEXT and must match SQLite's ``datetime('now')`` output
# byte-for-byte: "YYYY-MM-DD HH:MM:SS", UTC, space-separated, no fractional/offset.
# A bare ``CURRENT_TIMESTAMP`` default casts a timestamptz to text and drifts
# (fractional seconds + "+00"), so we format explicitly at the default level.
SQLITE_NOW_DEFAULT = "to_char((now() AT TIME ZONE 'UTC'), 'YYYY-MM-DD HH24:MI:SS')"


class Base(DeclarativeBase):
    pass


class TraceRow(Base):
    __tablename__ = "traces"

    trace_id = mapped_column(Text, primary_key=True)
    run_id = mapped_column(Text, nullable=False)
    timestamp = mapped_column(Text, nullable=False)
    prompt = mapped_column(Text, nullable=False)
    league = mapped_column(Text)
    matchup = mapped_column(Text)
    execution_mode = mapped_column(Text)
    simulation_seed = mapped_column(Integer)
    aggregate_quality = mapped_column(Float)
    predictions = mapped_column(Text)
    recommendations = mapped_column(Text)
    odds_snapshot = mapped_column(Text)
    downgrades = mapped_column(Text)
    full_trace = mapped_column(Text, nullable=False)
    schema_version = mapped_column(Integer, nullable=False, server_default=text("1"))
    created_at = mapped_column(Text, nullable=False, server_default=text(SQLITE_NOW_DEFAULT))
    session_id = mapped_column(Text)
    # V20 (Phase 8): governed backend parameter-profile provenance, mirroring the
    # SQLite traces.parameter_profile_ref column so the two backends stay at parity
    # and provenance is not silently dropped on a Postgres persist.
    parameter_profile_ref = mapped_column(Text)

    __table_args__ = (
        Index("idx_traces_league", "league"),
        Index("idx_traces_timestamp", "timestamp"),
        Index("idx_traces_matchup", "matchup"),
        Index("idx_traces_session_id", "session_id"),
    )


class OutcomeRow(Base):
    __tablename__ = "outcomes"

    outcome_id = mapped_column(Text, primary_key=True)
    trace_id = mapped_column(Text, ForeignKey("traces.trace_id"), nullable=False)
    home_score = mapped_column(Integer, nullable=False)
    away_score = mapped_column(Integer, nullable=False)
    result = mapped_column(Text, nullable=False)
    attached_at = mapped_column(Text, nullable=False, server_default=text(SQLITE_NOW_DEFAULT))
    source = mapped_column(Text, nullable=False, server_default=text("'manual'"))

    __table_args__ = (
        UniqueConstraint("trace_id", name="uq_outcomes_trace_id"),
        Index("idx_outcomes_trace_id", "trace_id"),
    )


class SchemaVersionRow(Base):
    __tablename__ = "schema_versions"

    version = mapped_column(Integer, primary_key=True)
    applied_at = mapped_column(Text, nullable=False, server_default=text(SQLITE_NOW_DEFAULT))
    description = mapped_column(Text)


class ClosingLineRow(Base):
    __tablename__ = "closing_lines"

    closing_id = mapped_column(Text, primary_key=True)
    trace_id = mapped_column(Text, ForeignKey("traces.trace_id"), nullable=False)
    market = mapped_column(Text, nullable=False)
    selection_descriptor = mapped_column(Text, nullable=False)
    closing_line = mapped_column(Float)
    closing_odds = mapped_column(Float, nullable=False)
    closing_timestamp = mapped_column(Text, nullable=False)
    source = mapped_column(Text, nullable=False)
    captured_at = mapped_column(Text, nullable=False, server_default=text(SQLITE_NOW_DEFAULT))

    __table_args__ = (
        UniqueConstraint(
            "trace_id",
            "market",
            "selection_descriptor",
            name="uq_closing_lines_trace_market_selection",
        ),
        Index("idx_closing_lines_trace_id", "trace_id"),
    )


class MarketSnapshotRow(Base):
    __tablename__ = "market_snapshots"

    snapshot_id = mapped_column(Text, primary_key=True)
    league = mapped_column(Text, nullable=False)
    provider = mapped_column(Text, nullable=False)
    provider_event_id = mapped_column(Text, nullable=False)
    home_team = mapped_column(Text, nullable=False)
    away_team = mapped_column(Text, nullable=False)
    commence_time = mapped_column(Text)
    bookmaker = mapped_column(Text, nullable=False)
    market = mapped_column(Text, nullable=False)
    selection = mapped_column(Text, nullable=False)
    player = mapped_column(Text)
    point = mapped_column(Float)
    price = mapped_column(Float, nullable=False)
    snapshot_timestamp = mapped_column(Text, nullable=False)
    provider_last_update = mapped_column(Text)
    source = mapped_column(Text, nullable=False)
    schema_version = mapped_column(Integer, nullable=False, server_default=text("1"))
    captured_at = mapped_column(Text, nullable=False, server_default=text(SQLITE_NOW_DEFAULT))

    __table_args__ = (
        Index(
            "idx_market_snapshots_event",
            "league",
            "provider_event_id",
            "market",
            "bookmaker",
        ),
        Index(
            "idx_market_snapshots_movement",
            "provider_event_id",
            "market",
            "selection",
            "bookmaker",
            "snapshot_timestamp",
        ),
    )


class PropOutcomeRow(Base):
    __tablename__ = "prop_outcomes"

    prop_outcome_id = mapped_column(Text, primary_key=True)
    trace_id = mapped_column(Text, ForeignKey("traces.trace_id"), nullable=False)
    player_name = mapped_column(Text, nullable=False)
    stat_type = mapped_column(Text, nullable=False)
    stat_value = mapped_column(Float, nullable=False)
    line = mapped_column(Float, nullable=False)
    side = mapped_column(Text, nullable=False)
    result = mapped_column(Text, nullable=False)
    attached_at = mapped_column(Text, nullable=False, server_default=text(SQLITE_NOW_DEFAULT))
    source = mapped_column(Text, nullable=False, server_default=text("'manual'"))

    __table_args__ = (
        UniqueConstraint("trace_id", "player_name", "stat_type", name="uq_prop_outcomes_identity"),
        Index("idx_prop_outcomes_trace_id", "trace_id"),
    )


class EvidenceSignalRow(Base):
    __tablename__ = "evidence_signals"

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id = mapped_column(Text, ForeignKey("traces.trace_id"), nullable=False)
    signal_type = mapped_column(Text, nullable=False)
    category = mapped_column(Text)
    plane = mapped_column(Text)
    source = mapped_column(Text)
    confidence = mapped_column(Float)
    obs_window = mapped_column(Text)
    direction = mapped_column(Text)
    stat_key = mapped_column(Text)
    league = mapped_column(Text)
    value_json = mapped_column(Text)
    applied = mapped_column(Integer, nullable=False, server_default=text("0"))
    applied_factor = mapped_column(Float)
    policy_version = mapped_column(Text)
    evidence_mode = mapped_column(Text)
    created_at = mapped_column(Text, nullable=False, server_default=text(SQLITE_NOW_DEFAULT))

    __table_args__ = (
        Index("idx_evidence_signals_trace_id", "trace_id"),
        Index("idx_evidence_signals_type", "signal_type", "league"),
    )


class SignalPerformanceRow(Base):
    __tablename__ = "signal_performance"

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_type = mapped_column(Text, nullable=False)
    source = mapped_column(Text, nullable=False)
    obs_window = mapped_column(Text, nullable=False)
    league = mapped_column(Text, nullable=False)
    sample_size = mapped_column(Integer, nullable=False)
    direction_correct = mapped_column(Integer, nullable=False)
    direction_accuracy = mapped_column(Float)
    mean_confidence = mapped_column(Float)
    realized_hit_rate = mapped_column(Float)
    calibration_gap = mapped_column(Float)
    brier = mapped_column(Float)
    dataset_hash = mapped_column(Text, nullable=False)
    scored_at = mapped_column(Text, nullable=False, server_default=text(SQLITE_NOW_DEFAULT))

    __table_args__ = (
        UniqueConstraint(
            "signal_type",
            "source",
            "obs_window",
            "league",
            "dataset_hash",
            name="uq_signal_performance_identity",
        ),
        Index("idx_signal_performance_key", "signal_type", "league"),
    )


class SimulationDistributionRow(Base):
    __tablename__ = "simulation_distributions"

    distribution_id = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id = mapped_column(Text, ForeignKey("traces.trace_id"), nullable=False)
    kind = mapped_column(Text)
    league = mapped_column(Text)
    target = mapped_column(Text, nullable=False)
    market = mapped_column(Text)
    stat_key = mapped_column(Text)
    distribution_type = mapped_column(Text, nullable=False)
    distribution_params = mapped_column(Text, nullable=False)
    params_schema_version = mapped_column(Integer, nullable=False, server_default=text("1"))
    sample_mean = mapped_column(Float)
    sample_std = mapped_column(Float)
    p10 = mapped_column(Float)
    p50 = mapped_column(Float)
    p90 = mapped_column(Float)
    n_iterations = mapped_column(Integer)
    seed = mapped_column(Integer)
    context_hash = mapped_column(Text)
    component_version = mapped_column(Text)
    created_at = mapped_column(Text, nullable=False, server_default=text(SQLITE_NOW_DEFAULT))

    __table_args__ = (
        Index("idx_sim_distributions_trace_id", "trace_id"),
        Index("idx_sim_distributions_lookup", "league", "kind", "market", "stat_key"),
    )


class EarlyMarketSnapshotRow(Base):
    __tablename__ = "early_market_snapshots"

    early_id = mapped_column(Text, primary_key=True)
    trace_id = mapped_column(Text)
    league = mapped_column(Text, nullable=False)
    market = mapped_column(Text, nullable=False)
    selection_descriptor = mapped_column(Text, nullable=False)
    early_line = mapped_column(Float)
    early_odds = mapped_column(Float, nullable=False)
    liquidity_profile = mapped_column(Text, nullable=False)
    captured_at = mapped_column(Text, nullable=False)
    source = mapped_column(Text, nullable=False)
    recorded_at = mapped_column(Text, nullable=False, server_default=text(SQLITE_NOW_DEFAULT))

    __table_args__ = (
        UniqueConstraint(
            "trace_id",
            "league",
            "market",
            "selection_descriptor",
            "captured_at",
            name="uq_early_market_snapshots_identity",
        ),
        Index("idx_early_market_snapshots_trace_id", "trace_id"),
        Index("idx_early_market_snapshots_league", "league", "captured_at"),
    )


class TraceQaVerdictRow(Base):
    __tablename__ = "trace_qa_verdicts"

    trace_id = mapped_column(Text, ForeignKey("traces.trace_id"), primary_key=True)
    session_id = mapped_column(Text)
    verdict = mapped_column(Text, nullable=False)
    scope = mapped_column(Text, nullable=False)
    gate_name = mapped_column(Text)
    reason = mapped_column(Text)
    event_id = mapped_column(Text)
    matched_trace_id = mapped_column(Text)
    ran_at = mapped_column(Text)
    created_at = mapped_column(Text, nullable=False, server_default=text(SQLITE_NOW_DEFAULT))

    __table_args__ = (
        Index("idx_trace_qa_verdicts_session", "session_id"),
        Index("idx_trace_qa_verdicts_verdict", "verdict", "scope"),
    )


class BetLedgerRow(Base):
    __tablename__ = "bet_ledger"

    ledger_id = mapped_column(Text, primary_key=True)
    trace_id = mapped_column(Text, ForeignKey("traces.trace_id"), nullable=False)
    bet_date = mapped_column(Text)
    league = mapped_column(Text)
    sport = mapped_column(Text)
    matchup = mapped_column(Text)
    market = mapped_column(Text, nullable=False)
    bookmaker = mapped_column(Text, nullable=False, server_default=text("'consensus'"))
    selection = mapped_column(Text, nullable=False)
    selection_descriptor = mapped_column(Text, nullable=False)
    line = mapped_column(Float)
    odds = mapped_column(Float, nullable=False)
    stake_amount = mapped_column(Float, nullable=False, server_default=text("25.0"))
    payout_amount = mapped_column(Float)
    net_pnl = mapped_column(Float)
    bankroll_at_open = mapped_column(Float, server_default=text("1000.0"))
    status = mapped_column(Text, nullable=False, server_default=text("'pending'"))
    provenance = mapped_column(Text, nullable=False)
    decision_timestamp = mapped_column(Text, nullable=False)
    graded_at = mapped_column(Text)
    session_id = mapped_column(Text)
    created_at = mapped_column(Text, nullable=False, server_default=text(SQLITE_NOW_DEFAULT))
    # Sizing audit (schema V15) — all nullable.
    staking_policy_id = mapped_column(Text)
    staking_policy_version = mapped_column(Integer)
    exposure_limits_version = mapped_column(Integer)
    sizing_reasons = mapped_column(Text)
    correlation_group = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint(
            "trace_id",
            "market",
            "selection_descriptor",
            name="uq_bet_ledger_trace_market_selection",
        ),
        Index("idx_bet_ledger_trace_id", "trace_id"),
        Index("idx_bet_ledger_status", "status"),
        Index("idx_bet_ledger_league", "league"),
        Index("idx_bet_ledger_sport", "sport"),
        Index("idx_bet_ledger_book", "bookmaker"),
        Index("idx_bet_ledger_date", "bet_date"),
    )
