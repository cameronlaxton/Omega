"""Initial Postgres schema equivalent to TraceStore SQLite V14."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0001_initial_v14_schema"
down_revision = None
branch_labels = None
depends_on = None

# Match SQLite's datetime('now') text format exactly (UTC, space-separated, no
# fractional seconds / offset). A bare CURRENT_TIMESTAMP default drifts on PG.
SQLITE_NOW_DEFAULT = "to_char((now() AT TIME ZONE 'UTC'), 'YYYY-MM-DD HH24:MI:SS')"


def _t(*args, **kwargs):
    return sa.Column(*args, **kwargs)


def upgrade() -> None:
    op.create_table(
        "traces",
        _t("trace_id", sa.Text(), primary_key=True),
        _t("run_id", sa.Text(), nullable=False),
        _t("timestamp", sa.Text(), nullable=False),
        _t("prompt", sa.Text(), nullable=False),
        _t("league", sa.Text()),
        _t("matchup", sa.Text()),
        _t("execution_mode", sa.Text()),
        _t("simulation_seed", sa.Integer()),
        _t("aggregate_quality", sa.Float()),
        _t("predictions", sa.Text()),
        _t("recommendations", sa.Text()),
        _t("odds_snapshot", sa.Text()),
        _t("downgrades", sa.Text()),
        _t("full_trace", sa.Text(), nullable=False),
        _t("schema_version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        _t("created_at", sa.Text(), nullable=False, server_default=sa.text(SQLITE_NOW_DEFAULT)),
        _t("session_id", sa.Text()),
    )
    op.create_index("idx_traces_league", "traces", ["league"])
    op.create_index("idx_traces_timestamp", "traces", ["timestamp"])
    op.create_index("idx_traces_matchup", "traces", ["matchup"])
    op.create_index("idx_traces_session_id", "traces", ["session_id"])

    op.create_table(
        "schema_versions",
        _t("version", sa.Integer(), primary_key=True),
        _t("applied_at", sa.Text(), nullable=False, server_default=sa.text(SQLITE_NOW_DEFAULT)),
        _t("description", sa.Text()),
    )

    op.create_table(
        "outcomes",
        _t("outcome_id", sa.Text(), primary_key=True),
        _t("trace_id", sa.Text(), sa.ForeignKey("traces.trace_id"), nullable=False),
        _t("home_score", sa.Integer(), nullable=False),
        _t("away_score", sa.Integer(), nullable=False),
        _t("result", sa.Text(), nullable=False),
        _t("attached_at", sa.Text(), nullable=False, server_default=sa.text(SQLITE_NOW_DEFAULT)),
        _t("source", sa.Text(), nullable=False, server_default=sa.text("'manual'")),
        sa.UniqueConstraint("trace_id", name="uq_outcomes_trace_id"),
    )
    op.create_index("idx_outcomes_trace_id", "outcomes", ["trace_id"])

    op.create_table(
        "closing_lines",
        _t("closing_id", sa.Text(), primary_key=True),
        _t("trace_id", sa.Text(), sa.ForeignKey("traces.trace_id"), nullable=False),
        _t("market", sa.Text(), nullable=False),
        _t("selection_descriptor", sa.Text(), nullable=False),
        _t("closing_line", sa.Float()),
        _t("closing_odds", sa.Float(), nullable=False),
        _t("closing_timestamp", sa.Text(), nullable=False),
        _t("source", sa.Text(), nullable=False),
        _t("captured_at", sa.Text(), nullable=False, server_default=sa.text(SQLITE_NOW_DEFAULT)),
        sa.UniqueConstraint(
            "trace_id",
            "market",
            "selection_descriptor",
            name="uq_closing_lines_trace_market_selection",
        ),
    )
    op.create_index("idx_closing_lines_trace_id", "closing_lines", ["trace_id"])

    op.create_table(
        "market_snapshots",
        _t("snapshot_id", sa.Text(), primary_key=True),
        _t("league", sa.Text(), nullable=False),
        _t("provider", sa.Text(), nullable=False),
        _t("provider_event_id", sa.Text(), nullable=False),
        _t("home_team", sa.Text(), nullable=False),
        _t("away_team", sa.Text(), nullable=False),
        _t("commence_time", sa.Text()),
        _t("bookmaker", sa.Text(), nullable=False),
        _t("market", sa.Text(), nullable=False),
        _t("selection", sa.Text(), nullable=False),
        _t("player", sa.Text()),
        _t("point", sa.Float()),
        _t("price", sa.Float(), nullable=False),
        _t("snapshot_timestamp", sa.Text(), nullable=False),
        _t("provider_last_update", sa.Text()),
        _t("source", sa.Text(), nullable=False),
        _t("schema_version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        _t("captured_at", sa.Text(), nullable=False, server_default=sa.text(SQLITE_NOW_DEFAULT)),
    )
    op.create_index(
        "idx_market_snapshots_event",
        "market_snapshots",
        ["league", "provider_event_id", "market", "bookmaker"],
    )
    op.create_index(
        "idx_market_snapshots_movement",
        "market_snapshots",
        ["provider_event_id", "market", "selection", "bookmaker", "snapshot_timestamp"],
    )

    op.create_table(
        "prop_outcomes",
        _t("prop_outcome_id", sa.Text(), primary_key=True),
        _t("trace_id", sa.Text(), sa.ForeignKey("traces.trace_id"), nullable=False),
        _t("player_name", sa.Text(), nullable=False),
        _t("stat_type", sa.Text(), nullable=False),
        _t("stat_value", sa.Float(), nullable=False),
        _t("line", sa.Float(), nullable=False),
        _t("side", sa.Text(), nullable=False),
        _t("result", sa.Text(), nullable=False),
        _t("attached_at", sa.Text(), nullable=False, server_default=sa.text(SQLITE_NOW_DEFAULT)),
        _t("source", sa.Text(), nullable=False, server_default=sa.text("'manual'")),
        sa.UniqueConstraint(
            "trace_id", "player_name", "stat_type", name="uq_prop_outcomes_identity"
        ),
    )
    op.create_index("idx_prop_outcomes_trace_id", "prop_outcomes", ["trace_id"])

    op.create_table(
        "evidence_signals",
        _t("id", sa.Integer(), primary_key=True, autoincrement=True),
        _t("trace_id", sa.Text(), sa.ForeignKey("traces.trace_id"), nullable=False),
        _t("signal_type", sa.Text(), nullable=False),
        _t("category", sa.Text()),
        _t("plane", sa.Text()),
        _t("source", sa.Text()),
        _t("confidence", sa.Float()),
        _t("obs_window", sa.Text()),
        _t("direction", sa.Text()),
        _t("stat_key", sa.Text()),
        _t("league", sa.Text()),
        _t("value_json", sa.Text()),
        _t("applied", sa.Integer(), nullable=False, server_default=sa.text("0")),
        _t("applied_factor", sa.Float()),
        _t("policy_version", sa.Text()),
        _t("evidence_mode", sa.Text()),
        _t("created_at", sa.Text(), nullable=False, server_default=sa.text(SQLITE_NOW_DEFAULT)),
    )
    op.create_index("idx_evidence_signals_trace_id", "evidence_signals", ["trace_id"])
    op.create_index("idx_evidence_signals_type", "evidence_signals", ["signal_type", "league"])

    op.create_table(
        "signal_performance",
        _t("id", sa.Integer(), primary_key=True, autoincrement=True),
        _t("signal_type", sa.Text(), nullable=False),
        _t("source", sa.Text(), nullable=False),
        _t("obs_window", sa.Text(), nullable=False),
        _t("league", sa.Text(), nullable=False),
        _t("sample_size", sa.Integer(), nullable=False),
        _t("direction_correct", sa.Integer(), nullable=False),
        _t("direction_accuracy", sa.Float()),
        _t("mean_confidence", sa.Float()),
        _t("realized_hit_rate", sa.Float()),
        _t("calibration_gap", sa.Float()),
        _t("brier", sa.Float()),
        _t("dataset_hash", sa.Text(), nullable=False),
        _t("scored_at", sa.Text(), nullable=False, server_default=sa.text(SQLITE_NOW_DEFAULT)),
        sa.UniqueConstraint(
            "signal_type",
            "source",
            "obs_window",
            "league",
            "dataset_hash",
            name="uq_signal_performance_identity",
        ),
    )
    op.create_index("idx_signal_performance_key", "signal_performance", ["signal_type", "league"])

    op.create_table(
        "simulation_distributions",
        _t("distribution_id", sa.Integer(), primary_key=True, autoincrement=True),
        _t("trace_id", sa.Text(), sa.ForeignKey("traces.trace_id"), nullable=False),
        _t("kind", sa.Text()),
        _t("league", sa.Text()),
        _t("target", sa.Text(), nullable=False),
        _t("market", sa.Text()),
        _t("stat_key", sa.Text()),
        _t("distribution_type", sa.Text(), nullable=False),
        _t("distribution_params", sa.Text(), nullable=False),
        _t("params_schema_version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        _t("sample_mean", sa.Float()),
        _t("sample_std", sa.Float()),
        _t("p10", sa.Float()),
        _t("p50", sa.Float()),
        _t("p90", sa.Float()),
        _t("n_iterations", sa.Integer()),
        _t("seed", sa.Integer()),
        _t("context_hash", sa.Text()),
        _t("component_version", sa.Text()),
        _t("created_at", sa.Text(), nullable=False, server_default=sa.text(SQLITE_NOW_DEFAULT)),
    )
    op.create_index("idx_sim_distributions_trace_id", "simulation_distributions", ["trace_id"])
    op.create_index(
        "idx_sim_distributions_lookup",
        "simulation_distributions",
        ["league", "kind", "market", "stat_key"],
    )

    op.create_table(
        "early_market_snapshots",
        _t("early_id", sa.Text(), primary_key=True),
        _t("trace_id", sa.Text()),
        _t("league", sa.Text(), nullable=False),
        _t("market", sa.Text(), nullable=False),
        _t("selection_descriptor", sa.Text(), nullable=False),
        _t("early_line", sa.Float()),
        _t("early_odds", sa.Float(), nullable=False),
        _t("liquidity_profile", sa.Text(), nullable=False),
        _t("captured_at", sa.Text(), nullable=False),
        _t("source", sa.Text(), nullable=False),
        _t("recorded_at", sa.Text(), nullable=False, server_default=sa.text(SQLITE_NOW_DEFAULT)),
        sa.UniqueConstraint(
            "trace_id",
            "league",
            "market",
            "selection_descriptor",
            "captured_at",
            name="uq_early_market_snapshots_identity",
        ),
    )
    op.create_index("idx_early_market_snapshots_trace_id", "early_market_snapshots", ["trace_id"])
    op.create_index(
        "idx_early_market_snapshots_league", "early_market_snapshots", ["league", "captured_at"]
    )

    op.create_table(
        "trace_qa_verdicts",
        _t("trace_id", sa.Text(), sa.ForeignKey("traces.trace_id"), primary_key=True),
        _t("session_id", sa.Text()),
        _t("verdict", sa.Text(), nullable=False),
        _t("scope", sa.Text(), nullable=False),
        _t("gate_name", sa.Text()),
        _t("reason", sa.Text()),
        _t("event_id", sa.Text()),
        _t("matched_trace_id", sa.Text()),
        _t("ran_at", sa.Text()),
        _t("created_at", sa.Text(), nullable=False, server_default=sa.text(SQLITE_NOW_DEFAULT)),
    )
    op.create_index("idx_trace_qa_verdicts_session", "trace_qa_verdicts", ["session_id"])
    op.create_index("idx_trace_qa_verdicts_verdict", "trace_qa_verdicts", ["verdict", "scope"])

    op.create_table(
        "bet_ledger",
        _t("ledger_id", sa.Text(), primary_key=True),
        _t("trace_id", sa.Text(), sa.ForeignKey("traces.trace_id"), nullable=False),
        _t("bet_date", sa.Text()),
        _t("league", sa.Text()),
        _t("sport", sa.Text()),
        _t("matchup", sa.Text()),
        _t("market", sa.Text(), nullable=False),
        _t("bookmaker", sa.Text(), nullable=False, server_default=sa.text("'consensus'")),
        _t("selection", sa.Text(), nullable=False),
        _t("selection_descriptor", sa.Text(), nullable=False),
        _t("line", sa.Float()),
        _t("odds", sa.Float(), nullable=False),
        _t("stake_amount", sa.Float(), nullable=False, server_default=sa.text("25.0")),
        _t("payout_amount", sa.Float()),
        _t("net_pnl", sa.Float()),
        _t("bankroll_at_open", sa.Float(), server_default=sa.text("1000.0")),
        _t("status", sa.Text(), nullable=False, server_default=sa.text("'pending'")),
        _t("provenance", sa.Text(), nullable=False),
        _t("decision_timestamp", sa.Text(), nullable=False),
        _t("graded_at", sa.Text()),
        _t("session_id", sa.Text()),
        _t("created_at", sa.Text(), nullable=False, server_default=sa.text(SQLITE_NOW_DEFAULT)),
        sa.UniqueConstraint(
            "trace_id",
            "market",
            "selection_descriptor",
            name="uq_bet_ledger_trace_market_selection",
        ),
    )
    op.create_index("idx_bet_ledger_trace_id", "bet_ledger", ["trace_id"])
    op.create_index("idx_bet_ledger_status", "bet_ledger", ["status"])
    op.create_index("idx_bet_ledger_league", "bet_ledger", ["league"])
    op.create_index("idx_bet_ledger_sport", "bet_ledger", ["sport"])
    op.create_index("idx_bet_ledger_book", "bet_ledger", ["bookmaker"])
    op.create_index("idx_bet_ledger_date", "bet_ledger", ["bet_date"])

    op.execute(
        """
        CREATE OR REPLACE VIEW v_distribution_outcomes AS
        SELECT
            d.distribution_id, d.trace_id, d.kind, d.league, d.target, d.market,
            d.stat_key, d.distribution_type, d.distribution_params,
            d.params_schema_version, d.sample_mean, d.sample_std, d.p10, d.p50,
            d.p90, d.n_iterations, d.seed, d.context_hash, d.component_version,
            o.home_score, o.away_score, o.result AS game_result,
            p.player_name, p.stat_type, p.stat_value, p.line, p.side,
            p.result AS prop_result
        FROM simulation_distributions d
        LEFT JOIN outcomes o ON o.trace_id = d.trace_id
        LEFT JOIN prop_outcomes p ON p.trace_id = d.trace_id
            AND (d.stat_key IS NULL OR p.stat_type = d.stat_key)
        """
    )
    op.execute(
        """
        CREATE OR REPLACE VIEW v_bet_ledger_dashboard AS
        SELECT
            l.ledger_id, l.trace_id, l.bet_date, l.league, l.sport, l.matchup,
            l.market, l.bookmaker, l.selection, l.selection_descriptor, l.line,
            l.odds, l.stake_amount, l.payout_amount, l.net_pnl,
            l.bankroll_at_open, l.status, l.provenance, l.decision_timestamp,
            l.graded_at, l.session_id, l.created_at,
            CASE WHEN l.status IN ('won', 'lost', 'push', 'void') AND l.stake_amount > 0
                 THEN l.net_pnl / l.stake_amount END AS return_pct,
            CASE WHEN l.odds > 0 THEN 'underdog' ELSE 'favorite' END AS odds_side,
            CASE
                WHEN l.odds BETWEEN -110 AND 110 THEN 'pickem'
                WHEN l.odds < -110 THEN 'heavy_fav'
                ELSE 'plus_money' END AS odds_bucket
        FROM bet_ledger l
        """
    )
    op.execute(
        """
        INSERT INTO schema_versions (version, description)
        VALUES (14, 'Postgres initial schema equivalent to SQLite V14')
        ON CONFLICT (version) DO NOTHING
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS v_bet_ledger_dashboard")
    op.execute("DROP VIEW IF EXISTS v_distribution_outcomes")
    for table in (
        "bet_ledger",
        "trace_qa_verdicts",
        "early_market_snapshots",
        "simulation_distributions",
        "signal_performance",
        "evidence_signals",
        "prop_outcomes",
        "market_snapshots",
        "closing_lines",
        "outcomes",
        "schema_versions",
        "traces",
    ):
        op.drop_table(table)
