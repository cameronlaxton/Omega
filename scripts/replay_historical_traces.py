"""
Trace Replay Engine (Lean Backtesting) - V2

Replays historical traces using an experimental calibration profile.
Integrates directly with omega.core.betting to ensure deterministic
staking yield recalculations. Uses cross-platform tempfile logic.
"""
import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone

# Add repo root to path to ensure omega module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omega.core.calibration.registry import CalibrationRegistry
from omega.core.calibration.probability import calibrate_probability
from omega.core.betting.kelly import recommend_stake
from omega.core.betting.odds import edge_percentage, expected_value_percent, implied_probability

logger = logging.getLogger("replay_engine")

def setup_local_db_copy(source_db_path: str) -> str:
    """Copy the database to a local temp dir using cross-platform paths."""
    temp_dir = tempfile.gettempdir()
    temp_db_path = os.path.join(temp_dir, "omega_traces_replay.db")
    logger.info(f"Copying DB from {source_db_path} to {temp_db_path}")
    shutil.copy2(source_db_path, temp_db_path)
    return temp_db_path

def main():
    parser = argparse.ArgumentParser(description="Trace Replay Engine V2")
    parser.add_argument("--profile-version", required=True, help="Calibration profile ID to test")
    parser.add_argument("--limit", type=int, default=None, help="Max traces to process")
    parser.add_argument("--offset", type=int, default=0, help="Offset for processing")
    parser.add_argument("--all", action="store_true", help="Process all traces asynchronously")
    parser.add_argument("--db", default="omega_traces.db", help="Path to source traces DB")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not os.path.exists(args.db):
        logger.error(f"Source DB {args.db} not found.")
        sys.exit(1)

    local_db_path = setup_local_db_copy(args.db)

    registry = CalibrationRegistry()
    profile = registry.get_profile(args.profile_version)
    if not profile:
        logger.error(f"Profile version '{args.profile_version}' not found in registry.")
        sys.exit(1)
    
    logger.info(f"Loaded profile: {profile.profile_id} (method: {profile.method})")

    # Connect to the local copy in read-only mode if supported, or standard if not.
    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT t.trace_id, t.full_trace, t.trace_timestamp, 
               COALESCE(o.result, po.result) as actual_result
        FROM traces t
        LEFT JOIN outcomes o ON t.trace_id = o.trace_id
        LEFT JOIN prop_outcomes po ON t.trace_id = po.trace_id
        WHERE t.execution_mode = 'native_sim'
    """
    
    # Apply deterministic ordering via timestamp
    query += " ORDER BY t.trace_timestamp DESC"

    if not args.all:
        if args.limit is not None:
            query += f" LIMIT {args.limit}"
        if args.offset > 0:
            query += f" OFFSET {args.offset}"

    logger.info(f"Executing extraction query... (limit={args.limit if not args.all else 'ALL'})")
    cursor.execute(query)
    rows = cursor.fetchall()
    
    results_report = []

    for row in rows:
        trace_id = row["trace_id"]
        actual_result = row["actual_result"]
        
        try:
            payload = json.loads(row["full_trace"])
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode trace {trace_id}")
            continue

        bankroll = payload.get("bankroll", 1000.0)
        result_obj = payload.get("result", {})
        
        if isinstance(result_obj, dict):
            # Strip previous legacy yield metrics natively from the trace object
            for field in ["ev_percent", "edge_percent", "kelly_fraction", "recommended_units", "confidence_tier"]:
                result_obj.pop(field, None)

            # Re-calibration & Downstream Staking Recalculation Hook
            edges = result_obj.get("edges", [])
            for edge in edges:
                if "true_prob" in edge and "market_odds" in edge:
                    raw_prob = edge["true_prob"]
                    market_odds = edge["market_odds"]
                    
                    # 1. Substitute the underlying probability
                    cal_result = calibrate_probability(
                        raw_prob, 
                        method=profile.method, 
                        **profile.params
                    )
                    calibrated_prob = cal_result["calibrated"]
                    
                    # 2. Pipe updated probabilities into downstream staking engine mechanics
                    market_prob = implied_probability(market_odds)
                    edge_pct = edge_percentage(calibrated_prob, market_prob)
                    ev_pct = expected_value_percent(calibrated_prob, market_odds)
                    
                    # Compute confidence tier (assuming standard A tier for completed native sims)
                    tier = "A"
                    if abs(edge_pct) < 3.0:
                        tier = "Pass"
                        
                    stake = recommend_stake(
                        true_prob=calibrated_prob,
                        odds=market_odds,
                        bankroll=bankroll,
                        confidence_tier=tier,
                    )
                    
                    # 3. Inject fully mutated metrics back into the edge
                    edge["calibrated_prob"] = calibrated_prob
                    edge["recalibrated_with"] = profile.profile_id
                    edge["market_implied"] = market_prob
                    edge["edge_pct"] = edge_pct
                    edge["ev_pct"] = ev_pct
                    edge["confidence_tier"] = tier
                    edge["recommended_units"] = stake["units"]
                    edge["kelly_fraction"] = stake["kelly_fraction"]

        results_report.append({
            "trace_id": trace_id,
            "actual_result": actual_result,
            "recalibrated_result": result_obj
        })

    conn.close()

    report_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"replay_results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "profile": args.profile_version, 
            "count": len(results_report), 
            "traces": results_report
        }, f, indent=2)
        
    logger.info(f"Replay complete. Processed {len(results_report)} traces.")
    logger.info(f"Report written to {report_path}")

if __name__ == "__main__":
    main()
