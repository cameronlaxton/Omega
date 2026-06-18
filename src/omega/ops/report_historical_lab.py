"""omega-report-historical-lab — (re)render REPORT.md for a historical lab run.

Reads the four net-new artifacts (plus the optional walk-forward backtest report)
from ``var/historical/lab_runs/<lab_run_id>/`` and renders the deterministic
Markdown report. The orchestrator already writes REPORT.md on each run; this CLI
exists to re-render after manual edits or for ad-hoc inspection.

Exit codes:
    0 — report rendered
    1 — lab run artifacts not found
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from omega.historical.contracts import BacktestReport
from omega.historical.lab.orchestrator import lab_dir
from omega.historical.lab.report import render
from omega.historical.lab.schemas import (
    AttemptedVariantLedger,
    HistoricalLabRun,
    PromotionEvidenceBundle,
)

logger = logging.getLogger("omega.ops.report_historical_lab")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render REPORT.md for a historical lab run.")
    parser.add_argument("--lab-run-id", required=True)
    parser.add_argument("--root", default=None, help="Artifact root (default var/historical)")
    parser.add_argument("--out", default=None, help="Output path (default <lab_dir>/REPORT.md)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = lab_dir(args.root, args.lab_run_id)
    lab_path = out_dir / "LAB_RUN.json"
    if not lab_path.exists():
        logger.error("No lab run at %s", lab_path)
        return 1

    try:
        lab_run = HistoricalLabRun.model_validate(_load(lab_path))
        ledger = AttemptedVariantLedger.model_validate(_load(out_dir / "ATTEMPTED_VARIANTS.json"))
        evidence = PromotionEvidenceBundle.model_validate(_load(out_dir / "PROMOTION_EVIDENCE.json"))
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load lab artifacts: %s", exc)
        return 1

    backtest_report = None
    bt_path = out_dir / "backtest_report.json"
    if bt_path.exists():
        backtest_report = BacktestReport.model_validate(_load(bt_path))

    md = render(lab_run, ledger, evidence, backtest_report)
    out_path = Path(args.out) if args.out else out_dir / "REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    logger.info("Wrote %s", out_path)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
