"""
DEPRECATED — use scripts/fetch_closing_lines.py instead.

This module remains only as a thin back-compat shim so existing automation
(cron jobs, allowlist entries in .claude/settings.local.json) keeps working.
It forces --league NBA and forwards everything else to the generalized script.

Remove this shim once all callers reference fetch_closing_lines.py directly.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-export public symbols other code may have imported historically.
from scripts.fetch_closing_lines import (  # noqa: E402,F401
    _MARKET_MAP,
    _is_supported_market,
    _match_outcome,
    _pending_bets_needing_close,
)
from scripts.fetch_closing_lines import (  # noqa: E402
    main as _generalized_main,
)


def main() -> int:
    warnings.warn(
        "scripts/fetch_closing_lines_nba.py is deprecated; "
        "use scripts/fetch_closing_lines.py (optionally with --league NBA).",
        DeprecationWarning,
        stacklevel=2,
    )
    # If the caller didn't already pin a league, default to NBA to preserve
    # the old behavior. Don't override an explicit --league NFL etc.
    if not any(a == "--league" or a.startswith("--league=") for a in sys.argv[1:]):
        sys.argv.append("--league")
        sys.argv.append("NBA")
    return _generalized_main()


if __name__ == "__main__":
    sys.exit(main())
