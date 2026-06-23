"""CLI wrapper for Omega's packaged Odds API resolver."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.integrations.odds_resolver import (  # noqa: E402,F401
    PROP_MARKET_MAP,
    normalize_book_odds,
    normalize_event_odds,
    provider_market_for_prop,
    resolve_odds,
)
from omega.integrations.odds_resolver import main as _main  # noqa: E402


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
