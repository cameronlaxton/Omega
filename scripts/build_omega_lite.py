"""
Rebuild omega_lite/ from the canonical omega/ source tree.

Run this whenever the canonical files in omega/core/ (or
omega/reasoning/evaluator.py) change so the sandbox port stays in sync.

Usage:
    python scripts/build_omega_lite.py            # rebuild in place
    python scripts/build_omega_lite.py --zip      # also produce omega_lite-v1.zip

The rebuild:
    1. Copies the verbatim-port files
    2. Rewrites their imports to point at omega_lite.*
    3. Truncates engine.py to drop the two markov-dependent methods
    4. Regenerates models.py (subset of omega/core/models.py)
    5. Leaves __init__.py, run.py, quality_gate.py header, and
       _quality_helpers.py alone — those are hand-written

The rebuild is idempotent: running it twice produces the same tree.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "omega"
DST = REPO_ROOT / "omega_lite"

# Files copied verbatim with a simple `from omega.X` → `from omega_lite.X` rewrite.
VERBATIM_COPIES = [
    (SRC / "core" / "betting" / "odds.py",         DST / "odds.py"),
    (SRC / "core" / "betting" / "kelly.py",        DST / "kelly.py"),
    (SRC / "core" / "calibration" / "probability.py", DST / "calibration.py"),
    (SRC / "core" / "config" / "leagues.py",       DST / "leagues.py"),
    (SRC / "core" / "simulation" / "archetypes.py", DST / "archetypes.py"),
    (SRC / "core" / "simulation" / "validation.py", DST / "validation.py"),
    (SRC / "core" / "contracts" / "schemas.py",    DST / "schemas.py"),
    (SRC / "core" / "contracts" / "service.py",    DST / "service.py"),
    (SRC / "reasoning" / "evaluator.py",           DST / "quality_gate.py"),
]

# Where engine.py should be cut off — keep up to and including run_fast_game_simulation.
ENGINE_KEEP_LINES = 899


def _rewrite_omega_imports(text: str) -> str:
    """Replace `from omega.core.X` and `from omega.reasoning.gatherer` imports
    with the omega_lite equivalents."""
    # Module-level remaps
    remaps = [
        (r"from omega\.core\.betting\.odds",          "from omega_lite.odds"),
        (r"from omega\.core\.betting\.kelly",         "from omega_lite.kelly"),
        (r"from omega\.core\.calibration\.probability", "from omega_lite.calibration"),
        (r"from omega\.core\.config\.leagues",        "from omega_lite.leagues"),
        (r"from omega\.core\.simulation\.archetypes", "from omega_lite.archetypes"),
        (r"from omega\.core\.simulation\.validation", "from omega_lite.validation"),
        (r"from omega\.core\.simulation\.engine",     "from omega_lite.engine"),
        (r"from omega\.core\.contracts\.schemas",     "from omega_lite.schemas"),
        (r"from omega\.core\.contracts\.service",     "from omega_lite.service"),
        (r"from omega\.core\.models",                 "from omega_lite.models"),
        (r"from omega\.reasoning\.gatherer",          "from omega_lite._quality_helpers"),
    ]
    for pattern, replacement in remaps:
        text = re.sub(pattern, replacement, text)

    # Logger names
    text = text.replace('logging.getLogger("omega.service")', 'logging.getLogger("omega_lite.service")')
    text = text.replace('logging.getLogger("omega.core.simulation.validation")', 'logging.getLogger("omega_lite.validation")')
    text = text.replace('logging.getLogger("omega.agent.quality_gate")', 'logging.getLogger("omega_lite.quality_gate")')

    return text


def _copy_engine() -> None:
    """Copy engine.py, rewrite imports, truncate to fast-path-only."""
    text = (SRC / "core" / "simulation" / "engine.py").read_text(encoding="utf-8")
    text = _rewrite_omega_imports(text)
    lines = text.splitlines(keepends=True)
    truncated = "".join(lines[:ENGINE_KEEP_LINES])
    (DST / "engine.py").write_text(truncated, encoding="utf-8")


def _copy_verbatim() -> None:
    for src, dst in VERBATIM_COPIES:
        text = src.read_text(encoding="utf-8")
        text = _rewrite_omega_imports(text)
        dst.write_text(text, encoding="utf-8")


def _regen_models() -> None:
    """The models.py subset is small enough to keep as-is; only flag if it
    drifts from what the quality gate imports. We don't auto-regenerate it
    because it's hand-curated."""
    # Sanity check: does our subset cover everything quality_gate.py imports?
    qg = (DST / "quality_gate.py").read_text(encoding="utf-8")
    needed = re.findall(r"from omega_lite\.models import \(([^)]+)\)", qg)
    if not needed:
        return
    imported = {n.strip().rstrip(",") for line in needed[0].splitlines() for n in [line.strip()] if n}
    mdl = (DST / "models.py").read_text(encoding="utf-8")
    for name in imported:
        if name and name not in mdl:
            print(f"WARNING: quality_gate.py imports {name!r} but omega_lite/models.py "
                  "does not define it. Update models.py by hand.")


def _build_zip(target: Path) -> None:
    """Create a flat zip of the omega_lite package ready to upload."""
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for py in sorted(DST.glob("*.py")):
            zf.write(py, arcname=f"omega_lite/{py.name}")
    print(f"Wrote {target} ({target.stat().st_size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", action="store_true", help="Also produce omega_lite-v1.zip")
    args = parser.parse_args()

    if not SRC.is_dir():
        print(f"ERROR: canonical source tree not found at {SRC}", file=sys.stderr)
        return 1
    DST.mkdir(exist_ok=True)

    _copy_verbatim()
    _copy_engine()
    _regen_models()

    print(f"Rebuilt {DST} from {SRC}.")

    if args.zip:
        target = REPO_ROOT / "omega_lite-v1.zip"
        _build_zip(target)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
