"""
Rebuild omega_lite/ from the canonical omega/ source tree.

Run this whenever the canonical files in omega/core/ (or
omega/reasoning/evaluator.py) change so the sandbox port stays in sync.

Usage:
    python scripts/build_omega_lite.py                  # rebuild omega_lite/ in place
    python scripts/build_omega_lite.py --single-file    # ALSO emit omega_lite_standalone.py
    python scripts/build_omega_lite.py --zip            # legacy: also produce omega_lite-v1.zip

The rebuild:
    1. Copies the verbatim-port files
    2. Rewrites their imports to point at omega_lite.*
    3. Truncates engine.py to drop the two markov-dependent methods
    4. Regenerates models.py (subset of omega/core/models.py)
    5. Leaves __init__.py, run.py, quality_gate.py header, and
       _quality_helpers.py alone — those are hand-written

The rebuild is idempotent: running it twice produces the same tree.

The --single-file mode additionally concatenates all omega_lite/*.py modules
into one self-contained omega_lite_standalone.py at the repo root. This is
the artifact the agent loads at session start (see prompts/system_prompt.txt §3).
No zip extraction required.
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
# Marker: the next method after run_fast_game_simulation is run_game_simulation
# (Markov-based, depends on omega.markov which isn't shipped to the sandbox).
# We truncate immediately before its `def run_game_simulation(` line.
ENGINE_TRUNCATE_MARKER = "    def run_game_simulation("


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
    """Copy engine.py, rewrite imports, truncate to fast-path-only.

    Truncation is marker-based: we slice off everything from the
    ENGINE_TRUNCATE_MARKER line onward, so the cutoff survives
    additions/removals above run_game_simulation without manual line-count
    bookkeeping. Falls back to a hard error if the marker is missing.
    """
    text = (SRC / "core" / "simulation" / "engine.py").read_text(encoding="utf-8")
    text = _rewrite_omega_imports(text)
    lines = text.splitlines(keepends=True)
    cut_idx: int | None = None
    for i, line in enumerate(lines):
        if line.startswith(ENGINE_TRUNCATE_MARKER):
            cut_idx = i
            break
    if cut_idx is None:
        raise RuntimeError(
            f"_copy_engine: truncation marker {ENGINE_TRUNCATE_MARKER!r} not found "
            "in engine.py. Update ENGINE_TRUNCATE_MARKER in scripts/build_omega_lite.py."
        )
    truncated = "".join(lines[:cut_idx])
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


# Order matters — modules listed earlier must not reference symbols from
# modules listed later. This mirrors the actual omega_lite import topology.
SINGLE_FILE_ORDER = [
    "models.py",
    "odds.py",
    "kelly.py",
    "archetypes.py",
    "leagues.py",
    "calibration.py",
    "validation.py",
    "schemas.py",
    "_quality_helpers.py",
    "quality_gate.py",
    "engine.py",
    "service.py",
    "run.py",
]


_INTERNAL_IMPORT_RE = re.compile(
    r"""^\s*(?:from\s+omega_lite(?:\.\w+)?\s+import\s+[^\n]+|import\s+omega_lite(?:\.\w+)?[^\n]*)$""",
    re.MULTILINE,
)
_FUTURE_IMPORT_RE = re.compile(r"^from __future__ import [^\n]+\n", re.MULTILINE)
_LEADING_DOCSTRING_RE = re.compile(r'^\s*"""[\s\S]*?"""\s*\n', re.MULTILINE)
_PARENTHESIZED_INTERNAL_IMPORT_RE = re.compile(
    r"^from\s+omega_lite(?:\.\w+)?\s+import\s*\([^)]*\)\s*\n",
    re.MULTILINE,
)


def _strip_module(text: str, is_first: bool) -> str:
    """Remove internal imports, future-imports, and (for non-first modules)
    the leading docstring. Keep external imports — they get hoisted later
    by Python's normal de-duplication when we just collect them all in the
    head; in practice Python tolerates duplicate `import X` lines anyway.
    """
    # Strip multi-line parenthesized internal imports first.
    text = _PARENTHESIZED_INTERNAL_IMPORT_RE.sub("", text)
    # Strip single-line internal imports.
    text = _INTERNAL_IMPORT_RE.sub("", text)
    # Strip all future-imports (we'll hoist exactly one at the top).
    text = _FUTURE_IMPORT_RE.sub("", text)
    if not is_first:
        # Drop only the very first leading docstring of this module.
        text = _LEADING_DOCSTRING_RE.sub("", text, count=1)
    return text.lstrip() + "\n"


def _build_single_file(target: Path) -> None:
    """Concatenate all omega_lite/*.py into one self-contained module.

    Output layout:
      1. Header docstring (purpose + provenance).
      2. `from __future__ import annotations`.
      3. Bodies of every omega_lite/*.py module in topological order.
         Internal `from omega_lite.X` imports are stripped — every symbol
         lives in the same module namespace after concatenation.
    """
    header = (
        '"""\n'
        'omega_lite_standalone — single-file deterministic engine for Omega.\n\n'
        'AUTO-GENERATED by scripts/build_omega_lite.py --single-file.\n'
        'Do not edit this file by hand; edit the canonical sources in omega/\n'
        'and rerun the build script. Bit-identical math to canonical Omega,\n'
        'verified by tests/omega_lite/test_parity.py.\n\n'
        'Usage from a sandbox at session start:\n'
        '    import os, sys\n'
        '    sys.path.insert(0, os.getcwd())\n'
        '    from omega_lite_standalone import analyze\n'
        '    smoke = analyze({"player_name": "X", "league": "NBA",\n'
        '                     "prop_type": "pts", "line": 20.0,\n'
        '                     "home_team": "Smoke Home",\n'
        '                     "away_team": "Smoke Away",\n'
        '                     "game_date": "2026-05-14",\n'
        '                     "odds_over": -110, "odds_under": -110,\n'
        '                     "player_context": {"pts_mean": 20.0, "pts_std": 5.0},\n'
        '                     "seed": 1},\n'
        '                    session_id="sess-20260518-smok", bankroll=1000.0)\n'
        '    assert smoke["trace_id"].startswith("sandbox-")\n'
        '"""\n\n'
        'from __future__ import annotations\n\n'
    )

    bodies = []
    for i, fname in enumerate(SINGLE_FILE_ORDER):
        path = DST / fname
        if not path.is_file():
            raise FileNotFoundError(
                f"Expected omega_lite module not found: {path}. "
                "Run scripts/build_omega_lite.py without --single-file first."
            )
        text = path.read_text(encoding="utf-8")
        body = _strip_module(text, is_first=(i == 0))
        bodies.append(
            f"# {'=' * 72}\n"
            f"# Inlined from omega_lite/{fname}\n"
            f"# {'=' * 72}\n\n"
            f"{body}\n"
        )

    content = header + "".join(bodies)
    target.write_text(content, encoding="utf-8")
    print(f"Wrote {target} ({target.stat().st_size:,} bytes, {len(SINGLE_FILE_ORDER)} modules inlined)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", action="store_true",
                        help="Legacy: also produce omega_lite-v1.zip")
    parser.add_argument("--single-file", action="store_true", dest="single_file",
                        help="Also produce omega_lite_standalone.py at repo root")
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

    if args.single_file:
        target = REPO_ROOT / "omega_lite_standalone.py"
        _build_single_file(target)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
