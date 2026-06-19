"""Structural guardrails: the lab reuses the single engine, never re-implements it.

These assert the plan's core constraint at the AST level — the lab package must
not define its own fitter, registry write, grading, staking, or promotion-gate
logic, and the grid must stay registry-free so it can never pollute profiles.json.
"""

from __future__ import annotations

import ast
from pathlib import Path

import omega.historical.lab as lab_pkg

LAB_DIR = Path(lab_pkg.__file__).parent

# Symbols whose presence would mean a primitive was re-implemented instead of reused.
FORBIDDEN = {
    "fit_isotonic",
    "fit_shrinkage",
    "kelly_fraction",
    "evaluate_promotion_gates",
    "_grade_selection",
    "_pool_adjacent_violators",
    "_adaptive_calibration_error",
    "compute_clv",
    "betting_metrics",
    "probability_metrics",
    "stratified_folds",
}


def _module_files() -> list[Path]:
    return [p for p in LAB_DIR.glob("*.py")]


def _defs_and_imports(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return names, imports


def test_no_reimplemented_primitives():
    for f in _module_files():
        names, _ = _defs_and_imports(f)
        clash = names & FORBIDDEN
        assert not clash, f"{f.name} re-defines {clash}; reuse the canonical implementation"


def test_reuses_the_single_engine():
    all_imports: set[str] = set()
    for f in _module_files():
        _, imps = _defs_and_imports(f)
        all_imports |= imps
    assert "omega.core.calibration.fitter" in all_imports, "must reuse the single fitter"
    assert "omega.core.calibration.registry" in all_imports, "must reuse the single registry"
    assert "omega.core.calibration.promotion" in all_imports, "must reuse the single gate"
    assert any(
        m in all_imports for m in ("omega.historical.metrics", "omega.historical.walk_forward")
    ), "must reuse the single metric/walk-forward path"


def test_grid_stays_registry_free():
    _, imports = _defs_and_imports(LAB_DIR / "grid.py")
    assert not any("registry" in m for m in imports), (
        "grid.py must not import the registry — a grid of N variants must never "
        "register N-1 junk candidates"
    )


def test_lab_library_defines_no_cli_main():
    # CLIs (the only place a fit/promote may be triggered) live in ops/, not here.
    for f in _module_files():
        names, _ = _defs_and_imports(f)
        assert "main" not in names, f"{f.name} should not define a CLI main()"
