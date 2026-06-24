"""
Safe evaluator for LLM-proposed feature-combo signals (issue #28 WS3).

A proposal is a *typed hypothesis over a whitelisted feature vocabulary the engine
already extracts* — never arbitrary code. Its ``feature_combo`` is a small dict
AST that this module validates (at proposal-registration time) and evaluates (at
application time) into a single capped factor. There is **no ``eval``/``exec``**:
the AST is walked recursively over a fixed whitelist of feature names and a fixed
set of operators.

Two grammars are supported:

  * ``predicate`` — boolean ``AND``/``OR``/``NOT`` over threshold comparisons,
    yielding ``true_factor`` when it fires else ``false_factor`` (default 1.0).
    This captures the non-linear, conditional edge the market under-prices, e.g.
    ``(usage > 0.30) AND (teammate_injured == true) AND (opponent_scheme == "ZONE")``.
  * ``linear`` — ``bias + Σ weight_i · feature_i`` over whitelisted features.

Feature *values* are supplied by the agent on the EvidenceSignal (``signal.value``
is a dict); the whitelist constrains which feature *names* a proposal may
reference, so a proposal can never require new extraction code. A missing feature
value makes its comparison False / contributes 0 to a linear spec (safe no-op).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

__all__ = [
    "FEATURE_WHITELIST",
    "FeatureComboError",
    "evaluate_feature_combo",
    "validate_feature_combo",
]


class FeatureComboError(ValueError):
    """A feature_combo spec is malformed or references a non-whitelisted feature."""


# Engine-extracted primitives a proposal may reference. The agent supplies the
# values; this set bounds the *names* so no proposal can require new code.
FEATURE_WHITELIST: frozenset[str] = frozenset(
    {
        "model_prob",
        "market_prob",
        "edge",
        "line_movement",
        "recent_residual",
        "rest_days",
        "usage",
        "minutes",
        "opponent_rank",
        "pace",
        "spread",
        "total",
        "implied_total",
        "days_since_last",
        "is_home",
        "is_playoff",
        "b2b",
        "teammate_injured",
        "opponent_scheme",
    }
)

_CMP_OPS: frozenset[str] = frozenset({">", "<", ">=", "<=", "==", "!="})
_BOOL_OPS: frozenset[str] = frozenset({"AND", "OR", "NOT"})
_MAX_DEPTH = 6  # bound recursion so a pathological spec cannot blow the stack
# Predicate factors are bounded at validation (the handler's ``cap`` bounds the
# applied deviation further); this keeps a single proposal from claiming a wild swing.
_MIN_FACTOR = 0.5
_MAX_FACTOR = 2.0


# ---------------------------------------------------------------------------
# Validation (registration time)
# ---------------------------------------------------------------------------


def validate_feature_combo(spec: Any, *, whitelist: frozenset[str] = FEATURE_WHITELIST) -> None:
    """Raise :class:`FeatureComboError` unless ``spec`` is a well-formed combo.

    Run at proposal-registration time so a malformed or off-whitelist spec is
    rejected before it is ever stored or evaluated.
    """
    if not isinstance(spec, Mapping):
        raise FeatureComboError("feature_combo must be a mapping")
    kind = spec.get("kind")
    if kind == "predicate":
        if "when" not in spec:
            raise FeatureComboError("predicate combo requires a 'when' node")
        _validate_bool_node(spec["when"], whitelist, depth=0)
        _validate_factor(spec.get("true_factor"), "true_factor")
        if "false_factor" in spec:
            _validate_factor(spec["false_factor"], "false_factor")
    elif kind == "linear":
        terms = spec.get("terms")
        if not isinstance(terms, list) or not terms:
            raise FeatureComboError("linear combo requires a non-empty 'terms' list")
        for t in terms:
            if not isinstance(t, Mapping):
                raise FeatureComboError("each linear term must be a mapping")
            _require_feature(t.get("feature"), whitelist)
            _require_number(t.get("weight"), "weight")
        if "bias" in spec:
            _require_number(spec["bias"], "bias")
    else:
        raise FeatureComboError(f"unknown feature_combo kind: {kind!r}")


def _validate_bool_node(node: Any, whitelist: frozenset[str], *, depth: int) -> None:
    if depth > _MAX_DEPTH:
        raise FeatureComboError(f"feature_combo nesting exceeds max depth {_MAX_DEPTH}")
    if not isinstance(node, Mapping):
        raise FeatureComboError("boolean node must be a mapping")
    op = node.get("op")
    if op in ("AND", "OR"):
        terms = node.get("terms")
        if not isinstance(terms, list) or not terms:
            raise FeatureComboError(f"{op} requires a non-empty 'terms' list")
        for t in terms:
            _validate_bool_node(t, whitelist, depth=depth + 1)
    elif op == "NOT":
        if "term" not in node:
            raise FeatureComboError("NOT requires a single 'term'")
        _validate_bool_node(node["term"], whitelist, depth=depth + 1)
    elif op in _CMP_OPS:
        _require_feature(node.get("feature"), whitelist)
        if "value" not in node:
            raise FeatureComboError(f"comparison {op!r} requires a 'value'")
        val = node["value"]
        if not isinstance(val, (int, float, bool, str)):
            raise FeatureComboError("comparison 'value' must be a number, bool, or string")
    else:
        raise FeatureComboError(f"unknown operator: {op!r}")


def _require_feature(name: Any, whitelist: frozenset[str]) -> None:
    if not isinstance(name, str) or name not in whitelist:
        raise FeatureComboError(f"feature {name!r} is not in the whitelist")


def _require_number(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FeatureComboError(f"{label} must be a finite number")


def _validate_factor(value: Any, label: str) -> None:
    _require_number(value, label)
    if not (_MIN_FACTOR <= float(value) <= _MAX_FACTOR):
        raise FeatureComboError(f"{label} must be within [{_MIN_FACTOR}, {_MAX_FACTOR}]")


# ---------------------------------------------------------------------------
# Evaluation (application time)
# ---------------------------------------------------------------------------


def evaluate_feature_combo(
    spec: Mapping[str, Any],
    features: Mapping[str, Any],
    *,
    whitelist: frozenset[str] = FEATURE_WHITELIST,
) -> float:
    """Evaluate a (pre-validated) combo against feature values into a factor.

    Defensive: re-validates structure cheaply and treats a missing feature as a
    False comparison / zero linear contribution, so a thin feature dict yields a
    neutral 1.0 rather than raising mid-prediction. The caller (the handler) caps
    the returned factor.
    """
    kind = spec.get("kind")
    if kind == "predicate":
        fired = _eval_bool_node(spec["when"], features, whitelist, depth=0)
        factor = spec.get("true_factor", 1.0) if fired else spec.get("false_factor", 1.0)
        return float(factor)
    if kind == "linear":
        total = float(spec.get("bias", 1.0))
        for t in spec["terms"]:
            value = _as_number(features.get(t["feature"]))
            if value is not None:
                total += float(t["weight"]) * value
        return total
    raise FeatureComboError(f"unknown feature_combo kind: {kind!r}")


def _eval_bool_node(
    node: Mapping[str, Any], features: Mapping[str, Any], whitelist: frozenset[str], *, depth: int
) -> bool:
    if depth > _MAX_DEPTH:
        raise FeatureComboError(f"feature_combo nesting exceeds max depth {_MAX_DEPTH}")
    op = node.get("op")
    if op == "AND":
        return all(_eval_bool_node(t, features, whitelist, depth=depth + 1) for t in node["terms"])
    if op == "OR":
        return any(_eval_bool_node(t, features, whitelist, depth=depth + 1) for t in node["terms"])
    if op == "NOT":
        return not _eval_bool_node(node["term"], features, whitelist, depth=depth + 1)
    # comparison leaf
    if node.get("feature") not in whitelist:
        raise FeatureComboError(f"feature {node.get('feature')!r} is not in the whitelist")
    return _eval_comparison(op, features.get(node["feature"]), node["value"])


def _eval_comparison(op: str, actual: Any, expected: Any) -> bool:
    if actual is None:
        return False  # missing feature -> comparison does not fire (safe no-op)
    if op in (">", "<", ">=", "<="):
        a, b = _as_number(actual), _as_number(expected)
        if a is None or b is None:
            return False
        if op == ">":
            return a > b
        if op == "<":
            return a < b
        if op == ">=":
            return a >= b
        return a <= b
    # equality: numeric (with tolerance) or exact non-numeric
    a_num, b_num = _as_number(actual), _as_number(expected)
    if a_num is not None and b_num is not None:
        equal = math.isclose(a_num, b_num, abs_tol=1e-9)
    else:
        equal = _norm(actual) == _norm(expected)
    return equal if op == "==" else not equal


def _as_number(value: Any) -> float | None:
    """Coerce to float; bools count as 1/0; non-numeric strings return None."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _norm(value: Any) -> str:
    """Case-insensitive string form for non-numeric equality (e.g. 'ZONE' == 'zone')."""
    return str(value).strip().lower()
