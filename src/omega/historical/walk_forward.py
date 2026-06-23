"""Chronological walk-forward calibration over replayed traces.

For each fold: select training traces strictly before the fold's test window, fit
a base profile plus eligible context-slice profiles, **freeze** them in memory
(never written to the production registry), then evaluate the future test window
reporting raw-vs-calibrated probability metrics. No random train/test split is
ever used. Each frozen profile's snapshot + hash is recorded on the fold for
reproducibility.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from omega.core.calibration.fitter import CalibrationFitter
from omega.core.calibration.probability import calibrate_probability
from omega.core.calibration.profiles import CalibrationProfile
from omega.historical.contracts import (
    BacktestReport,
    FoldResult,
    FrozenProfileRef,
    MetricBlock,
    ReplayCandidateSelection,
    ReplayEventRecord,
    WalkForwardConfig,
    stable_hash,
)
from omega.historical.metrics import betting_metrics, health_metrics, probability_metrics
from omega.trace.store import TraceStore

UTC = timezone.utc
_FIT_MIN_SAMPLES = 30  # CalibrationFitter floor


# ---------------------------------------------------------------------------
# Pair extraction per calibration plane
# ---------------------------------------------------------------------------


def _game_pair(trace: dict) -> tuple[float, int] | None:
    p = (trace.get("predictions") or {}).get("home_win_prob")
    res = (trace.get("_outcome") or {}).get("result")
    if p is None or res not in ("home_win", "away_win", "draw"):
        return None
    return (p / 100.0 if p > 1 else float(p)), (1 if res == "home_win" else 0)


def _draw_pair(trace: dict) -> tuple[float, int] | None:
    p = (trace.get("predictions") or {}).get("draw_prob")
    res = (trace.get("_outcome") or {}).get("result")
    if p is None or res not in ("home_win", "away_win", "draw"):
        return None
    return (p / 100.0 if p > 1 else float(p)), (1 if res == "draw" else 0)


_PAIR_FN = {"game": _game_pair, "draw": _draw_pair}


# ---------------------------------------------------------------------------
# Slice predicates (derived from the trace's context labels / game_context)
# ---------------------------------------------------------------------------


def _labels_of(trace: dict) -> dict:
    labels = dict(trace.get("context_labels") or {})
    gc = (trace.get("input_snapshot") or {}).get("game_context") or {}
    return {**gc, **labels}


_SLICE_PREDICATES = {
    "playoff": lambda m: bool(m.get("is_playoff")),
    "neutral_site": lambda m: bool(m.get("neutral_site")),
    "back_to_back": lambda m: (
        0 in (m.get("rest_days"), m.get("home_rest_days"), m.get("away_rest_days"))
    ),
    "short_week": lambda m: m.get("rest_days") is not None and m["rest_days"] <= 3,
    "congested_fixture": lambda m: m.get("rest_days") is not None and m["rest_days"] <= 3,
    "rest_disadvantage": lambda m: (
        m.get("home_rest_days") is not None
        and m.get("away_rest_days") is not None
        and abs(m["home_rest_days"] - m["away_rest_days"]) >= 2
    ),
    "three_in_four": lambda m: bool(m.get("three_in_four")),
    "best_of_5": lambda m: m.get("best_of") == 5,
    "serve_dominant": lambda m: bool(m.get("serve_dominant")),
    "surface_hard": lambda m: str(m.get("surface")).lower() == "hard",
    "surface_clay": lambda m: str(m.get("surface")).lower() == "clay",
    "surface_grass": lambda m: str(m.get("surface")).lower() == "grass",
    "surface_indoor_hard": lambda m: str(m.get("surface")).lower() == "indoor_hard",
    "park_factor_extreme": lambda m: (
        m.get("park_factor") is not None and (m["park_factor"] >= 1.1 or m["park_factor"] <= 0.9)
    ),
}


def _slice_of(trace: dict, slices: list[str]) -> str | None:
    """Return the first configured slice the trace belongs to, else None (base)."""
    if not slices:
        return None
    m = _labels_of(trace)
    for s in slices:
        pred = _SLICE_PREDICATES.get(s, lambda mm, _s=s: bool(mm.get(_s)))
        try:
            if pred(m):
                return s
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------


def _dt(iso: str) -> datetime:
    s = iso[:-1] + "+00:00" if iso.endswith("Z") else iso
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def partition_fold(
    graded: list[dict], ts_iso: str, te_iso: str, config: WalkForwardConfig
) -> tuple[list[dict], list[dict], str | None]:
    """Split graded traces into (train, test, train_start) for one fold.

    Training is everything strictly before ``ts_iso`` (expanding) or within the
    rolling window ``[ts_iso - train_window, ts_iso)``. Test is ``[ts_iso, te_iso)``.
    This is the single place the no-future-leak rule is enforced.
    """
    if config.mode == "rolling" and config.train_window_days:
        tr_start_iso = (_dt(ts_iso) - timedelta(days=config.train_window_days)).isoformat()
        train = [t for t in graded if tr_start_iso <= t["_dt"] < ts_iso]
    else:
        tr_start_iso = None
        train = [t for t in graded if t["_dt"] < ts_iso]
    test = [t for t in graded if ts_iso <= t["_dt"] < te_iso]
    return train, test, tr_start_iso


def _generate_folds(
    dates_iso: list[str], config: WalkForwardConfig
) -> list[tuple[datetime, datetime]]:
    if not dates_iso:
        return []
    ordered = sorted(_dt(d) for d in dates_iso)
    start, end = ordered[0], ordered[-1]
    test_w = timedelta(days=config.test_window_days)
    step = timedelta(days=config.step_days or config.test_window_days)
    folds: list[tuple[datetime, datetime]] = []
    test_start = start + test_w  # leave the first window for training
    while test_start <= end:
        folds.append((test_start, test_start + test_w))
        test_start += step
    return folds


# ---------------------------------------------------------------------------
# Profile fitting + freezing
# ---------------------------------------------------------------------------


def _fit(
    pairs: list[tuple[float, int]], league: str, market: str, *, context_slice: str | None = None
) -> CalibrationProfile | None:
    if len(pairs) < _FIT_MIN_SAMPLES:
        return None
    preds = [p for p, _ in pairs]
    outs = [y for _, y in pairs]
    try:
        profile = CalibrationFitter().fit_isotonic(preds, outs, league=league, market=market)
    except ValueError:
        return None
    if context_slice:
        profile = profile.model_copy(update={"context_slice": context_slice})
    return profile


def _ref(market: str, context_slice: str | None, profile: CalibrationProfile) -> FrozenProfileRef:
    profile_hash = stable_hash(
        {
            "league": profile.league,
            "market": market,
            "context_slice": context_slice,
            "method": profile.method,
            "params": profile.params,
        }
    )
    return FrozenProfileRef(
        market=market,
        context_slice=context_slice,
        method=profile.method,
        profile_id=profile.profile_id,
        profile_hash=profile_hash,
        sample_size=profile.sample_size,
        params_snapshot=profile.params,
    )


def _apply(profile: CalibrationProfile | None, p: float) -> float:
    if profile is None:
        return p
    return calibrate_probability(p, method=profile.method, **profile.params)["calibrated"]


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_walk_forward(
    store: TraceStore,
    *,
    config: WalkForwardConfig,
    league: str,
    replay_id: str,
    dataset_manifest_id: str,
    selections: list[ReplayCandidateSelection] | None = None,
    replay_records: list[ReplayEventRecord] | None = None,
) -> BacktestReport:
    """Run a chronological walk-forward backtest and return a :class:`BacktestReport`."""
    graded = [
        t
        for t in store.get_graded_traces(league=league, limit=100_000)
        if t.get("historical_replay")
    ]
    for t in graded:
        t["_dt"] = t.get("decision_time") or t.get("timestamp")
    graded.sort(key=lambda t: t["_dt"])

    selections = selections or []
    outcomes_by_event = {
        t["event_id"]: t["_outcome"] for t in graded if t.get("event_id") and t.get("_outcome")
    }
    closing_by_trace = {sel.trace_id: store.get_closing_lines(sel.trace_id) for sel in selections}

    folds: list[FoldResult] = []
    agg_pairs: dict[str, tuple[list[float], list[float], list[int]]] = defaultdict(
        lambda: ([], [], [])
    )
    agg_fb = 0
    agg_slice_total = 0

    for idx, (ts, te) in enumerate(_generate_folds([t["_dt"] for t in graded], config)):
        ts_iso, te_iso = ts.isoformat(), te.isoformat()
        train, test, tr_start_iso = partition_fold(graded, ts_iso, te_iso, config)

        if len(train) < config.min_train_samples or not test:
            continue
        # Hard guard: training must be strictly before the test window.
        assert all(t["_dt"] < ts_iso for t in train), "walk-forward train/future leak"

        metrics_by_market: dict[str, MetricBlock] = {}
        frozen_refs: list[FrozenProfileRef] = []
        fold_fb = 0
        fold_slice_total = 0

        for market in config.markets:
            pair_fn = _PAIR_FN.get(market)
            if pair_fn is None:
                continue

            base_pairs = [pf for t in train if (pf := pair_fn(t))]
            base_profile = _fit(base_pairs, league, market)
            frozen: dict[str | None, CalibrationProfile | None] = {None: base_profile}
            if base_profile is not None:
                frozen_refs.append(_ref(market, None, base_profile))

            for s in config.slices:
                s_pairs = [pf for t in train if _slice_of(t, [s]) == s and (pf := pair_fn(t))]
                if len(s_pairs) >= config.min_slice_samples:
                    prof = _fit(s_pairs, league, market, context_slice=s)
                    if prof is not None:
                        frozen[s] = prof
                        frozen_refs.append(_ref(market, s, prof))

            raw: list[float] = []
            cal: list[float] = []
            outs: list[int] = []
            per_slice: dict[str, tuple[list[float], list[float], list[int]]] = defaultdict(
                lambda: ([], [], [])
            )
            for t in test:
                pf = pair_fn(t)
                if not pf:
                    continue
                p, y = pf
                label = _slice_of(t, config.slices)
                if label is not None:
                    fold_slice_total += 1
                    if frozen.get(label) is None:
                        fold_fb += 1
                profile = frozen.get(label) or frozen.get(None)
                c = _apply(profile, p)
                raw.append(p)
                cal.append(c)
                outs.append(y)
                key = label or "base"
                ps = per_slice[key]
                ps[0].append(p)
                ps[1].append(c)
                ps[2].append(y)
                ap = agg_pairs[market]
                ap[0].append(p)
                ap[1].append(c)
                ap[2].append(y)
                if label is not None:
                    apk = agg_pairs[f"{market}:{label}"]
                    apk[0].append(p)
                    apk[1].append(c)
                    apk[2].append(y)

            if not raw:
                continue
            metrics_by_market[market] = probability_metrics(raw, cal, outs)
            for key, (rp, cp, oo) in per_slice.items():
                if key == "base":
                    continue
                metrics_by_market[f"{market}:{key}"] = probability_metrics(rp, cp, oo)

        # betting + health for the fold window
        fold_sels = [s for s in selections if ts_iso <= s.decision_time < te_iso]
        fold_betting = (
            betting_metrics(fold_sels, outcomes_by_event, closing_by_trace) if fold_sels else None
        )
        fold_records = (
            [r for r in (replay_records or []) if ts_iso <= r.decision_time < te_iso]
            if replay_records
            else []
        )
        fallback_rate = fold_fb / fold_slice_total if fold_slice_total else 0.0
        fold_health = health_metrics(fold_records, fallback_profile_rate=fallback_rate)

        agg_fb += fold_fb
        agg_slice_total += fold_slice_total

        folds.append(
            FoldResult(
                fold_index=idx,
                train_start=tr_start_iso,
                train_end=ts_iso,
                test_start=ts_iso,
                test_end=te_iso,
                n_train=len(train),
                n_test=len(test),
                metrics_by_market=metrics_by_market,
                betting=fold_betting,
                health=fold_health,
                frozen_profiles=frozen_refs,
            )
        )

    aggregate_metrics = {
        market: probability_metrics(rp, cp, oo) for market, (rp, cp, oo) in agg_pairs.items()
    }
    aggregate_betting = (
        betting_metrics(selections, outcomes_by_event, closing_by_trace) if selections else None
    )
    aggregate_health = health_metrics(
        replay_records or [],
        fallback_profile_rate=(agg_fb / agg_slice_total if agg_slice_total else 0.0),
    )

    return BacktestReport(
        manifest_id=dataset_manifest_id,
        replay_id=replay_id,
        league=league.upper(),
        walk_forward_config=config,
        folds=folds,
        aggregate_metrics_by_market=aggregate_metrics,
        aggregate_betting=aggregate_betting,
        aggregate_health=aggregate_health,
    )
