# Historical Validation Lab — lab_golden

## Provenance
- created_at: 2023-08-01T00:00:00+00:00
- code_version: omega-test
- git_commit: deadbeefdeadbeefdeadbeefdeadbeefdeadbeef
- working_tree_dirty: False
- league / plane / market: FIFA_INTL / draw / draw
- dataset_manifest_id: mfest
- dataset_hash: dshash
- replay_id: rep1
- replay_config_hash: cfg
- profile_grid_hash: grid

## Windows
- train:      2023-01-01 .. 2023-04-30
- validation: 2023-05-01 .. 2023-06-15
- holdout:    2023-06-16 .. 2023-07-31
- holdout_sealed: True (access_count=1)

## Promotion
- status: **shadow_only**
- auto_promote_armed: False
- candidate_id: —
- incumbent_id: —
- recommended: False

### Parity verdicts
- backtest_parity: no_incumbent
- clv_walk_forward: INCONCLUSIVE
- historical_live_parity: INCONCLUSIVE
- promotion_gate: not evaluated
- clv_coherent (incremental-edge risk flag, non-gating): True

## Winner's-curse
- attempted variants (N): 3
- validation→holdout ECE delta: 0.0050
- risk: **elevated**

## Attempted variants

| variant_id | family | slice | n_train | n_val | val_brier | val_ece | cv_ece | holdout_ece | status |
|---|---|---|---|---|---|---|---|---|---|
| draw_isotonic_base | isotonic | base | 200 | 60 | 0.2100 | 0.0300 | 0.0320 | 0.0350 | selected |
| draw_shrinkage_base | shrinkage | base | 200 | 60 | 0.2250 | 0.0500 | 0.0520 | — | rejected |
| draw_isotonic_playoff | isotonic | playoff | 10 | 2 | — | — | — | — | skipped |

