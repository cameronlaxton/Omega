# Context-Slicing Calibration Runbook

Context slicing improves the accuracy of simulation engine outputs by fitting calibration profiles to specific high-impact scenarios (e.g., `playoff`, `back_to_back`), correcting systemic biases that exist only in those contexts.

## 1. Audit Input Quality

Before deciding to fit a new context slice, verify that traces contain the required contextual data:
```bash
python -m omega.ops.report_input_quality --league NBA --db data/omega_traces.db
```
If traces are missing `context_labels`, `market`, or `team_context`, investigate the upstream data gatherers (e.g., `priors_nba`) before proceeding.

## 2. Check Raw Simulation Bias

Report the baseline (pre-calibration) bias to see if specific contexts are structurally under-performing.
```bash
python -m omega.ops.report_calibration_bias --league NBA --db data/omega_traces.db
```
This evaluates the raw probabilities and calculates ECE, Brier score, and Log Loss for each context slice (e.g., `base`, `playoff`). 

## 3. Fit a Context-Sliced Profile

If a context slice shows significant bias and has sufficient sample size (e.g. >1000 observations), fit a dedicated profile:
```bash
python -m omega.ops.fit_calibration \
    --league NBA \
    --context-slice playoff \
    --db data/omega_traces.db \
    --method isotonic
```
This generates a profile strictly targeting traces where `context_slice_for_trace(...) == "playoff"`.

## 4. Promotion and the Isolated Incumbent Rule

Promote the profile using the standard tool:
```bash
python -m omega.ops.promote_profile --profile-id <GENERATED_PROFILE_ID>
```
**Important:** Under the slice-isolated promotion logic, a `playoff` profile only competes against the incumbent `playoff` profile. It will never fall back to evaluate against the `base` profile.

## 5. Runtime Resolution

Once promoted to `production`, any new trace exhibiting the `playoff` hint (via `labels_from_trace`) will exactly match and use the `playoff` profile during `omega.core.contracts.service.analyze()`. Traces with no hints will resolve to `base`.

## Simulation Dispersion Tuning
If bias persists after calibration (i.e. calibration fixes probability magnitudes but ECE shows consistent over/under-confidence), you may adjust the explicit `DispersionPolicy` in the backend parameters, which directly scales model variance (e.g. Poisson lambdas, negative binomial $k$, or Markov pressure states) before simulation.
