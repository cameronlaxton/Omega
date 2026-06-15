# Input Quality Audit

The `report_input_quality.py` tool provides an offline audit of data gaps in historical simulation traces. Before fitting context-sliced calibration profiles, it is critical to verify that the upstream data population supports the intended slicing.

## Rationale
Context-sliced calibration relies entirely on the presence and accuracy of `context_labels` and `market` data within each trace. If the game context or prior payload is missing, the simulation traces cannot be correctly assigned to specific context slices (e.g., `playoff`, `back_to_back`, `short_week`). 
Consequently, these traces would fall back to the generic base profile, polluting the baseline and starving the specialized slices of data.

## Usage
Run the script to analyze the missing components of a specific league:
```bash
python -m omega.ops.report_input_quality --league NBA --db data/omega_traces.db
```

## Report Output
The report analyzes all "graded" traces for the league and logs the frequency of:
- `missing_context_labels`: No context labels present (required for slicing).
- `missing_market`: Missing target market probabilities.
- `missing_team_context`: Missing offensive/defensive context payload for home/away sides.

## Mitigation
If high percentages of data gaps are observed:
1. Do not proceed to fit calibration slices.
2. Investigate upstream data gatherers (`priors_nba`, `priors_soccer`, etc.).
3. Ensure traces are fully formed before the `engine.run` stage.
