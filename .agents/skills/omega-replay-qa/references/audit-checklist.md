# Omega Replay QA Checklist

## Replay Failure Modes

- Replay silently calls live evidence providers.
- Replay seed derives from wall-clock date instead of historical decision date.
- Replay fixtures include post-outcome information.
- Replay metrics are confused with quant benchmark metrics.
- Replay mutates frozen evidence.

## Trace Failure Modes

- Trace persistence depends on request/response wrapper objects.
- Trace schema versions are missing.
- Trace queries cannot filter by league, time, outcome state, or execution mode.
- Outcome attachment mutates source records instead of using `TraceStore`.

## Calibration Failure Modes

- Backtest and production use different calibration policy.
- Preview tools write or promote profiles.
- Dataset hash, sample size, or league is missing from candidate metadata.

## Useful Checks

```powershell
python -m pytest tests/mcp tests/trace/test_trace_store.py tests/core/test_calibration_profiles.py -q
```
