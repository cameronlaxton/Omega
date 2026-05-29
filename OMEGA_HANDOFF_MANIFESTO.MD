# OMEGA ARCHITECTURAL HANDOFF: PHASE 6.1 (POST-HARDENING)

## 1. System State & Recently Landed Architecture
The core infrastructure has been hardened to remove network-mount friction; stabilize MCP bridge connections; and enforce strict data typing. The following features are live and must not be rebuilt:

*   **FUSE Workarounds:** `scripts/cowork_preflight.py` automatically relocates `.git/index.lock` collisions via `os.rename`.
*   **Odds Resolver:** `omega_resolve_odds` is successfully decoupled into `omega/integrations/odds_resolver.py`. Import paths are pinned to prevent OneDrive directory bleed.
*   **MCP Safeguards:** Tools now enforce Pydantic enum validation for `EvidenceSignal`; exploratory iterations are capped (`fast_score`: 300, `markov_state`: 100, `props`: 500) to prevent 180s timeouts; replay bundles return `result` payloads.
*   **Data Gates:** MLB Pitcher props (K/outs) strictly require `sample_size >= 5`, plus explicit numeric mean/std. Missing data forces a `skipped` status and strips calibration eligibility.
*   **Linux Native Path:** A `.devcontainer` configuration exists for future migration.
*   **Git Protocol:** `git cat-file -p HEAD:path` is the mandatory standard for source reads.

## 2. Immediate Residual Blockers (Action Required)
Before initiating new feature development, the following environment blockers must be cleared:

1.  **Parity Failure:** `cowork_preflight.py --direct-only` is currently failing due to uncommitted modifications in `omega/core/contracts/service.py`. 
    *   *Action:* Review and commit the current working tree to clear the parity guard.
2.  **Test Suite Obstruction:** A full `python -m pytest -q` run is blocked by an untracked, broken test file (`tests/scripts/test_backfill_trace_quality.py`) attempting to import a missing module. 
    *   *Action:* Delete or repair this untracked test to restore the CI pipeline.

## 3. Horizon 2 Strategic Directives
With the foundation secured, development must pivot to leveraging existing resources safely and building zero-cost testing pipelines.

### 3.1 Odds API Maximization (Stop Web Scraping)
The `omega_resolve_odds` MCP tool is now stable. The system has a 30,000-request quota via The Odds API that is currently underutilized.
*   **Directive:** Cease utilizing web search functions to scrape ESPN or CBS for pre-game odds.
*   **Execution:** Default all pre-decision market queries to the `omega_resolve_odds` tool. For maximum EV discovery; execute the resolver with `all_books=true` to mathematically line-shop rather than manually reading sportsbook pages.

### 3.2 Trace Replay Engine (Lean Backtesting)
Full time-series backtesting requires excessive historical tick data. "Trace Replay" leverages existing database logs for offline calibration testing.
*   **Directive:** Build `scripts/replay_historical_traces.py`.
*   **Execution:** The script must iterate through historical traces in `omega_traces.db`; strip prior EV/Kelly calculations; inject experimental calibration profiles; and re-score the trace accuracy against known actual outcomes without executing new network requests.

### 3.3 API Circuit Breakers
Automated LLM loops introduce the risk of rapid quota exhaustion through hallucinated retries.
*   **Directive:** Implement a hard limit enforcer for The Odds API.
*   **Execution:** Build a strict SQLite-based TTL cache for external requests. If the daily API request count hits a predefined threshold, automatically freeze network requests and force the system into narrative-only fallback mode.

### 3.4 Correlated Exposure Guard
The current `kelly.py` module evaluates single-bet sizing but ignores portfolio concentration risk.
*   **Directive:** Implement a risk aggregator to prevent overexposure on a single game narrative.
*   **Execution:** Build `omega/core/portfolio/risk_guard.py` to calculate joint exposure across correlated bets (e.g., Team ML + Star Player Prop) and systematically reduce Kelly fractions to enforce a maximum daily loss boundary per matchup.

## 4. Operating Constraints
*   **Git Operations:** Never execute `git show HEAD:path`; it triggers FUSE lock collisions.
*   **Output Formatting:** Generate strictly typed responses; favor tables and code blocks over conversational prose.
*   **Error Handling:** Do not attempt to mask or hallucinate missing data; immediately trigger `null_data_audit` downgrades when API or context parameters fail.