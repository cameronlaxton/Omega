"""
Agent orchestrator — end-to-end query processing pipeline.

Flow:
    1. Intent understanding (heuristic + optional LLM)
    2. Answer strategy (deterministic)
    3. Requirement planning (deterministic)
    4. Fact gathering (data pipeline)
    5. Quality gate (deterministic)
    6. Execution (simulation / research)
    7. Response composition (deterministic + optional LLM narrative)

This orchestrator is stateless — session management is handled by the
API layer (omega.api.session). The orchestrator receives a prompt and
optional conversation history, and returns a structured response.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from omega.core.models import (
    AnswerPlan,
    ExecutionMode,
    ExecutionResult,
    ExecutionTrace,
    GatheredFact,
    OutputPackage,
    QueryUnderstanding,
    Subject,
)
from omega.core.contracts.service import analyze_game
from omega.core.simulation.validation import validate_sim_context
from omega.core.contracts.schemas import GameAnalysisRequest, OddsInput
from omega.reasoning.intent import understand, parse_heuristic
from omega.reasoning.router import build_answer_plan
from omega.reasoning.planner import build_gather_list
from omega.reasoning.gatherer import (
    gather_facts,
    compute_aggregate_quality,
    critical_inputs_filled,
    important_inputs_filled,
    build_data_completeness,
)
from omega.reasoning.evaluator import apply_quality_gate
from omega.reasoning.llm.client import LLMClient
from omega.synthesis.composer import compose_response

logger = logging.getLogger("omega.reasoning.orchestrator")


class OrchestratorConfig:
    """Configuration for the orchestrator."""

    def __init__(
        self,
        llm_provider: str = "anthropic",
        llm_model: str = "claude-sonnet-4-20250514",
        llm_api_key: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.strict = strict


class Orchestrator:
    """Stateless query processing pipeline."""

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.config = config or OrchestratorConfig()
        self._llm: Optional[LLMClient] = None

    @property
    def llm(self) -> Optional[LLMClient]:
        if self._llm is None and self.config.llm_api_key:
            self._llm = LLMClient(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                api_key=self.config.llm_api_key,
            )
        return self._llm

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def handle_query(
        self,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        progress_callback: Optional[Callable[[str, Any], None]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a user query end-to-end. Returns structured response dict."""
        pipeline_start = time.monotonic()
        trace = ExecutionTrace(prompt=prompt, session_id=session_id)

        def emit(stage: str, data: Any = None) -> None:
            if progress_callback:
                progress_callback(stage, data)

        try:
            # 1. Intent understanding
            emit("understanding", {"message": "Analyzing your question..."})
            t0 = time.monotonic()
            understanding = understand(prompt, self.llm)
            trace.stage_timings["understanding"] = time.monotonic() - t0
            trace.understanding = understanding.model_dump()
            trace.league = understanding.league
            logger.info(
                "Intent: subjects=%s league=%s goal=%s betting=%s",
                [s.value for s in understanding.subjects],
                understanding.league,
                understanding.goal.value,
                understanding.wants_betting_advice,
            )

            # 2. Answer strategy
            emit("strategy", {"message": "Planning analysis approach..."})
            t0 = time.monotonic()
            plan = build_answer_plan(understanding, self.llm)
            trace.stage_timings["strategy"] = time.monotonic() - t0
            trace.answer_plan = plan.model_dump()

            if plan.clarification_needed:
                return self._clarification_response(
                    plan.clarification_question or "Could you be more specific?",
                    understanding,
                )

            # 3. Requirement planning
            emit("planning", {"message": "Determining data requirements..."})
            t0 = time.monotonic()
            slots = build_gather_list(understanding, plan, self.llm)
            trace.stage_timings["planning"] = time.monotonic() - t0
            trace.gather_slots = [s.model_dump() for s in slots]

            # 4. Fact gathering
            emit("gathering", {
                "message": f"Gathering data ({len(slots)} requirements)...",
                "slot_count": len(slots),
            })
            t0 = time.monotonic()
            facts = gather_facts(slots)
            trace.stage_timings["gathering"] = time.monotonic() - t0
            trace.gathered_facts = [f.model_dump() for f in facts]
            trace.aggregate_quality = compute_aggregate_quality(facts)
            filled_facts = [f for f in facts if f.filled]
            sources_used = list({f.result.source for f in filled_facts if f.result})
            trace.facts_summary = {
                "total_slots": len(slots),
                "filled": len(filled_facts),
                "critical_filled": critical_inputs_filled(facts),
                "sources_used": sources_used,
            }

            # 5. Quality gate
            emit("quality_gate", {"message": "Evaluating data quality..."})
            t0 = time.monotonic()
            revised_plan = apply_quality_gate(plan, facts)
            trace.stage_timings["quality_gate"] = time.monotonic() - t0
            trace.revised_plan = revised_plan.model_dump()
            trace.downgrades = revised_plan.downgrades

            # 6. Execution
            emit("executing", {"message": "Running analysis..."})
            t0 = time.monotonic()
            execution = self._execute(understanding, revised_plan, facts, trace)
            trace.stage_timings["execution"] = time.monotonic() - t0
            trace.execution_mode = execution.mode.value
            trace.execution_result = execution.model_dump()

            # Populate backtest-ready fields from execution
            if execution.simulation:
                trace.predictions = {
                    "home_win_prob": execution.simulation.get("home_win_prob"),
                    "away_win_prob": execution.simulation.get("away_win_prob"),
                    "predicted_spread": execution.simulation.get("predicted_spread"),
                    "predicted_total": execution.simulation.get("predicted_total"),
                }
            if execution.best_bet:
                trace.recommendations = [execution.best_bet]
            if execution.edges:
                trace.recommendations = execution.edges
            # Build matchup string
            home_name = away_name = ""
            for entity in understanding.entities:
                if entity.role.value == "home":
                    home_name = entity.name
                elif entity.role.value == "away":
                    away_name = entity.name
            if home_name or away_name:
                trace.matchup = f"{away_name} @ {home_name}"

            # 7. Response composition
            emit("composing", {"message": "Composing response..."})
            t0 = time.monotonic()
            response = compose_response(
                understanding, revised_plan, facts, execution,
            )
            trace.stage_timings["composition"] = time.monotonic() - t0
            trace.output_packages = [p.value for p in revised_plan.output_packages]
            trace.narrative_length = len(response.get("narrative", ""))

            # Finalize trace
            trace.total_duration_ms = (time.monotonic() - pipeline_start) * 1000
            response["trace"] = trace.model_dump(mode="json")

            emit("done", None)
            return response

        except Exception as e:
            logger.error("Orchestrator error: %s", e, exc_info=True)
            trace.error = str(e)
            trace.total_duration_ms = (time.monotonic() - pipeline_start) * 1000
            resp = self._error_response(str(e))
            resp["trace"] = trace.model_dump(mode="json")
            return resp

    # ------------------------------------------------------------------
    # Async streaming entry point (for SSE)
    # ------------------------------------------------------------------

    async def handle_query_stream(
        self,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a query and yield SSE-compatible stage events."""
        stages: List[Dict[str, Any]] = []

        def collect_progress(stage: str, data: Any) -> None:
            stages.append({"stage": stage, "data": data})

        # Run synchronous pipeline (will be made fully async in future)
        response = self.handle_query(prompt, history, progress_callback=collect_progress)

        # Yield stage updates
        for stage_info in stages:
            yield {
                "event_type": "stage_update",
                "data": stage_info,
            }

        # Yield structured data
        if response.get("simulation"):
            yield {
                "event_type": "structured_data",
                "data": {
                    "type": "simulation",
                    "payload": response["simulation"],
                },
            }

        if response.get("edges"):
            yield {
                "event_type": "structured_data",
                "data": {
                    "type": "edges",
                    "payload": response["edges"],
                },
            }

        # Yield final text
        yield {
            "event_type": "partial_text",
            "data": response.get("narrative", response.get("text", "")),
        }

        # Yield trace if present
        if response.get("trace"):
            yield {
                "event_type": "trace",
                "data": response["trace"],
            }

        # Done
        yield {
            "event_type": "done",
            "data": {
                "final_text": response.get("narrative", response.get("text", "")),
                "metadata": response.get("metadata", {}),
            },
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute(
        self,
        understanding: QueryUnderstanding,
        plan: AnswerPlan,
        facts: List[GatheredFact],
        trace: Optional[ExecutionTrace] = None,
    ) -> ExecutionResult:
        """Execute the analysis plan against gathered data."""
        mode = plan.execution_modes[0] if plan.execution_modes else ExecutionMode.RESEARCH
        quality = compute_aggregate_quality(facts)
        completeness = build_data_completeness(facts)

        if mode == ExecutionMode.NATIVE_SIM:
            return self._execute_simulation(understanding, facts, quality, completeness, trace)
        else:
            return ExecutionResult(
                mode=mode,
                simulation=None,
                edges=[],
                best_bet=None,
                research_facts=[f for f in facts if f.filled],
                data_quality_score=quality,
                data_completeness=completeness,
            )

    def _execute_simulation(
        self,
        understanding: QueryUnderstanding,
        facts: List[GatheredFact],
        quality: float,
        completeness: Dict[str, str],
        trace: Optional[ExecutionTrace] = None,
    ) -> ExecutionResult:
        """Build team contexts from facts and run simulation."""
        home_ctx: Dict[str, Any] = {}
        away_ctx: Dict[str, Any] = {}
        odds_data: Dict[str, Any] = {}

        # Extract team contexts and odds from gathered facts
        for fact in facts:
            if not fact.filled or fact.result is None:
                continue
            data = fact.result.data or {}
            key = fact.slot.key

            if "home" in key and "stat" in fact.slot.data_type:
                home_ctx.update(data)
            elif "away" in key and "stat" in fact.slot.data_type:
                away_ctx.update(data)
            elif fact.slot.data_type == "odds":
                odds_data.update(data)

        # Build request
        home_team = ""
        away_team = ""
        for entity in understanding.entities:
            if entity.role.value == "home":
                home_team = entity.name
            elif entity.role.value == "away":
                away_team = entity.name

        league = understanding.league or "NBA"

        # Validate contexts before they enter the simulation engine
        home_ctx = validate_sim_context(home_ctx, league, "home", strict=self.config.strict)
        away_ctx = validate_sim_context(away_ctx, league, "away", strict=self.config.strict)

        odds_input = None
        if odds_data:
            odds_input = OddsInput(
                moneyline_home=odds_data.get("moneyline_home"),
                moneyline_away=odds_data.get("moneyline_away"),
                spread_home=odds_data.get("spread_home"),
                over_under=odds_data.get("over_under"),
            )

        # Deterministic seed for reproducibility
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        seed = int(hashlib.sha256(f"{understanding.raw_prompt}:{today}".encode()).hexdigest(), 16) % (2**31)
        if trace:
            trace.simulation_seed = seed
            trace.odds_snapshot = odds_data or None

        try:
            req = GameAnalysisRequest(
                home_team=home_team,
                away_team=away_team,
                league=league,
                odds=odds_input,
                home_context=home_ctx or None,
                away_context=away_ctx or None,
                seed=seed,
            )
            result = analyze_game(req)
            result_dict = result.model_dump() if hasattr(result, "model_dump") else result

            return ExecutionResult(
                mode=ExecutionMode.NATIVE_SIM,
                simulation=result_dict.get("simulation"),
                edges=[e for e in (result_dict.get("edges") or [])],
                best_bet=result_dict.get("best_bet"),
                research_facts=[f for f in facts if f.filled],
                data_quality_score=quality,
                data_completeness=completeness,
            )
        except Exception as e:
            logger.warning("Simulation failed: %s", e)
            return ExecutionResult(
                mode=ExecutionMode.RESEARCH,
                simulation=None,
                edges=[],
                best_bet=None,
                research_facts=[f for f in facts if f.filled],
                data_quality_score=quality,
                data_completeness=completeness,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clarification_response(
        self, question: str, understanding: QueryUnderstanding,
    ) -> Dict[str, Any]:
        return {
            "type": "clarification",
            "text": question,
            "narrative": question,
            "metadata": {
                "execution_mode": "clarification",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

    def _error_response(self, message: str) -> Dict[str, Any]:
        return {
            "type": "error",
            "text": f"Something went wrong: {message}",
            "narrative": f"Something went wrong: {message}",
            "metadata": {
                "execution_mode": "error",
                "error": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
