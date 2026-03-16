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

import logging
import os
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from omega.core.models import (
    AnswerPlan,
    ExecutionMode,
    ExecutionResult,
    GatheredFact,
    OutputPackage,
    QueryUnderstanding,
    Subject,
)
from omega.core.contracts.service import analyze_game
from omega.core.contracts.schemas import GameAnalysisRequest, OddsInput
from omega.research.agent.intent import understand, parse_heuristic
from omega.research.agent.strategist import build_answer_plan
from omega.research.agent.planner import build_gather_list
from omega.research.agent.fact_gatherer import (
    gather_facts,
    compute_aggregate_quality,
    critical_inputs_filled,
    important_inputs_filled,
    build_data_completeness,
)
from omega.research.agent.quality_gate import apply_quality_gate
from omega.research.llm.client import LLMClient

logger = logging.getLogger("omega.research.orchestrator")


class OrchestratorConfig:
    """Configuration for the orchestrator."""

    def __init__(
        self,
        llm_provider: str = "anthropic",
        llm_model: str = "claude-sonnet-4-20250514",
        llm_api_key: Optional[str] = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key or os.getenv("ANTHROPIC_API_KEY", "")


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
    ) -> Dict[str, Any]:
        """Process a user query end-to-end. Returns structured response dict."""
        def emit(stage: str, data: Any = None) -> None:
            if progress_callback:
                progress_callback(stage, data)

        try:
            # 1. Intent understanding
            emit("understanding", {"message": "Analyzing your question..."})
            understanding = understand(prompt, self.llm)
            logger.info(
                "Intent: subjects=%s league=%s goal=%s betting=%s",
                [s.value for s in understanding.subjects],
                understanding.league,
                understanding.goal.value,
                understanding.wants_betting_advice,
            )

            # 2. Answer strategy
            emit("strategy", {"message": "Planning analysis approach..."})
            plan = build_answer_plan(understanding)

            if plan.clarification_needed:
                return self._clarification_response(
                    plan.clarification_question or "Could you be more specific?",
                    understanding,
                )

            # 3. Requirement planning
            emit("planning", {"message": "Determining data requirements..."})
            slots = build_gather_list(understanding, plan)

            # 4. Fact gathering
            emit("gathering", {
                "message": f"Gathering data ({len(slots)} requirements)...",
                "slot_count": len(slots),
            })
            facts = gather_facts(slots)

            # 5. Quality gate
            emit("quality_gate", {"message": "Evaluating data quality..."})
            revised_plan = apply_quality_gate(plan, facts)

            # 6. Execution
            emit("executing", {"message": "Running analysis..."})
            execution = self._execute(understanding, revised_plan, facts)

            # 7. Response composition
            emit("composing", {"message": "Composing response..."})
            response = self._compose_response(
                understanding, revised_plan, facts, execution,
            )

            emit("done", None)
            return response

        except Exception as e:
            logger.error("Orchestrator error: %s", e, exc_info=True)
            return self._error_response(str(e))

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
    ) -> ExecutionResult:
        """Execute the analysis plan against gathered data."""
        mode = plan.execution_modes[0] if plan.execution_modes else ExecutionMode.RESEARCH
        quality = compute_aggregate_quality(facts)
        completeness = build_data_completeness(facts)

        if mode == ExecutionMode.NATIVE_SIM:
            return self._execute_simulation(understanding, facts, quality, completeness)
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

        odds_input = None
        if odds_data:
            odds_input = OddsInput(
                moneyline_home=odds_data.get("moneyline_home"),
                moneyline_away=odds_data.get("moneyline_away"),
                spread_home=odds_data.get("spread_home"),
                over_under=odds_data.get("over_under"),
            )

        try:
            req = GameAnalysisRequest(
                home_team=home_team,
                away_team=away_team,
                league=league,
                odds=odds_input,
                home_context=home_ctx or None,
                away_context=away_ctx or None,
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
    # Response composition
    # ------------------------------------------------------------------

    def _compose_response(
        self,
        understanding: QueryUnderstanding,
        plan: AnswerPlan,
        facts: List[GatheredFact],
        execution: ExecutionResult,
    ) -> Dict[str, Any]:
        """Build structured response from execution results."""
        sections: Dict[str, Any] = {}
        narrative_parts: List[str] = []

        for pkg in plan.output_packages:
            if pkg == OutputPackage.BET_CARD and execution.best_bet:
                sections["bet_card"] = {
                    "edges": execution.edges,
                    "best_bet": execution.best_bet,
                    "data_completeness": execution.data_completeness,
                }

            elif pkg == OutputPackage.GAME_BREAKDOWN and execution.simulation:
                sections["game_breakdown"] = {
                    "simulation": execution.simulation,
                    "key_factors": self._extract_key_factors(facts),
                }

            elif pkg == OutputPackage.KEY_FACTORS:
                sections["key_factors"] = self._extract_key_factors(facts)

            elif pkg == OutputPackage.RESEARCH_REPORT:
                sections["research"] = {
                    "facts": [
                        {
                            "key": f.slot.key,
                            "data": f.result.data if f.result else None,
                            "source": f.result.source if f.result else None,
                            "quality": f.quality_score,
                        }
                        for f in facts if f.filled
                    ],
                }

            elif pkg == OutputPackage.PLAIN_EXPLANATION:
                narrative_parts.append(
                    f"Analysis of {understanding.raw_prompt}"
                )

        # Build text summary
        text = self._build_text_summary(understanding, execution, narrative_parts)

        return {
            "type": "answer",
            "text": text,
            "narrative": text,
            "sections": sections,
            "simulation": execution.simulation,
            "edges": execution.edges,
            "best_bet": execution.best_bet,
            "metadata": {
                "execution_mode": execution.mode.value,
                "data_quality": execution.data_quality_score,
                "sources_used": len([f for f in facts if f.filled]),
                "total_slots": len(facts),
                "league": understanding.league,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

    def _build_text_summary(
        self,
        understanding: QueryUnderstanding,
        execution: ExecutionResult,
        narrative_parts: List[str],
    ) -> str:
        """Build a human-readable text summary."""
        parts: List[str] = []

        if execution.simulation:
            sim = execution.simulation
            home = sim.get("home_team", "Home")
            away = sim.get("away_team", "Away")
            parts.append(
                f"**{away} @ {home}** — "
                f"Model: {home} {sim.get('home_win_prob', '?')}% / "
                f"{away} {sim.get('away_win_prob', '?')}%"
            )
            if sim.get("predicted_spread"):
                parts.append(
                    f"Projected spread: {sim['predicted_spread']:+.1f} | "
                    f"Total: {sim.get('predicted_total', '?')}"
                )

        if execution.edges:
            best = execution.edges[0] if execution.edges else None
            if best:
                parts.append(
                    f"Best edge: {best.get('side', '?')} "
                    f"({best.get('edge_pct', 0):+.1f}% edge, "
                    f"{best.get('confidence_tier', '?')} confidence)"
                )

        if not parts:
            if narrative_parts:
                parts.extend(narrative_parts)
            else:
                parts.append(
                    f"Analysis complete for: {understanding.raw_prompt}"
                )

        return "\n\n".join(parts)

    def _extract_key_factors(self, facts: List[GatheredFact]) -> List[Dict[str, Any]]:
        """Extract key factors from gathered facts."""
        factors = []
        for fact in facts:
            if fact.filled and fact.result:
                factors.append({
                    "key": fact.slot.key,
                    "importance": fact.slot.importance.value,
                    "data": fact.result.data,
                    "source": fact.result.source,
                    "quality": fact.quality_score,
                })
        return factors

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
