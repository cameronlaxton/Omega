"""
LLM-powered web search -- the primary data acquisition path for slots
that direct APIs cannot serve.

Uses Perplexity Sonar as primary backend (structured JSON output),
falling back to Anthropic web_search tool if Perplexity is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from omega.core.models import GatherSlot
from omega.evidence.models import SearchResult

logger = logging.getLogger("omega.data.acquisition.search")

# Sports-focused domains to prioritize
_SPORTS_DOMAINS = [
    "espn.com",
    "basketball-reference.com",
    "baseball-reference.com",
    "pro-football-reference.com",
    "hockey-reference.com",
    "covers.com",
    "vegasinsider.com",
    "actionnetwork.com",
    "oddshark.com",
    "rotowire.com",
    "cbssports.com",
    "nba.com",
    "nfl.com",
    "mlb.com",
    "nhl.com",
]


def search_for_slot(slot: GatherSlot, queries: Optional[List[str]] = None) -> List[SearchResult]:
    """Execute web searches for a gather slot.

    Args:
        slot: The gather slot that needs data.
        queries: Pre-planned search queries. If None, auto-generates them.

    Returns:
        List of SearchResult objects.
    """
    if queries is None:
        queries = _auto_generate_queries(slot)

    if not queries:
        return []

    results: List[SearchResult] = []
    for query in queries:
        try:
            batch = _execute_search(query, slot)
            results.extend(batch)
        except Exception as exc:
            logger.debug("Search failed for query '%s': %s", query, exc)

    # Deduplicate by URL
    seen_urls: set = set()
    unique: List[SearchResult] = []
    for r in results:
        if r.url not in seen_urls:
            seen_urls.add(r.url)
            unique.append(r)

    logger.debug("Web search returned %d unique results for slot %s", len(unique), slot.key)
    return unique


def _auto_generate_queries(slot: GatherSlot) -> List[str]:
    """Generate search queries from slot metadata."""
    entity = slot.entity
    league = slot.league.upper() if slot.league else ""
    data_type = slot.data_type

    queries = []
    if data_type == "odds":
        queries.append(f"{entity} {league} betting odds spread moneyline today")
    elif data_type == "team_stat":
        queries.append(f"{entity} {league} team statistics current season")
    elif data_type == "player_stat":
        queries.append(f"{entity} {league} player stats current season")
    elif data_type == "player_game_log":
        queries.append(f"{entity} {league} recent game log last 10 games")
    elif data_type == "injury":
        queries.append(f"{entity} {league} injury report today")
    elif data_type == "schedule":
        queries.append(f"{entity} {league} schedule today upcoming games")
    else:
        queries.append(f"{entity} {league} {data_type}")

    return queries


def _execute_search(query: str, slot: Optional[GatherSlot] = None) -> List[SearchResult]:
    """Execute a single search query via the configured backend.

    Priority:
    1. Perplexity Sonar structured (returns JSON matching extractor schema)
    2. Anthropic web_search tool (fallback, returns prose)
    """
    perplexity_key = os.environ.get("PERPLEXITY_API_KEY")
    if perplexity_key and slot is not None:
        results = _search_perplexity_structured(query, perplexity_key, slot)
        if results:
            return results

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        results = _search_anthropic(query, anthropic_key)
        if results:
            return results

    logger.warning("No search backend configured (set PERPLEXITY_API_KEY or ANTHROPIC_API_KEY)")
    return []


# ---------------------------------------------------------------------------
# Perplexity Sonar -- structured JSON search (primary)
# ---------------------------------------------------------------------------

_TYPE_INSTRUCTIONS: Dict[str, str] = {
    "odds": (
        "Search for current betting odds for {entity} in {league}. "
        "Find moneyline, point spread, and over/under totals from major sportsbooks. "
        "Use American odds format (e.g., -110, +150). "
        "Spreads should be numeric (e.g., -3.5, +7). "
        "Totals should be the over/under number (e.g., 224.5)."
    ),
    "team_stat": (
        "Search for current {league} season statistics for {entity}. "
        "Find offensive rating, defensive rating, pace, shooting percentages, "
        "rebounds, assists, turnovers per game, and win-loss record."
    ),
    "player_stat": (
        "Search for current {league} season statistics for {entity}. "
        "Find points, rebounds, assists, steals, blocks per game, "
        "shooting percentages, and games played this season."
    ),
    "player_game_log": (
        "Search for recent game logs for {entity} in {league}. "
        "Find the last 5-10 games with points, rebounds, assists, and minutes."
    ),
    "injury": (
        "Search for the current {league} injury report for {entity}. "
        "Find all injured or questionable players, their injury type, "
        "and their status (out, doubtful, questionable, probable, day-to-day)."
    ),
    "schedule": (
        "Search for today's {league} schedule involving {entity}. "
        "Find the opponent, game time, venue, and home/away designation."
    ),
}


def _build_perplexity_prompt(slot: GatherSlot) -> str:
    """Build a system prompt for Perplexity structured search."""
    data_type = slot.data_type
    entity = slot.entity
    league = slot.league.upper() if slot.league else ""

    template = _TYPE_INSTRUCTIONS.get(data_type, "Search for current {data_type} data for {entity} in {league}.")
    instruction = template.format(entity=entity, league=league, data_type=data_type)

    return (
        "You are a sports data retrieval agent. Your ONLY job is to search the web "
        "and return structured data.\n\n"
        f"{instruction}\n\n"
        "Return ONLY a valid JSON object with relevant fields.\n\n"
        "RULES:\n"
        "- Return ONLY the JSON object, no explanation or markdown\n"
        "- Use null for any field you cannot find\n"
        "- Numbers must be numeric (not strings)\n"
        "- Use the most recent data available\n"
        "- If multiple sources disagree, use the consensus or most reputable source"
    )


def _search_perplexity_structured(
    query: str, api_key: str, slot: GatherSlot
) -> List[SearchResult]:
    """Search using Perplexity Sonar and request structured JSON output."""
    import httpx

    system_prompt = _build_perplexity_prompt(slot)

    try:
        resp = httpx.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                "temperature": 0.0,
            },
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        citations = data.get("citations", [])

        if not content:
            logger.warning("Perplexity returned empty content for: %s", query[:60])
            return []

        parsed_json = _try_parse_json(content)

        results: List[SearchResult] = []

        if parsed_json is not None:
            logger.info(
                "Perplexity structured search succeeded for slot %s (%s): %d fields",
                slot.key, slot.data_type, len(parsed_json),
            )
            results.append(SearchResult(
                url="perplexity://structured",
                title=f"Perplexity Structured: {slot.data_type} for {slot.entity}",
                snippet=json.dumps(parsed_json),
                domain="perplexity.structured",
            ))
        else:
            logger.info("Perplexity returned prose for slot %s, usable by extractors", slot.key)
            results.append(SearchResult(
                url="perplexity://search",
                title=f"Perplexity: {query[:80]}",
                snippet=content,
                domain="perplexity.ai",
            ))

        for url in citations:
            domain = _extract_domain(url)
            results.append(SearchResult(
                url=url,
                title=f"Source: {domain}",
                snippet="",
                domain=domain,
            ))

        return results

    except Exception as exc:
        logger.warning("Perplexity structured search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Anthropic web_search tool (fallback)
# ---------------------------------------------------------------------------

def _search_anthropic(query: str, api_key: str) -> List[SearchResult]:
    """Search using Anthropic's server-side web_search tool."""
    import anthropic

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=(
                "You are a sports data research assistant. "
                "Return factual, current sports statistics, odds, and betting lines. "
                "Include specific numbers (spreads, moneylines, totals, stats). "
                "Be precise and data-focused."
            ),
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5,
                },
            ],
            messages=[{"role": "user", "content": query}],
        )

        results: List[SearchResult] = []

        for block in response.content:
            if block.type == "text":
                citations = getattr(block, "citations", None) or []
                cited_urls: List[str] = []
                for cite in citations:
                    url = getattr(cite, "url", None)
                    if url:
                        cited_urls.append(url)

                if block.text.strip():
                    results.append(SearchResult(
                        url="anthropic://web_search",
                        title=f"Web Search: {query[:80]}",
                        snippet=block.text,
                        domain="anthropic.web_search",
                    ))

                for url in cited_urls:
                    domain = _extract_domain(url)
                    results.append(SearchResult(
                        url=url,
                        title=f"Source: {domain}",
                        snippet="",
                        domain=domain,
                    ))

            elif block.type == "web_search_tool_result":
                search_results = getattr(block, "search_results", [])
                for sr in search_results:
                    url = getattr(sr, "url", "")
                    title = getattr(sr, "title", "")
                    snippet = getattr(sr, "page_snippet", "") or getattr(sr, "snippet", "")
                    domain = _extract_domain(url) if url else ""
                    if url:
                        results.append(SearchResult(
                            url=url,
                            title=title,
                            snippet=snippet,
                            domain=domain,
                        ))

        logger.info("Anthropic web search returned %d results for: %s", len(results), query[:60])
        return results

    except Exception as exc:
        logger.warning("Anthropic web search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON from Perplexity response, handling markdown code blocks."""
    text = text.strip()

    # Strip markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the response
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


def _extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Collector-protocol wrapper
# ---------------------------------------------------------------------------

class FallbackSearchCollector:
    """Last-resort web search collector.

    Serves *all* evidence types but with an explicit quality discount:
    - ``method`` is always ``"llm_extraction"``
    - ``trust_tier`` is capped at 3 (never treated as tier-1/2)
    - ``confidence`` is capped at 0.75

    Implements the :class:`~omega.evidence.collectors.base.Collector` protocol.
    """

    _ALL_TYPES = {
        "schedule", "odds", "team_stat", "player_stat",
        "player_game_log", "injury", "environment", "news_signal",
    }

    @property
    def name(self) -> str:
        return "fallback_search"

    @property
    def evidence_types(self) -> set[str]:
        return self._ALL_TYPES

    @property
    def supported_leagues(self) -> set[str]:
        # Web search can attempt any league
        return {
            "NBA", "NFL", "MLB", "NHL", "NCAAB", "NCAAF", "WNBA",
            "MLS", "EPL", "UFC", "ATP", "PGA", "CS2",
        }

    @property
    def trust_tier(self) -> int:
        return 3

    def collect(
        self,
        entity: str,
        league: str,
        data_type: str,
    ):
        """Run web search for the given evidence request.

        Returns a :class:`CollectorResult` or ``None``.
        """
        from omega.evidence.collectors.base import CollectorResult

        try:
            slot = GatherSlot(
                key=f"search_{data_type}_{entity}",
                data_type=data_type,
                entity=entity,
                league=league,
            )
            results = search_for_slot(slot)
            if not results:
                return None

            # Combine all snippets into a single data dict
            structured_data: Dict[str, Any] = {}
            raw_snippets: list[str] = []

            for sr in results:
                if sr.domain == "perplexity.structured":
                    try:
                        parsed = json.loads(sr.snippet)
                        if isinstance(parsed, dict):
                            structured_data.update(parsed)
                    except (ValueError, TypeError):
                        pass
                elif sr.snippet:
                    raw_snippets.append(sr.snippet)

            if not structured_data and not raw_snippets:
                return None

            data: Dict[str, Any] = {}
            if structured_data:
                data.update(structured_data)
            if raw_snippets:
                data["_raw_text"] = "\n\n".join(raw_snippets)

            # Count surviving numeric values; cap confidence if too few
            numeric_count = sum(
                1 for v in data.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            )
            confidence = min(0.75, 0.70)
            if numeric_count < 2:
                confidence = min(confidence, 0.30)

            return CollectorResult(
                data=data,
                source="web_search",
                method="llm_extraction",
                trust_tier=max(self.trust_tier, 3),
                confidence=confidence,
                entity_matched=entity,
            )

        except Exception as exc:
            logger.debug("FallbackSearchCollector.collect failed: %s", exc)
            return None
