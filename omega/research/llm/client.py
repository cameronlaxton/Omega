"""
Provider-agnostic LLM client for the agent pipeline.

Supports multiple LLM providers (Anthropic, OpenAI) via an abstract
``LLMProvider`` base class. The ``LLMClient`` facade picks the right
provider based on ``AgentConfig.llm_provider`` and delegates all calls.

Switching providers is a config change, not a code change.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger("omega.agent.llm")


# ---------------------------------------------------------------------------
# Abstract provider interface
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Base class for LLM provider implementations."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider is configured and reachable."""

    @abstractmethod
    def call_with_tools(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send a message with tool definitions and return the parsed tool-use block.

        Returns the ``input`` dict from the first tool-use content block,
        or ``None`` on failure.
        """

    @abstractmethod
    def stream_message(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Yield text chunks from a streaming completion."""

    @abstractmethod
    def generate_text(
        self,
        system: str,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        """Simple text generation. Returns the full text or None on failure."""


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider using the official SDK."""

    def __init__(self, api_key: str, default_model: str) -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def is_available(self) -> bool:
        return bool(self._api_key)

    def call_with_tools(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            client = self._get_client()
            response = client.messages.create(
                model=model or self._default_model,
                max_tokens=1024,
                system=system,
                messages=messages,
                tools=tools,
            )
            for block in response.content:
                if block.type == "tool_use":
                    return block.input
            return None
        except Exception:
            logger.warning("Anthropic tool call failed", exc_info=True)
            return None

    def stream_message(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Generator[str, None, None]:
        try:
            client = self._get_client()
            with client.messages.stream(
                model=model or self._default_model,
                max_tokens=2048,
                system=system,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception:
            logger.warning("Anthropic streaming failed", exc_info=True)

    def generate_text(
        self,
        system: str,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        try:
            client = self._get_client()
            response = client.messages.create(
                model=model or self._default_model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text if response.content else None
        except Exception:
            logger.warning("Anthropic text generation failed", exc_info=True)
            return None


# ---------------------------------------------------------------------------
# OpenAI provider (stub — ready to implement)
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider stub.

    The interface is ready; fill in when you need OpenAI support.
    """

    def __init__(self, api_key: str, default_model: str) -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def is_available(self) -> bool:
        return bool(self._api_key)

    def call_with_tools(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            client = self._get_client()
            # Convert Anthropic-style tools to OpenAI function-calling format
            functions = []
            for tool in tools:
                functions.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["input_schema"],
                    },
                })

            # Prepend system message to messages list
            oai_messages = [{"role": "system", "content": system}]
            oai_messages.extend(messages)

            response = client.chat.completions.create(
                model=model or self._default_model,
                messages=oai_messages,
                tools=functions,
                tool_choice="auto",
            )

            choice = response.choices[0]
            if choice.message.tool_calls:
                call = choice.message.tool_calls[0]
                return json.loads(call.function.arguments)
            return None
        except Exception:
            logger.warning("OpenAI tool call failed", exc_info=True)
            return None

    def stream_message(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Generator[str, None, None]:
        try:
            client = self._get_client()
            oai_messages = [{"role": "system", "content": system}]
            oai_messages.extend(messages)

            stream = client.chat.completions.create(
                model=model or self._default_model,
                messages=oai_messages,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception:
            logger.warning("OpenAI streaming failed", exc_info=True)

    def generate_text(
        self,
        system: str,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=model or self._default_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception:
            logger.warning("OpenAI text generation failed", exc_info=True)
            return None


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDERS: Dict[str, type] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

class LLMClient:
    """Provider-agnostic LLM client.

    Instantiates the correct provider based on the provider name and
    delegates all calls. Lazy-initializes on first use so the system
    starts fine without an API key.

    Usage::

        client = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-...")
        if client.is_available():
            result = client.call_with_tools(system, messages, tools)
    """

    def __init__(self, provider: str, model: str, api_key: str) -> None:
        self._provider_name = provider
        self._model = model
        self._api_key = api_key
        self._provider: Optional[LLMProvider] = None

    def _get_provider(self) -> LLMProvider:
        if self._provider is None:
            cls = _PROVIDERS.get(self._provider_name)
            if cls is None:
                raise ValueError(
                    f"Unknown LLM provider: {self._provider_name!r}. "
                    f"Available: {list(_PROVIDERS.keys())}"
                )
            self._provider = cls(api_key=self._api_key, default_model=self._model)
        return self._provider

    def is_available(self) -> bool:
        """Check if the provider is configured (has an API key)."""
        try:
            return self._get_provider().is_available()
        except Exception:
            return False

    def call_with_tools(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Call LLM with tool definitions. Returns tool input dict or None."""
        try:
            return self._get_provider().call_with_tools(system, messages, tools, model)
        except Exception:
            logger.warning("LLMClient.call_with_tools failed", exc_info=True)
            return None

    def stream_message(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream text chunks from the LLM."""
        try:
            yield from self._get_provider().stream_message(system, messages, model)
        except Exception:
            logger.warning("LLMClient.stream_message failed", exc_info=True)

    def generate_text(
        self,
        system: str,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        """Generate a complete text response. Returns text or None."""
        try:
            return self._get_provider().generate_text(system, prompt, model, max_tokens)
        except Exception:
            logger.warning("LLMClient.generate_text failed", exc_info=True)
            return None
