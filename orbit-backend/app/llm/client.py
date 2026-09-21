"""Freebuff LLM client.

A single, shared OpenAI-compatible client used by every agent. All provider
settings (base URL, API key, model, timeout, retries) come from configuration —
nothing is hardcoded and no OpenAI SDK is used.

Both call styles are supported:
    await client.chat(messages)        # async
    client.chat_sync(messages)         # sync (agents run in worker threads)
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import httpx

from app.config import get_settings

logger = logging.getLogger("orbit.llm")

RETRYABLE_STATUS = (429, 500, 502, 503, 504)


class LLMError(RuntimeError):
    """Raised when the LLM call fails in a way agents should surface."""


class LLMConfigError(LLMError):
    """Raised when required Freebuff configuration is missing or malformed."""


@dataclass
class LLMResponse:
    """Normalized completion result."""

    content: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0


@dataclass
class LLMMessage:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def _normalize(messages: Iterable[LLMMessage | dict[str, str]]) -> list[dict[str, str]]:
    return [m.as_dict() if isinstance(m, LLMMessage) else dict(m) for m in messages]


def _tail(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[-limit:]


class FreebuffLLMClient:
    """Client for the Freebuff provider's OpenAI-compatible endpoint.

    Provider settings resolve in this order:
      1. Explicit constructor arguments (tests / advanced use).
      2. FREEBUFF_* environment variables.
      3. Freebuff Desktop's BYOK store (connections.json + OS keychain via
         Bun.secrets) — the same integration Freebuff's orchestrator uses.
    """

    def __init__(self, *, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float | None = None,
                 max_retries: int | None = None) -> None:
        resolved = None
        if base_url is None or api_key is None or model is None:
            from app.llm.freebuff_byok import resolve_freebuff_provider

            resolved = resolve_freebuff_provider(settings=get_settings())

        from app.llm.freebuff_byok import (
            FALLBACK_BASE_URL,
            FALLBACK_MODEL,
        )

        self.base_url = (base_url or (resolved.base_url if resolved else None) or
                         FALLBACK_BASE_URL).rstrip("/")
        self.api_key = api_key or (resolved.api_key if resolved else "")
        self.model = model or (resolved.model if resolved else None) or FALLBACK_MODEL
        self.provider_source = resolved.source if resolved else "none"
        self.provider_detail = resolved.detail if resolved else {}
        s = get_settings()
        self.timeout = timeout if timeout is not None else s.freebuff_timeout_seconds
        self.max_retries = max_retries if max_retries is not None else s.freebuff_max_retries

    # -- shared helpers ------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages: Iterable[LLMMessage | dict[str, str]], *, model: str | None,
                 temperature: float | None, max_tokens: int | None) -> dict[str, Any]:
        if not self.base_url:
            raise LLMConfigError("FREEBUFF_BASE_URL is not configured.")
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": _normalize(messages),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def _handle_response(self, resp: httpx.Response, *, model: str,
                         latency: float) -> LLMResponse:
        if resp.status_code in RETRYABLE_STATUS:
            raise _Retryable(LLMError(
                f"Freebuff returned HTTP {resp.status_code}: {_tail(resp.text)}"
            ))
        if resp.status_code >= 400:
            raise LLMError(f"Freebuff returned HTTP {resp.status_code}: {_tail(resp.text)}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise _Retryable(LLMError(f"Freebuff returned invalid JSON: {exc}")) from exc
        return self._parse(body, model=model, latency=latency)

    def _parse(self, body: dict[str, Any], *, model: str, latency: float) -> LLMResponse:
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Malformed Freebuff response: {exc}") from exc
        if not isinstance(content, str):
            raise LLMError("Freebuff response content was not a string.")
        return LLMResponse(
            content=content,
            model=body.get("model", model),
            finish_reason=choice.get("finish_reason"),
            usage=body.get("usage") or {},
            latency_seconds=round(latency, 3),
        )

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(2 ** attempt, 8) + random.uniform(0, 0.25)

    # -- async API -------------------------------------------------------------
    async def chat(self, messages: Iterable[LLMMessage | dict[str, str]],
                   *, model: str | None = None, temperature: float | None = None,
                   max_tokens: int | None = None) -> LLMResponse:
        payload = self._payload(messages, model=model, temperature=temperature,
                                max_tokens=max_tokens)
        last_error: Exception | None = None
        async with httpx.AsyncClient(base_url=self.base_url, headers=self._headers(),
                                     timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                if attempt:
                    logger.warning("LLM async retry %s/%s after error: %s",
                                   attempt, self.max_retries, last_error)
                    await asyncio.sleep(self._backoff(attempt))
                started = time.monotonic()
                try:
                    resp = await client.post("/chat/completions", json=payload)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = LLMError(f"Freebuff request failed: {exc}")
                    continue
                try:
                    return self._handle_response(resp, model=payload["model"],
                                                 latency=time.monotonic() - started)
                except _Retryable as exc:
                    last_error = exc.inner
        raise last_error or LLMError("Freebuff request failed for an unknown reason.")

    # -- sync API ---------------------------------------------------------------
    def chat_sync(self, messages: Iterable[LLMMessage | dict[str, str]],
                  *, model: str | None = None, temperature: float | None = None,
                  max_tokens: int | None = None) -> LLMResponse:
        payload = self._payload(messages, model=model, temperature=temperature,
                                max_tokens=max_tokens)
        last_error: Exception | None = None
        with httpx.Client(base_url=self.base_url, headers=self._headers(),
                          timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                if attempt:
                    logger.warning("LLM sync retry %s/%s after error: %s",
                                   attempt, self.max_retries, last_error)
                    time.sleep(self._backoff(attempt))
                started = time.monotonic()
                try:
                    resp = client.post("/chat/completions", json=payload)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = LLMError(f"Freebuff request failed: {exc}")
                    continue
                try:
                    return self._handle_response(resp, model=payload["model"],
                                                 latency=time.monotonic() - started)
                except _Retryable as exc:
                    last_error = exc.inner
        raise last_error or LLMError("Freebuff request failed for an unknown reason.")

    # -- lifecycle (kept for API compatibility) ----------------------------------
    async def aclose(self) -> None:
        return None


class _Retryable(Exception):
    """Internal marker: the wrapped error is retryable."""

    def __init__(self, inner: LLMError) -> None:
        super().__init__(str(inner))
        self.inner = inner


_default_client: FreebuffLLMClient | None = None


def get_llm_client() -> FreebuffLLMClient:
    """Process-wide shared client (all agents use this)."""
    global _default_client
    if _default_client is None:
        _default_client = FreebuffLLMClient()
    return _default_client


def reset_llm_client() -> None:
    """Drop the shared client (used by tests to inject fakes)."""
    global _default_client
    _default_client = None
