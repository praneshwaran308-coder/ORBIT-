"""Phase 1 gate — Freebuff connectivity test.

Usage (from orbit-backend/):
    ./.venv/Scripts/python.exe scripts/freebuff_connectivity_test.py

Resolves the provider in this order:
  1. FREEBUFF_* environment variables
  2. Freebuff Desktop's BYOK store (connections.json + OS keychain via Bun)

Then performs ONE minimal chat completion and prints the result.
Exit code 0 = success. The API key is never printed.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings  # noqa: E402
from app.llm.client import FreebuffLLMClient, LLMConfigError, LLMError, LLMMessage  # noqa: E402
from app.llm.freebuff_byok import resolve_freebuff_provider  # noqa: E402


def _print_resolved(resolved) -> None:
    print(f"  provider source: {resolved.source}")
    for k, v in resolved.detail.items():
        if v:
            print(f"  {k}: {v}")
    print(f"  base_url:        {resolved.base_url}")
    print(f"  model:           {resolved.model}")
    print(f"  api_key:         configured ({len(resolved.api_key)} chars, hidden)")


async def main() -> int:
    print("ORBIT Phase 1 — Freebuff connectivity test")
    print("-" * 48)

    resolved = resolve_freebuff_provider(settings=get_settings())
    if resolved is None:
        print("FAIL: no Freebuff provider available.")
        print("Tried, in order:")
        print("  1. FREEBUFF_API_KEY environment variable (not set)")
        print("  2. Settings file (.data/.env) FREEBUFF_API_KEY (not set)")
        print("  3. Freebuff BYOK store (~/.config/freebuff/byok/connections.json")
        print("     + OS keychain via Bun.secrets) — not usable")
        print("\nFix: open Freebuff Desktop and add a BYOK connection, or set")
        print("FREEBUFF_API_KEY / FREEBUFF_BASE_URL / FREEBUFF_MODEL explicitly.")
        return 2

    _print_resolved(resolved)

    client = FreebuffLLMClient()
    print(f"\nSending one minimal chat completion to {client.base_url}/chat/completions ...")
    try:
        response = await client.chat(
            [
                LLMMessage(role="system", content="You are a connectivity probe."),
                LLMMessage(role="user", content="Reply with exactly: ORBIT ONLINE"),
            ],
            max_tokens=20,
            temperature=0,
        )
    except LLMConfigError as exc:
        print(f"\nFAIL: configuration error: {exc}")
        return 2
    except LLMError as exc:
        print(f"\nFAIL: request failed: {exc}")
        return 1
    finally:
        await client.aclose()

    print(f"  reply:          {response.content.strip()[:200]!r}")
    print(f"  served by:      {response.model} (latency {response.latency_seconds}s)")
    print("\nPASS: Freebuff provider is reachable and responding.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
