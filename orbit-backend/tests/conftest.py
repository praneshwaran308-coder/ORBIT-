from __future__ import annotations

import io
from typing import Any, Iterable

import pytest

from app.config import Settings, get_settings
from app.core.files import FileContext


class FakeLLMClient:
    """Deterministic stand-in for FreebuffLLMClient (async + sync surfaces)."""

    name = "fake"

    def __init__(self) -> None:
        self.api_key = "test-key"
        self.base_url = "http://fake.local/v1"
        self.calls: list[list[dict[str, str]]] = []

    def _record(self, messages: Iterable[Any]) -> None:
        self.calls.append([m.as_dict() if hasattr(m, "as_dict") else dict(m) for m in messages])

    def _respond(self):
        from app.llm.client import LLMResponse

        return LLMResponse(content="FAKE-REPLY", model="fake-model", latency_seconds=0.001)

    async def chat(self, messages: Iterable[Any], **kwargs: Any):
        self._record(messages)
        return self._respond()

    def chat_sync(self, messages: Iterable[Any], **kwargs: Any):
        self._record(messages)
        return self._respond()

    async def aclose(self) -> None:
        return None


@pytest.fixture()
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture()
def isolated_settings(monkeypatch):
    """Point settings at a scratch dir and strip provider env."""
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="orbit-test-"))
    monkeypatch.setenv("FREEBUFF_API_KEY", "test-key")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1048576")
    monkeypatch.setattr("app.config.DATA_DIR", tmp)
    monkeypatch.setattr("app.config.get_settings.cache_clear", lambda: None)
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def make_csv_context(rows: int = 30, filename: str = "sample.csv") -> FileContext:
    """Deterministic CSV: numeric x,y,z correlated with y = 2x + noise-free."""
    buf = io.StringIO()
    buf.write("x,y,z,category\n")
    for i in range(rows):
        x = i * 1.0
        buf.write(f"{x},{2 * x + (i % 3)},{10 - x % 10},'cat{i % 2}'\n")
    raw = buf.getvalue().encode("utf-8")
    return FileContext.from_upload(filename, raw)
