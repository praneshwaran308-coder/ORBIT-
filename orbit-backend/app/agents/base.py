"""BaseAgent — the abstraction every specialized agent extends."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from app.core.results import AGENT_LABELS, AgentResult

if TYPE_CHECKING:  # pragma: no cover
    from app.core.files import FileContext
    from app.llm.client import FreebuffLLMClient


logger = logging.getLogger("orbit.agents")

INSUFFICIENT_DATA: str = "insufficient_data"


class InsufficientDataError(Exception):
    """Raised by agents when the provided data cannot support the task."""


class BaseAgent(ABC):
    """Common run flow: validation -> execute -> envelope with timing/errors."""

    name: str = "base"

    def __init__(self, llm: "FreebuffLLMClient | None" = None) -> None:
        from app.llm.client import get_llm_client

        self.llm = llm or get_llm_client()

    @property
    def label(self) -> str:
        return AGENT_LABELS.get(self.name, self.name.replace("_", " ").title())

    # -- template method -----------------------------------------------------
    def run(self, task: str, file_context: "FileContext | None" = None,
            history: list[dict] | None = None) -> AgentResult:
        started = time.monotonic()
        try:
            result = self._run(task, file_context=file_context, history=history)
        except InsufficientDataError as exc:
            result = AgentResult.create(self.name, INSUFFICIENT_DATA, str(exc))
        except Exception as exc:  # noqa: BLE001 - every failure is surfaced in the envelope
            logger.exception("Agent '%s' failed: %s", self.name, exc)
            result = AgentResult.create(self.name, "failed", f"{type(exc).__name__}: {exc}")
        result.metadata.setdefault("agent_label", self.label)
        result.metadata["duration_seconds"] = round(time.monotonic() - started, 3)
        return result

    @abstractmethod
    def _run(self, task: str, file_context: "FileContext | None" = None,
             history: list[dict] | None = None) -> AgentResult:
        """Agent-specific implementation; raise to map to failure statuses."""
