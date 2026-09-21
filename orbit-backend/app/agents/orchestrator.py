"""Orchestrator — deterministic task routing to the specialized agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.ai_agent import AIAgent
from app.agents.base import BaseAgent
from app.agents.data_agent import DataAgent
from app.agents.ml_agent import MLAgent
from app.agents.research_agent import ResearchAgent
from app.core.results import AgentResult, FAILED

if TYPE_CHECKING:  # pragma: no cover
    from app.core.files import FileContext
    from app.llm.client import FreebuffLLMClient


class Orchestrator:
    """Routes tasks deterministically; knows nothing about HTTP."""

    def __init__(self, llm: "FreebuffLLMClient | None" = None,
                 agents: dict[str, BaseAgent] | None = None) -> None:
        from app.llm.client import get_llm_client

        self.llm = llm or get_llm_client()
        self.agents = agents or {
            "ai": AIAgent(llm=self.llm),
            "data": DataAgent(llm=self.llm),
            "ml": MLAgent(llm=self.llm),
            "research": ResearchAgent(llm=self.llm),
        }

    def run(self, task: str, file_context: "FileContext | None" = None,
            agent_override: str | None = None,
            history: list[dict] | None = None) -> AgentResult:
        try:
            decision = route_task(task, has_file=file_context is not None,
                                  agent_override=agent_override)
        except ValueError as exc:
            return AgentResult.create("orchestrator", FAILED, str(exc))

        agent = self.agents.get(decision.agent)
        if agent is None:  # pragma: no cover - defensive
            return AgentResult.create(
                "orchestrator", FAILED, f"No agent registered for '{decision.agent}'."
            )

        result = agent.run(task, file_context=file_context, history=history)
        result.metadata["routing"] = {
            "agent": decision.agent,
            "agent_label": decision.agent_label,
            "reason": decision.reason,
            "matched_keywords": decision.matched_keywords,
        }
        result.metadata["orchestrator"] = "deterministic-rules-v1"
        return result


# Imported late to keep the routing module free of agent imports.
from app.core.routing import route_task  # noqa: E402
