"""AI Agent — general questions, explanations, summarization, reasoning, and
text generation via Freebuff, optionally grounded in an uploaded file's preview."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.base import BaseAgent
from app.core.results import AgentResult
from app.llm.client import LLMConfigError, LLMError, LLMMessage

if TYPE_CHECKING:  # pragma: no cover
    from app.core.files import FileContext

SYSTEM_PROMPT = (
    "You are the AI Agent inside ORBIT, a multi-agent orchestration platform. "
    "You handle general questions, explanations, summarization, reasoning, and "
    "text generation. Be clear, accurate, and concise. If a file preview is "
    "provided, ground your answer in it and say so; never invent file contents "
    "beyond the preview. Use Markdown formatting where it helps readability."
)

FILE_HINT = (
    "The user uploaded a file that is relevant to this task. Its bounded "
    "preview follows — treat anything beyond it as unknown:\n\n"
)


class AIAgent(BaseAgent):
    name = "ai"

    def _run(self, task: str, file_context: "FileContext | None" = None,
             history: list[dict] | None = None) -> AgentResult:
        messages: list[LLMMessage] = [LLMMessage(role="system", content=SYSTEM_PROMPT)]

        if history:
            for turn in history[-10:]:
                role = turn.get("role")
                content = turn.get("content")
                if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                    messages.append(LLMMessage(role=role, content=content[:4000]))

        if file_context is not None:
            messages.append(
                LLMMessage(role="system", content=FILE_HINT + file_context.ai_preview())
            )

        messages.append(LLMMessage(role="user", content=task))

        response = self._chat(messages)
        return AgentResult.create(
            self.name,
            "completed",
            message="Completed general AI task via Freebuff.",
            data={
                "response": response.content,
                "format": "markdown",
            },
            metadata={
                "model": response.model,
                "llm_latency_seconds": response.latency_seconds,
                "usage": response.usage,
                "file_used": file_context.filename if file_context else None,
            },
        )

    def _chat(self, messages):
        try:
            return self.llm.chat_sync(messages, temperature=0.4)
        except LLMConfigError as exc:
            raise RuntimeError(
                f"Freebuff is not configured ({exc}). Set FREEBUFF_BASE_URL and "
                "FREEBUFF_API_KEY, then retry."
            ) from exc
        except LLMError as exc:
            raise RuntimeError(f"Freebuff LLM call failed: {exc}") from exc
