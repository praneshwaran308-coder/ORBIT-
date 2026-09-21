"""Deterministic routing rules for the Orchestrator.

Priority order matters: research > ML > data > AI. The first matching rule
wins, so a task like "train a classifier on this CSV and chart the accuracy"
routes to the ML Agent (training is the harder, more specific intent).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.results import AGENT_LABELS

SUPPORTED_AGENTS = ("ai", "data", "ml", "research")


@dataclass(frozen=True)
class RoutingRule:
    agent: str
    keywords: tuple[str, ...] = ()
    all_keywords: tuple[str, ...] = ()
    description: str = ""


RULES: tuple[RoutingRule, ...] = (
    RoutingRule(
        agent="research",
        keywords=(
            "research", "current", "latest", "news", "today", "recent",
            "sources", "cite", "citation", "who won", "stock price",
            "weather", "release date", "published", "2024", "2025", "2026",
        ),
        description="Research / current-information tasks -> Research Agent",
    ),
    RoutingRule(
        agent="ml",
        keywords=(
            "train", "training", "model", "predict", "prediction", "forecast",
            "classify", "classification", "regression", "machine learning",
            "ml ", "accuracy", "f1", "confusion matrix", "feature", "label",
            "overfitting", "cross-validation",
        ),
        description="Machine-learning tasks -> ML Agent",
    ),
    RoutingRule(
        agent="data",
        keywords=(
            "csv", "spreadsheet", "column", "columns", "row", "rows",
            "mean", "median", "average", "std", "standard deviation",
            "correlation", "correlations", "missing values", "duplicates",
            "statistics", "stats", "dataset", "summarize the data",
            "analyze the data", "data analysis", "distribution", "outliers",
        ),
        description="CSV / data-analysis tasks -> Data Agent",
    ),
)


@dataclass
class RoutingDecision:
    agent: str
    reason: str
    matched_keywords: list[str] = field(default_factory=list)

    @property
    def agent_label(self) -> str:
        return AGENT_LABELS.get(self.agent, self.agent)


def route_task(task: str, *, has_file: bool = False,
               agent_override: str | None = None) -> RoutingDecision:
    """Deterministically pick the agent for a task."""
    if agent_override:
        override = agent_override.strip().lower()
        if override in SUPPORTED_AGENTS:
            return RoutingDecision(
                agent=override,
                reason="Explicit agent override supplied with the request.",
            )
        raise ValueError(
            f"Unknown agent '{agent_override}'. Supported: {', '.join(SUPPORTED_AGENTS)}."
        )

    text = " ".join(task.lower().split())

    for rule in RULES:
        matched = [k for k in rule.keywords if k in text]
        if not matched and rule.all_keywords:
            matched = [" ".join(combo) for combo in rule.all_keywords
                       if all(k in text for k in combo)]
        if matched:
            return RoutingDecision(
                agent=rule.agent,
                reason=rule.description,
                matched_keywords=matched,
            )

    if has_file:
        file_reference = (
            "file", "csv", "spreadsheet", "dataset", "upload", "uploaded",
            "attachment", "column", "columns", "row", "rows", "data",
            "analyze", "statistics", "stats",
        )
        if any(k in text for k in file_reference) or len(text.split()) <= 6:
            return RoutingDecision(
                agent="data",
                reason="A file was uploaded and the task appears to reference it; "
                       "defaulting to the Data Agent.",
            )
        return RoutingDecision(
            agent="ai",
            reason="General AI task -> AI Agent (uploaded file attached as context).",
        )

    return RoutingDecision(agent="ai", reason="General AI task -> AI Agent (default).")
