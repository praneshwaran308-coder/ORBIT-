"""Data Agent — deterministic CSV analysis with pandas/NumPy.

All numerical work happens locally. The LLM (Freebuff) only narrates the
already-computed statistics and never produces the numbers itself.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from app.agents.base import BaseAgent, InsufficientDataError
from app.core.results import AgentResult

if TYPE_CHECKING:  # pragma: no cover
    from app.core.files import FileContext

HIGH_CORRELATION_THRESHOLD = 0.75
MISSING_COLUMN_ALERT = 0.5


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars into JSON-safe primitives."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {k: _json_safe(v) for k, v in record.items()}


class DataAgent(BaseAgent):
    name = "data"

    def _run(self, task: str, file_context: "FileContext | None" = None,
             history: list[dict] | None = None) -> AgentResult:
        if file_context is None:
            return AgentResult.create(
                self.name,
                "failed",
                "The Data Agent needs a CSV file. Upload a .csv file and ask a "
                "data question about it (rows, columns, means, correlations, ...).",
            )

        try:
            df = file_context.frame()
        except Exception as exc:  # noqa: BLE001
            raise InsufficientDataError(f"Could not read the uploaded CSV: {exc}") from exc

        if df.empty:
            raise InsufficientDataError("The uploaded CSV contains no data rows.")

        report = self._analyze(df)
        narrative = self._narrate(task, report)

        return AgentResult.create(
            self.name,
            "completed",
            message=f"Analyzed {report['shape']['rows']} rows x "
                    f"{report['shape']['columns']} columns from {file_context.filename}.",
            data={"report": report, "narrative": narrative, "format": "data_report"},
            metadata={
                "filename": file_context.filename,
                "computed_locally": True,
                "llm_role": "narration_only",
            },
        )

    # -- local, deterministic computation ------------------------------------
    def _analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        numeric = df.select_dtypes(include=[np.number])
        report: dict[str, Any] = {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns": [
                {
                    "name": name,
                    "dtype": str(dtype),
                    "missing": int(df[name].isna().sum()),
                    "missing_pct": round(float(df[name].isna().mean()) * 100, 2),
                    "unique": int(df[name].nunique(dropna=True)),
                }
                for name, dtype in df.dtypes.items()
            ],
            "total_missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        }

        if numeric.empty:
            report["numeric_summary"] = {}
            report["correlations"] = []
            report["insights"] = self._insights(report, None)
            return report

        desc = numeric.describe().to_dict()
        report["numeric_summary"] = {
            col: {
                "mean": _json_safe(vals.get("mean")),
                "median": _json_safe(numeric[col].median()),
                "min": _json_safe(vals.get("min")),
                "max": _json_safe(vals.get("max")),
                "std": _json_safe(vals.get("std")),
            }
            for col, vals in desc.items()
        }

        corr = numeric.corr(numeric_only=True)
        pairs: list[dict[str, Any]] = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r = corr.loc[a, b]
                if pd.notna(r):
                    pairs.append({"a": a, "b": b, "r": _json_safe(r)})
        pairs.sort(key=lambda p: abs(p["r"]), reverse=True)
        report["correlations"] = pairs[:10]
        report["insights"] = self._insights(report, corr)

        return report

    def _insights(self, report: dict[str, Any],
                  corr: pd.DataFrame | None) -> list[str]:
        insights: list[str] = []

        if report["duplicate_rows"]:
            insights.append(
                f"{report['duplicate_rows']} duplicate row(s) detected — consider deduplicating."
            )

        for col in report["columns"]:
            if col["missing_pct"] > MISSING_COLUMN_ALERT * 100:
                insights.append(
                    f"Column '{col['name']}' is {col['missing_pct']:.0f}% missing."
                )

        if report["total_missing_cells"] == 0 and report["shape"]["rows"] > 0:
            insights.append("No missing values anywhere in the dataset.")

        numeric_cols = [c for c in report["columns"] if c["dtype"].startswith(("float", "int"))]
        if not numeric_cols:
            insights.append("No numeric columns found; statistical summaries are unavailable.")

        if corr is not None:
            for pair in report["correlations"]:
                if abs(pair["r"]) >= HIGH_CORRELATION_THRESHOLD:
                    direction = "positive" if pair["r"] > 0 else "negative"
                    insights.append(
                        f"'{pair['a']}' and '{pair['b']}' show a strong {direction} "
                        f"correlation (r={pair['r']:.2f})."
                    )

        if not insights:
            insights.append("No notable issues detected; the dataset looks clean.")
        return insights

    # -- narration via Freebuff ------------------------------------------------
    def _narrate(self, task: str, report: dict[str, Any]) -> str:
        import json

        from app.llm.client import LLMConfigError, LLMError, LLMMessage

        compact = json.dumps(report, default=str)
        if len(compact) > 6000:
            compact = compact[:6000]

        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are the Data Agent inside ORBIT. You are given analysis "
                    "statistics that were computed locally with pandas/NumPy. "
                    "Summarize them for the user in Markdown, clearly, in a few "
                    "short sections. CRITICAL: never invent or alter any number — "
                    "only use the provided statistics. If a requested analysis is "
                    "missing from the statistics, say so."
                ),
            ),
            LLMMessage(role="user", content=f"User task: {task}\n\nComputed statistics:\n{compact}"),
        ]
        try:
            response = self.llm.chat_sync(messages, temperature=0.2)
            return response.content
        except (LLMConfigError, LLMError) as exc:
            return (
                "_Automatic narration unavailable (Freebuff LLM call failed: "
                f"{exc}). The statistics above were still computed locally._"
            )
