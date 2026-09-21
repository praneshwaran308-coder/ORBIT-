"""ML Agent — scikit-learn training and evaluation.

Validates the dataset, infers regression vs classification, trains three
models per mode, and reports training/testing info and metrics. All
computation is local; insufficient data is reported gracefully.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.agents.base import BaseAgent, InsufficientDataError
from app.core.results import AgentResult

if TYPE_CHECKING:  # pragma: no cover
    from app.core.files import FileContext

MIN_ROWS = 20
CORR_TARGET_TOP_K = 5
CONFUSION_MAX_SIZE = 8
RANDOM_STATE = 42

# A text column where (almost) every row has a distinct value is an identifier
# (name, id, email...) — never a useful ML feature by default.
IDENTIFIER_UNIQUE_RATIO = 0.95


@dataclass
class FeaturePlan:
    """Resolved target + feature set for an ML task."""

    target: str | None
    frame: pd.DataFrame | None          # features (in order) + target
    features: list[str]
    issues: list[str] = field(default_factory=list)
    explicit_target: bool = False


def _is_identifier_column(series: pd.Series) -> bool:
    """True for non-numeric columns where values are (almost) all distinct."""
    if pd.api.types.is_numeric_dtype(series):
        return False
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    return non_null.nunique(dropna=True) / len(non_null) >= IDENTIFIER_UNIQUE_RATIO


def _finite(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class MLAgent(BaseAgent):
    name = "ml"

    def _run(self, task: str, file_context: "FileContext | None" = None,
             history: list[dict] | None = None) -> AgentResult:
        if file_context is None:
            return AgentResult.create(
                self.name,
                "failed",
                "The ML Agent needs a CSV dataset. Upload a .csv file and ask to "
                "train, classify, or predict (e.g. 'train a model to predict "
                "price from this CSV').",
            )

        plan = self._plan_features(file_context, task)
        if plan.target is None:
            raise InsufficientDataError("; ".join(plan.issues))

        target = plan.target
        df = plan.frame
        X = plan.frame.drop(columns=[target])
        y = df[target]

        mode = self._infer_mode(task, y)

        numeric_features = list(X.select_dtypes(include=[np.number]).columns)
        categorical_features = [c for c in X.columns if c not in numeric_features]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), numeric_features),
                ("cat", Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
                ]), categorical_features),
            ],
            remainder="drop",
        )

        stratify = y if mode == "classification" and y.nunique() > 1 and all(
            (y == c).sum() >= 2 for c in y.unique()
        ) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
        )

        models = self._models(mode)
        results, trained = [], []
        for label, estimator in models:
            pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
            try:
                pipeline.fit(X_train, y_train)
            except Exception as exc:  # noqa: BLE001 - a single model failing shouldn't kill the run
                results.append({"model": label, "error": f"{type(exc).__name__}: {exc}"})
                continue
            metrics = self._metrics(pipeline, X_test, y_test, mode)
            results.append({"model": label, **metrics})
            trained.append((label, pipeline, metrics))

        ok = [r for r in results if "error" not in r]
        if not ok:
            raise InsufficientDataError(
                "No model could be trained on this dataset. " +
                ("; ".join(r.get("error", "") for r in results))
            )

        best = max(ok, key=lambda r: r.get("r2", -1e9) if mode == "regression"
                   else r.get("f1_weighted", -1e9))
        best_label = best["model"]
        best_pipeline = next(p for l, p, _ in trained if l == best_label)

        return AgentResult.create(
            self.name,
            "completed",
            message=f"Trained and compared {len(ok)} {mode} model(s) on "
                    f"'{file_context.filename}' targeting '{target}'.",
            data={
                "format": "ml_report",
                "mode": mode,
                "target": target,
                "dataset": {
                    "filename": file_context.filename,
                    "rows": int(len(df)),
                    "train_rows": int(len(X_train)),
                    "test_rows": int(len(X_test)),
                    "features": list(X.columns),
                    "numeric_features": numeric_features,
                    "categorical_features": categorical_features,
                    "class_balance": self._class_balance(y) if mode == "classification" else None,
                },
                "models": results,
                "best_model": best_label,
                "confusion_matrix": next(
                    (r.get("confusion_matrix") for r in ok if r["model"] == best_label), None
                ),
            },
            metadata={"computed_locally": True, "random_state": RANDOM_STATE},
        )

    # -- validation -------------------------------------------------------------
    def _validate(self, file_context: "FileContext", task: str):
        """Backward-compatible wrapper: returns (df, target, mode, issues)."""
        plan = self._plan_features(file_context, task)
        if plan.target is None:
            return None, None, None, plan.issues

        y = plan.frame[plan.target]
        if y.isna().all():
            return None, None, None, [f"Target column '{plan.target}' is entirely empty."]

        mode = "classification" if (self._task_says_classification(task) or not
                                    pd.api.types.is_numeric_dtype(y)) else "regression"
        if mode == "classification" and y.nunique() > max(20, len(plan.frame) // 10):
            # High-cardinality numeric column treated as an ID, not a class label.
            return None, None, None, [
                f"Target '{plan.target}' has too many distinct values ({y.nunique()}) for "
                "classification; if it is continuous, ask for regression instead."
            ]

        return plan.frame, plan.target, mode, plan.issues

    def _plan_features(self, file_context: "FileContext", task: str) -> "FeaturePlan":
        """Resolve the target and the exact feature set for the task.

        Rules:
        - An explicitly named target must exist exactly; no substitution.
        - Explicitly requested features are honored (target excluded).
        - Non-numeric identifier-like columns (unique text columns) are never
          used as features unless explicitly requested.
        - Without an explicit target, the last column is used (the conventional
          CSV target position).
        """
        try:
            df = file_context.frame()
        except Exception as exc:  # noqa: BLE001
            return FeaturePlan(None, None, [], [f"Could not read CSV: {exc}"], False)

        issues: list[str] = []
        if df.empty:
            return FeaturePlan(None, None, [], ["The CSV has no data rows."], False)
        if len(df) < MIN_ROWS:
            return FeaturePlan(None, None, [], [
                f"Only {len(df)} data rows found; at least {MIN_ROWS} are required for "
                "a meaningful train/test split."
            ], False)

        text = " ".join(task.lower().split())
        explicit_target = self._explicit_target(df, text)
        if explicit_target is not None and explicit_target not in df.columns:
            return FeaturePlan(None, None, [], [
                f"Target column '{explicit_target}' was not found. Available "
                f"columns: {', '.join(map(str, df.columns))}."
            ], True)

        target = explicit_target or df.columns[-1]

        requested = self._explicit_features(df, text)
        if requested is not None:
            missing = [c for c in requested if c not in df.columns]
            if missing:
                issues.append(
                    f"Requested feature column(s) not found: {', '.join(missing)}. "
                    f"Available columns: {', '.join(map(str, df.columns))}."
                )
            features = [c for c in requested if c in df.columns and c != target]
            if not features:
                return FeaturePlan(None, None, [], [
                    "No usable feature columns remain after applying the requested "
                    f"feature selection (target: '{target}')."
                ], explicit_target is not None)
        else:
            # Default features: everything except the target, minus identifier-like
            # columns (text columns with a distinct value for every row).
            features = [c for c in df.columns if c != target
                        and not _is_identifier_column(df[c])]
            if not features:
                return FeaturePlan(None, None, [], [
                    "No usable feature columns remain after excluding identifier-like "
                    f"columns (target: '{target}')."
                ], explicit_target is not None)

        working = df[[*features, target]]
        y = working[target]
        if y.isna().all():
            return FeaturePlan(None, None, [],
                               [f"Target column '{target}' is entirely empty."],
                               explicit_target is not None)

        return FeaturePlan(target=target, frame=working, features=features,
                           issues=issues, explicit_target=explicit_target is not None)

    def _explicit_target(self, df: pd.DataFrame, text: str) -> str | None:
        """The user-named target, or None when not explicitly specified.

        Only treats a column as the explicit target when the task clearly asks
        to predict/estimate/forecast it — incidental mentions never count.
        """
        patterns = (
            r"\bpredict\s+(?:the\s+)?[`\"']?([\w][\w \-]*?)[`\"']?\s*(?:from\b|using\b|with\b|based on\b|,|$|\.)",
            r"\b(?:estimate|forecast)\s+(?:the\s+)?[`\"']?([\w][\w \-]*?)[`\"']?\s*(?:from\b|using\b|with\b|based on\b|,|$|\.)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            candidate = match.group(1).strip(" `\"'.,")
            for col in df.columns:
                if str(col).lower() == candidate.lower():
                    return str(col)
            return candidate  # explicitly named but absent -> caller errors
        return None

    def _explicit_features(self, df: pd.DataFrame, text: str) -> list[str] | None:
        """Explicit feature list, e.g. 'using age and experience as features'."""
        match = (re.search(r"\busing\s+(.+?)\s+as\s+(?:the\s+)?features?\b", text)
                 or re.search(r"\bwith\s+(.+?)\s+as\s+(?:the\s+)?features?\b", text))
        if not match:
            return None
        raw = match.group(1)
        parts = [p.strip(" `\"'") for p in re.split(r"\s*(?:,|\band\b)\s*", raw)]
        resolved: list[str] = []
        for part in (p for p in parts if p):
            for col in df.columns:
                if str(col).lower() == part.lower() and col not in resolved:
                    resolved.append(col)
        return resolved

    def _infer_mode(self, task: str, y: pd.Series) -> str:
        """Regression vs classification from the task text and the target dtype."""
        return "classification" if (self._task_says_classification(task) or not
                                     pd.api.types.is_numeric_dtype(y)) else "regression"

    def _task_says_classification(self, task: str) -> bool:
        t = task.lower()
        return any(k in t for k in (
            "classif", "classify", "class", "category", "label", "churn", "survived",
        ))

    # -- training -----------------------------------------------------------------
    def _models(self, mode: str):
        if mode == "regression":
            return [
                ("Linear Regression", LinearRegression()),
                ("Decision Tree", DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=6)),
                ("Random Forest", RandomForestRegressor(
                    n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        return [
            ("Logistic Regression", LogisticRegression(max_iter=1000)),
            ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6)),
            ("Random Forest", RandomForestClassifier(
                n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)),
        ]

    def _metrics(self, pipeline: Pipeline, X_test, y_test, mode: str) -> dict[str, Any]:
        pred = pipeline.predict(X_test)
        if mode == "regression":
            return {
                "r2": _finite(r2_score(y_test, pred)),
                "mae": _finite(mean_absolute_error(y_test, pred)),
                "rmse": _finite(math.sqrt(mean_squared_error(y_test, pred))),
            }
        average = "binary" if len(np.unique(y_test)) == 2 else "weighted"
        out: dict[str, Any] = {
            "accuracy": _finite(accuracy_score(y_test, pred)),
            "precision": _finite(precision_score(y_test, pred, average=average, zero_division=0)),
            "recall": _finite(recall_score(y_test, pred, average=average, zero_division=0)),
            "f1_weighted": _finite(f1_score(y_test, pred, average=average, zero_division=0)),
        }
        cm = confusion_matrix(y_test, pred)
        if cm.shape[0] <= CONFUSION_MAX_SIZE:
            out["confusion_matrix"] = {
                "labels": [str(l) for l in np.unique(y_test)],
                "matrix": cm.tolist(),
            }
        return out

    def _class_balance(self, y: pd.Series) -> dict[str, float]:
        counts = y.value_counts(normalize=True)
        return {str(k): round(float(v), 4) for k, v in counts.items()}
