import io

import pytest

from app.agents.ml_agent import MLAgent
from app.core.files import FileContext
from tests.conftest import FakeLLMClient, make_csv_context


def _regression_csv(rows: int = 120, noise: bool = True) -> FileContext:
    import random

    rng = random.Random(7)
    buf = io.StringIO()
    buf.write("sqft,bedrooms,age,price\n")
    for _ in range(rows):
        sqft = rng.uniform(500, 3000)
        bedrooms = rng.randint(1, 5)
        age = rng.uniform(0, 50)
        price = 120 * sqft + 10_000 * bedrooms - 800 * age + (rng.uniform(-5e3, 5e3) if noise else 0)
        buf.write(f"{sqft:.1f},{bedrooms},{age:.1f},{price:.1f}\n")
    return FileContext.from_upload("houses.csv", buf.getvalue().encode())


def _classification_csv(rows: int = 120) -> FileContext:
    import random

    rng = random.Random(3)
    buf = io.StringIO()
    buf.write("feature1,feature2,churn\n")
    for _ in range(rows):
        f1 = rng.uniform(0, 10)
        f2 = rng.uniform(0, 10)
        label = 1 if (f1 + f2) + rng.uniform(-1.2, 1.2) > 10 else 0
        buf.write(f"{f1:.2f},{f2:.2f},{label}\n")
    return FileContext.from_upload("churn.csv", buf.getvalue().encode())


def test_regression_happy_path():
    result = MLAgent(llm=FakeLLMClient()).run(
        "train a model to predict price from this csv", file_context=_regression_csv()
    )
    assert result.status == "completed"
    data = result.data
    assert data["mode"] == "regression"
    assert data["target"] == "price"
    assert data["best_model"] in ("Linear Regression", "Decision Tree", "Random Forest")
    assert len(data["models"]) == 3
    for m in data["models"]:
        assert 0.9 <= m["r2"] <= 1.0  # synthetic data is strongly linear
        assert m["mae"] is not None and m["rmse"] is not None
    assert data["dataset"]["train_rows"] + data["dataset"]["test_rows"] == 120


def test_classification_happy_path():
    result = MLAgent(llm=FakeLLMClient()).run(
        "classify churn using this dataset", file_context=_classification_csv()
    )
    assert result.status == "completed"
    data = result.data
    assert data["mode"] == "classification"
    assert data["target"] == "churn"
    assert {m["model"] for m in data["models"]} == {
        "Logistic Regression", "Decision Tree", "Random Forest",
    }
    for m in data["models"]:
        assert 0.6 <= m["accuracy"] <= 1.0
        assert 0 <= m["f1_weighted"] <= 1.0
    assert data["best_model"] in {m["model"] for m in data["models"] if "error" not in m}


def test_too_few_rows_is_insufficient_data():
    ctx = make_csv_context(rows=10)
    result = MLAgent(llm=FakeLLMClient()).run("train a model on this csv", file_context=ctx)
    assert result.status == "insufficient_data"
    assert "at least 20" in result.message


def test_no_file_fails_gracefully():
    result = MLAgent(llm=FakeLLMClient()).run("train a model to predict price")
    assert result.status == "failed"
    assert "needs a CSV dataset" in result.message


def test_explicit_nonexistent_target_is_rejected():
    ctx = _regression_csv()
    result = MLAgent(llm=FakeLLMClient()).run(
        "train a model to predict zzz from this csv", file_context=ctx
    )
    assert result.status == "insufficient_data"
    assert "'zzz' was not found" in result.message
    assert "price" in result.message  # available columns are listed
    # No training happened: no model metrics in the payload.
    assert "models" not in (result.data or {}) or not result.data.get("models")


# -- Issue 1: explicit target / feature selection (ml_test_data.csv shape) ------


def _people_csv(rows: int = 60) -> FileContext:
    """name, age, salary, experience — mirrors the ml_test_data.csv scenario."""
    import random

    rng = random.Random(11)
    buf = io.StringIO()
    buf.write("name,age,salary,experience\n")
    for i in range(rows):
        age = rng.randint(22, 60)
        experience = max(0, age - 22 - rng.randint(0, 3))
        salary = 30_000 + experience * 2_500 + age * 300 + rng.randint(-2_000, 2_000)
        buf.write(f"person{i},{age},{salary},{experience}\n")
    return FileContext.from_upload("ml_test_data.csv", buf.getvalue().encode())


def test_explicit_valid_target_used_exactly():
    result = MLAgent(llm=FakeLLMClient()).run(
        "train a model to predict salary from this CSV",
        file_context=_people_csv(),
    )
    assert result.status == "completed"
    assert result.data["target"] == "salary"
    assert result.data["mode"] == "regression"


def test_explicit_nonexistent_target_no_silent_substitute():
    # name/age/salary/experience columns; 'price' does not exist. The agent
    # must NOT silently pick 'experience' (the last column) instead.
    result = MLAgent(llm=FakeLLMClient()).run(
        "Train a model to predict price from this CSV",
        file_context=_people_csv(),
    )
    assert result.status == "insufficient_data"
    assert "Target column 'price' was not found" in result.message
    assert "name, age, salary, experience" in result.message


def test_explicit_feature_selection_honored():
    result = MLAgent(llm=FakeLLMClient()).run(
        "Train a model to predict salary from this CSV using age and experience "
        "as features",
        file_context=_people_csv(),
    )
    assert result.status == "completed"
    data = result.data
    assert data["target"] == "salary"
    assert sorted(data["dataset"]["features"]) == ["age", "experience"]
    assert "name" not in data["dataset"]["features"]


def test_identifier_column_excluded_by_default():
    result = MLAgent(llm=FakeLLMClient()).run(
        "train a model to predict salary from this CSV",
        file_context=_people_csv(),
    )
    assert result.status == "completed"
    features = result.data["dataset"]["features"]
    assert "name" not in features          # unique text column -> identifier
    assert set(features) == {"age", "experience"}


def test_target_inference_only_without_explicit_target():
    # No explicit target: conventional last-column inference still applies.
    result = MLAgent(llm=FakeLLMClient()).run(
        "train a model on this CSV",
        file_context=_people_csv(),
    )
    assert result.status == "completed"
    assert result.data["target"] == "experience"  # last column
    assert "name" not in result.data["dataset"]["features"]


def test_incidental_column_mention_is_not_a_target():
    # 'age' appears in the text but is not the prediction target; salary is.
    result = MLAgent(llm=FakeLLMClient()).run(
        "train a model to predict salary from this CSV using age as the feature",
        file_context=_people_csv(),
    )
    assert result.status == "completed"
    assert result.data["target"] == "salary"
    assert result.data["dataset"]["features"] == ["age"]
