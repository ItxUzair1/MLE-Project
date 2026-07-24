import pytest
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from src.model_evaluation.model_evaluation import ModelEvaluation
from src.config import PREPROCESSOR_PATH, MODEL_PATH, TARGET_COLUMN


@pytest.fixture
def clean_transformed_data():
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        "num_col1": np.random.randn(n_samples),
        "num_col2": np.random.randn(n_samples),
        TARGET_COLUMN: np.random.choice([0, 1], size=n_samples)
    })
    return df


@pytest.fixture
def setup_artifacts(clean_transformed_data, tmp_path):
    preprocessor = ColumnTransformer(
        transformers=[("scaler", StandardScaler(), ["num_col1", "num_col2"])]
    )
    X = clean_transformed_data[["num_col1", "num_col2"]]
    y = clean_transformed_data[TARGET_COLUMN]
    
    preprocessor.fit(X)
    X_scaled = preprocessor.transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    os.makedirs("artifacts", exist_ok=True)

    with open(PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(preprocessor, f)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_scaled, y


def test_model_evaluation_fallback_and_evaluate(setup_artifacts):
    model, X_test, y_test = setup_artifacts
    evaluator = ModelEvaluation(X_test=X_test, y_test=y_test, model_name="telco-churn-model")
    
    # Test model loading (should fall back to local artifact if not in MLflow registry)
    loaded_model = evaluator.load_model()
    assert loaded_model is not None

    # Test evaluation metrics
    metrics = evaluator.evaluate()
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
