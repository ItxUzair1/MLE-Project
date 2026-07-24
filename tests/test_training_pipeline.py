import pytest
from src.pipeline.training_pipeline import TrainingPipeline
from src.config import RAW_PATH


def test_training_pipeline_execution():
    pipeline = TrainingPipeline(raw_csv_path=RAW_PATH)
    results = pipeline.run_pipeline()

    assert isinstance(results, dict)
    assert "best_model_name" in results
    assert "best_train_f1" in results
    assert "evaluation_metrics" in results
    assert "f1_score" in results["evaluation_metrics"]
