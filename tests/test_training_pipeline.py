import pytest
import os
from src.pipeline.training_pipeline import TrainingPipeline

def test_training_pipeline_execution(dummy_data, tmp_path):
    dummy_csv_path = os.path.join(tmp_path, "dummy_telcom.csv")
    dummy_data.to_csv(dummy_csv_path, index=False)
    
    pipeline = TrainingPipeline(raw_csv_path=dummy_csv_path)
    results = pipeline.run_pipeline()

    assert isinstance(results, dict)
    assert "best_model_name" in results
    assert "best_train_f1" in results
    assert "evaluation_metrics" in results
    assert "f1_score" in results["evaluation_metrics"]
