import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import mlflow
import mlflow.sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient

from src.logger.logger import get_logger
from src.exception.exception import CustomException
from src.config import (
    MODEL_PATH,
    MODEL_NAME,
    EXPERIMENT_NAME
)

logger = get_logger(__name__)


class ModelEvaluation:
    def __init__(self, X_test, y_test, model_name: str = MODEL_NAME):
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name

    def load_model(self):
        """
        Attempts to load model from MLflow Model Registry.
        If not available in MLflow Model Registry, falls back to loading from local MODEL_PATH artifact.
        """
        try:
            logger.info(f"Attempting to load model '{self.model_name}' from MLflow Model Registry")
            client = MlflowClient()
            
            # Search registered model versions
            latest_versions = client.get_latest_versions(name=self.model_name)
            if latest_versions:
                latest_version = latest_versions[0].version
                model_uri = f"models:/{self.model_name}/{latest_version}"
                logger.info(f"Loading registered model version {latest_version} via URI: {model_uri}")
                model = mlflow_sklearn.load_model(model_uri)
                logger.info("Successfully loaded model from MLflow Model Registry")
                return model
            else:
                logger.warning(f"No versions found for model '{self.model_name}' in MLflow Model Registry")
        except Exception as e:
            logger.warning(f"Could not load model from MLflow Model Registry: {e}")

        # Fallback to local MODEL_PATH artifact
        try:
            logger.info(f"Fallback: Loading model from local artifact path '{MODEL_PATH}'")
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, "rb") as f:
                    model = pickle.load(f)
                logger.info(f"Successfully loaded model from local path: {MODEL_PATH}")
                return model
            else:
                raise FileNotFoundError(f"Local model artifact not found at: {MODEL_PATH}")
        except Exception as e:
            logger.error("Failed to load model from both MLflow registry and local artifacts")
            raise CustomException(e, sys)  # type: ignore

    def evaluate(self) -> dict:
        try:
            logger.info("Starting model evaluation process")

            # 1. Load model
            model = self.load_model()

            # 2. Generate predictions on provided transformed test dataset
            logger.info(f"Generating predictions on test set of shape {np.shape(self.X_test)}")
            # pyrefly: ignore [missing-attribute]
            y_pred = model.predict(self.X_test)
            
            y_proba = None
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(self.X_test)[:, 1]

            # 3. Compute metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            
            roc_auc = float(roc_auc_score(self.y_test, y_proba)) if y_proba is not None else None

            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            }
            if roc_auc is not None:
                metrics["roc_auc"] = float(roc_auc)

            logger.info(
                f"Evaluation Metrics — Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, F1: {f1:.4f}" + 
                (f", ROC AUC: {roc_auc:.4f}" if roc_auc is not None else "")
            )

            # 4. Create Confusion Matrix Plot
            cm = confusion_matrix(self.y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots(figsize=(6, 6))
            # pyrefly: ignore [bad-argument-type]
            disp.plot(ax=ax, cmap="Blues")
            plt.title(f"Confusion Matrix ({self.model_name})")

            os.makedirs("artifacts", exist_ok=True)
            cm_plot_path = os.path.join("artifacts", "confusion_matrix.png")
            plt.savefig(cm_plot_path, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Confusion matrix plot saved to {cm_plot_path}")

            # 5. Log metrics and artifacts to MLflow
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="model-evaluation"):
                mlflow.log_param("evaluated_model", self.model_name)
                for metric_name, val in metrics.items():
                    mlflow.log_metric(metric_name, val)
                
                mlflow.log_artifact(cm_plot_path)

            logger.info("Evaluation metrics and confusion matrix plot logged to MLflow successfully")

            return metrics

        except Exception as e:
            logger.error("Model evaluation failed")
            raise CustomException(e, sys)  # type: ignore
