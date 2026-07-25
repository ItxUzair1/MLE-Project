import sys
import pickle
import tempfile

import numpy as np
import pandas as pd
import shap
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.logger.logger import get_logger
from src.exception.exception import CustomException
from src.config import MODEL_NAME, EXPERIMENT_NAME
from src.utils.dagshub_init import init_dagshub

logger = get_logger(__name__)


class PredictionPipeline:
    def __init__(self):
        # Point all MLflow calls to DAGsHub before loading anything
        init_dagshub()
        self.model            = self.load_model()
        self.preprocessor     = self.load_preprocessor()
        self.shap_background  = self.load_shap_background()

    # ------------------------------------------------------------------
    # Helper: find the most recent MLflow run tagged as pipeline-artifacts
    # ------------------------------------------------------------------
    def _get_pipeline_artifacts_run_id(self) -> str | None:
        try:
            client     = MlflowClient()
            experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if experiment is None:
                logger.warning(f"Experiment '{EXPERIMENT_NAME}' not found in MLflow.")
                return None

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.artifact_type = 'pipeline_artifacts'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                return runs[0].info.run_id
            logger.warning("No 'pipeline-artifacts' run found in MLflow.")
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not query MLflow for pipeline-artifacts run: {e}")
            return None

    # ------------------------------------------------------------------
    # Model loading — from MLflow / DAGsHub Model Registry
    # ------------------------------------------------------------------
    def load_model(self):
        try:
            logger.info(f"Loading model '{MODEL_NAME}' from MLflow Model Registry")
            client          = MlflowClient()
            latest_versions = client.get_latest_versions(name=MODEL_NAME)
            if latest_versions:
                latest_version = latest_versions[0].version
                model_uri      = f"models:/{MODEL_NAME}/{latest_version}"
                model          = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Model v{latest_version} loaded from MLflow Registry")
                return model
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to load model from MLflow Registry: {e}")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Preprocessor loading — downloaded from DAGsHub MLflow artifacts
    # ------------------------------------------------------------------
    def load_preprocessor(self):
        run_id = self._get_pipeline_artifacts_run_id()
        if run_id:
            try:
                logger.info(f"Downloading preprocessor from MLflow (run_id={run_id})")
                local_path = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/pipeline/preprocessor.pkl"
                )
                with open(local_path, "rb") as f:
                    preprocessor = pickle.load(f)
                logger.info("Preprocessor loaded remotely from DAGsHub ✓")
                return preprocessor
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Remote preprocessor download failed: {e}")

        raise RuntimeError(
            "Could not load the preprocessor from DAGsHub. "
            "Please run the training pipeline first to register artifacts."
        )

    # ------------------------------------------------------------------
    # SHAP background loading — downloaded from DAGsHub MLflow artifacts
    # ------------------------------------------------------------------
    def load_shap_background(self):
        run_id = self._get_pipeline_artifacts_run_id()
        if run_id:
            try:
                logger.info(f"Downloading SHAP background from MLflow (run_id={run_id})")
                local_path = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/pipeline/shap_background.npy"
                )
                background = np.load(local_path)
                logger.info(f"SHAP background loaded remotely: {background.shape} ✓")
                return background
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Remote SHAP background download failed: {e}")

        logger.warning("SHAP background unavailable — SHAP explanations may be degraded.")
        return None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame):
        try:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"], errors="coerce"
            ).fillna(0.0)

            # Transform
            transformed_data = self.preprocessor.transform(df)

            # Predict
            prediction  = self.model.predict(transformed_data)[0]  # type: ignore[union-attr]
            probability = 0.0
            if hasattr(self.model, "predict_proba"):
                probability = self.model.predict_proba(transformed_data)[0][1]

            # -----------------------------------------------------------
            # SHAP Explainability
            # Try TreeExplainer → LinearExplainer → KernelExplainer
            # -----------------------------------------------------------
            import scipy.sparse as sp

            feature_names = self.preprocessor.get_feature_names_out()
            shap_result   = []

            dense_data = (
                transformed_data.toarray()
                if sp.issparse(transformed_data)
                else np.array(transformed_data)
            )

            # Use saved training background; fall back to prediction row
            background = (
                self.shap_background if self.shap_background is not None else dense_data
            )

            def _extract_top_features(shap_vals_row):
                importance = list(zip(feature_names, shap_vals_row))
                importance.sort(key=lambda x: abs(x[1]), reverse=True)
                return [
                    {"feature": fname.split("__")[-1], "impact": float(val)}
                    for fname, val in importance[:3]
                ]

            try:
                explainer   = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(dense_data)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                shap_result = _extract_top_features(shap_values[0])
                logger.info("SHAP via TreeExplainer")
            except Exception as tree_err:  # noqa: BLE001
                logger.warning(f"TreeExplainer failed: {tree_err}")
                try:
                    masker      = shap.maskers.Independent(background)
                    explainer   = shap.LinearExplainer(self.model, masker)
                    shap_values = explainer.shap_values(dense_data)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    shap_result = _extract_top_features(shap_values[0])
                    logger.info("SHAP via LinearExplainer")
                except Exception as lin_err:  # noqa: BLE001
                    logger.warning(f"LinearExplainer failed: {lin_err}")
                    try:
                        bg_sample = shap.sample(background, min(50, background.shape[0]))
                        predict_fn = (
                            (lambda x: self.model.predict_proba(x)[:, 1])  # noqa: E731
                            if hasattr(self.model, "predict_proba")
                            else self.model.predict
                        )
                        explainer   = shap.KernelExplainer(predict_fn, bg_sample)
                        shap_values = explainer.shap_values(dense_data, nsamples=100)
                        shap_result = _extract_top_features(shap_values[0])
                        logger.info("SHAP via KernelExplainer")
                    except Exception as kernel_err:  # noqa: BLE001
                        logger.warning(f"All SHAP explainers failed: {kernel_err}")

            return {
                "prediction":  int(prediction),
                "probability": float(probability),
                "top_features": shap_result,
            }

        except Exception as e:  # noqa: BLE001
            logger.error("Prediction failed")
            raise CustomException(e, sys)
