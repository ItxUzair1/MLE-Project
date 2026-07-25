import os
import sys
import pickle
import pandas as pd
import numpy as np
import shap
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.logger.logger import get_logger
from src.exception.exception import CustomException
from src.config import MODEL_PATH, PREPROCESSOR_PATH, MODEL_NAME

logger = get_logger(__name__)

class PredictionPipeline:
    def __init__(self):
        self.model = self.load_model()
        self.preprocessor = self.load_preprocessor()
        
    def load_model(self):
        try:
            logger.info(f"Attempting to load model '{MODEL_NAME}' from MLflow Model Registry")
            client = MlflowClient()
            latest_versions = client.get_latest_versions(name=MODEL_NAME)
            
            if latest_versions:
                latest_version = latest_versions[0].version
                model_uri = f"models:/{MODEL_NAME}/{latest_version}"
                model = mlflow.sklearn.load_model(model_uri)
                logger.info("Successfully loaded model from MLflow Registry")
                return model
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not load from MLflow Registry, falling back to local: {e}")

        try:
            logger.info(f"Loading model from local artifact: {MODEL_PATH}")
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load model locally.")
            raise CustomException(e, sys)

    def load_preprocessor(self):
        try:
            with open(PREPROCESSOR_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load preprocessor")
            raise CustomException(e, sys)

    def predict(self, df: pd.DataFrame):
        try:
            # TotalCharges is float, ensure it's not empty if not validated by schema
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

            # Transform features
            transformed_data = self.preprocessor.transform(df)

            # Predict
            # pyrefly: ignore [missing-attribute]
            prediction = self.model.predict(transformed_data)[0]
            probability = 0.0
            if hasattr(self.model, "predict_proba"):
                probability = self.model.predict_proba(transformed_data)[0][1]

            # SHAP Explainability
            # Try TreeExplainer → LinearExplainer → KernelExplainer (fallback)
            feature_names = self.preprocessor.get_feature_names_out()
            shap_result = []

            def _extract_top_features(shap_vals_row):
                feature_importance = list(zip(feature_names, shap_vals_row))
                feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                result = []
                for fname, val in feature_importance[:3]:
                    clean_name = fname.split("__")[-1]
                    result.append({"feature": clean_name, "impact": float(val)})
                return result

            try:
                # Tree models (RandomForest, XGBoost, DecisionTree, etc.)
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(transformed_data)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # positive class for binary
                shap_result = _extract_top_features(shap_values[0])
                logger.info("SHAP via TreeExplainer")
            except Exception:  # noqa: BLE001
                try:
                    # Linear models (LogisticRegression, etc.)
                    explainer = shap.LinearExplainer(self.model, transformed_data)
                    shap_values = explainer.shap_values(transformed_data)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    shap_result = _extract_top_features(shap_values[0])
                    logger.info("SHAP via LinearExplainer")
                except Exception:  # noqa: BLE001
                    try:
                        # Generic fallback for any model
                        explainer = shap.KernelExplainer(
                            self.model.predict_proba,
                            shap.sample(transformed_data, 50)
                        )
                        shap_values = explainer.shap_values(transformed_data, nsamples=100)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]
                        shap_result = _extract_top_features(shap_values[0])
                        logger.info("SHAP via KernelExplainer")
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"All SHAP explainers failed: {e}")
                
            return {
                "prediction": int(prediction),
                "probability": float(probability),
                "top_features": shap_result
            }

        except Exception as e:  # noqa: BLE001
            logger.error("Prediction failed")
            raise CustomException(e, sys)
