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
from src.config import MODEL_PATH, PREPROCESSOR_PATH, MODEL_NAME, SHAP_BACKGROUND_PATH

logger = get_logger(__name__)

class PredictionPipeline:
    def __init__(self):
        self.model = self.load_model()
        self.preprocessor = self.load_preprocessor()
        self.shap_background = self.load_shap_background()
        
    def load_model(self):
        try:
            logger.info(f"Attempting to load model '{MODEL_NAME}' from MLflow Model Registry")
            client = MlflowClient()
            # pyrefly: ignore [deprecated]
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

    def load_shap_background(self):
        try:
            background = np.load(SHAP_BACKGROUND_PATH)
            logger.info(f"SHAP background loaded: {background.shape}")
            return background
        except Exception as e:  # noqa: BLE001
            logger.warning(f"SHAP background not found, will use prediction row as fallback: {e}")
            return None

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

            # Convert sparse matrix → dense numpy (required by all SHAP explainers)
            import scipy.sparse as sp
            dense_data = transformed_data.toarray() if sp.issparse(transformed_data) else np.array(transformed_data)

            # Use saved training background; fall back to prediction row if not available
            background = self.shap_background if self.shap_background is not None else dense_data

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
                shap_values = explainer.shap_values(dense_data)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # positive class for binary
                shap_result = _extract_top_features(shap_values[0])
                logger.info("SHAP via TreeExplainer")
            except Exception as tree_err:  # noqa: BLE001
                logger.warning(f"TreeExplainer failed: {tree_err}")
                try:
                    # Linear models (LogisticRegression, LinearSVC, etc.)
                    # pyrefly: ignore [implicit-import]
                    masker = shap.maskers.Independent(background)
                    explainer = shap.LinearExplainer(self.model, masker)
                    shap_values = explainer.shap_values(dense_data)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    shap_result = _extract_top_features(shap_values[0])
                    logger.info("SHAP via LinearExplainer")
                except Exception as lin_err:  # noqa: BLE001
                    logger.warning(f"LinearExplainer failed: {lin_err}")
                    try:
                        # Generic fallback for any model (KNN, SVM, NaiveBayes, etc.)
                        bg_sample = shap.sample(background, min(50, background.shape[0]))

                        # Wrap predict_proba to return only positive class probability (1D)
                        # to avoid "truth value of an array is ambiguous" in SHAP internals
                        if hasattr(self.model, "predict_proba"):
                            predict_fn = lambda x: self.model.predict_proba(x)[:, 1]  # noqa: E731
                        else:
                            # pyrefly: ignore [missing-attribute]
                            predict_fn = self.model.predict

                        explainer = shap.KernelExplainer(predict_fn, bg_sample)
                        shap_values = explainer.shap_values(dense_data, nsamples=100)
                        # shap_values is now a plain 2D array (not a list) since predict_fn is 1D
                        shap_result = _extract_top_features(shap_values[0])
                        logger.info("SHAP via KernelExplainer")
                    except Exception as kernel_err:  # noqa: BLE001
                        logger.warning(f"All SHAP explainers failed: {kernel_err}")

            return {
                "prediction": int(prediction),
                "probability": float(probability),
                "top_features": shap_result
            }

        except Exception as e:  # noqa: BLE001
            logger.error("Prediction failed")
            raise CustomException(e, sys)
