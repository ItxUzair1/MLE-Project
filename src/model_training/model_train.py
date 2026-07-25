from src.logger.logger import get_logger
from src.exception.exception import CustomException
import sys
import os
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn as mlflow_sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from src.config import PREPROCESSOR_PATH, TARGET_COLUMN, TEST_SIZE, MODEL_PATH, SHAP_BACKGROUND_PATH, EXPERIMENT_NAME

logger = get_logger(__name__)


class ModelTraining:
    def __init__(self, clean_df: pd.DataFrame):
        self.df = clean_df

    def train(self):
        try:
            # 1. prepare data
            X_train, X_test, y_train, y_test = self.transform_data()

            logger.info("Starting model training")

            models = {
                "LogisticRegression": LogisticRegression(),
                "RandomForest":       RandomForestClassifier(),
                "XGBoost":            XGBClassifier(),
                "DecisionTree":       DecisionTreeClassifier(),
                "KNN":                KNeighborsClassifier(),
                "SVM":                SVC(),
                "NaiveBayes":         GaussianNB()
            }

            best_name  = ""
            best_f1    = 0

            # 2. set experiment once before loop
            mlflow.set_experiment("Telco Churn Prediction")

            # 3. train all models
            for name, model in models.items():
                logger.info(f"Training model: {name}")

                with mlflow.start_run(run_name=name):

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    accuracy  = accuracy_score(y_test, y_pred)
                    recall    = recall_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    f1        = f1_score(y_test, y_pred)

                    logger.info(
                        f"{name} -> accuracy: {accuracy:.4f}  "
                        f"recall: {recall:.4f}  "
                        f"precision: {precision:.4f}  "
                        f"f1: {f1:.4f}"
                    )

                    # log to MLflow
                    mlflow.log_param("model",     name)
                    mlflow.log_metric("accuracy",  float(accuracy))
                    mlflow.log_metric("recall",    float(recall))
                    mlflow.log_metric("precision", float(precision))
                    mlflow.log_metric("f1",        float(f1))
                    mlflow_sklearn.log_model(model, name, serialization_format="cloudpickle")

                    # track best
                    if f1 > best_f1:
                        best_f1    = f1
                        best_name  = name

            logger.info(f"Best model: {best_name} with F1: {best_f1:.4f}")

            # Register the baseline model just for record keeping
            with mlflow.start_run(run_name=f"{best_name}-baseline"):
                mlflow.log_param("baseline_model", best_name)
                mlflow.log_metric("baseline_f1", float(best_f1))
            
            # --- HYPERPARAMETER TUNING ---
            logger.info(f"Starting hyperparameter tuning for {best_name}")
            from src.model_training.hyperparameter_tuning import HyperparameterTuning
            tuner = HyperparameterTuning(
                model_name=best_name, 
                X_train=X_train, X_test=X_test, 
                y_train=y_train, y_test=y_test
            )
            
            best_params, _tuned_f1 = tuner.tune_model(n_trials=10) # 10 trials for speed
            tuned_model, final_f1 = tuner.train_best_model(best_params)
            
            return tuned_model, f"{best_name} (Tuned)", final_f1, X_test, y_test

        except Exception as e:  # noqa: BLE001
            logger.error("Model training failed")
            raise CustomException(e, sys)  # type: ignore

    def transform_data(self):
        try:
            logger.info("Starting scaling and encoding of data")
            df = self.df

            X = df.drop(columns=[TARGET_COLUMN])
            y = df[TARGET_COLUMN]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=TEST_SIZE,
                random_state=42
            )
            logger.info(f"Data split — X_train: {X_train.shape}, X_test: {X_test.shape}")

            logger.info("Applying preprocessor")
            with open(PREPROCESSOR_PATH, "rb") as f:
                preprocessor = pickle.load(f)

            X_train = preprocessor.fit_transform(X_train)
            X_test  = preprocessor.transform(X_test)  # only transform never fit

            with open(PREPROCESSOR_PATH, "wb") as f:
                pickle.dump(preprocessor, f)

            # Save a background sample for SHAP explanations at inference time
            import scipy.sparse as sp
            X_train_dense = X_train.toarray() if sp.issparse(X_train) else np.array(X_train)
            n_samples = min(100, X_train_dense.shape[0])
            indices = np.random.choice(X_train_dense.shape[0], n_samples, replace=False)
            background = X_train_dense[indices]
            np.save(SHAP_BACKGROUND_PATH, background)
            logger.info(f"SHAP background dataset saved ({background.shape[0]} samples)")

            # Log preprocessor and SHAP background to DAGsHub/MLflow so the
            # prediction pipeline can load them remotely (no local artifacts needed)
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="pipeline-artifacts") as artifacts_run:
                mlflow.set_tag("artifact_type", "pipeline_artifacts")
                mlflow.log_artifact(PREPROCESSOR_PATH, artifact_path="pipeline")
                mlflow.log_artifact(SHAP_BACKGROUND_PATH, artifact_path="pipeline")
                logger.info(
                    f"Preprocessor & SHAP background logged to MLflow "
                    f"(run_id: {artifacts_run.info.run_id})"
                )

            logger.info("Preprocessor applied and fitted preprocessor saved successfully")

            return X_train, X_test, y_train, y_test

        except Exception as e:  # noqa: BLE001
            logger.error("Data transformation failed in model training")
            raise CustomException(e, sys)  # type: ignore