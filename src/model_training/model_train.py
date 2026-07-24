from src.logger.logger import get_logger
from src.exception.exception import CustomException
import sys
import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn as mlflow_sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
from src.config import PREPROCESSOR_PATH, TARGET_COLUMN, TEST_SIZE, MODEL_PATH

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
                "XGBoost":            XGBClassifier()
            }

            best_model = None
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
                        best_model = model
                        best_name  = name

            logger.info(f"Best model: {best_name} with F1: {best_f1:.4f}")


            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(best_model, f)
            logger.info(f"Best model saved to {MODEL_PATH}")

            with mlflow.start_run(run_name=f"{best_name}-final-registered"):
                mlflow.log_param("best_model", best_name)
                mlflow.log_metric("best_f1",   float(best_f1))
                mlflow_sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="best_model",
                    registered_model_name="telco-churn-model",
                    serialization_format="cloudpickle"
                )
            logger.info("Best model registered to MLflow Model Registry")

            return best_model, best_name, best_f1, X_test, y_test

        except Exception as e:
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
            logger.info("Preprocessor applied successfully")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error("Data transformation failed in model training")
            raise CustomException(e, sys)  # type: ignore