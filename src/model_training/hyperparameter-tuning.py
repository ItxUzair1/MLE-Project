from src.logger.logger import get_logger
from src.exception.exception import CustomException
import optuna
import pickle
from src.config import MODEL_PATH
import mlflow
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import mlflow.sklearn as mlflow_sklearn
import os
logger=get_logger(__name__)
class HyperparameterTuning:
    def __init__(self,model_name,X_train,X_test,y_train,y_test):
        self.model_name=model_name
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test


    def objective(self,trial):
        try:

            if self.model_name == "RandomForest":
                params = {
                    "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
                    "max_depth":         trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 5),
                    "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                }
                model=RandomForestClassifier(**params,random_state=42)
            
            elif self.model_name == "XGBoost":
                params = {
                "n_estimators":  trial.suggest_int("n_estimators", 50, 500),
                "max_depth":     trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }
                model = XGBClassifier(**params, random_state=42, eval_metric="logloss")
            
            else:
                params = {
                "C":        trial.suggest_float("C", 0.01, 10.0),
                "solver":   trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
                "max_iter": trial.suggest_int("max_iter", 100, 500),
                }
                model = LogisticRegression(**params, random_state=42)

            model.fit(self.X_train,self.y_train)
            y_pred=model.predict(self.X_test)
            return f1_score(self.y_test,y_pred)


        
        except Exception as e:
            raise CustomException(e,sys)


    def tune_model(self,n_trials:int=50):
        try:
            
            logger.info(f"Starting the tunning of best model {self.model_name}")

            study=optuna.create_study(direction="maximize")
            study.optimize(self.objective,n_trials)

            logger.info(f"The best paramas are {study.best_params}")
            logger.info(f"best f1 score: {study.best_value}")

            return study.best_params,study.best_value
        except Exception as e:
            logger.error("Tuning crashed")
            raise CustomException(e,sys)


    def train_best_model(self, best_params: dict):

        try:
            logger.info(f"Retraining {self.model_name} with best params")

            if self.model_name == "RandomForest":
                model = RandomForestClassifier(**best_params, random_state=42)

            elif self.model_name == "XGBoost":
                model = XGBClassifier(**best_params, random_state=42, eval_metric="logloss")

            elif self.model_name == "LogisticRegression":
                model = LogisticRegression(**best_params, random_state=42)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            f1 = f1_score(self.y_test, y_pred)

            logger.info(f"Tuned model F1: {f1:.4f}")

            mlflow.set_experiment("Telco Churn Prediction")

            with mlflow.start_run(run_name=f"{self.model_name}-tuned-final"):

                mlflow.log_params(best_params)
                mlflow.log_param("model", self.model_name)
                mlflow.log_metric("f1_score", float(f1))

                mlflow.log_artifact(MODEL_PATH)

                mlflow_sklearn.log_model(
                    sk_model=model,
                    artifact_path="tuned_model",
                    registered_model_name="telco-churn-model"  # new version
                )

            logger.info("Tuned model registered to MLflow Model Registry")

            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)

            logger.info(f"Tuned model saved to {MODEL_PATH}")

            return model, f1

        except Exception as e:
            logger.error("Failed to save tuned model")
            raise CustomException(e, sys)  # type: ignore

