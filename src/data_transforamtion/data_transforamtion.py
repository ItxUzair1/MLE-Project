import os
import sys
import pickle
import pandas as pd
from src.logger.logger import get_logger
from src.exception.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import PREPROCESSOR_PATH

logger = get_logger(__name__)

class DataTransformation:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def transform_data(self) -> pd.DataFrame:
        try:
            logger.info("Starting data transformation")
            df = self.df.copy()

            # 1. drop customerID
            logger.info("Dropping customerID")
            df = df.drop(columns=["customerID"])

            logger.info("Fixing TotalCharges column")
            df["TotalCharges"] = df["TotalCharges"].str.strip()
            df["TotalCharges"] = df["TotalCharges"].replace("", float("nan"))
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df.dropna(inplace=True)                                  
            logger.info("TotalCharges fixed")

            # 3. fix redundant values
            logger.info("Removing redundant values")
            internet_cols = [
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"
            ]
            df[internet_cols] = df[internet_cols].replace("No internet service", "No")
            df[["MultipleLines"]] = df[["MultipleLines"]].replace("No phone service", "No")
            logger.info("Redundant values removed")

            # 4. map Churn
            logger.info("Mapping Churn column")
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
            df["Churn"] = df["Churn"].astype("int32")
            logger.info("Churn mapped successfully")

            logger.info(f"Transformation complete. Shape: {df.shape}")
            
            logger.info("Saving transformed data to data/processed")
            os.makedirs("data/processed", exist_ok=True)
            df.to_csv("data/processed/transformed_data.csv", index=False)
            logger.info("Transformed data saved successfully")

            return df

        except Exception as e:
            logger.error("Data transformation failed")
            raise CustomException(e, sys)  # type: ignore

    def get_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        try:
            logger.info("Creating preprocessor")


            numeric_columns = [
                col for col in df.select_dtypes(exclude="object").columns.tolist()
                if col != "Churn"
            ]
            categorical_columns = df.select_dtypes(include="object").columns.tolist()

            logger.info(f"Numeric columns: {numeric_columns}")
            logger.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(transformers=[
                ("scaler", StandardScaler(), numeric_columns),
                ("encoder", OneHotEncoder(), categorical_columns)
            ])

            os.makedirs("artifacts", exist_ok=True)
            with open(PREPROCESSOR_PATH, "wb") as f:
                pickle.dump(preprocessor, f)

            logger.info("Preprocessor saved to artifacts/preprocessor.pkl")
            return preprocessor

        except Exception as e:
            logger.error("Failed to create preprocessor")
            raise CustomException(e, sys)  # type: ignore