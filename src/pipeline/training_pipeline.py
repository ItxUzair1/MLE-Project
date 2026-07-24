import sys
from src.logger.logger import get_logger
from src.exception.exception import CustomException
from src.config import RAW_PATH
from src.data_ingestion.data_ingestion import DataIngestion
from src.data_validation.data_validation import DataValidation
from src.data_transforamtion.data_transforamtion import DataTransformation
from src.model_training.model_train import ModelTraining
from src.model_evaluation.model_evaluation import ModelEvaluation

logger = get_logger(__name__)


class TrainingPipeline:
    def __init__(self, raw_csv_path: str = RAW_PATH):
        self.raw_csv_path = raw_csv_path

    def run_pipeline(self) -> dict:
        """
        Executes the end-to-end training pipeline:
        1. Data Ingestion
        2. Data Validation
        3. Data Transformation
        4. Model Training & MLflow Registration
        5. Model Evaluation
        """
        try:
            logger.info("==========================================")
            logger.info(">>> STARTING TRAINING PIPELINE <<<")
            logger.info("==========================================")

            # Stage 1: Data Ingestion
            logger.info(">>> Stage 1: Data Ingestion <<<")
            ingestion = DataIngestion(csv_path=self.raw_csv_path)
            ingestion.load_data_to_database()
            df = ingestion.load_data_from_database()
            logger.info(f"Stage 1 Complete — Raw data loaded with shape {df.shape}")

            # Stage 2: Data Validation
            logger.info(">>> Stage 2: Data Validation <<<")
            validation = DataValidation(dataset=df)
            is_valid = validation.validate_data()
            if not is_valid:
                logger.warning("Data validation reported issues. Proceeding with caution...")
            else:
                logger.info("Stage 2 Complete — Data validation passed successfully")

            # Stage 3: Data Transformation
            logger.info(">>> Stage 3: Data Transformation <<<")
            transformation = DataTransformation(df=df)
            transformed_df = transformation.transform_data()
            transformation.get_preprocessor(transformed_df)
            logger.info(f"Stage 3 Complete — Data transformed shape {transformed_df.shape}")

            # Stage 4: Model Training
            logger.info(">>> Stage 4: Model Training <<<")
            trainer = ModelTraining(clean_df=transformed_df)
            best_model, best_name, best_f1, X_test, y_test = trainer.train()
            logger.info(f"Stage 4 Complete — Best Model: {best_name} (F1 Score: {best_f1:.4f})")

            # Stage 5: Model Evaluation
            logger.info(">>> Stage 5: Model Evaluation <<<")
            evaluator = ModelEvaluation(X_test=X_test, y_test=y_test)
            metrics = evaluator.evaluate()
            logger.info(f"Stage 5 Complete — Evaluation Metrics: {metrics}")

            logger.info("==========================================")
            logger.info(">>> TRAINING PIPELINE COMPLETED SUCCESSFULLY <<<")
            logger.info("==========================================")

            return {
                "best_model_name": best_name,
                "best_train_f1": float(best_f1),
                "evaluation_metrics": metrics
            }

        except Exception as e:
            logger.error("Training Pipeline failed during execution")
            raise CustomException(e, sys)  # type: ignore


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
