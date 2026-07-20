import pandas as pd
from src.logger.logger import get_logger
from src.exception.exception import CustomException
import sys
import great_expectations as ge

logger = get_logger(__name__)

class DataValidation:
    def __init__(self, dataset: pd.DataFrame):
        self.df = dataset

    def validate_data(self) -> bool:
        try:
            logger.info("Starting the validation of data")

            results = []

            ge_df = ge.from_pandas(self.df)

            # ── 1. columns must exist ──────────────────────────
            results.append(ge_df.expect_column_to_exist("customerID"))
            results.append(ge_df.expect_column_to_exist("gender"))
            results.append(ge_df.expect_column_to_exist("tenure"))
            results.append(ge_df.expect_column_to_exist("MonthlyCharges"))
            results.append(ge_df.expect_column_to_exist("TotalCharges"))
            results.append(ge_df.expect_column_to_exist("Churn"))

            # ── 2. no nulls in critical columns ───────────────
            results.append(ge_df.expect_column_values_to_not_be_null("customerID"))
            results.append(ge_df.expect_column_values_to_not_be_null("Churn"))
            results.append(ge_df.expect_column_values_to_not_be_null("tenure"))
            results.append(ge_df.expect_column_values_to_not_be_null("MonthlyCharges"))

            # ── 3. unique customerID ───────────────────────────
            results.append(ge_df.expect_column_values_to_be_unique("customerID"))

            # ── 4. allowed values — raw values before cleaning ─
            results.append(ge_df.expect_column_values_to_be_in_set(
                "gender", ["Male", "Female"]
            ))
            results.append(ge_df.expect_column_values_to_be_in_set(
                "Churn", ["Yes", "No"]
            ))
            results.append(ge_df.expect_column_values_to_be_in_set(
                "Partner", ["Yes", "No"]
            ))
            results.append(ge_df.expect_column_values_to_be_in_set(
                "Dependents", ["Yes", "No"]
            ))
            results.append(ge_df.expect_column_values_to_be_in_set(
                "PhoneService", ["Yes", "No"]
            ))
            results.append(ge_df.expect_column_values_to_be_in_set(
                "MultipleLines", ["Yes", "No", "No phone service"]  # raw value
            ))
            results.append(ge_df.expect_column_values_to_be_in_set(
                "InternetService", ["DSL", "Fiber optic", "No"]
            ))
            results.append(ge_df.expect_column_values_to_be_in_set(
                "OnlineSecurity", ["Yes", "No", "No internet service"]  # raw value
            ))
            results.append(ge_df.expect_column_values_to_be_in_set(
                "OnlineBackup", ["Yes", "No", "No internet service"]  # raw value
            ))
            results.append(ge_df.expect_column_values_to_be_in_set(
                "DeviceProtection", ["Yes", "No", "No internet service"]  # raw value
            ))
            results.append(ge_df.expect_column_values_to_be_in_set(
                "TechSupport", ["Yes", "No", "No internet service"]  # raw value
            ))
            results.append(ge_df.expect_column_values_to_be_in_set(
                "StreamingTV", ["Yes", "No", "No internet service"]  # raw value
            ))
            results.append(ge_df.expect_column_values_to_be_in_set(
                "StreamingMovies", ["Yes", "No", "No internet service"]  # raw value
            ))
            results.append(ge_df.expect_column_values_to_be_in_set(
                "Contract", ["Month-to-month", "One year", "Two year"]
            ))
            results.append(ge_df.expect_column_values_to_be_in_set(
                "PaperlessBilling", ["Yes", "No"]
            ))
            results.append(ge_df.expect_column_values_to_be_in_set(
                "PaymentMethod", [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            ))

            # ── 5. numeric ranges ──────────────────────────────
            results.append(ge_df.expect_column_values_to_be_between(
                "tenure", 0, 72
            ))
            results.append(ge_df.expect_column_values_to_be_between(
                "MonthlyCharges", 0, 999
            ))

            # ── 6. TotalCharges — just check it exists ─────────
            # dont check type or range here because raw data has spaces
            results.append(ge_df.expect_column_to_exist("TotalCharges"))

            # ── 7. row count ───────────────────────────────────
            results.append(ge_df.expect_table_row_count_to_be_between(100, 100000))

            # ── check all results ──────────────────────────────
            all_passed = all(r.success for r in results)

            for r in results:
                expectation = r.expectation_config.expectation_type
                column = r.expectation_config.kwargs.get("column", "table")
                status = "PASSED" if r.success else "FAILED"
                logger.info(f"{status} — {expectation} — [{column}]")

            if all_passed:
                logger.info("All validations passed ")
            else:
                logger.warning("Some validations failed  — check logs above")

            return all_passed

        except Exception as e:
            logger.error("Data validation crashed")
            raise CustomException(e, sys) # type: ignore