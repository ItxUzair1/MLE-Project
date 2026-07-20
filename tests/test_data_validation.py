import pytest
# pyrefly: ignore [missing-import]
from src.data_validation.data_validation import DataValidation

def test_data_validation_success(dummy_data):
    validation = DataValidation(dummy_data)
    result = validation.validate_data()
    assert result is True

def test_data_validation_failure(dummy_data):
    # Mess up the dummy_data to cause validation failure
    dummy_data.loc[0, "gender"] = "InvalidGender"
    validation = DataValidation(dummy_data)
    result = validation.validate_data()
    assert result is False
