import pytest
import pandas as pd
import os
from unittest.mock import patch, mock_open
# pyrefly: ignore [missing-import]
from src.data_transforamtion.data_transforamtion import DataTransformation
from sklearn.compose import ColumnTransformer

@patch("os.makedirs")
@patch("pandas.DataFrame.to_csv")
def test_transform_data(mock_to_csv, mock_makedirs, dummy_data):
    transformer = DataTransformation(dummy_data)
    transformed_df = transformer.transform_data()
    
    assert "customerID" not in transformed_df.columns
    assert transformed_df["TotalCharges"].dtype == "float64"
    assert "No internet service" not in transformed_df["OnlineSecurity"].values
    assert transformed_df["Churn"].isin([0, 1]).all()

@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
@patch("pickle.dump")
def test_get_preprocessor(mock_pickle_dump, mock_file_open, mock_makedirs, dummy_data):
    transformer = DataTransformation(dummy_data)
    transformed_df = transformer.transform_data()
    preprocessor = transformer.get_preprocessor(transformed_df)
    
    assert isinstance(preprocessor, ColumnTransformer)
    mock_pickle_dump.assert_called_once()
