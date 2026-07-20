import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
# pyrefly: ignore [missing-import]
from src.data_ingestion.data_ingestion import DataIngestion

@patch("src.data_ingestion.data_ingestion.pd.read_csv")
@patch("src.data_ingestion.data_ingestion.create_engine")
@patch("os.makedirs")
def test_load_data_to_database(mock_makedirs, mock_create_engine, mock_read_csv, dummy_data):
    mock_read_csv.return_value = dummy_data
    
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine
    
    ingestion = DataIngestion("fake_path.csv")
    ingestion.load_data_to_database()
    
    mock_read_csv.assert_called_once_with("fake_path.csv")
    mock_create_engine.assert_called_once()
    
@patch("src.data_ingestion.data_ingestion.pd.read_sql")
@patch("src.data_ingestion.data_ingestion.create_engine")
def test_load_data_from_database(mock_create_engine, mock_read_sql, dummy_data):
    mock_read_sql.return_value = dummy_data
    
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine
    
    ingestion = DataIngestion("fake_path.csv")
    df = ingestion.load_data_from_database()
    
    mock_read_sql.assert_called_once()
    assert df.equals(dummy_data)
