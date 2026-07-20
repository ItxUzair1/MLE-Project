import pytest
# pyrefly: ignore [missing-import]
from src.config import load_yaml
from unittest.mock import patch, mock_open

@patch("builtins.open", new_callable=mock_open, read_data="data:\n  database_path: dummy_path")
def test_load_yaml_success(mock_file):
    result = load_yaml("dummy.yaml")
    assert result == {"data": {"database_path": "dummy_path"}}
    mock_file.assert_called_once_with("dummy.yaml", "r")

def test_load_yaml_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_yaml("non_existent.yaml")
