import pytest
import sys
# pyrefly: ignore [missing-import]
from src.exception.exception import CustomException, get_detailed_message

def test_custom_exception():
    try:
        # pyrefly: ignore [division-by-zero]
        1 / 0
    except Exception as e:
        custom_exc = CustomException(e, sys)
        assert "division by zero" in str(custom_exc)
        assert "test_exception.py" in str(custom_exc)
