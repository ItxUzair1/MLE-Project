import pytest
import logging
# pyrefly: ignore [missing-import]
from src.logger.logger import get_logger

def test_get_logger():
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    
    # Verify handlers
    has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    
    assert has_stream_handler
    assert has_file_handler
