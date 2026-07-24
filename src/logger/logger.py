import logging
import os
import sys
from datetime import datetime
from src.config import LOG_DIR

os.makedirs(LOG_DIR,exist_ok=True)

log_file=os.path.join(LOG_DIR,f"{datetime.now().strftime('%d-%m-%Y')}.log")



def get_logger(name:str)-> logging.Logger:
    
    
    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    
    if logger.handlers:
        return logger
    
    
    formatting=logging.Formatter(
        fmt="[%(asctime)s %(levelname)s %(name)s - %(message)s]",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Use utf-8 encoded stdout to avoid cp1252 UnicodeEncodeError on Windows
    console_handler=logging.StreamHandler(stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False))
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatting)
    
    
    file_handler=logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatting)
    
    
    logger.addHandler(console_handler)
    
    logger.addHandler(file_handler)
    
    return logger

