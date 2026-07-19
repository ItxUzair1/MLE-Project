import pandas as pd
from src.logger.logger import get_logger
from src.exception.exception import CustomException
from dotenv import load_dotenv
import os
import sys
from sqlalchemy import create_engine

load_dotenv()
logger=get_logger(__name__)

DATABASE_PATH=os.getenv("DATABASE_PATH")
TABLE_NAME=os.getenv("TABLE_NAME")


class DataIngestion():
    def __init__(self,csv_path:str):
        self.path=csv_path
        self.database_path=DATABASE_PATH
        self.table_name=TABLE_NAME
        
        
    
    def load_data_to_database(self):
        try:
            logger.info("Loading the data")
            df=pd.read_csv(self.path)
            logger.info(F"Data loaded succesfully:  {df.shape}")
            
            os.makedirs(os.path.dirname(self.database_path),exist_ok=True)
            engine=create_engine(f"sqlite:///{self.database_path}")
            
            df.to_sql(self.table_name,engine,if_exists="replace",index=False)
            logger.info(f"Data pushed to database table '{self.table_name}' successfully")
            
            
        except Exception as e:
            logger.error("Data did not laod to the database")
            raise CustomException(e,sys)
        
        
    
    def load_data_from_database(self)-> pd.DataFrame:
        try:
            
            logger.info("Feteching the data from database")
            engine=create_engine(f"sqlite:///{self.database_path}")
            
            df=pd.read_sql(sql=f"SELECT * FROM {self.table_name}",con=engine)
            
            logger.info("Data loaded sucessfully from the database")
            
            return df
        
        except Exception as e:
            logger.error("Data did not laod from the database")
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    ingestor=DataIngestion("data/raw/telcom.csv")
    ingestor.load_data_to_database()
    df=ingestor.load_data_from_database()
    print(df.shape)
            
