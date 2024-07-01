import os 
import sys 
import pandas as pd 

from src.logger import logger
from src.exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("data","train.csv")
    test_data_path = os.path.join("data","test.csv")
    raw_data_path = os.path.join("data", "raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def ingest_data(self):
            logger.info("Ingesting data")
            try:
                logger.info("Reading data from ")
                df = pd.read_csv("C:/Users/MN/Documents/ml_projects/end_to_end_ml/assert/stud.csv")
                # print(df.head())

                train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

                os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

                train_set.to_csv(self.config.train_data_path, index=False, header=True)
                test_set.to_csv(self.config.test_data_path, index=False, header=True)
                df.to_csv(self.config.raw_data_path, index=False, header=True)

                logger.info("Data ingestion completed successfully")

                return self.config.train_data_path, self.config.test_data_path 
            
            except Exception as e:
                CustomException(e)

