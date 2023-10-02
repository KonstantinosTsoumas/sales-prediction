import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, RAW_DATA_PATH, INPUT_DATA_CSV
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = TRAIN_DATA_PATH
    test_data_path: str = TEST_DATA_PATH
    raw_data_path: str = RAW_DATA_PATH

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("The data ingestion phase has started")
        try:
            df = pd.read_csv(INPUT_DATA_CSV)
            logging.info('The dataset is read as a dataframe')

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Both the train and the test dataset have been initiated')
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('The data ingestion phase is complete')
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    # Initialize data ingestion
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    # Start data transformation
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    # Initialize model training
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))