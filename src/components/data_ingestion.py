import os
import sys
from dataclasses import dataclass


import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'DataCoSupplyChainDataset.csv')

class DataIngestion:
    try:
        df = pd.read_csv('input/DataCoSupplyChainDataset.csv')
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

