import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from typing import List, Dict
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from src.logger import logging
from config import ARTIFACTS_DIR, TRANSFORMED_DATA_PATH
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')
    transformed_data_csv_path = os.path.join(ARTIFACTS_DIR, "transformed_data.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        logging.info("The data transformation phase has started.")

    def split_date_feature(self, df: pd.DataFrame, date_cols: List[str], date_features: Dict[str, List[str]]) -> pd.DataFrame:
        """
        This function extract features from date columns.

        Args:
        df: df, the df containing date columns to be split.
        date_cols: list, a list of datetime columns to split.
        date_features: dictionary, a dictionary containing the features to be extracted for each datetime column

        Returns:
        df: a df with date columns as replaced by the extracted features.
        """
        logging.info(f" Starting to split date features. Initial DataFrame shape: {df.shape}")
        for date_col in date_cols:
            # Check if any element in the date column is of integer type
            if df[date_col].apply(type).eq(int).any():
                raise CustomException(f"Invalid data type for date column {date_col}", sys)

             df[date_col] = pd.to_datetime(df[date_col])
            for feature in date_features.get(date_col, []):
                df[f"{date_col}_{feature}"] = getattr(df[date_col].dt, feature)
            df.drop(date_col, axis=1, inplace=True)
        logging.info(f"Date features split. New DataFrame shape: {df.shape}")
        return df

    def dynamic_imputer(self, df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
        """
        This function dynamically impute missing values based on column data types and missing value ratios. The
        pre-defined threshold is 0.4. The imputation is only applied if the impure do not exceed the 40% of the total
        rows, the whole rows are deleted otherwise.

        Args:
        df: df, the original df
        threshold: float, the missing value ratio threshold for column dropping

        Returns:
        df: df, a df with imputed or dropped columns
        """
        logging.info(f"Starting dynamic imputation on column. Initial DataFrame shape: {df.shape}")
        for col in df.columns:
            missing_ratio = df[col].isna().mean()

            if missing_ratio > threshold:
                df.drop(columns=[col], inplace=True)
                print(f"Column {col} dropped due to high missing ratio.")
                continue

            if missing_ratio > 0:
                if np.issubdtype(df[col].dtype, np.number):
                    imputer = SimpleImputer(strategy='median')
                    print(f"Column {col} imputed with median.")
                else:
                    imputer = SimpleImputer(strategy='most_frequent')
                    print(f"Column {col} imputed with most frequent value.")

                df[col] = imputer.fit_transform(df[[col]]).ravel()

        logging.info(f" Feature imputation done. New DataFrame shape: {df.shape}")
        return df

    def initiate_data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Initiating the process of reading train and test data files.")
            df = pd.read_csv(df)
            logging.info(f"Successfully read train data from {df}.")
            logging.info("Starting missing value curation.")

            # Split date features
            date_cols = ['order date (DateOrders)']
            # Define the features we want to extract from the date column
            date_features = {"order date (DateOrders)": ["year", "month", 'day']}
            df = self.split_date_feature(df, date_cols, date_features)

            logging.info("Successfully split date columns into their respective features in both train and test datasets.")

            logging.info("Initiating dynamic imputation on train and test datasets.")
            df = self.dynamic_imputer(df)
            logging.info("Successfully applied dynamic imputation on missing values in both train and test datasets.")
            logging.info('Starting saving the dataset as a csv file in the "artifacts" directory.')

            # Save transformed file to directory
            df.to_csv(self.data_transformation_config.transformed_data_csv_path, index=False)
            logging.info('Saving the dataset has been successfully completed.')

            # Save transformed DataFrame to directory using pickle
            save_object(self.data_transformation_config.preprocessor_obj_file_path, df)
            return df

        except Exception as e:
            raise CustomException(e, sys)
