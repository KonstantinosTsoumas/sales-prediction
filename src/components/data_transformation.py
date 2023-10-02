import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from typing import List, Dict
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        logging.info("The data transformation phase has started.")

    def validate_dataframe(self, df: pd.DataFrame, date_cols: List[str]):
        """
        This function validates the df to ensure that columns intended to be of type datetime.

        Args:
            df: df, the df to validate.
            date_cols: List[str], a list of column names that should be of datetime type.

        Raises:
            CustomException: If a column that's supposed to be datetime is not.
        """

        for date_col in date_cols:
            if df[date_col].dtype != 'datetime64[ns]':
                error_message = f"Expected datetime64[ns] dtype for column {date_col}, found {df[date_col].dtype}"
                raise CustomException(error_message, sys.exc_info())

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
        self.validate_dataframe(df, date_cols)
        logging.info(" The dataframe contains columns of type datetime, validated.")
        for date_col in date_cols:
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

