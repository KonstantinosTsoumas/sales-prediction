from typing import Optional
import sys
import os
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from config import ARTIFACTS_DIR


@dataclass
class DataEncodingConfig:
    artifacts_dir: str = ARTIFACTS_DIR


class DataEncoding:
    def __init__(self, unique_value_threshold: Optional[int] = 20):
        self.data_encoding_config = DataEncodingConfig()
        self.unique_value_threshold = unique_value_threshold

    def handle_categorical_encoding(self, df: pd.DataFrame, output_file_name: str) -> pd.DataFrame:
        """
        This function handles the label and one-hot encoding for the specified columns.
        It performs label encoding on columns listed in 'label_encode_cols' and
        one-hot encoding on those in 'one_hot_encode_cols'. The encoding for the categorical features
        is based on the amount of unique values the column has.

        Args:
        df: df, The original data frame

        Returns:
        df: A df with encoded values
        """
        try:
            logging.info(
                f"Starting the encoding columns functionality. Initial DataFrame shape: {df.shape}"
            )
            # Initialize label encoder and its lists
            label_encoder = LabelEncoder()
            label_encode_cols = []
            one_hot_encode_cols = []

            # Distinguish which columns needs label or one-hot encoding based on pre (or user) defined threshold.
            for col in df.select_dtypes(include=["object"]).columns:
                if df[col].nunique() > self.unique_value_threshold:
                    label_encode_cols.append(col)
                else:
                    one_hot_encode_cols.append(col)

            logging.info(
                f"Starting encoding columns. Columns to label encode: {label_encode_cols}"
            )
            logging.info(
                f"Starting encoding columns. Columns to one-hot encode: {one_hot_encode_cols}"
            )

            # Perform label encoding on categorical columns with a variety of different values.
            for col in label_encode_cols:
                df[col] = label_encoder.fit_transform(df[col])

            # Perform one-hot encoding on categorical columns with a small number of unique values.
            encoded_data = pd.get_dummies(df, columns=one_hot_encode_cols, drop_first=True)
            logging.info("Encoding completed successfully.")

            logging.info(
                'Starting saving the dataset as a csv file in the "artifacts" directory.'
            )

            encoded_data_path = os.path.join(self.data_encoding_config.artifacts_dir, output_file_name)
            encoded_data.to_csv(encoded_data_path, index=False)
            logging.info("Saving the encoded dataset has been successfully completed.")

            return encoded_data

        except Exception as e:
            raise CustomException(e, sys)
