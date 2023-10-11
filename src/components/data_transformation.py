import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.stats import skew, boxcox

from src.exception import CustomException
from sklearn.impute import SimpleImputer
from src.logger import logging
from config import ARTIFACTS_DIR
from src.utils import save_object
import os


@dataclass
class DataTransformationConfig:
    data_transformation_path = os.path.join(ARTIFACTS_DIR, "data_transformation.pkl")
    artifacts_dir = ARTIFACTS_DIR


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        logging.info("The data transformation phase has started.")

    def split_date_feature(
        self,
        df: pd.DataFrame,
        date_cols: List[str],
        date_features: Dict[str, List[str]],
            ) -> pd.DataFrame:
        """
        This function extract features from date columns.

        Args:
        df: df, the df containing date columns to be split.
        date_cols: list, a list of datetime columns to split.
        date_features: dictionary, a dictionary containing the features to be extracted for each datetime column

        Returns:
        df: a df with date columns as replaced by the extracted features.
        """
        logging.info(
            f" Starting to split date features. Initial DataFrame shape: {df.shape}"
        )
        for date_col in date_cols:
            # Check if any element in the date column is of integer type
            if df[date_col].apply(type).eq(int).any():
                raise CustomException(
                    f"Invalid data type for date column {date_col}", sys
                )

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
        logging.info(
            f"Starting dynamic imputation on column. Initial DataFrame shape: {df.shape}"
        )
        for col in df.columns:
            missing_ratio = df[col].isna().mean()

            if missing_ratio > threshold:
                df.drop(columns=[col], inplace=True)
                print(f"Column {col} dropped due to high missing ratio.")
                continue

            if missing_ratio > 0:
                if np.issubdtype(df[col].dtype, np.number):
                    imputer = SimpleImputer(strategy="median")
                    print(f"Column {col} imputed with median.")
                else:
                    imputer = SimpleImputer(strategy="most_frequent")
                    print(f"Column {col} imputed with most frequent value.")

                df[col] = imputer.fit_transform(df[[col]]).ravel()

        logging.info(f" Feature imputation done. New DataFrame shape: {df.shape}")
        return df

    def apply_log_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        This functions applies log transformation to specified columns in a DataFrame.

        Args:
        df: df, the original df
        columns: list, columns to be log-transformed

        Returns:
        df: df, a df with log-transformed columns
        """
        for col in columns:
            df[col] = np.log1p(df[col])
        return df

    def apply_box_cox(self, df: pd.DataFrame, columns: List[str], skew_threshold: float = 0.5) -> pd.DataFrame:
        """
        This function applies Box-Cox transformation to specified columns in a df, if their skewness is above a given threshold.

        Args:
        df: DataFrame, the original DataFrame
        columns: list, columns to be Box-Cox transformed
        skew_threshold: float, skewness threshold to apply Box-Cox

        Returns:
        df: DataFrame, a DataFrame with Box-Cox transformed columns
        """
        for col in columns:
            if (df[col] <= 0).any():
                raise CustomException("Box-Cox transformation only works for positive values", sys)

            # Check skewness
            col_skewness = skew(df[col])
            if abs(col_skewness) > skew_threshold:
                df[col], _ = boxcox(df[col])
                print(f"Applied Box-Cox to {col} with original skewness {col_skewness}")
            else:
                print(f"Skipped {col} as its skewness {col_skewness} is within the threshold")
        return df

    def save(self, file_path: str):
        try:
            save_object(file_path, self)
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, df: pd.DataFrame, output_file_name: str) -> pd.DataFrame:
        """
        This function initiates a series of data transformations including missing value imputation,
        date feature splitting, and Box-Cox transformation for the target variable.
        The function also saves the transformed DataFrame as a CSV file.

        Args:
        df: df, the DataFrame containing the dataset that needs transformation.
        output_file_name: str, the name of the transformed data file

        Returns:
        pd: df, the DataFrame after the transformations have been applied.

        Raises:
        CustomException
            An exception raised if any error occurs during the data transformation process.

        Side Effects:
        Logs various stages of the data transformation process.
        Saves the transformed DataFrame as a CSV file.
        Applies Box-Cox transformation on the target variable if applicable.
        """
        try:
            logging.info("Initiating the process of reading train and test data files.")
            transformed_df = df.copy()
            logging.info(f"Successfully read train data from {transformed_df}.")
            logging.info("Starting missing value curation.")

            # Split date features
            if 'order date (DateOrders)' in df.columns:
                date_cols = ["order date (DateOrders)"]
                # Define the features we want to extract from the date column
                date_features = {"order date (DateOrders)": ["year", "month", "day"]}
                transformed_df = self.split_date_feature(transformed_df, date_cols, date_features)
            logging.info(
                "Successfully split date columns into their respective features in both train and test datasets."
            )

            logging.info("Initiating dynamic imputation on train and test datasets.")
            transformed_df = self.dynamic_imputer(transformed_df)
            logging.info(
                "Successfully applied dynamic imputation on missing values in both train and test datasets."
            )
            logging.info(
                'Starting saving the dataset as a csv file in the "artifacts" directory.'
            )

            # Apply Box-Cox Transformation on target variable)
            target_variable = "Sales"
            if (transformed_df["Sales"] <= 0).any():
                logging.warning("Box-Cox transformation only works for positive values")
            else:
                transformed_df[target_variable], _ = boxcox(transformed_df[target_variable])
                logging.info(f"Successfully applied Box-Cox transformation on {target_variable}.")

            transformed_data_path = os.path.join(self.data_transformation_config.artifacts_dir, output_file_name)

            # Save transformed file to directory
            transformed_df.to_csv(
                transformed_data_path, index=False
            )
            logging.info("Saving the dataset has been successfully completed.")

            save_object(
                self.data_transformation_config.data_transformation_path, self
            )

            return transformed_df

        except Exception as e:
            raise CustomException(e, sys)
