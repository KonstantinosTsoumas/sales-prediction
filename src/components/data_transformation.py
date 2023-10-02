import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from typing import List, Dict
from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def split_date_feature(self, df: pd.DataFrame, date_cols: List[str], date_features: Dict[str, List[str]]) -> pd.DataFrame:
        """
        This function extract features from date columns.

        Args:
        df: df, the df containing the data
        date_cols: list, a list of columns that are of datetime type
        date_features: dictionary, a dictionary containing the features to be extracted for each datetime column

        Returns:
        df: a df with new features
        """
        for date_col in date_cols:
            df[date_col] = pd.to_datetime(df[date_col])
            for feature in date_features.get(date_col, []):
                df[f"{date_col}_{feature}"] = getattr(df[date_col].dt, feature)
            df.drop(date_col, axis=1, inplace=True)
        return df

