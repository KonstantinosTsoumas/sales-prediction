import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from src.components.data_transformation import DataTransformation, CustomException  # Replace 'your_module' with the actual module name


class TestDataTransformation(unittest.TestCase):

    def setUp(self):
        self.data_transformation = DataTransformation()

    @patch("pandas.read_csv")
    def test_initiate_data_transformation(self, mock_read_csv):
        # Mock dfs
        mock_df_train = pd.DataFrame({
            'order date (DateOrders)': ['2022-01-01', '2022-01-02'],
            'value': [1, 2]
        })
        mock_df_test = pd.DataFrame({
            'order date (DateOrders)': ['2022-01-03', '2022-01-04'],
            'value': [3, 4]
        })

        mock_read_csv.side_effect = [mock_df_train, mock_df_test]

        try:
            self.data_transformation.initiate_data_transformation("mock_train_path")
        except CustomException as e:
            self.fail(f"initiate_data_transformation() raised CustomException unexpectedly: {e}")

    def test_split_date_feature_valid(self):
        df = pd.DataFrame({
            'date_column': ['2022-01-01', '2022-01-02'],
            'value': [1, 2]
        })
        date_features = {'date_column': ['year', 'month', 'day']}

        new_df = self.data_transformation.split_date_feature(df, ['date_column'], date_features)

        self.assertIn('date_column_year', new_df.columns)
        self.assertIn('date_column_month', new_df.columns)
        self.assertIn('date_column_day', new_df.columns)

    def test_split_date_feature_invalid(self):
        df = pd.DataFrame({
            'date_column': [1, 2],
            'value': [1, 2]
        })
        date_features = {'date_column': ['year', 'month', 'day']}

        with self.assertRaises(CustomException):
            self.data_transformation.split_date_feature(df, ['date_column'], date_features)

    def test_dynamic_imputer(self):
        df = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4],
            'categorical_col': ['a', 'b', 'a', np.nan],
        })

        new_df = self.data_transformation.dynamic_imputer(df)

        self.assertFalse(new_df['numeric_col'].isnull().any())
        self.assertFalse(new_df['categorical_col'].isnull().any())


if __name__ == '__main__':
    unittest.main()
