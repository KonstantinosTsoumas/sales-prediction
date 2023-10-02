import unittest
import pandas as pd
from src.components.data_transformation import DataTransformation

class TestDataTransformation(unittest.TestCase):

    def test_split_date_feature(self):
        # Instantiate the class
        data_transformation = DataTransformation()

        # Sample df
        data = {'date_col1': ['2022-01-01', '2022-01-02', '2022-01-03'],
                'another_col': [1, 2, 3]}
        df = pd.DataFrame(data)

        # Extract features and run
        date_features = {'date_col1': ['year', 'month', 'day']}
        new_df = data_transformation.split_date_feature(df, ['date_col1'], date_features)

        # Validation tests
        self.assertIn('date_col1_year', new_df.columns)
        self.assertIn('date_col1_month', new_df.columns)
        self.assertIn('date_col1_day', new_df.columns)
        self.assertEqual(new_df['date_col1_year'].iloc[0], 2022)
        self.assertEqual(new_df['date_col1_month'].iloc[0], 1)
        self.assertEqual(new_df['date_col1_day'].iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
