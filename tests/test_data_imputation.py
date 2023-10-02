import unittest
import pandas as pd
import numpy as np
from src.components.data_transformation import DataTransformation  # replace 'your_module' with the actual module name

class TestDataTransformation(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': ['a', 'b', 'c', np.nan, 'e'],
            'C': [np.nan] * 5
        })
        self.dt = DataTransformation()

    def test_dynamic_imputer_drops_columns(self):
        df_transformed = self.dt.dynamic_imputer(self.df.copy())
        self.assertNotIn('C', df_transformed.columns)

    def test_dynamic_imputer_imputes_values(self):
        df_transformed = self.dt.dynamic_imputer(self.df.copy())
        self.assertFalse(df_transformed['A'].isna().any())
        self.assertFalse(df_transformed['B'].isna().any())

    def test_dynamic_imputer_dataframe_shape(self):
        df_transformed = self.dt.dynamic_imputer(self.df.copy())
        self.assertEqual(df_transformed.shape, (5, 2))

if __name__ == '__main__':
    unittest.main()
