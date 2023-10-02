import unittest
import pandas as pd
import os
from src.components.data_encoding import DataEncoding, DataEncodingConfig

class TestDataEncoding(unittest.TestCase):

    def setUp(self):
        self.data_encoding = DataEncoding()

        self.test_data = pd.DataFrame({
            'A': ['apple', 'banana', 'apple', 'apple', 'banana'],
            'B': ['dog', 'cat', 'dog', 'dog', 'bird'],
            'C': [1, 2, 3, 1, 2]
        })

    def test_handle_categorical_encoding(self):
        encoded_df = self.data_encoding.handle_categorical_encoding(self.test_data)

        self.assertTrue('A' not in encoded_df.columns)
        self.assertTrue('A_apple' not in encoded_df.columns)
        self.assertTrue('B' not in encoded_df.columns)

        self.assertTrue('A_banana' in encoded_df.columns)
        self.assertTrue('B_dog' in encoded_df.columns)
        self.assertTrue('B_cat' in encoded_df.columns)

        self.assertFalse('3' in encoded_df.columns)
        self.assertFalse('10' in encoded_df.columns)
        self.assertFalse('1' in encoded_df.columns)

        # Clean up the generated CSV file§
        if os.path.exists(DataEncodingConfig.artifacts_dir):
            os.remove(DataEncodingConfig.artifacts_dir)


if __name__ == '__main__':
    unittest.main()
