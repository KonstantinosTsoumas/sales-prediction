import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.components.feature_selection import FeatureSelection


class TestFeatureSelection(unittest.TestCase):

    @patch('src.components.feature_selection.save_object')
    @patch('src.components.feature_selection.logging')
    @patch('sklearn.ensemble.RandomForestRegressor')
    def test_execute_feature_selection(self, MockRF, MockLogging, MockSaveObject):
        # mock objects
        mock_rf_instance = MagicMock()
        mock_rf_instance.feature_importances_ = np.array([0.2, 0.4, 0.1, 0.3])
        MockRF.return_value = mock_rf_instance

        # test data
        test_df = pd.DataFrame({
            'Sales': [1, 2, 3, 4, 5],
            'Household consumption': [2, 4, 1, 5, 7],
            'Demand in numbers': [3, 1, 2, 7, 8],
            'Gender': ["Male", "Male", "Female", "Male", "Female"]
        })

        # test the actual method
        fs = FeatureSelection()

        # capture any exception raised by the method
        try:
            fs.execute_feature_selection(test_df)
            exception_raised = False
        except Exception as e:
            exception_raised = True

        # assert that no exceptions were raised
        self.assertFalse(exception_raised, "Exception was raised during feature selection.")
