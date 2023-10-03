import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import config
from src.exception import CustomException
from src.logger import logging
from config import ARTIFACTS_DIR, TRANSFORMED_DATA_PATH
import os
from src.utils import save_object

class FeatureSelectionConfig:
    correlation_matrix_path = os.path.join(ARTIFACTS_DIR, "correlation_matrix.pkl")
    feature_importance_path = os.path.join(ARTIFACTS_DIR, "feature_importance.pkl")
    selected_features_path = os.path.join(ARTIFACTS_DIR, "selected_features.pkl")
    correlated_features_path = os.path.join(ARTIFACTS_DIR, "correlated_features.pkl")

class FeatureSelection:
    def __init__(self):
        self.data_analysis_config = FeatureSelectionConfig()

    def calculate_correlation(self, numerical_features):
        correlation_matrix = numerical_features.corr()
        return correlation_matrix

    def plot_correlation_with_target(self, correlation_matrix, target_column, save_path):
        """
        This function plots and saves the correlation of all numerical features with the specified target column.

        Args:
        correlation_matrix: df, the correlation matrix of the dataset.
        target_column: str, the name of the target column.
        save_path: str, the path where the plot will be saved.

        Returns: -
        """
        target_corr = correlation_matrix[target_column].sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=target_corr.values, y=target_corr.index)
        plt.title(f'Correlation of Numerical Features with {target_column}')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Features')
        plt.savefig(save_path)
        plt.show()

