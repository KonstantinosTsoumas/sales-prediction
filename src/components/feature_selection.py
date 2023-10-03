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

    def identify_correlated_features(self, corr_matrix, threshold=0.8):
        """
        This function identifies features that are highly correlated based on the provided threshold.

        Args:
        corr_matrix: df, the correlation matrix of the dataset.
        threshold: float, the correlation coefficient threshold for identifying features (mind: the default is 0.8).

        Returns:
        set, a set of column names that are correlated beyond the specified threshold.
        """
        # Store the correlated features in a set
        correlated_features = set()
        # Loop over the columns in the corr matrix and select only the highly correlated
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]
                    correlated_features.add(colname)
        return correlated_features

    def get_correlated_features(self, correlation_matrix, target_column, threshold=0.3):
        """
         This function filters out features based on their correlation with the target column.

         Args:
         correlation_matrix: df, the correlation matrix of the numerical features.
         target_column: str, the column we're looking to predict.
         threshold: float, the absolute value under which correlations are ignored (default is 0.3).

         Returns:
         List, the column names that are correlated above the threshold with the target.
         """
        target_corr = correlation_matrix[target_column].sort_values(ascending=False)
        return target_corr[target_corr.abs() > threshold].index.tolist()
