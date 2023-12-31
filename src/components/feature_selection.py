import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import config
from src.exception import CustomException
from src.logger import logging
from config import ARTIFACTS_DIR
import os
from src.utils import save_object


class FeatureSelectionConfig:
    correlation_matrix_path = os.path.join(ARTIFACTS_DIR, "correlation_matrix.pkl")
    feature_importance_path = os.path.join(ARTIFACTS_DIR, "feature_importance.pkl")
    selected_features_path = os.path.join(ARTIFACTS_DIR, "selected_features.pkl")
    correlated_features_path = os.path.join(ARTIFACTS_DIR, "correlated_features.pkl")
    artifacts_dir = ARTIFACTS_DIR


class FeatureSelection:
    def __init__(self):
        self.feature_selection_config = FeatureSelectionConfig()

    def calculate_correlation(self, numerical_features):
        correlation_matrix = numerical_features.corr()
        return correlation_matrix

    def plot_correlation_with_target(
        self, correlation_matrix, target_column, save_path
    ):
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
        plt.title(f"Correlation of Numerical Features with {target_column}")
        plt.xlabel("Correlation Coefficient")
        plt.ylabel("Features")
        plt.savefig(save_path + 'correlation_plot_sales.png')

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

    def train_rf_for_feature_importance(self, X, y):
        """
        This function trains a Random Forest Regressor model to identify feature importance.

        Args:
        X: df, the features df.
        y: Series, the target variable Series.

        Returns:
        array: an array of feature importances.
        """
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        return rf_model.feature_importances_

    def get_important_features_from_rf(
        self, feature_importances, X_columns, threshold=0.03
    ):
        """
        This function filters out features based on their importance as determined by Random Forest Regressor.

        Args:
        feature_importances: array, the feature importances obtained from a trained Random Forest model.
        X_columns: List, the column names of the features.
        threshold: float, the importance under which features are ignored (default is 0.03).

        Returns:
        List, the column names that are deemed important by Random Forest.
        """
        features_df = pd.DataFrame(
            {"Feature": X_columns, "Importance": feature_importances}
        )
        return features_df[features_df["Importance"] > threshold]["Feature"].tolist()

    def execute_feature_selection(self, df, output_file_name: str):
        """
        This function executes all the feature selection processes and saves the output.

        Args:
        df: df, the df for the feature selection to be performed on .

        Returns: -
        """

        try:
            logging.info("Starting the feature selection process.")

            # Calculate correlations
            numerical_features = df.select_dtypes(include=[np.number, 'number'])
            correlation_matrix = self.calculate_correlation(numerical_features)

            # Save corr matrix
            self.plot_correlation_with_target(
                correlation_matrix, "Sales", self.feature_selection_config.artifacts_dir
            )

            # Remove highly correlated features among themselves
            correlated_features_among_themselves = self.identify_correlated_features(
                correlation_matrix
            )
            # Remove 'Sales' col from the set if it's in there
            correlated_features_among_themselves.discard("Sales")
            df_reduced = df.drop(columns=list(correlated_features_among_themselves))

            # Capturing feature importance from Random Forest
            X = df_reduced.select_dtypes(include=[np.number, 'number']).drop(columns=["Sales"], axis =1)
            y = df_reduced["Sales"]
            feature_importances = self.train_rf_for_feature_importance(X, y)

            # Identify and remove highly-correlated features
            correlated_features = self.get_correlated_features(
                correlation_matrix, "Sales"
            )
            important_features = self.get_important_features_from_rf(
                feature_importances, X.columns
            )

            # Merging correlated and important features, removing duplicates to create the final selected features list
            final_selected_features = list(
                set(correlated_features) | set(important_features)
            )

            # Filtering df_reduced to only have final_selected_features
            final_df = df[final_selected_features]
            # Check if "Sales" column exists in final_df
            if "Sales" not in final_df.columns:
                final_df["Sales"] = df["Sales"]

            # Save this DataFrame to a CSV or any format you prefer
            final_df.to_csv(os.path.join(self.feature_selection_config.artifacts_dir, output_file_name),
                            index=False)

            # Save results
            save_object(
                self.feature_selection_config.correlation_matrix_path, correlation_matrix
            )
            save_object(
                self.feature_selection_config.feature_importance_path, feature_importances
            )
            save_object(
                self.feature_selection_config.correlated_features_path, correlated_features
            )

            # Save the correlation plot
            self.plot_correlation_with_target(
                correlation_matrix, "Sales", self.feature_selection_config.artifacts_dir
            )

            logging.info(f"Final selected features: {final_selected_features}")
            logging.info("The feature selection procedure has been completed.")

            return final_df
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Initiate class instance
    feature_selection = FeatureSelection()
    df = pd.read_csv(config.TRANSFORMED_DATA_PATH)
    feature_selection.execute_feature_selection(df)
