import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.exception import CustomException
from src.logger import logging
from config import RAW_DATA_PATH, ARTIFACTS_DIR
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.components.data_encoding import DataEncoding, DataEncodingConfig
from src.components.feature_selection import FeatureSelection, FeatureSelectionConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path: str = RAW_DATA_PATH
    artifacts_dir: str = ARTIFACTS_DIR

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_stratified_k_fold(self, n_splits: int = 5) -> None:
        """
        This function performs Stratified K-Fold splitting on a given dataset.

        Parameters:
        n_splits: int, the number of folds (must be at least 2 - defaults to 5).

        Raises:
        CustomException: An exception raised if any error occurs during the process.
        """
        logging.info("Starting Stratified K-Fold splitting.")
        try:
            df = pd.read_csv(INPUT_DATA_CSV)
            X = df.drop('target', axis=1)  # Replace 'target' with your target column
            y = df['target']  # Replace 'target' with your target column

            skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

            for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
                train_set = df.iloc[train_index]
                test_set = df.iloc[test_index]

                train_set.to_csv(f"{self.ingestion_config.artifacts_dir}_fold_{fold}.csv", index=False)
                test_set.to_csv(f"{self.ingestion_config.artifacts_dir}_fold_{fold}.csv", index=False)

            logging.info(f"Completed Stratified K-Fold splitting into {n_splits} folds.")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_splitting(self):
        """
        This function is responsible for splitting the dataset into train/test.

        Args: -

        Raises:
        CustomException: An exception raised if any error occurs during the process.
        """
        logging.info("The data splitting phase has started.")
        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path, encoding='ISO-8859-1')
            logging.info("The dataset is read as a dataframe.")

            # Create directory if it doesn't exist
            os.makedirs(
                os.path.dirname(self.ingestion_config.artifacts_dir), exist_ok=True
            )

            # Drop unuseful features
            df = df.drop(['Product Image', 'Sales per customer'], axis=1)

            # Perform 70/30 split to train,test set and save both.
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Sales'])
            train_path = os.path.join(self.ingestion_config.artifacts_dir, 'train_data.csv')
            test_path = os.path.join(self.ingestion_config.artifacts_dir, 'test_data.csv')

            train_set.to_csv(
                train_path, index=False, header=True
            )
            test_set.to_csv(
                test_path, index=False, header=True
            )

            logging.info("Both the train and the test dataset have been initiated")

            logging.info("The data ingestion process is complete.")

            return (
                train_set,
                test_set
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Initialize the required classes
    data_ingestion = DataIngestion()
    data_transformation = DataTransformation()
    data_encoding = DataEncoding()
    feature_selection = FeatureSelection()
    model_trainer = ModelTrainer()

    # Perform data splitting into train, test sets.
    train_set, test_set = data_ingestion.initiate_data_splitting()

    # Separate features and targets, given that 'Sales' is the target
    X_train = train_set.drop('Sales', axis=1)
    y_train = train_set['Sales']
    X_test = test_set.drop('Sales', axis=1)
    y_test = test_set['Sales']

    # Perform data transformation
    transformed_train_df = data_transformation.initiate_data_transformation(train_set, "train_data_transformed")
    transformed_test_df = data_transformation.initiate_data_transformation(test_set, "test_data_transformed")

    # Perform encoding
    encoded_train_set = data_encoding.handle_categorical_encoding(transformed_train_df, "train_data_encoded")
    encoded_test_set = data_encoding.handle_categorical_encoding(transformed_test_df, "test_data_encoded")

    # Perform feature selection
    final_features_train_set = feature_selection.execute_feature_selection(encoded_train_set, 'train_data_feat_selected')
    final_features_test_set = feature_selection.execute_feature_selection(encoded_test_set, 'test_data_feat_selected')

    # Initialize model training
    r_squared = model_trainer.initiate_model_trainer(final_features_train_set, final_features_test_set)
    print(f"Final R-squared value: {r_squared}")
