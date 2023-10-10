import os
import sys
from dataclasses import dataclass

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, save_best_params
from config import ARTIFACTS_DIR

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(ARTIFACTS_DIR, "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_df, test_df):
        """
        This function initialize the model training process.

        Args:
            train_df: df, The training dataset.
            test_df: df, The test dataset.

        Raises:
            CustomException: If any step in the process fails.
        """
        try:
            if train_df is None or test_df is None:
                raise CustomException("BEWARE! Train set or test set is of a None type.")

            logging.info("Split training and test input data")

            # Fetch target and feature cols
            target_column = 'Sales'
            feature_columns = [col for col in train_df.columns if col != target_column]

            X_train = train_df[feature_columns]
            y_train = train_df[target_column].iloc[:, 0]
            X_test = test_df[feature_columns]
            y_test = test_df[target_column].iloc[:, 0]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['friedman_mse'],
                    'splitter': ['best'],
                    'max_depth': [None, 10]
                },
                "Random Forest": {
                    'n_estimators': [50, 100],
                    'criterion': ['friedman_mse'],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1],
                    'subsample': [0.9]
                },
                "XGBRegressor": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1],
                    'max_depth': [3]
                },
                "CatBoosting Regressor": {
                    'iterations': [500],
                    'learning_rate': [0.1]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50],
                    'learning_rate': [0.1]
                }
            }

            # Evaluate the models
            evaluation_report, best_model, best_params = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)
            save_best_params(best_params)
            logging.info(f"Model evaluation report: {evaluation_report}")

            # Get the best model
            best_model_score = max([val['R2'] for val in evaluation_report.values()])
            best_model_name = [k for k, v in evaluation_report.items() if v['R2'] == best_model_score][0]

            if best_model_score < 0.6:
                raise CustomException("You don't have a good model running")
            logging.info(f"The search for the best model is completed.")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            if best_model:
                predicted = best_model.predict(X_test)
                r_squared = r2_score(y_test, predicted)
            return r_squared

        except Exception as e:
            raise CustomException(e, sys)

