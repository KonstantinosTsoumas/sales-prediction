import os
import sys

import numpy as np
import pandas as pd
import json
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_log_error
from src.exception import CustomException
from config import ARTIFACTS_DIR

def save_object(file_path: str, obj: object) -> None:
    """
    This function saves a Python object to a file using pickle.

    Args:
        file_path: str, the path where the object should be saved.
        obj: object, the Python object to save.

    Raises:
        CustomException: Custom exception that wraps any thrown exceptions.
    """
    try:
        # Get the directory path and create it if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(
            os.path.dirname(dir_path), exist_ok=True
        )
        # Save the object to the specified file
        with open(file_path, "wb") as output_file:
            pickle.dump(obj, output_file)

    except Exception as e:
        # Raise a custom exception if anything goes wrong
        raise CustomException(e, sys)

def load_object(file_path: str) -> object:
    """
    This function loads a Python object from a file using pickle.
    Args:
        file_path: str, the path where the object is saved.
    Returns:
        object: the loaded Python object.
    Raises:
        CustomException: A custom exception for any issues that arise.
    """
    try:
        with open(file_path, "rb") as file_obj:
            loaded_obj = pickle.load(file_obj)
        return loaded_obj
    except Exception as e:
        raise CustomException(e, sys)


def save_best_params(best_params: dict, file_name: str = "best_params.json") -> None:
    """
    This fuctnion saves best parameters to a JSON file.
    Args:
        best_params: dict, the best parameters to be saved.
        file_name: str, the name of the JSON file to save to. Defaults to "best_params.json".
    Side Effects:
        Writes the best parameters to a JSON file.
    """
    try:
        with open(file_name, "w") as f:
            json.dump(best_params, f)
    except Exception as e:
        raise CustomException(f"Failed to save best parameters: {e}", sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    This function evaluate various machine learning models using R2 score and RMSLE.

    Parameters:
        X_train: df, the features for the training set.
        y_train: df, the target for the training set.
        X_test: df, the features for the test set.
        y_test: df, the target for the test set.
        models: dict, the dictionary containing the machine learning models to evaluate.
        param: dict, the dictionary containing the hyperparameters for the models.

    Returns:
        dict: The full report containing the R2 score and RMSLE of each model on the test set.

    Raises:
        CustomException: If any exception occurs during model evaluation.
    """
    try:
        report = {}
        best_params = {}

        # Loop over the models for hyperparameter tuning
        for model_name in models.keys():
            model = models[model_name]
            model_params = param[model_name]

            print(f"The model name is : {model_name}")
            # Perform grid search
            gs = GridSearchCV(model, model_params, cv=5, error_score='raise', n_jobs=-1)
            gs.fit(X_train, y_train)

            print(f"The y train is {y_train}")
            # Fetch the best model and its parameters
            best_model = gs.best_estimator_
            best_params[model_name] = gs.best_params_

            # Save the best model
            save_object(f"{ARTIFACTS_DIR}best_model.pkl", best_model)

            # Make predictions on test set
            y_test_pred = best_model.predict(X_test)

            # Calculate R2 Score, RMSLE
            test_model_score = r2_score(y_test, y_test_pred)
            rmsle_test = np.sqrt(mean_squared_log_error(y_test, np.abs(y_test_pred)))

            # Save to the report dict
            report[model_name] = {
                'R2': test_model_score,
                'RMSLE': rmsle_test
            }

        return report, best_model, best_params

    except Exception as e:
        raise CustomException(e, sys)
