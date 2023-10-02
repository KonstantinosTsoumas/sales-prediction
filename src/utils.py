import os
import sys

import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException

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
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the specified file
        with open(file_path, "wb") as output_file:
            pickle.dump(obj, output_file)

    except Exception as e:
        # Raise a custom exception if anything goes wrong
        raise CustomException(e, sys)