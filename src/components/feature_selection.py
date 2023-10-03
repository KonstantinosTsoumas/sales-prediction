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

