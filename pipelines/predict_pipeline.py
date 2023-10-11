import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformation
from src.components.data_encoding import DataEncoding
from src.components.feature_selection import FeatureSelection
from sklearn.preprocessing import LabelEncoder

class CustomData:
    def __init__(self,
                 Order_Item_Product_Price: float,
                 Order_Item_Total: float,
                 Order_Item_Quantity: float,
                 Product_Price: float,
                 Order_Item_Discount: float,
                 Category_Name: str,
                 Product_Name: str,
                 ):

        self.Order_Item_Product_Price = Order_Item_Product_Price
        self.Order_Item_Total = Order_Item_Total
        self.Order_Item_Quantity = Order_Item_Quantity
        self.Product_Price = Product_Price
        self.Order_Item_Discount = Order_Item_Discount
        self.Category_Name = Category_Name
        self.Product_Name = Product_Name

    def get_data_as_data_frame(self):
        try:
            dict_input_custom_data = {
                "Product Price": [self.Product_Price],
                "Category Name": [self.Category_Name],
                "Order Item Discount": [self.Order_Item_Discount],
                "Order Item Product Price": [self.Order_Item_Product_Price],
                "Product Name": [self.Product_Name],
                "Order Item Total": [self.Order_Item_Total],
                "Order Item Quantity": [self.Order_Item_Quantity],
            }

            return pd.DataFrame(dict_input_custom_data)

        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    def __init__(self):
        self.data_transformation = DataTransformation()
        self.data_encoding = DataEncoding(unique_value_threshold=1000)
        self.feature_selection = FeatureSelection()

    def predict(self, custom_data: CustomData):
        """
        This function performs prediction using a trained model and preprocessor.

        Args:
            custom_data: The instance of 'CustomData' containing the features for prediction.

        Returns:
            preds: array, The predictions from the model.
        """
        try:
            features_df = custom_data.get_data_as_data_frame()

            # Load model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            print("Loading the model...")
            model = load_object(file_path=model_path)
            print("Model loaded successfully.")

            # Perform encoding on the input
            label_encoder = LabelEncoder()
            for col in features_df.select_dtypes(include=["object"]).columns:
                features_df[col] = label_encoder.fit_transform(features_df[col])

            print("Performing predictions...")

            # Make Prediction
            preds = model.predict(features_df)

            print("Predictions completed.")

            return preds

        except Exception as e:
            raise CustomException(e)
