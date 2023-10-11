from flask import Flask, request, render_template
import pandas as pd
import numpy as np

import pipelines.predict_pipeline
from pipelines.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

app.debug = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        input_data=pipelines.predict_pipeline.CustomData(
        Order_Item_Product_Price= float(request.form.get('Order_Item_Product_Price')),
        Order_Item_Total= float(request.form.get('Order_Item_Total')),
        Order_Item_Quantity= float(request.form.get('Order_Item_Quantity')),
        Product_Price = float(request.form.get('Product_Price')),
        Order_Item_Discount = float(request.form.get('Order_Item_Discount')),
        Category_Name = request.form.get('Category_Name'),
        Product_Name= request.form.get('Product_Name')
        )

        df_pred = input_data.get_data_as_data_frame()
        print(f"DataFrame Content:\n{df_pred}")

        prediction_pipeline = PredictPipeline()
        print("Initializing Prediction Pipeline")
        results = prediction_pipeline.predict(input_data)
        print("Prediction Complete!")


