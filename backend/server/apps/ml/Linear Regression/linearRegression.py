import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression


class LinReg:
    def __init__(self):
        self.modelRedWine = joblib.load(
            "../../research/RedWineLinearRegression.joblib")
        self.modelWhiteWine = joblib.load(
            "../../research/WhiteWineLinearRegression.joblib")

    def getDataFrame(self, inputData):
        inputData = pd.DataFrame(inputData, index=[0])
        return inputData

    def predict_red(self, inputData):
        return self.modelRedWine.predict(inputData)

    def predict_white(self, inputData):
        return self.modelWhiteWine.predict(inputData)

    def postprocessing(self, inputData):
        return {"Quality": inputData[1], "status": "OK"}

    def compute_prediction_red(self, inputData):
        try:
            inputData = self.getDataFrame(inputData)
            prediction = self.predict_red(inputData)[0]
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}
        return prediction

    def compute_prediction_white(self, inputData):
        try:
            inputData = self.getDataFrame(inputData)
            prediction = self.predict_white(inputData)[0]
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}
        return prediction
