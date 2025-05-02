"""all the important functions"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression


#####################################################################
# Globale Dateninstanz
class Data:
    def __init__(self):
        self.data = pd.DataFrame()

    def add_data(self, new_data: dict):
        new_data['timestamp'] = datetime.now()
        df = pd.DataFrame([new_data])
        self.data = pd.concat([self.data, df], ignore_index=True)

    def all_data(self)-> pd.DataFrame:
        return self.data

data = Data()
######################################################################    


# ==== ETL ====
# ==== E ====
def get_sensbox_ip():
    """The sense_box_i we choice wisely after analysing all sense_boxes"""
    ip = "abcde"
    return ip

def scrape_sense_box_data(sense_box_ip):
    """get data from the internet (probably as json)"""
    return {"temperature": (12,24,4,24,2,4,24), "luftfeuchtigkeit": (1,4,44,244,52,54,5), "regen":(1,1,4,2,5,5,5)}

# ==== T ====
def transform_sense_box_data(sense_box_data_data: dict)->pd.DataFrame:
    """transform the data in a DataFrame as well changing columns"""
    if sense_box_data_data is None: 
        new_sense_box_data_data = {}
        return new_sense_box_data_data
    
# ==== L ====
def store_data(new_data):
    """load new data in Data"""
    pass

def all_Data()->pd.DataFrame:
    """gives you all the data stored in 'Data' !"""
    return data.all_data()







# ==== Analyse ====
def plot_data(data: pd.DataFrame, column=None):
    print("plotting...")
    if column and column in data.columns:
        data[column].plot(title=f"Plot of {column}")
    else:
        print("No valid column provided for plotting")


def correlation_between_columns(data:pd.DataFrame, columns:list):
    print("show correlation")








# ==== ML ====
def train_model(df: pd.DataFrame):
    if 'temperature' in df and 'humidity' in df:
        X = df[['humidity']].values
        y = df['temperature'].values
        model = LinearRegression().fit(X, y)
        return model
    else:
        print("Required columns missing")
        return None

def predict(model, humidity_value):
    return model.predict([[humidity_value]])[0] if model else None