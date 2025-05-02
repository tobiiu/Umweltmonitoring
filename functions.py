"""all the important funktions"""

import pandas, numpy, matplotlib, plotly, seaborn, dash, \
        sklearn, requests

def get_sensbox_ip():
    """The sense_box_i we choice wisely after analysing all sense_boxes"""
    ip = "abcde"
    return ip

def scrape_sense_box_data(sense_box_ip):
    """get data from the internet (probably as json)"""
    return {"temperatur": 12, "luftfeuchtigkeit": 78, "regen":1}



def transform_sense_box_data(sense_box_data_data):
    """transform the data in a way we can add the data and it is congruent to the rest"""
    if sense_box_data_data is None: 
        new_sense_box_data_data = {}
        return new_sense_box_data_data
    
def plot_data(data:pandas.DataFrame, column: int|str):
    print("plotting column X")

def correlation_between_columns(data:pandas.DataFrame, columns:list):
    print("show correlation")


class Data():
    """stores the Date information from 1 Sensebox as a DataFrame
    'add_data()' can add datapoints
    'all_data()' gives you all data
    """
    def __init__(self)->None:
        self.data = pandas.DataFrame()

    def add_data(self)->None:
        print("Datenhinzufügen")

    def all_data(self) -> pandas.DataFrame:
        return self.data
        

def store_data(new_data):
    """load new data in Data"""
    pass

def all_Data()->pandas.DataFrame:
    """gives you all the data stored in 'Data' !"""
    return data.data



data = Data()