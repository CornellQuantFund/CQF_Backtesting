import abc
import numpy as np
import pandas as pd

# Abstract class for user defined strategy.


"""
The Strategy Class used within the Execution class. Virtual Class 

Parameters
----------
name: str
    Creates the Name of the strategy class

Virtual Method:
Abstract Method:
"""
class Strategy(metaclass=abc.ABCMeta):
    data_history = []
    # list of dataframes, each entry is the data at specified date date
    name = ''

    def __init__(self, name=''):
        self.name = name
        return

    def get_name(self):
        return self.name

    def append_data_history(self, data):
        self.data_history.append(data)

    @abc.abstractmethod
    def execute(date, data, portfolio_values, capital, orders, error):
        pass
