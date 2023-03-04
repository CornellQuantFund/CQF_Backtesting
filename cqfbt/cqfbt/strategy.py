import abc
import numpy as np
import pandas as pd

# Abstract class for user defined strategy.


class Strategy(metaclass=abc.ABCMeta):
    data_history = []
    # list of dataframes, each entry is the data at specified date date

    def __init__(self):
        return

    def append_data_history(self, data):
        self.data_history.append(data)

    @abc.abstractmethod
    def execute(date, data, portfolio_values, capital, orders, error):
        pass
