import abc
import numpy as np

# Abstract class for user defined strategy.


class Strategy(metaclass=abc.ABCMeta):
    data_history = [[]]
    # list of lists, each entry is a list of the form
    # [date, np_array], where the np array is the data at that date

    def __init__(self):
        return

    def init_data_shape(self, data):
        self.data_history[0] = list(data)

    def append_data_history(self, data):
        self.data_history = self.data_history + list(data)

    @abc.abstractmethod
    def execute(date, data, portfolio_values, capital, orders, error):
        pass
