import abc
import numpy as np

# Abstract class for user defined strategy.


class Strategy(metaclass=abc.ABCMeta):
    data_history = []

    def __init__(self):
        return

    def init_data_shape(self, data):
        self.data_history = np.zeros(np.shape(data)+(1,))

    def append_data_history(self, data):
        self.data_history = np.append(self.data_history, data)

    @abc.abstractmethod
    def execute(date, data, portfolio_values, capital, orders, error):
        pass
