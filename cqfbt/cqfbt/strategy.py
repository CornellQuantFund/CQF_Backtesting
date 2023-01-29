import abc


# Abstract class for user defined strategy.
class Strategy(metaclass=abc.ABCMeta):
    def __init__(self):
        return

    @abc.abstractmethod
    def execute(date, data, portfolio_values, capital, orders, error):
        pass
