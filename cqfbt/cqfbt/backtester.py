# CQF Backtester 0.0.1

import yfinance as yf
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *


class engine:
    counter = 0
    dates = []
    portfolio = []
    portfolio_values = []
    portfolio_history = []
    capital = 0
    orders = None
    arr = []
    arr_length = 0
    strategies = []

    # Initializes sizes of vectors, csv files, etc, this function is kind of
    # fragile right now
    def __init__(self, ptfl=[], start_date='', end_date='', interval='', init_capital=5e4):
        self.capital = init_capital
        self.portfolio = ptfl
        self.portfolio_values = np.zeros(len(ptfl))
        engine.get_info_on_stock(ptfl, start_date, end_date, interval)

        # for now, we are restrained in the sense that all assets must
        # have the exact same length of data
        # unfortunately this means cryptocurrencies cannot be in the
        # portfolio while regular stocks are, I will work on this for version 0.0.2
        if len(ptfl) != 0:
            length = len(pd.read_csv("cqfbt\\data\\"+ptfl[0]+"_hist.csv"))
            self.arr_length = length
            self.arr = np.ndarray((len(ptfl), self.arr_length, 7))

        idx = 0
        for i in range(0, len(ptfl)):
            ticker = ptfl[i]
            df = pd.read_csv("cqfbt\\data\\"+f"{ticker}_hist.csv")
            self.arr[idx, :, :] = df.loc[0:self.arr_length,
                                         'Open':'Stock Splits'].to_numpy()
            if idx == 0:
                self.dates = df.Date.to_list()
                self.arr_length = len(self.dates)
            idx += 1
        self.portfolio_history = np.zeros(
            (self.arr_length, len(self.portfolio)+1))

    # Gets yf data on stocks in optional portfolio argument
    # Saves csv files in data folder
    def get_info_on_stock(ptfl, start, end, interval):
        for i in range(0, len(ptfl)):
            ticker = ptfl[i]
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start, end=end, interval=interval)
            hist.to_csv("cqfbt\\data\\" + f"{ticker}_hist.csv")

    # Takes in path to csv data and adds it to the data folder,
    # edits engine.arr, engine.portfolio and engine.dates
    def add_data(self, path):
        pass

    # Sets capital to desired amount
    def set_capital(self, capital):
        self.capital = capital

    # Adds strategy to list of strategies
    # Callable by user
    def add_strategy(self, strategy):
        self.strategies.append(strategy)

    # Returns the next time-slice of data
    # If an error was thrown due to failure to execute orders,
    # the counter is not updated and the same data is returned.
    def next_data(self, error):
        if (error):
            return self.dates[self.counter], self.arr[:, self.counter, :]
        else:
            self.counter += 1
            return self.dates[self.counter], self.arr[:, self.counter, :]

    # Updates portfolio and capital according to executed orders
    def update_portfolio(self, portfolio_delta, capital_delta):
        self.capital += capital_delta
        for i in range(0, len(self.portfolio)-1):
            self.portfolio_values[i] += portfolio_delta[i]

    # TODO: Executes orders at prices found in the data, and returns change in
    #   portfolio and capital as a result. Returns any orders which did not execute
    #   yet due to the limit price.
    def execute_orders(self, data):
        #   For now, just use one of the open or close values as the 'true price'
        #   at which orders are executed
        # Since we do not have access to live data, we may in the future approximate
        # the 'true price' with a gaussian centered between the high and low with a
        # standard deviation derived from other data
        outstanding_orders = None
        capital_delta = 0
        portfolio_delta = np.zeros(len(self.portfolio))

        if self.capital+capital_delta < 0:
            raise InsufficientFundsException()
        return outstanding_orders, portfolio_delta, capital_delta

    # Main loop, feeds data to strategy, tries to execute specified orders then
    # returns the results back to the strategy. Repeats for all available data
    def run(self):
        error = False
        for i in range(0, self.arr_length-1):
            for j in range(0, len(self.portfolio_values-1)):
                self.portfolio_history[i, j] = self.portfolio_values[j]
            self.portfolio_history[i, len(
                self.portfolio_values)] = self.capital
            date, data = self.next_data(error)
            print(str(i) + ' ' + date + ' ::: ' +
                  list_to_str(self.portfolio_history[i, :]))
            # for now supports one strategy   V
            self.orders = self.strategies[0].execute(
                date, data, self.portfolio_values, self.capital, self.orders, error)
            error = False
            # Add logging of transactions for 0.0.2
            try:
                outstanding_orders, delta_p, delta_c = self.execute_orders(
                    data)
                self.update_portfolio(delta_p, delta_c)
                self.orders = outstanding_orders
            except InsufficientFundsException:
                i -= 1
                error = True
        self.plot_history()

    # Plots history of portfolio value, summing assets and capital
    def plot_history(self):
        print("\n ------------ Plotting History ------------ \n\n")

    # Uses data to price the cash value of a portfolio by summing bid prices
    # for each asset in the portfolio, capital should be included.
    def get_portfolio_cash_value(self, data):
        pass


class InsufficientFundsException(Exception):
    def __init__(self, message='Insufficient Funds to perform this transaction.'):
        super(InsufficientFundsException, self).__init__(message)


# int / float list to string
def list_to_str(lst):
    out = "["
    for i in range(0, len(lst)-2):
        out += str(lst[i]) + ', '
    return out + str(lst[len(lst)-1])+']'
