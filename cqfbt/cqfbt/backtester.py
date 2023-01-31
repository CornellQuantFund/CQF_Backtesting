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
        engine.capital = init_capital
        engine.portfolio = ptfl
        engine.portfolio_values = np.zeros(len(ptfl))
        engine.get_info_on_stock(ptfl, start_date, end_date, interval)

        # for now, we are restrained in the sense that all assets must
        # have the exact same length of data
        # unfortunately this means cryptocurrencies cannot be in the
        # portfolio while regular stocks are, I will work on this for version 0.0.2
        if len(ptfl) != 0:
            length = len(pd.read_csv("cqfbt\\data\\"+ptfl[0]+"_hist.csv"))
            engine.arr_length = length
            engine.arr = np.ndarray((len(ptfl), engine.arr_length, 7))

        idx = 0
        for i in range(0, len(ptfl)):
            ticker = ptfl[i]
            df = pd.read_csv("cqfbt\\data\\"+f"{ticker}_hist.csv")
            engine.arr[idx, :, :] = df.loc[0:engine.arr_length,
                                           'Open':'Stock Splits'].to_numpy()
            if idx == 0:
                engine.dates = df.Date.to_list()
                engine.arr_length = len(engine.dates)
            idx += 1
        engine.portfolio_history = np.zeros(
            (engine.arr_length, len(engine.portfolio)+1))

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
    def set_capital(capital):
        engine.capital = capital

    # Adds strategy to list of strategies
    # Callable by user
    def add_strategy(self, strategy):
        engine.strategies.append(strategy)

    # Returns the next time-slice of data
    # If an error was thrown due to failure to execute orders,
    # the counter is not updated and the same data is returned.
    def next_data(error):
        if (error):
            return engine.dates[engine.counter], engine.arr[:, engine.counter, :]
        else:
            engine.counter += 1
            return engine.dates[engine.counter], engine.arr[:, engine.counter, :]

    # Updates portfolio and capital according to executed orders
    def update_portfolio(portfolio_delta, capital_delta):
        engine.capital += capital_delta
        for i in range(0, len(engine.portfolio)-1):
            engine.portfolio_values[i] += portfolio_delta[i]

    # TODO: Executes orders at prices found in the data, and returns change in
    #   portfolio and capital as a result. Returns any orders which did not execute
    #   yet due to the limit price.
    def execute_orders(data):
        #   For now, just use one of the open or close values as the 'true price'
        #   at which orders are executed
        # Since we do not have access to live data, we may in the future approximate
        # the 'true price' with a gaussian centered between the high and low with a
        # standard deviation derived from other data
        outstanding_orders = None
        capital_delta = 0
        portfolio_delta = np.zeros(len(engine.portfolio))

        if engine.capital+capital_delta < 0:
            raise InsufficientFundsException()
        return outstanding_orders, portfolio_delta, capital_delta

    # Main loop, feeds data to strategy, tries to execute specified orders then
    # returns the results back to the strategy. Repeats for all available data
    def run(self):
        error = False
        for i in range(0, engine.arr_length-1):
            for j in range(0, len(engine.portfolio_values-1)):
                engine.portfolio_history[i, j] = engine.portfolio_values[j]
            engine.portfolio_history[i, len(
                engine.portfolio_values)] = engine.capital
            date, data = engine.next_data(error)
            print(str(i) + ' ' + date + ' ::: ' +
                  list_to_str(engine.portfolio_history[i, :]))
            # for now supports one strategy   V
            engine.orders = engine.strategies[0].execute(
                date, data, engine.portfolio_values, engine.capital, engine.orders, error)
            error = False
            # Add logging of transactions for 0.0.2
            try:
                outstanding_orders, delta_p, delta_c = engine.execute_orders(
                    data)
                engine.update_portfolio(delta_p, delta_c)
                engine.orders = outstanding_orders
            except InsufficientFundsException:
                i -= 1
                error = True
        engine.plot_history()

    # Plots history of portfolio value, summing assets and capital
    def plot_history():
        print("\n ------------ Plotting History ------------ \n\n")

    # Uses data to price the cash value of a portfolio by summing bid prices
    # for each asset in the portfolio, capital should be included.
    def get_portfolio_cash_value(data):
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
