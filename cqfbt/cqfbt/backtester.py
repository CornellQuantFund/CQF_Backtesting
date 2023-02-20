# CQF Backtester 0.0.1

import yfinance as yf
import datetime as dt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
import logging

# Design a class to represent an order, should be labeled with buy/sell, which 
# asset is being traded, is it a limit or market order? at what price is 
# the limit? etc
class Order():
    buyT_sellF = None
    asset = -1
    limit = -1
    quantity = -1
    def __init__(self, type, asset_index, limit, quantity):
        self.buyT_sellF = type
        self.asset = asset_index
        self.limit = limit
        self.quantity = quantity


class Engine():
    counter = 0
    dates = []
    portfolio_assets = []
    portfolio_allocations = []
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
        self.portfolio_assets = ptfl
        self.portfolio_allocations = np.zeros((len(ptfl), 1))
        self.get_info_on_stocks(start_date, end_date, interval)
        logging.basicConfig(filename='strategy_log.log', filemode='w', format='%(levelname)s - %(message)s',
                            level=logging.INFO)

        # for now, we are restrained in the sense that all assets must
        # have the exact same length of data
        # unfortunately this means cryptocurrencies cannot be in the
        # portfolio while regular stocks are
        if len(ptfl) != 0:
            length = len(pd.read_csv("cqfbt\\data\\"+ptfl[0]+"_hist.csv"))
            self.arr_length = length
            # This 7 is hard coding for the structure of the yf data
            self.arr = np.zeros((self.arr_length, 7, len(ptfl)))

        idx = 0
        for i in range(0, len(ptfl)):
            ticker = ptfl[i]
            df = pd.read_csv("cqfbt\\data\\"+f"{ticker}_hist.csv")
            self.arr[:, :, i] = df.loc[0:self.arr_length,
                                       'Open':'Stock Splits'].to_numpy()
            if idx == 0:
                self.dates = list(map(str_to_dt, df.Date.to_list()))
                self.arr_length = len(self.dates)
            idx = 1

        self.portfolio_history = np.ndarray(
            (self.arr_length, len(self.portfolio_assets)+2, 1))

    # Gets yf data on stocks in optional portfolio argument
    # Saves csv files in data folder
    def get_info_on_stocks(self, start, end, interval):
        for i in range(0, len(self.portfolio_assets)):
            ticker = self.portfolio_assets[i]
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start, end=end, interval=interval)
            hist.to_csv("cqfbt\\data\\" + f"{ticker}_hist.csv")

    # TODO: (II) Takes in path to csv data and adds it to the data folder,
    # edits / reformats engine.arr, engine.portfolio_assets and engine.dates
    def add_data(self, path):
        pass

    # Sets capital to desired amount
    def set_capital(self, capital):
        self.capital = capital

    # Adds strategy to list of strategies
    # Callable by user
    def add_strategy(self, strategy):
        self.strategies.append(strategy)
        # Exend portfolio_history into 3rd dimension +1

    # Returns the next time-slice of data
    # If an error was thrown due to failure to execute orders,
    # the counter is not updated and the same data is returned.
    def next_data(self, error):
        if (error):
            return self.dates[self.counter], self.arr[self.counter, :, :]
        else:
            self.counter += 1
            return self.dates[self.counter], self.arr[self.counter, :, :]

    # Updates portfolio and capital according to executed orders
    def update_portfolio(self, portfolio_delta, capital_delta):
        self.capital = round(self.capital + capital_delta, 2)
        for i in range(0, len(self.portfolio_assets)):
            self.portfolio_allocations[i, 0] = round(
                self.portfolio_allocations[i, 0] + portfolio_delta[i, 0], 2)

    # TODO: Group 1
    #   Executes orders at prices found in the data, and returns change in
    #   portfolio and capital as a result. Returns any orders which did not execute
    #   yet due to the limit price.
    def execute_orders(self, data):
        # For now, just use one of the open or close values as the 'true price'
        # at which orders are executed
        outstanding_orders = self.orders
        capital_delta = 0
        portfolio_delta = np.zeros((len(self.portfolio_assets), 1))
        logging.info("Bought X shares of Y, Sold A shares of B, etc")
        
        if self.capital+capital_delta < 0:
            message = "Transaction attempted with insufficient funds: " + str(self.capital) + ' + ' + str(capital_delta) + ' = ' + str(self.capital+capital_delta) + " < 0"
            logging.warning(message)
            raise InsufficientFundsException()
        return outstanding_orders, portfolio_delta, capital_delta

    # Main loop, feeds data to strategy, tries to execute specified orders then
    # returns the results back to the strategy. Repeats for all available data
    def run(self):
        # If there is insufficient funds to execute a transaction, the engine will 
        # re-feed the same data and try again with the new logic provided by the
        # strategy, if the strategy does nothing different the engine will quit 
        # to avoid an infinite loop
        num_errors = 0
        if np.shape(self.arr != (0,)):
            error = False
            for strategy in self.strategies:
                strategy.init_data_shape(self.next_data(True))
            self.counter = -1
            for i in range(0, self.arr_length-1):
                date, data = self.next_data(error)
                strategy.append_data_history(data)
                # Portfolio History keeps track of asset allocations
                for j in range(0, len(self.portfolio_assets)):
                    self.portfolio_history[i, j,
                                           0] = self.portfolio_allocations[j, 0]
                # And capital
                self.portfolio_history[i, len(
                    self.portfolio_assets), 0] = self.capital
                # And total portfolio value
                self.portfolio_history[i, len(
                    self.portfolio_assets)+1, 0] = self.get_portfolio_cash_value(date)
                
                # Testing purposes
                print(str(i) + ' ::: ' + str(date) + ' ::: ' +
                      list_to_str(self.portfolio_history[i, :, 0]))
                
                # for now supports only one   V   strategy
                self.orders = self.strategies[0].execute(
                    date, data, self.portfolio_allocations, self.capital, self.orders, error)
                error = False
                try:
                    outstanding_orders, delta_p, delta_c = self.execute_orders(
                        data)
                    self.update_portfolio(delta_p, delta_c)
                    self.orders = outstanding_orders
                    num_errors = 0
                except InsufficientFundsException:
                    i -= 1
                    num_errors += 1
                    if (num_errors >= 2):
                        raise InsufficientFundsException()
                    error = True
            self.plot_history()
        else:
            print('No data to test.\n')

    # TODO: Group 2
    # Plot history of portfolio value, summing assets and capital with
    # get_portfolio_cash_value
    def plot_history(self):
        print("\n ------------ Plotting History ------------ \n\n")

    # TODO: Group 3
    # Uses data (self.arr) to price the cash value of a portfolio by summing bid prices
    # for each asset in the portfolio, capital should be included. There may be
    # no data for the given date, if this is the case use the most recent previous bid
    def get_portfolio_cash_value(self, date: dt.datetime) -> float:
        data_idx = 3 # use close price
        date_idx = self.dates.index(date)
        assets_value = np.dot(self.portfolio_allocations[:, 0], self.arr[date_idx][data_idx])
        return self.capital + assets_value

    # Clears all data files in data folder
    def clear_data(self):
        for i in range(0, len(self.portfolio_assets)):
            self.remove_data(self.portfolio_assets[i])

    # Removes a specific data file from data folder
    # Can specify a ticker, or if it is user-added data, the index
    # of the asset in the portfolio
    # TODO: (II) remove user-added data files by name
    def remove_data(self, ticker='', assetNo=-1):
        if (assetNo >= 0):
            ticker = self.portfolio_assets[assetNo]
        os.remove("cqfbt\\data\\" + f"{ticker}_hist.csv")


class InsufficientFundsException(Exception):
    def __init__(self, message='Insufficient Funds to perform this transaction.'):
        super(InsufficientFundsException, self).__init__(message)


# int / float list to string
def list_to_str(lst):
    out = "["
    for i in range(0, len(lst)-1):
        out += str(lst[i]) + ', '
    return out + str(lst[len(lst)-1])+']'

# string to datetime
# TODO (II) add support for UNIX ts and other date formats
def str_to_dt(s: str) -> dt.datetime:
    date = s.split(' ')[0].split('-')
    time = s.split(' ')[1].split('-')[0].split(':')
    return dt.datetime(int(date[0]), int(date[1]), int(date[2]), int(time[0]), int(time[1]), int(time[2]))
