# CQF Backtester 0.0.2

import yfinance as yf
import datetime as dt
import os
import dateutil.parser
import datetime as dt
import numpy as np
import pandas as pd
import seaborn as sns
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
    quantity = 0

    def __init__(self, type, asset_index, quantity,  limit=-1):
        self.buyT_sellF = type
        self.asset = asset_index
        self.limit = limit
        self.quantity = quantity

    def __repr__(self):
        return "asset index is " + str(self.asset) + " this is a buy: " + str(self.buyT_sellF) + " for quantity " + str(self.quantity)


class Engine():
    counter = 0 # Keeps track of where in the time series data we are
    dates = [] # List of datetime objects, from start date to end date separated by interval
    interval = '' # String representing time between data points "1d", "1w", "2h", etc.
    portfolio_assets = [] # List of strings representing the tickers or data filenames we are testing on
    portfolio_allocations = [] # List of lists of rational numbers representing our strategies current holding quantities for each asset
    portfolio_history = [] # List of np arrays, keeps track of our allocations, capital and portfolio value for each strategy
    capital = [] # List of capital values for each strategy

    orders = [] # List of lists, each containing the orders each strategy is hoping to execute
    arr = [] # List of data frames, one for each asset
    data_arr = [] # List of data frames, one for each trading interval, containing all relevent data on that cycle
    arr_length = 0 # 
    strategies = [] # List of strategy objects, each acting independently

    # Initializes sizes of vectors, csv files, etc.
    def __init__(self, ptfl=[], start_date='', end_date='', interval=''):
        self.interval = interval
        self.get_info_on_stocks(start_date, end_date, ptfl)
        logging.basicConfig(filename='backtesting_log.log', filemode='w', format='%(levelname)s - %(message)s',
                            level=logging.INFO)
        self.dates = pd.date_range(start=start_date, end=end_date, freq=str.upper(interval))
        for i in range(0, len(ptfl)):
            ticker = ptfl[i]
            self.add_data("cqfbt\\data\\"+f"{ticker}_hist.csv")


    # Gets yf data on stocks in optional portfolio argument
    # Saves csv files in data folder
    def get_info_on_stocks(self, start, end, ptfl):
        for i in range(0, len(ptfl)):
            yf.Ticker(ptfl[i]).history(start=start, end=end, interval=self.interval).to_csv("cqfbt\\data\\" + f"{ptfl[i]}_hist.csv")

    # TODO: (II) Takes in path to csv data and adds it to the data folder,
    # edits / reformats engine.arr, engine.portfolio_assets and engine.dates
    def add_data(self, path):
        print("adding " + path)
        path_context = path.split('\\')
        self.portfolio_assets.append(path_context[len(path_context)-1].split('.')[0])
        data = pd.read_csv(path)
        for i in range(len(data)):
            data.at[i, 'Date'] = str_to_dt(data.at[i, 'Date']).replace(tzinfo=dt.timezone.utc)
        timestamp_range = pd.date_range(
            start=data['Date'][0], end=data['Date'][data.shape[0]-1], freq=str.upper(self.interval), tz=dt.timezone.utc).tolist()
        newData = pd.DataFrame().reindex_like(data)
        newData = newData.reindex(np.linspace(0, len(timestamp_range)-1, len(timestamp_range), dtype=int))
        didx = 0
        for i in range(len(timestamp_range)):
            newData.at[i, 'Date'] = timestamp_range[i]

            if(newData.iloc[i].loc['Date'] == data.iloc[didx].loc['Date']):
                newData.iloc[i] = data.iloc[didx]
                didx+=1
        newData = newData.fillna(-1.0)
        self.arr.append(newData)
        self.arr_length+=1


    # Adds strategy to list of strategies
    # Callable by user
    def add_strategy(self, strategy, init_capital):
        self.capital.append(init_capital)
        self.orders.append([])
        self.strategies.append(strategy)
        # Exend portfolio_history into 3rd dimension +1

    
    def data_at_idx(self, idx):
        return self.dates[idx], self.data_arr[idx]

    # Updates portfolio and capital according to executed orders
    def update_portfolio(self, portfolio_delta, capital_delta):
        self.capital = np.around(np.add(self.capital, capital_delta), decimals=2)
        self.portfolio_allocations = np.add(
            self.portfolio_allocations, portfolio_delta)

    #   Executes orders at prices found in the data, and returns change in
    #   portfolio and capital as a result. Returns any orders which did not execute
    #   yet due to the limit price.
    def execute_orders(self, data, idx):
        # For now, just use one of the open or close values as the 'true price'
        # at which orders are executed
        outstanding_orders = self.orders
        capital_delta = np.zeros(len(self.strategies))
        portfolio_delta = np.zeros((len(self.strategies), len(self.portfolio_assets)))
        logging.info("Bought X shares of Y, Sold A shares of B, etc")
        for i in range(len(self.strategies)):
            executed_orders = []
            id = idx
            date, data = self.data_at_idx(id)
            for order in outstanding_orders[i]:
                if(order.buyT_sellF):
                    price = data.at[order.asset, 'Close']
                    portfolio_delta[i, order.asset] += order.quantity
                    while(price <= 0 and id >= 0):
                        date, data = self.data_at_idx(id)
                        price = data.at[order.asset, 'Close']
                        id -= 1
                    capital_delta[i] -= order.quantity * price

                elif(~order.buyT_sellF):
                    portfolio_delta[i, order.asset] -= order.quantity
                    price = data.at[order.asset, 'Close']
                    while(price <= 0 and id >= 0):
                        date, data = self.data_at_idx(id)
                        price = data.at[order.asset, 'Close']
                        id -= 1
                    capital_delta[i] += order.quantity * price
                executed_orders.append(order)

            if self.capital[i]+capital_delta[i] < 0:
                message = "Transaction attempted with insufficient funds: strategy " + str(i) + ":" + \
                    str(self.capital[i]) + ' + ' + str(capital_delta[i]) + \
                    ' = ' + str(self.capital[i]+capital_delta[i]) + " < 0"
                logging.warning(message)
                self.orders[i]=[]
                raise InsufficientFundsException()
            
            for order in executed_orders:
                self.orders[i].remove(order)

        return portfolio_delta, capital_delta


    def make_data(self):
        for i in range(len(self.arr[0])):
            data = self.arr[0].iloc[i].to_frame().transpose()
            for j in range(1, len(self.arr)):
                a_data = self.arr[j].iloc[i].to_frame().transpose()
                data = pd.concat(
                    [data, a_data], axis=0, ignore_index=True)
            self.data_arr.append(data)
        

    def reformat_arr(self, max_len_idx):
        timestamp_range = pd.date_range(
            start=self.arr[max_len_idx]['Date'][0], \
                end=self.arr[max_len_idx]['Date'][self.arr[max_len_idx].shape[0]-1], \
                    freq=str.upper(self.interval), tz=dt.timezone.utc).tolist()
        self.dates = timestamp_range
        
        self.arr[max_len_idx] = self.arr[max_len_idx].fillna(-1.0)
        for j in range(0, len(self.arr)):
            self.arr[j].reindex_like(self.arr[max_len_idx])
            if(j != max_len_idx):
                newData = pd.DataFrame().reindex_like(self.arr[max_len_idx])
                newData = newData.reindex(np.linspace(0, len(timestamp_range)-1, len(timestamp_range), dtype=int))
                didx = 0
                for i in range(len(timestamp_range)):
                    newData.at[i, 'Date'] = timestamp_range[i]
                    if(didx < len(self.arr[j]) and newData.at[i, 'Date'] == self.arr[j].at[didx, 'Date']):
                        newData.iloc[i] = self.arr[j].iloc[didx]
                        didx+=1
                newData = newData.fillna(-1.0)
                self.arr[j] = newData
        self.make_data()

    def setup_run(self):
        if self.arr != []:
            max_len = self.arr[0].shape[0]
            max_idx = 0
            for i in range(0, len(self.arr)):
                if(self.arr[i].shape[0] > max_len):
                    max_len = self.arr[i].shape[0]
                    max_idx = i
                if(self.arr[i].shape[1] < 9):
                    self.arr[i]['Capital Gains'] = np.nan
            self.reformat_arr(max_idx)
            self.portfolio_allocations = np.zeros((len(self.strategies), len(self.portfolio_assets)))
            self.portfolio_history = np.ndarray(
                (len(self.strategies), max_len, len(self.portfolio_assets)+2, 1))
        else:
            print('No data to test.\n')


    # Main loop, feeds data to strategy, tries to execute specified orders then
    # returns the results back to the strategy. Repeats for all available data
    def run(self):
        # If there is insufficient funds to execute a transaction, the engine will
        # re-feed the same data and try again with the new logic provided by the
        # strategy, if the strategy does nothing different the engine will quit
        # to avoid an infinite loop
        print("Setting up run for:")
        print(self.portfolio_assets)
        self.setup_run()

        num_errors = 0
        error = False
        print("Running:")
        printer=1
        for i in range(0, len(self.data_arr)):
            if((i+1) % round(len(self.data_arr)/8) == 0):
                print(str(printer*12.5) + '%')
                printer+=1
            for j in range(len(self.strategies)):
                date, data = self.data_at_idx(i)
                self.strategies[j].append_data_history(data)
                # Portfolio History keeps track of asset allocations
                for k in range(0, len(self.portfolio_assets)):
                    self.portfolio_history[j, i, k] = self.portfolio_allocations[j, k]
                # And capital
                self.portfolio_history[j, i, len(
                    self.portfolio_assets)] = self.capital[j]
                # And total portfolio value
                self.portfolio_history[j, i, len(
                    self.portfolio_assets)+1] = self.get_portfolio_cash_value(i, j)

                # Testing purposes
                #print(str(i) + ' ::: ' + str(date) + ' ::: ' +
                #       list_to_str(self.portfolio_history[j, i, :]))

                # for now supports only one   V   strategy
                self.orders[j] = self.strategies[j].execute(
                    date, data, self.portfolio_allocations[j], self.capital[j], self.orders[j], error, self.portfolio_assets)
                error = False
                try:
                    delta_p, delta_c = self.execute_orders(
                        data, i)
                    self.update_portfolio(delta_p, delta_c)
                    num_errors = 0
                except InsufficientFundsException: 
                    i -= 1
                    num_errors += 1
                    error = True
                    if (num_errors >= 2):
                        print("Strategy " + str(j) + ':')
                        raise InsufficientFundsException()
                    else: continue
        self.plot_history()

    # Plot history of portfolio value, summing assets and capital with
    # get_portfolio_cash_value
    def plot_history(self) -> None:
        dates = self.dates
        sns.set_theme()
        for j in range(len(self.strategies)):
            values = np.add(self.portfolio_history[j, :, len(
                self.portfolio_assets)+1], self.portfolio_history[j, :, len(self.portfolio_assets)])

                
            font = {'family' : 'Times New Roman',
                    'weight' : 'bold',
                    'size'   : '18'}
            fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
            ax.set_title("Portfolio History: " + self.strategies[j].get_name(), font=font)
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.plot(dates, values)
            fig.savefig("strategy" + str(j) + "_performance.pdf")

    # Uses data (self.arr) to price the cash value of a portfolio by summing close prices
    # for each asset in the portfolio, capital should be included. There may be
    # no data for the given date, if this is the case use the most recent previous close
    def get_portfolio_cash_value(self, idx, strategy_num) -> float:
        asset_values = 0
        for i in range(len(self.portfolio_allocations[strategy_num])):
            id=idx
            date, data = self.data_at_idx(idx)
            price = data.at[i, 'Close']
            while(price <= 0 and id >= 0):
                date, data = self.data_at_idx(id)
                price = data.at[i, 'Close']
                id -= 1
            price = data.at[i, 'Close']
            asset_values += price*self.portfolio_allocations[strategy_num, i]
        return round(asset_values, 2)

    # Clears all data files in data folder
    def clear_data(self):
        for i in range(0, len(self.portfolio_assets)):
            self.remove_data(self.portfolio_assets[i], i)

    # Removes a specific data file from data folder
    # Can specify a ticker, or if it is user-added data, the index
    # of the asset in the portfolio
    # TODO: (II) remove user-added data files by name
    def remove_data(self, ticker='', assetNo=-1):
        if (assetNo >= 0):
            file = self.portfolio_assets[assetNo]
            path = "cqfbt\\data\\" + f"{file}.csv"
            if(os.path.exists(path)):
                os.remove(path)


class InsufficientFundsException(Exception):
    def __init__(self, message='Insufficient Funds to perform transaction.'):
        super(InsufficientFundsException, self).__init__(message)


# list to string
def list_to_str(lst):
    out = "["
    for i in range(0, len(lst)-1):
        out += str(lst[i]) + ', '
    return out + str(lst[len(lst)-1])+']'

# string to datetime
# TODO (II) add support for UNIX ts and other date formats
def str_to_dt(s: str) -> dt.datetime:
    return dateutil.parser.parse(s)

