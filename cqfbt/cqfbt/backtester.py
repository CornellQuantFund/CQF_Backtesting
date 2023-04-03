# CQF Backtester 0.0.2

import time
import yfinance as yf
import datetime as dt
import os
import dateutil.parser
import datetime as dt
import numpy as np
import polars as pl
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

    init_time_start =0
    init_time_end =0

    setup_time_start=0
    setup_time_end =0

    runtime_start =0
    runtime_end =0

    # Initializes sizes of vectors, csv files, etc.
    def __init__(self, ptfl=[], start_date='', end_date='', interval=''):
        print("Initializing engine:")
        self.init_time_start = time.time()
        self.interval = interval
        self.get_info_on_stocks(start_date, end_date, ptfl)
        logging.basicConfig(filename='backtesting_log.log', filemode='w', format='%(levelname)s - %(message)s',
                            level=logging.INFO)
        self.dates = pl.date_range(low=str_to_dt(start_date), high=str_to_dt(end_date), interval=str.lower(interval))
        for i in range(0, len(ptfl)):
            ticker = ptfl[i]
            self.add_data("cqfbt\\data\\"+f"{ticker}_hist.csv")
        self.init_time_end = time.time()
        print("Initialization time: " + str(self.init_time_end-self.init_time_start)+"s")


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
        data = pl.read_csv(path)
        
        data.replace('Date', data.get_column('Date').apply(str_to_dt))

        timestamp_range = pl.date_range(low=data['Date'][0], high=data['Date'][data.shape[0]-1], interval=str.lower(self.interval)) 
        newData = pl.DataFrame({'Date':timestamp_range})        
        newData = newData.join(data, on='Date', how='outer')
        newData = newData.fill_null(-1.0)
        newData.lazy()
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
                    price = data.select([pl.col('Close')])[order.asset].item()
                    portfolio_delta[i, order.asset] += order.quantity
                    while(price <= 0 and id >= idx - 10 and id >= 0):
                        date, data = self.data_at_idx(id)
                        price = data.select([pl.col('Close')])[order.asset].item()
                        id -= 1
                    capital_delta[i] -= order.quantity * price

                elif(~order.buyT_sellF):
                    portfolio_delta[i, order.asset] -= order.quantity
                    price = data.select([pl.col('Close')])[order.asset].item()
                    while(price <= 0 and id >= idx - 10 and id >= 0):
                        date, data = self.data_at_idx(id)
                        price = data.select([pl.col('Close')])[order.asset].item()
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
        for i in range(len(self.arr)):
            idx = 0
            for row in self.arr[i].iter_rows(named=True):
                if i == 0:
                    self.data_arr.append(pl.from_dict(row))
                else:
                    self.data_arr[idx] = self.data_arr[idx].vstack(pl.from_dict(row))
                idx+=1
        

    def reformat_arr(self):
        timestamp_range = pl.date_range(low=self.dates[0], \
                high=self.dates[len(self.dates)-1], \
                    interval=str.lower(self.interval)).to_list()
        
        for j in range(0, len(self.arr)):
            if(self.arr[j].shape[0] != len(timestamp_range)):
                newData = pl.DataFrame({'Date':timestamp_range}) 
                newData = self.arr[j].join(newData, on='Date', how='outer')
                newData = newData.fill_null(-1.0)
                self.arr[j] = newData
        self.make_data()

    def setup_run(self):
        if self.arr != []:
            cols = set()
            for i in range(0, len(self.arr)):
                cols.update(self.arr[i].columns)
                
            for i in range(0, len(self.arr)):
                for col in cols:
                    if col not in self.arr[i].columns:
                        self.arr[i] = self.arr[i].select([pl.all(), pl.lit(-1.0).alias(col)])

            self.reformat_arr()
            self.portfolio_allocations = np.zeros((len(self.strategies), len(self.portfolio_assets)))
            self.portfolio_history = np.ndarray(
                (len(self.strategies), len(self.dates), len(self.portfolio_assets)+2, 1))
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
        self.setup_time_start = time.time()
        self.setup_run()
        self.setup_time_end = time.time()
        print("Setup time: " + str(self.setup_time_end-self.setup_time_start)+"s")

        self.runtime_start = time.time()
        num_errors = 0
        error = False
        print("Running:")
        printer=1
        for i in range(0, len(self.data_arr)):
            if((i+1) % round(len(self.data_arr)/8) == 0):
                #print(str(printer*12.5) + '%')
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
        self.runtime_end = time.time()
        print("Run time: " + str(self.runtime_end-self.runtime_start)+"s")

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
            price = data.select([pl.col('Close')])[i].item()
            while(price < 0 and id >= idx - 10 and id >= 0):
                date, data = self.data_at_idx(id)
                price = data.select([pl.col('Close')])[i].item()
                id -= 1
            if price < 0:
                price = 0
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
    return dateutil.parser.parse(s, ignoretz=True)

