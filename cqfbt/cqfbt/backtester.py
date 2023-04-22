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
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
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
        """ 
        Initialize an Order.
    
        Parameters
        ------------
            type: bool
                True-Buy :: False-Sell
            asset_index: int
                The index of the asset being traded in the portfolio, should be 
                in range(0, len(portfolio_assets)).
            quantity: float
                Quantity of asset looking to be traded. Supports fractional 
                orders for all assets, to limit this functionality add logic
                within your strategy.
            limit: float (Optional)
                Limit price at which to excecute this order 
                default is -1, or no limit.
        """
        self.buyT_sellF = type
        self.asset = asset_index
        self.limit = limit
        self.quantity = quantity

    def __repr__(self):
        out = ''
        if self.buyT_sellF:
            out = out + 'Buy '
        else:
            out = out + 'Sell '
        out = out + str(self.quantity) + ' Asset ' + str(self.asset)

        return out







class Engine():
    dates = [] # List of datetime objects, from start date to end date separated by interval
    interval = '' # String representing time between data points "1d", "1w", "2h", etc.
    date_interval = None # Tuple of int representing num intervals, and string representing interval, '(1, "h")'
    portfolio_assets = [] # List of strings representing the tickers or data filenames we are testing on
    portfolio_allocations = [] # List of lists of rational numbers representing our strategies current holding quantities for each asset
    portfolio_history = [] # List of np arrays, keeps track of our allocations, capital and portfolio value for each strategy
    strategy_capital = [] # List of initial capital values for each strategy
    capital = [] # List of capital values for each strategy
    MRC_prices = [] # List of most recent close prices for eachh asset in portfolio
    transaction_costs = []

    orders = [] # List of lists, each containing the orders each strategy is hoping to execute
    arr = [] # List of data frames, one for each asset

    market = ('SPY', -1) # Ticker used as market reference, index in asset portfolio. default SPY
    market_arr = pl.DataFrame() # Data frame for market data, used as benchmark in sharpe
    data_arr = [] # List of data frames, one for each trading interval, containing all relevent data on that cycle
    arr_length = 0 # 
    strategies = [] # List of strategy objects, each acting independently

    init_time_start =0
    init_time_end =0

    setup_time_start=0
    setup_time_end =0

    runtime_start =0
    runtime_end =0

    setup_required = True

    # Initializes sizes of vectors, csv files, etc.
    def __init__(self, start_date, end_date, interval, ptfl=[]):
        """ 
        Initialize an Engine
    
        Parameters
        ------------
            ptfl: list (Optional)
                List of yfinance tickers for which to pull data.
            start_date: string
                first day for which to pull data and test strategies.
            end_date: string 
                Last day for which to pull data and backtest.
            interval: string
                Periodicity of data between start and end date, if ptfl is
                nonempty, should be one of “1m”, “2m”, “5m”, “15m”, “30m”, 
                “60m”, “90m”, “1h”, “1d”, “5d”, “1wk”, “1mo”, “3mo”.
        """
        print("Initializing engine:")
        self.init_time_start = time.time()
        self.interval = interval
        num = "".join([char for char in self.interval if char.isnumeric()])
        intvl = "".join([char for char in self.interval if char.isalpha()])
        if intvl.lower() == 'wk':
            intvl = 'w'
        self.date_interval = (intvl, num)
        self.get_info_on_stocks(start_date, end_date, ptfl)
        logging.basicConfig(filename='backtesting_log.log', filemode='w', format='%(levelname)s - %(message)s',
                            level=logging.INFO)
        self.dates = pl.date_range(low=self.str_to_dt(start_date), high=self.str_to_dt(end_date), interval=str(num)+str.lower(intvl))
        for i in range(0, len(ptfl)):
            ticker = ptfl[i]
            if ticker == 'SPX' or ticker == 'DJI' or  ticker =='NDAQ':
                self.market = (ticker, i)
                print(ticker)
            self.add_data("cqfbt\\data\\"+f"{ticker}_hist.csv")
        
        self.init_time_end = time.time()
        print("Initialization time: " + str(self.init_time_end-self.init_time_start)+"s")




    # Gets yf data on stocks in optional portfolio argument
    # Saves csv files in data folder
    def get_info_on_stocks(self, start, end, ptfl):
        for i in range(0, len(ptfl)):
            if(~os.path.isfile("cqfbt\\data\\" + f"{ptfl[i]}_hist.csv")):
                yf.Ticker(ptfl[i]).history(start=start, end=end, interval=self.interval).to_csv("cqfbt\\data\\" + f"{ptfl[i]}_hist.csv")




    # TODO: (II) Takes in path to csv data and adds it to the data folder,
    # edits / reformats engine.arr, engine.portfolio_assets and engine.dates
    def add_data(self, path):
        """ 
        Add data file to the Engine.
    
        Parameters
        ------------
            path: string
                The path to the data file on your filesystem.
        """
        self.setup_required = True
        print("adding " + path)
        path_context = path.split('\\')
        self.portfolio_assets.append(path_context[len(path_context)-1].split('.')[0])
        data = pl.read_csv(path)
        
        try:
            data = data.rename({'Datetime': 'Date'})
        except:
            pass
            
        data.replace('Date', data.get_column('Date').apply(self.str_to_dt))

        timestamp_range = pl.date_range(low=data['Date'][0], high=data['Date'][data.shape[0]-1], interval=str(self.date_interval[1])+str.lower(self.date_interval[0])) 
        newData = pl.DataFrame({'Date':timestamp_range})        
        newData = newData.join(data, on='Date', how='outer')
        newData = newData.fill_null(-1.0)
        newData.lazy()
        self.arr.append(newData)
        self.arr_length+=1


    # Adds strategy to list of strategies
    # Callable by user
    def add_strategy(self, strategy, init_capital):
        """ 
        Add a strategy object to the Engine, for its performance to be 
        evaluated.
    
        Parameters
        ------------
            strategy: Strategy
                Your strategy object.
            init_capital: float
                The initial amount of liquid cash you are affording to your
                strategy.
    """
        self.strategy_capital.append(init_capital)
        self.capital.append(init_capital)
        self.orders.append([])
        self.strategies.append(strategy)
        if(len(self.portfolio_history)>0):
            self.portfolio_history.append(np.zeros_like(self.portfolio_history[0]))

    
    def data_at_idx(self, idx):
        return self.dates[idx], self.data_arr[idx]

    # Updates portfolio and capital according to executed orders
    def update_portfolio(self, portfolio_delta, capital_delta):
        for i in range(len(self.strategies)):
            self.capital[i] = np.around(self.capital[i]+capital_delta[i], decimals = 2)
            self.portfolio_allocations[i] = np.add(
                self.portfolio_allocations[i], portfolio_delta[i])

    #   Executes orders at prices found in the data, and returns change in
    #   portfolio and capital as a result. Returns any orders which did not execute
    #   yet due to the limit price.
    def execute_orders(self, data, idx):
        # For now, just use one of the open or close values as the 'true price'
        # at which orders are executed
        outstanding_orders = self.orders
        capital_delta = np.zeros(len(self.strategies))
        portfolio_delta = np.zeros((len(self.strategies), len(self.portfolio_assets)))
        date, data = self.data_at_idx(idx)
        logging.info("::: " + str(date) + " :::")
        for i in range(len(self.strategies)):
            executed_orders = []
            for order in outstanding_orders[i]:
                if(order.buyT_sellF):
                    price = data.select([pl.col('Close')])[order.asset].item()
                    portfolio_delta[i, order.asset] += order.quantity
                    if price < 0:
                        price = self.MRC_prices[order.asset]
                    else:
                        self.MRC_prices[order.asset] = price
                    capital_delta[i] -= order.quantity * price
                    capital_delta[i] -= order.quantity * price * self.transaction_costs[order.asset]
                    logging.info(self.strategies[i].name + " Bought " + str(order.quantity) +
                                 " shares of " + self.portfolio_assets[order.asset] + " at " + str(price))

                elif(~order.buyT_sellF):
                    portfolio_delta[i, order.asset] -= order.quantity
                    price = data.select([pl.col('Close')])[order.asset].item()
                    if price < 0:
                        price = self.MRC_prices[order.asset]
                    else:
                        self.MRC_prices[order.asset] = price
                    capital_delta[i] += order.quantity * price 
                    capital_delta[i] -= order.quantity * price * self.transaction_costs[order.asset]
                    logging.info(self.strategies[i].name + " Sold " + str(order.quantity) +
                                 " shares of " + self.portfolio_assets[order.asset] + " at " + str(price))
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
                high=self.dates[-1], \
                    interval=str(self.date_interval[1])+str.lower(self.date_interval[0])).to_list()
        
        for j in range(0, len(self.arr)):
            if(self.arr[j].shape[0] != len(timestamp_range)):
                newData = pl.DataFrame({'Date':timestamp_range}) 
                newData = self.arr[j].join(newData, on='Date', how='outer')
                newData = newData.fill_null(-1.0)
                self.arr[j] = newData
                if(j == self.market[1]):
                    self.market_arr = self.arr[j][['Date', 'Close']]

        if self.market_arr.is_empty():
            newData = pl.DataFrame({'Date':timestamp_range})
            yf.Ticker(self.market[0]).history(start=self.dates[0], end=self.dates[-1], interval=self.interval).to_csv("cqfbt\\data\\" + "market_benchmark_internal_hist.csv")
            data = pl.read_csv("cqfbt\\data\\" + "market_benchmark_internal_hist.csv")
            try:
                data = data.rename({'Datetime': 'Date'})
            except:
                pass
            data.replace('Date', data.get_column('Date').apply(self.str_to_dt))
            newData = newData.join(data, on='Date', how='outer')
            self.market_arr = newData[['Date', 'Close']]
        self.market_arr = self.market_arr.fill_null(-1.0)
        self.make_data()


    def setup_run(self):
        if self.arr != []:
            cols = set()
            self.MRC_prices=np.zeros(len(self.arr))
            for i in range(0, len(self.arr)):
                cols.update(self.arr[i].columns)
                
            for i in range(0, len(self.arr)):
                for col in cols:
                    if col not in self.arr[i].columns:
                        self.arr[i] = self.arr[i].select([pl.all(), pl.lit(-1.0).alias(col)])

            self.reformat_arr()
            
        else:
            print('No data to test.\n')


    # Main loop, feeds data to strategy, tries to execute specified orders then
    # returns the results back to the strategy. Repeats for all available data
    def run(self):
        """ 
        Run all added strategies on the data.
        """
        # If there is insufficient funds to execute a transaction, the engine will
        # re-feed the same data and try again with the new logic provided by the
        # strategy, if the strategy does nothing different the engine will quit
        # to avoid an infinite loop
        print("Setting up run for:")
        print(self.portfolio_assets)
        self.setup_time_start = time.time()
        self.portfolio_allocations = []
        self.portfolio_history=[]
        for i in range(len(self.strategies)):
                self.capital[i] = self.strategy_capital[i]
                self.portfolio_allocations.append(np.zeros(len(self.portfolio_assets)))
                self.portfolio_history.append(np.zeros((len(self.dates), len(self.portfolio_assets)+2, 1)))
        if self.setup_required == True:
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
                
            date, data = self.data_at_idx(i)
            for j in range(len(self.strategies)):
                self.strategies[j].append_data_history(data)
                # Portfolio History keeps track of asset allocations
                for k in range(0, len(self.portfolio_assets)):
                    self.portfolio_history[j][i, k] = self.portfolio_allocations[j][k]
                # And capital
                self.portfolio_history[j][i, len(
                    self.portfolio_assets)] = self.capital[j]
                # And total portfolio value
                self.portfolio_history[j][i, len(
                    self.portfolio_assets)+1] = self.get_portfolio_cash_value(i, j)

                # Testing purposes
                if(i % 5 ==0 and False):
                    print(str(i) + ' ::: ' + str(date) + ' ::: ' +
                       list_to_str(self.portfolio_history[j][i, :]))

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
        self.setup_required = False
        self.runtime_end = time.time()
        print("Run time: " + str(self.runtime_end-self.runtime_start)+"s")
        self.print_metrics()
        self.performance_plot()

    def print_metrics(self):
        print("\n ****** Performance Metrics ******")
        print("Sharpe Ratio:     " + str(self.get_sharpe_ratio()))
        print("Info Ratio:       " + str(self.get_info_ratio()))
        print("Max Drawdown:     " + str(self.get_max_drawdown()))
        print("Sortino Ratio:    " + str(self.get_sortino_ratio()))
        print("Calmar Ratio:     " + str(self.get_calmar_ratio()))
        print("Upside Potential: " + str(self.get_upside_potential_ratio()))

        ## Group 3 todo
    def set_transaction_costs(self, transaction_costs):
        ## if its a float
        if isinstance(transaction_costs, float):
            self.transaction_costs = [transaction_costs] * len(self.portfolio_assets)
        ## if its a float list
        elif isinstance(transaction_costs, list):
            ## make sure the list is the same length
            assert len(transaction_costs) == len(self.portfolio_assets), 'Length of transaction_costs must match the number of assets'
            self.transaction_costs = transaction_costs
        else:
            raise TypeError('Transaction costs must be of type float or list of floats')

    # Plot history of portfolio value, summing assets and capital 
    def plot_strategies(self, names='all', orders=False, order_threshold=0.1, suffix = '') -> None:
        """ 
        Generate a plot for the performance of a single strategy, or specified 
        list of strategies. If the engine has not been run, this will fail.
    
        Parameters
        ------------
            names: list / string (Optional)
                The list of names of each strategy to be plotted.
                Default: all
        """
        dates = self.dates
        sns.set_theme()
        for j in range(len(self.strategies)):
            if names == 'all' or names.__contains__(self.strategies[j].name):
                values = np.add(self.portfolio_history[j][:, len(
                    self.portfolio_assets)+1], self.portfolio_history[j][:, len(self.portfolio_assets)])
                font = {'family' : 'Times New Roman',
                        'weight' : 'bold',
                        'size'   : '18'}
                fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)

                if(orders):
                        order_deltas = np.diff(self.portfolio_history[j][:, len(self.portfolio_allocations[j])], axis=0)
                        order_deltas = np.sum(order_deltas, axis=1)
                        order_deltas_norm =  order_deltas / np.max(abs(order_deltas))
                        cmap_pos = plt.cm.get_cmap('YlGn')
                        cmap_neg = plt.cm.get_cmap('YlOrRd')
                        for i in range(len(values)-1):
                            if order_deltas_norm[i] >= order_threshold:
                                color = cmap_pos(order_deltas_norm[i])
                                marker = '^'
                                ax.scatter(dates[i+1], values[i+1], color=color, marker=marker, edgecolors='black', linewidths=0.1)
                            elif order_deltas_norm[i] <= -order_threshold:
                                color = cmap_neg(-order_deltas_norm[i])
                                marker = 'v'
                                ax.scatter(dates[i+1], values[i+1], color=color, marker=marker, edgecolors='black', linewidths=0.1)

                ax.set_title("Portfolio History: " + self.strategies[j].get_name(), font=font)
                ax.set_xlabel("Date")
                ax.set_ylabel("Value")
                ax.plot(dates, values)
                if(suffix != ''):
                    suffix = '_' + suffix
                fig.savefig(self.strategies[j].get_name() + "_performance"+ suffix +".pdf")

        # Plot history of portfolio value, summing assets and capital with


    def plot_aggregate(self, names='all', orders=False, order_threshold=0.1, title = '', mkt_ref = False) -> None:
        """ 
        Generate a plot for the performance of a single strategy, or specified 
        list of strategies. If the engine has not been run, this will fail.
    
        Parameters
        ------------
            names: list / string (Optional)
                The list of names of each strategy to be plotted.
                Default: all
        """
        dates = self.dates
        sns.set_theme()
        font = {'family' : 'Times New Roman',
                        'weight' : 'bold',
                        'size'   : '18'}
        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
        for j in range(len(self.strategies)):
            if names == 'all' or names.__contains__(self.strategies[j].name):
                values = np.add(self.portfolio_history[j][:, len(
                    self.portfolio_assets)+1], self.portfolio_history[j][:, len(self.portfolio_assets)])

                if(orders):
                        order_deltas = np.diff(self.portfolio_history[j][:, len(self.portfolio_allocations[j])], axis=0)
                        order_deltas = np.sum(order_deltas, axis=1)
                        order_deltas_norm =  order_deltas / np.max(abs(order_deltas))
                        cmap_pos = plt.cm.get_cmap('YlGn')
                        cmap_neg = plt.cm.get_cmap('YlOrRd')
                        for i in range(len(values)-1):
                            if order_deltas_norm[i] >= order_threshold:
                                color = cmap_pos(order_deltas_norm[i])
                                marker = '^'
                                ax.scatter(dates[i+1], values[i+1], color=color, marker=marker, edgecolors='black', linewidths=0.1)
                            elif order_deltas_norm[i] <= -order_threshold:
                                color = cmap_neg(-order_deltas_norm[i])
                                marker = 'v'
                                ax.scatter(dates[i+1], values[i+1], color=color, marker=marker, edgecolors='black', linewidths=0.1)
                ax.plot(dates, values, label = self.strategies[j].name)
        if(mkt_ref):
            axlims = ax.get_ylim()
            center = (axlims[0]+axlims[1]) / 2
            ax2 = ax.twinx()  
            mkt_dates = self.market_arr['Date'].to_numpy()
            mkt_vals = self.market_arr['Close'].to_numpy()
            ax2.axis(ymin=axlims[0]/center * np.min(mkt_vals[mkt_vals > 0]), ymax=axlims[1]/center * np.max(mkt_vals[mkt_vals > 0]))
            ax2.plot(mkt_dates[mkt_vals > 0], mkt_vals[mkt_vals > 0], color='black', label=self.market[0])
            ax2.legend(loc='upper left')
        if(title == ''):
            title = 'performance'
        ax.set_title(title, font=font)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
            
        ax.legend(loc='upper right')
        fig.savefig(title + ".pdf")

    def performance_plot(self):
        dates = self.dates
        sns.set_theme()
        font = {'family' : 'Times New Roman',
                        'weight' : 'bold',
                        'size'   : '18'}
        
        for j in range(len(self.strategies)):
            cmap = plt.hsv
            fig = plt.figure(figsize=(12, 10))
            grid = plt.GridSpec(4, 2, hspace=1, wspace=0.5)
            performance = fig.add_subplot(grid[:2, 1:])
            values = np.add(self.portfolio_history[j][:, len(
                self.portfolio_assets)+1], self.portfolio_history[j][:, len(self.portfolio_assets)])
            performance.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=9))
            performance.plot(dates, values)
            performance.set_title(self.strategies[j].name, font=font)
            performance.set_xlabel("Date")
            performance.set_ylabel("Value")

            allocs = self.portfolio_history[j][:, :len(self.portfolio_allocations[j])]
            alloc_sums = np.sum(allocs, axis = 0)/ len(self.data_arr) + 1

            alloc_values = allocs
            alloc_value_sums = np.zeros_like(alloc_values[:, 0])

            scatter_plot = fig.add_subplot(grid[:2, :1])
            alloc_plot = fig.add_subplot(grid[2:, :])
            for i in range(len(self.arr)):
                end_idx = len(self.data_arr)-1
                end_date, data = self.data_at_idx(end_idx)
                end_price = data.select([pl.col('Close')])[i].item()
                while end_price <= 0:
                    end_idx -= 1
                    end_date, data = self.data_at_idx(end_idx)
                    end_price = data.select([pl.col('Close')])[i].item()
                start_idx = 0
                start_date, data = self.data_at_idx(start_idx)
                start_price = data.select([pl.col('Close')])[i].item()
                while start_price <= 0:
                    start_idx += 1
                    start_date, data = self.data_at_idx(start_idx)
                    start_price = data.select([pl.col('Close')])[i].item()
                    
                for k in range(len(alloc_values)):
                    idx = k
                    date, data = self.data_at_idx(idx)
                    price = data.select([pl.col('Close')])[i].item()
                    while price <=0 and idx > 0:
                        idx -= 1
                        date, data = self.data_at_idx(idx)
                        price = data.select([pl.col('Close')])[i].item()
                    alloc_values[k, i] *= price

                years = (end_date-start_date).total_seconds() / 31536000

                yearly_returns = ((end_price - start_price) / start_price) ** (1/years) * 100

                time_diff_days = (self.dates[1] - self.dates[0]).total_seconds()*len(self.dates) / 86400

                nonzero_vals = []
                nonzero_dates = []
                sum = 0
                num_nonzero_vals = 0
                for k in range(len(self.arr[i])):
                    val = self.arr[i].select([pl.col('Close')])[k].item()
                    if val > 0:
                        nonzero_vals.append(val)
                        sum += val
                        nonzero_dates.append(self.dates[k])
                        num_nonzero_vals += 1
                mean_price = sum / num_nonzero_vals

                nonzero_vals = pl.Series(nonzero_vals)
                volatility = nonzero_vals.pct_change().std() * (time_diff_days ** 0.5) * 100
                
                hsv = cm.get_cmap('hsv')
                scatter_plot.scatter(volatility.real, yearly_returns.real, marker='o', s=mean_price*alloc_sums[i]/max(alloc_sums), color=hsv(i/len(self.portfolio_assets)), alpha=0.6)
                scatter_plot.text(volatility.real, yearly_returns.real, self.portfolio_assets[i], ha='center', va='bottom', fontsize= 8)
                scatter_plot.set_title("Annualized % Returns vs. Volatility")
                scatter_plot.set_xlabel("Annualized Volatility (%)")
                scatter_plot.set_ylabel("Annualized Returns (%)")

                new_dates = self.dates
                group_avg = alloc_values[:, i, 0]
                group_avg_sum = alloc_value_sums[:, 0]

                if(len(self.dates) > 400):
                    num_vals_per_bar = len(self.dates)//400
                    num_values_to_pad = num_vals_per_bar - len(alloc_values[:, i, 0]) % num_vals_per_bar

                    padded_alloc_vals = np.pad(alloc_values[:, i, 0], (0, num_values_to_pad), mode='constant')
                    padded_alloc_value_sums = np.pad(alloc_value_sums[:, 0], (0, num_values_to_pad), mode='constant')

                    groups = padded_alloc_vals.reshape(-1, num_vals_per_bar)
                    group_sums = padded_alloc_value_sums.reshape(-1, num_vals_per_bar)

                    group_avg = np.sum(groups, axis=1) / num_vals_per_bar
                    group_avg_sum = np.sum(group_sums, axis=1) / num_vals_per_bar
                    new_dates = []
                    for k in range(0, len(self.dates), num_vals_per_bar):
                            new_dates.append(self.dates[k])

                alloc_plot.bar(new_dates[:], group_avg[:], 1, bottom=group_avg_sum[:], align='center', label = self.portfolio_assets[i], color=hsv(i/len(self.portfolio_assets)), edgecolor='black', linewidth = 0.1)
                alloc_value_sums = np.add(alloc_value_sums, alloc_values[:, i])
                alloc_plot.set_ylabel("Value in Asset")
                alloc_plot.set_xlabel("Date")
            #alloc_plot.set_ylim([0, self.strategy_capital[j]])
            alloc_plot.legend(loc='upper left')
        plt.show()


    # Uses data (self.arr) to price the cash value of a portfolio by summing close prices
    # for each asset in the portfolio, capital should be included. There may be
    # no data for the given date, if this is the case use the most recent previous close
    def get_portfolio_cash_value(self, idx, strategy_num) -> float:
        asset_values = 0
        for i in range(len(self.portfolio_allocations[strategy_num])):
            id=idx
            date, data = self.data_at_idx(idx)
            price = data.select([pl.col('Close')])[i].item()
            if price < 0:
                price = self.MRC_prices[i]
            else:
                self.MRC_prices[i] = price
            asset_values += price*self.portfolio_allocations[strategy_num][i]
        return round(asset_values, 2)


    def get_sharpe_ratio(self):
        # get SPY values
        mkt_close = self.market_arr['Close'].to_numpy()
        mkt_close = mkt_close[mkt_close > 0]
        mkt_returns = np.diff(mkt_close) / mkt_close[:-1]
        mkt_avg_annual_return = np.mean(mkt_returns) * 252
        mkt_std_dev = np.std(mkt_returns) * np.sqrt(252)

        # get portfolio values
        sharpe_ratio = []
        for i in range(len(self.strategies)):
            portfolio_value = np.add(self.portfolio_history[i][:, len(self.portfolio_assets)], self.portfolio_history[i][:, len(self.portfolio_assets)+1])
            strategy_returns = np.divide(np.diff(portfolio_value, axis=0), portfolio_value[:-1])
            avg_annual_return = np.mean(strategy_returns) * 252
            std_dev = np.std(strategy_returns) * np.sqrt(252)
            # relative sharpe ratio
            sharpe_ratio.append((avg_annual_return - mkt_avg_annual_return)/np.sqrt(std_dev**2 + mkt_std_dev**2))
        
        return sharpe_ratio


    def get_info_ratio(self, benchmark=''):
        if benchmark == '':
            benchmark = self.market[0]
            
        idx = 0
        benchmark_value = self.market_arr['Close'].to_numpy(writable=True)
        while benchmark_value[idx] < 0:
            idx+=1
        for i in range(len(benchmark_value)):
            if i < idx:
                benchmark_value[i] = benchmark_value[idx]
            elif benchmark_value[i] < 0:
                benchmark_value[i] = benchmark_value[i-1]
        benchmark_returns = np.divide(np.diff(benchmark_value, axis=0), benchmark_value[:-1])
        info_ratio = []
        for i in range(len(self.strategies)):
            portfolio_value = np.add(self.portfolio_history[i][:, len(self.portfolio_assets)], self.portfolio_history[i][:, len(self.portfolio_assets)+1])
            strategy_returns = np.divide(np.diff(portfolio_value, axis=0), portfolio_value[:-1])
            
            excess_returns = np.subtract(strategy_returns, benchmark_returns)
            info_ratio.append(np.mean(excess_returns) / np.std(excess_returns))
        
        return info_ratio


    def get_max_drawdown(self):
        max_drawdowns = []
        for i in range(len(self.strategies)):
            portfolio_value = np.add(self.portfolio_history[i][:, len(self.portfolio_assets)], self.portfolio_history[i][:, len(self.portfolio_assets)+1])
            peaks = np.maximum.accumulate(portfolio_value)
            drawdowns = (peaks-portfolio_value) / peaks
            max_drawdowns.append(np.max(drawdowns))
        return max_drawdowns


    def get_sortino_ratio(self):
        mkt_close = self.market_arr['Close'].to_numpy()
        mkt_close = mkt_close[mkt_close > 0]
        mkt_returns = np.diff(mkt_close) / mkt_close[:-1]
        mkt_avg_annual_return = np.mean(mkt_returns) * 252

        # get portfolio values
        sortino_ratio = []
        for i in range(len(self.strategies)):
            portfolio_value = np.add(self.portfolio_history[i][:, len(
                self.portfolio_assets)], self.portfolio_history[i][:, len(self.portfolio_assets)+1])
            strategy_returns = np.divide(
                np.diff(portfolio_value, axis=0), portfolio_value[:-1])
            downside_dev = 0
            num_neg_returns = 0
            for ret in strategy_returns:
                if ret - np.mean(mkt_returns) < 0:
                    downside_dev += (ret - np.mean(mkt_returns))**2
                    num_neg_returns += 1
            downside_dev /= num_neg_returns
            avg_annual_return = np.mean(strategy_returns) * 252
            sortino_ratio.append(
                (avg_annual_return - mkt_avg_annual_return)/downside_dev)

        return sortino_ratio

    def get_calmar_ratio(self):
        mkt_close = self.market_arr['Close'].to_numpy()
        mkt_close = mkt_close[mkt_close > 0]
        mkt_returns = np.diff(mkt_close) / mkt_close[:-1]
        mkt_avg_annual_return = np.mean(mkt_returns) * 252

        calmar_ratio = []
        for i in range(len(self.strategies)):
            portfolio_value = np.add(self.portfolio_history[i][:, len(
                self.portfolio_assets)], self.portfolio_history[i][:, len(self.portfolio_assets)+1])
            strategy_returns = np.divide(
                np.diff(portfolio_value, axis=0), portfolio_value[:-1])
            avg_annual_return = np.mean(strategy_returns) * 252
            calmar_ratio.append(
                (avg_annual_return - mkt_avg_annual_return)/self.get_max_drawdown()[i])

        return calmar_ratio

    def get_upside_potential_ratio(self, mar=-1):
        # if mar = -1, use risk-free-rate/market return
        mkt_close = self.market_arr['Close'].to_numpy()
        mkt_close = mkt_close[mkt_close > 0]
        mkt_returns = np.diff(mkt_close) / mkt_close[:-1]
        mkt_avg_annual_return = np.mean(mkt_returns) * 252
        if (mar == -1):
            mar = mkt_avg_annual_return
        upside_potential_ratio = []
        for i in range(len(self.strategies)):
            portfolio_value = np.add(self.portfolio_history[i][:, len(
                self.portfolio_assets)], self.portfolio_history[i][:, len(self.portfolio_assets)+1])
            strategy_returns = np.divide(
                np.diff(portfolio_value, axis=0), portfolio_value[:-1])
            upside = 0
            downside_dev = 0
            num_neg_returns = 0
            for ret in strategy_returns:
                if ret - np.mean(mkt_returns) < 0:
                    downside_dev += (ret - mar)**2
                    num_neg_returns += 1
                else:
                    upside += (ret - mar)
            downside_dev /= num_neg_returns
            upside /= (252 - num_neg_returns)
            upside_potential_ratio.append(
                upside/downside_dev)

        return upside_potential_ratio

    # Clears all data files in data folder
    def clear_data(self):
        """
        Clear all data from the engine.
        """
        self.arr = []
        self.data_arr = []
        self.dates = []
        for i in range(0, len(self.portfolio_assets)):
            self.remove_data(self.portfolio_assets[i])
        self.remove_data('market_benchmark_internal_hist')

    # Removes a specific data file from data folder
    # Can specify a ticker, or if it is user-added data, the index
    # of the asset in the portfolio
    # TODO: (II) remove user-added data files by name
    def remove_data(self, data=''):
        """ 
        Remove a specified data file from the engine.
    
        Parameters
        ------------
            data: string (Optional)
                The name of the file to be removed, the filename should be of
                the format data.csv, the data parameter will be directly placed 
                into a string that already contains '.csv'
            assetNo: int (Optional)
                The index of the asset to be removed, indices are arranged by 
                the order in which they were added to the engine, with tickers
                passed in the constructor given priority.
        """
        path = "cqfbt\\data\\" + f"{data}.csv"
        if(os.path.exists(path)):
            os.remove(path)
        
    # string to datetime
    # TODO (II) add support for UNIX ts and other date formats
    # Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
    def str_to_dt(self, s: str) -> dt.datetime:
        date = dateutil.parser.parse(s, ignoretz=True)

        if self.date_interval[0].lower() == 'h':
                return date.replace(second=0, microsecond=0, minute=0, hour=date.minute//30+ date.hour) 
            
        return date



    
    def __del__(self):
        self.clear_data()
    

class InsufficientFundsException(Exception):
    def __init__(self, message='Insufficient Funds to perform transaction.'):
        super(InsufficientFundsException, self).__init__(message)


# list to string
def list_to_str(lst):
    out = "["
    for i in range(0, len(lst)-1):
        out += str(lst[i]) + ', '
    return out + str(lst[len(lst)-1])+']'



