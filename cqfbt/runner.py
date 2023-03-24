# Test runner, not to be included in final package
from cqfbt import backtester, strategy
# These imports will not be needed, as the user will have imported
# the cqfbt package wherin these are contained
import yfinance as yf
import numpy as np
import pandas as pd
import os

if __name__ == "__main__":

    class Strat0(strategy.Strategy):
        def execute(self, date, data, portfolio_allocations, capital, orders, error, tickers):
            newOrders =[]
            open = data.loc[:, 'Open'].to_numpy()
            close = data.loc[:, 'Close'].to_numpy()
            if(~(open[0] < 0)):
                pct_change = (close-open)/open

                buy_indices = []
                buy_quantities = []
                sell_indices = []
                sell_quantities = []
                for i in range(len(pct_change)):
                    if pct_change[i] > 0.01:
                        quantity = capital / (close[i])*pct_change[i]
                        buy_indices.append(i)
                        if (i >= len(portfolio_allocations)-2):  # If this is Bitcoin / Eth
                            buy_quantities.append(quantity) # do not round to nearest int
                        else:
                            buy_quantities.append(round(quantity))
                    elif pct_change[i] < -0.01:
                        quantity = 200000 / (close[i])*pct_change[i]
                        sell_indices.append(i)
                        if (portfolio_allocations[i] <= -0.5*quantity):
                            sell_quantities.append(portfolio_allocations[i])
                        elif (i >= len(portfolio_allocations)-2):  # If this is Bitcoin / Eth
                            sell_quantities.append(-0.5*quantity) # again, do not round
                        else:
                            sell_quantities.append(round(-0.5*quantity))

                if(~error):
                    for i in range(len(buy_indices)):
                        newOrders.append(backtester.Order(True, buy_indices[i], buy_quantities[i]))
                for i in range(len(sell_indices)):
                    newOrders.append(backtester.Order(False, sell_indices[i], sell_quantities[i]))
            return newOrders
        

    class Strat1(strategy.Strategy):
        def execute(self, date, data, portfolio_allocations, capital, orders, error, tickers):
            newOrders =[]
            open = data.loc[:, 'Open'].to_numpy()
            close = data.loc[:, 'Close'].to_numpy()
            if(~(open[0] < 0)):
                pct_change = (close-open)/open

                buy_indices = []
                buy_quantities = []
                sell_indices = []
                sell_quantities = []
                for i in range(len(pct_change)):
                    if pct_change[i] > 0.02:
                        quantity = 2*capital / (close[i])*pct_change[i]
                        buy_indices.append(i)
                        if (i >= len(portfolio_allocations)-2):  # If this is Bitcoin / Eth
                            buy_quantities.append(quantity) # do not round to nearest int
                        else:
                            buy_quantities.append(round(quantity))
                    elif pct_change[i] < -0.03:
                        sell_indices.append(i)
                        sell_quantities.append(portfolio_allocations[i])
                if(~error):
                    for i in range(len(buy_indices)):
                        newOrders.append(backtester.Order(True, buy_indices[i], buy_quantities[i]))
                for i in range(len(sell_indices)):
                    newOrders.append(backtester.Order(False, sell_indices[i], sell_quantities[i]))
            return newOrders

    s0 = Strat0()
    s1= Strat1()

    
    start = '2020-01-01'
    end = '2023-01-01'
    intvl = '1d'
    yf.Ticker("NFLX").history(start=start, end=end, interval=intvl).to_csv("netflix.csv")
    yf.Ticker("AMZN").history(start=start, end=end, interval=intvl).to_csv("amazon.csv")
    yf.Ticker("BTC-USD").history(start=start, end=end, interval=intvl).to_csv("bitcoin.csv")
    yf.Ticker("ETH-USD").history(start=start, end=end, interval=intvl).to_csv("ethereum.csv")
    

    eng = backtester.Engine(
        ['SPY', 'TSLA', 'AAPL', 'JPM'], start, end, intvl)
    eng.add_data('netflix.csv')
    eng.add_data('amazon.csv')
    eng.add_data('bitcoin.csv')
    eng.add_data('ethereum.csv')
    eng.add_strategy(s0, 200000)
    eng.add_strategy(s1, 200000)
    eng.run()

    eng.clear_data()

    os.remove("data4.csv")
    os.remove("bitcoin.csv")
    os.remove("ethereum.csv")
    os.remove("amazon.csv")
    os.remove("netflix.csv")

    print('... done.\n')
