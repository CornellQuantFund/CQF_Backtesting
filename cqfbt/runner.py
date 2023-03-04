# Test runner, not to be included in final package
from cqfbt import backtester, strategy
# These imports will not be needed, as the user will have imported
# the cqfbt package wherin these are contained
import yfinance as yf
import numpy as np
import pandas as pd

if __name__ == "__main__":

    # TODO: Group 4
    # write a 

    class Strat0(strategy.Strategy):
        def execute(self, date, data, portfolio_values, capital, orders, error, tickers):
            if orders == None:
                orders = []
            newOrders =[]

            # print(tickers)
            # for row in data:
            #     print(row)
            # print(portfolio_values)
            open = data.loc['Open'].to_numpy()
            close = data.loc['Close'].to_numpy()
            pct_change = (close-open)/open
            totalMomentum = sum(pct_change)
            buy_indices = np.where(pct_change > 0)[0]
            sell_indices = np.where(pct_change < 0)[0]
            posSize = abs(pct_change)/abs(totalMomentum)
            orderAmount = capital * posSize / open
            # print(tickers)
            # print(pct_change)
            # print(buy_indices)
            # print(sell_indices)

            for buy in buy_indices:
                newOrders.append(backtester.Order(True , int(buy), close[buy], orderAmount[buy]))
            
            for sell in sell_indices:
                newOrders.append(backtester.Order(False , int(sell), close[sell], orderAmount[sell]))

            # If error == True, it means that the same date and data as last time will be
            # coming in, the capital and portfolio will be unchanged, and the orders
            # coming in will be the same as the orders sent out on the last iteration.
            # If this error is not adressed the engine will stop running
            return newOrders

    s0 = Strat0()

    # Testing optional arguments for multiple constructors
    #eng0 = backtester.Engine()
    #eng0.set_capital(8e4)
    #eng0.add_data('data1.csv')
    #eng0.add_data('data2.csv')
    #eng0.add_strategy(s0)
    #eng0.run()

    eng = backtester.Engine(
        ['SPY', 'TSLA', 'AAPL', 'JPM', 'AMZN'], '2021-01-01', '2022-01-01', '1d', 100000)
    eng.add_data('data1.csv')
    eng.add_data('data2.csv')
    eng.add_strategy(s0)
    eng.run()
    eng.clear_data()

    print('... running !\n')
