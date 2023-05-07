# Test runner, not to be included in final package
from cqfbt import backtester, strategy
# These imports will not be needed, as the user will have imported
# the cqfbt package wherin these are contained
import yfinance as yf
import numpy as np
import polars as pl
import os

if __name__ == "__main__":
    class Strat0(strategy.Strategy):
        def execute(self, date, data, portfolio_allocations, capital, orders, error, tickers):
            """
            Executes All Strategies on a given date given the data, current portfolio allocation, capital,
            inflight orders, errors, and tickers

            Parameters
            ----------
            date : str
                The date we are executing our strategy on.
            data : Polars Dataframe
               The dataframe that we execute our strategy on
            porfolio_allocations : int[]
                The Current assets that are available to tade and their current allocation in the portfolio
            capital: int
                The amount of capital to spend
            orders: Order[]
                An array of limit orders that still haven't been executed
            error: boolean
                If there is an error
            tickers: str[]
                List of tickers we can execute our strategy on.
            """
            newOrders = orders
            open = data.get_column('Open').to_numpy()
            close = data.get_column('Close').to_numpy()
            if(~(open.any() < 0)):
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
                        quantity = 1000000 / (close[i])*pct_change[i]
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
            newOrders = orders
            open = data.get_column('Open').to_numpy()
            close = data.get_column('Close').to_numpy()
            if(~(open.any() < 0)):
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

    s0 = Strat0('Momentum Trader')
    s1= Strat1('Nervous Momentum Trader')

    start = '2020-01-01'
    end = '2023-04-01'
    intvl = '1d'
    yf.Ticker("NFLX").history(start=start, end=end, interval=intvl).to_csv("netflix.csv")
    yf.Ticker("AMZN").history(start=start, end=end, interval=intvl).to_csv("amazon.csv")
    yf.Ticker("BTC-USD").history(start=start, end=end, interval=intvl).to_csv("bitcoin.csv")
    yf.Ticker("ETH-USD").history(start=start, end=end, interval=intvl).to_csv("ethereum.csv")
    
    eng = backtester.Engine(start, end, intvl, 
        ['SPY', 'TSLA', 'AAPL', 'JPM'])
    eng.add_data('netflix.csv')
    eng.add_data('amazon.csv')
    eng.add_data('bitcoin.csv')
    eng.add_data('ethereum.csv')

    eng.add_strategy(s0, 100000)
    eng.add_strategy(s1, 100000)
    eng.set_transaction_costs([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.003, 0.0025])
    eng.run()
    eng.plot_strategies(orders = True, order_threshold = .2, suffix='run1')
    eng.plot_aggregate(title = 'All Strategies', mkt_ref = True)

    os.remove("bitcoin.csv")
    os.remove("ethereum.csv")
    os.remove("amazon.csv")
    os.remove("netflix.csv")