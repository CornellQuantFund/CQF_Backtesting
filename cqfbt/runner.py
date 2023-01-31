# Test runner, not to be included in final package
from cqfbt import backtester, strategy
# These imports will not be needed, as the user will have imported
# the cqfbt package wherin these are contained

if __name__ == "__main__":

    class Strat0(strategy.Strategy):
        def execute(self, date, data, portfolio_values, capital, orders, error):
            # If error == True, it means that the same date and data as last time will be
            # coming in, the capital and portfolio will be unchanged, and the orders
            # coming in will be the same as the orders sent out on the last iteration.
            # If this error is not adressed there will be infinite recursion.
            return orders

    s0 = Strat0()

    # Testing optional arguments for multiple constructors
    eng0 = backtester.engine()
    eng0.set_capital(8e4)
    eng0.add_data('data1.csv')
    eng0.add_data('data2.csv')
    eng0.add_strategy(s0)
    eng0.run()

    eng = backtester.engine(
        ['SPY', 'TSLA', 'AAPL', 'JPM', 'AMZN'], '2021-01-01', '2022-01-01', '1d', 100000)
    eng.add_data('data1.csv')
    eng.add_data('data2.csv')
    eng.add_strategy(s0)
    eng.run()

    print('... running !\n')
