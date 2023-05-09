# Cornell Quant Fund Backtester (cqfbt)

## Engine

An `Engine` is the main component in our backtester. It accepts strategies and simulates them on the price data it is initialized with, and can output a variety of metrics on their performance.

The `Engine` is initialized with a start date and end date to pull data for, as well as the periodicity of the data. Price data is pulled from Yahoo Finance via the `yfinance` package. Optionally, the `Engine` can also be initialized with a list of Yahoo Finance tickers to load price data for.

A feature of the `Engine` is the ability to set transaction costs. These must be set via the `Engine.set_transaction_costs` method, which takes either a float representing a uniform transaction cost or a list of floats for each portfolio asset.

The `Engine` also contains methods that compare portfolio returns to a market benchmark. This market benchmark is set to the `SPX`, `DJI`, or `NDAQ` ticker, depending on which one is included in the list of tickers to load price data for during initialization. If more than one benchmark ticker appears in the ticker list, the market benchmark is set to the benchmark ticker that appears last in the ticker list. If no benchmark ticker is included in the ticker list, the default is the `SPY` ticker.