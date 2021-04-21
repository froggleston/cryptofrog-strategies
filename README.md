# cryptofrog-strategies
CryptoFrog - My First Strategy for freqtrade

Heavily borrowing ideas from:

- https://github.com/werkkrew/freqtrade-strategies : Amazing work on Solipsis that influenced my general framework and custom_stoploss
- https://github.com/brookmiles/freqtrade-stuff : Great Ichi based strat from Obelisk
- https://github.com/hansen1015/freqtrade_strategy/blob/main/heikin.py : Using the smoothed Heiken Ashi on the CryptoFrog 1hr informative timeframe

# Things to Know

- **Fairly conservative strategy** focusing on longer holds to find large peaks - *don't be surprised if you don't get buys for hours or days*
- Designed to trade altcoins against stablecoins, and I've used USDT intentionally to gain relative stability within BTC/ETH dump cycles
- Hyperopted with ~~Sharpe~~ ShortTradeDurHyperOptLoss - if not, it'll simply end up with thousands of tiny profit trades which will likely end in slippage.
- Protections need to be enabled. Happy to share my config - hit me up on the freqtrade discord
- Included a live_plotting.ipynb notebook that can be used to immediately and easily view backtest results

# TODO

- Better buy signals
- Better informative pair work looking for BTC/ETH trends
- More testing

# Preprequisites

You'll need:
- Python 3.7+
- Jupyter Notebook for the live_plotting.ipynb
- Solipsis_v4 custom_indicators.py
- finta
- TA-Lib
- Pandas
- Numpy
