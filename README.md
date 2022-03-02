# cryptofrog-strategies
CryptoFrog - My First Strategy for freqtrade

**_DO NOT USE THIS FOR LIVE TRADING_**

# "Release" Notes

- 2021-04-26: The informatives branch now includes a big refactor to include new KAMA and Madrid Squeeze code. Hyperopting now in the main strategy. I'll pull this into main whenever I feel it's ready.
- 2021-04-20: You'll need the latest freqtrade develop branch otherwise you might see weird "supersell" results in your backtraces. Head to the freqtrade discord for more info.

Heavily borrowing ideas from:

- https://github.com/werkkrew/freqtrade-strategies : Amazing work on Solipsis that influenced my general framework and custom_stoploss
- https://github.com/brookmiles/freqtrade-stuff : Great Ichi based strat from Obelisk
- https://github.com/hansen1015/freqtrade_strategy/blob/main/heikin.py : Using the smoothed Heiken Ashi on the CryptoFrog 1hr informative timeframe

# Things to Know

- Fairly conservative strategy focusing on fewer buys and longer holds to find large peaks.
- Designed to trade altcoins against stablecoins, and I've used USDT intentionally to gain relative stability within BTC/ETH dump cycles
- Hyperopting is now available for most of the key indicator thresholds.
- Protections need to be enabled. I've included a basic template config - hit me up on the freqtrade discord for any info but no surprises expected really
- Included a live_plotting.ipynb notebook that can be used to immediately and easily view backtest results

# TODO

- Better buy signals
- Better informative pair work looking for BTC/ETH trends
- More testing

# Preprequisites

You'll need:
- Python 3.7+
- Jupyter Notebook for the live_plotting.ipynb
- finta
- TA-Lib (I run my bot on a Raspberry Pi 400, so you'll need to build TA-Lib as per the Freqtrade docs if you're doing the same)
- Pandas
- Numpy
- Pandas-TA indicator library
- ~~Solipsis_v4 custom_indicators.py (now included in this repo - thanks for the go-ahead @werkkrew)~~
