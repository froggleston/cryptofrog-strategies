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

# Things what I Found Out

- The Bollinger Band expansion function has a multiplier - it can *drastically* change BT and presumably dry/live results. Higher multiplier means more conservative buys and sells as the bands have to widen faster (UNLESS you also increase the rolling window size for bb_width, but only to a point - it's like a balancing act between the two).
- The lower the expansion multiplier, the wider the BB expansion window will be, and the more buys and sells you should see
- Coreful! Too low and you'll just make more crappier buys. This would be great to optimise dynamically based on market volatility I think
- Hyperopting is the Devil's Game, and I'm not ready to play yet. Therefore, the values I've set seem to perform OK from looking at ENDLESS plots manually
- The BB expansion is probably not quite right for sells so that's something to tweak
- import custom_indicators.py can do one.

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
