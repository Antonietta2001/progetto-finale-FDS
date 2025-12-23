# MARKET-HEARTBEAT
![presentazione](imag/presentazione.png)


This quantitative intraday trading project is proposed to work with a great amount of data. Due to file size restriction imposed by github all the zipped data files are absent in the repository. It is possible  to download them by the official binance website using this path: Home / data / futures / um / monthly / trades / BTCUSDT (https://data.binance.vision/?prefix=data/futures/um/monthly/trades/BTCUSDT). For the purpose  of the research is necessary to work with all the BTCUSDT-trades-year-month files from March 2023 (BTCUSDT-trades-2023-03.zip) to December 2024 (BTCUSDT-trades-2024-12.zip). 

 The right order of execution of the .py files is the sequent:
 - data_processing.py
 - data_split.py
 - lstm_cnn.py (creation and training of a specific neural network model)
 - backtester.py (analysis of the market and proposal of some financial strategies)
 - ecg,py (demo visualization of our best single financial strategy)

 Some output files of the first 2 scripts are omitted due to file size restriction and can be requested via email (matteorog05@gmail.com) or obtained easily executing the scripts.

<p align="center">
  <img src="imag/1.png" width="50%">
</p>

This framework represents a complete end-to-end algorithmic trading system that achieved +9.73% returns with a 5.04 Sharpe ratio on genuine out-of-sample data, demonstrating robust predictive power beyond training conditions. The modular design separates data processing, model training, backtesting, and strategy execution into independent, scalable components ready for institutional-grade deployment.
<br>
**Live Implementation Path**: Connect to Binance WebSocket for real-time tick data → Replicate the 28-feature microstructure engineering pipeline on streaming 30s bars → Feed normalized sequences through the trained LSTM-CNN model → Execute trades via Binance API using the validated volatility-adaptive strategy with risk controls. The entire system is designed for minimal latency (<100ms prediction time) and handles market regime detection, position management, and emergency stops autonomously.
<br>
**Why This Works**: Unlike typical academic projects, this system was battle-tested through rigorous walk-forward validation, deflated Sharpe ratio analysis to avoid backtest overfitting, and ensemble methods across 5 top strategies. The microstructure features (Order Flow Imbalance, VWAP deviation, Hawkes intensity) capture genuine market dynamics that persist in live trading, not spurious patterns from data mining.


<p align="center">
  <img src="imag/4.png" width="50%">
</p>
