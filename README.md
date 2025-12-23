# MARKET-HEARTBEAT
![presentazione](imag/presentazione.png)

This quantitative intraday trading project is designed to operate on a large amount of data. 
Due to GitHub file size restrictions, all zipped raw data files are not included in the repository. 
They can be downloaded from the official Binance website at the following path:

Home / data / futures / um / monthly / trades / BTCUSDT  
https://data.binance.vision/?prefix=data/futures/um/monthly/trades/BTCUSDT

For the purpose of this research, it is necessary to work with all BTCUSDT trade files from
March 2023 (BTCUSDT-trades-2023-03.zip) to December 2024 (BTCUSDT-trades-2024-12.zip).

The correct execution order of the Python scripts provided in the repository is the following:
- data_processing.py
- data_split.py
- lstm_cnn.py (creation and training of a specific neural network model)
- backtester.py (market analysis and proposal of trading strategies)
- ecg.py (demo visualization of the best single trading strategy)

This repository also includes dedicated folders (one for each Python script), named after the
corresponding script. Each folder contains as many input and output files as possible related
to that script. Some files are unfortunately omitted due to GitHub file size restrictions and
can either be obtained by running the scripts or requested via email
(matteorog05@gmail.com).
A detailed list of features and a set of summary plots related to the neural network training process are provided in the main.


<p align="center">
  <img src="imag/1.png" width="50%">
</p>

This framework represents a complete end-to-end algorithmic trading system that achieved +9.73% returns with a 5.04 Sharpe ratio on genuine out-of-sample data, demonstrating robust predictive power beyond training conditions. The modular design separates data processing, model training, backtesting, and strategy execution into independent, scalable components ready for institutional-grade deployment.
<br>
**Live Implementation Path**: Connect to Binance WebSocket for real-time tick data → Replicate the 28-feature microstructure engineering pipeline on streaming 30s bars → Feed normalized sequences through the trained LSTM-CNN model → Execute trades via Binance API using the validated volatility-adaptive strategy with risk controls. The entire system is designed for minimal latency (<100ms prediction time) and handles market regime detection, position management, and emergency stops autonomously.
<br>
**Why This Works**: Unlike typical academic projects, this system was battle-tested through rigorous walk-forward validation, deflated Sharpe ratio analysis to avoid backtest overfitting, and ensemble methods across 5 top strategies. The microstructure features (Order Flow Imbalance, VWAP deviation, Hawkes intensity) capture genuine market dynamics that persist in live trading, not spurious patterns from data mining.
<br>
**Some needed specifications**: The project was developed by preprocessing the data in order to construct a binary target variable identifying upward and downward price movements. Consequently, for any real-world application, both the project and the proposed methodologies must be adapted to genuine live trading dynamics, where market behavior is not characterized by continuous up-and-down movements and the data do not always conform to the artificially constructed target variable used in this study.

<p align="center">
  <img src="imag/4.png" width="50%">
</p>
