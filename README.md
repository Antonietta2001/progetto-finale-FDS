# MARKET-HEARTBEAT
This quantitative intraday trading project is proposed to work with a great amount of data. Due to file size restriction imposed by github all the zipped data files are absent in the repository. It is possible  to download them by the official binance website using this path: Home / data / futures / um / monthly / trades / BTCUSDT (https://data.binance.vision/?prefix=data/futures/um/monthly/trades/BTCUSDT). For the purpose  of the research is necessary to work with all the BTCUSDT-trades-year-month files from March 2023 (BTCUSDT-trades-2023-03.zip) to December 2024 (BTCUSDT-trades-2024-12.zip). 

 The right order of execution of the .py files is the sequent:
 - data_processing.py
 - data_split.py
 - lstm_cnn.py (creation and training of a specific neural network model)
 - backtester.py (analysis of the market and proposal of some financial strategies)
 - ecg,py (demo visualization of our best single financial strategy)

 Some output files of the first 2 scripts are omitted due to file size restriction and can be requested via email (matteorog05@gmail.com) or obtained easily executing the scripts.

