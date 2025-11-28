# data_processor.py - Pipeline Dati HFT per LSTM (10 Feature Ottimizzate)

import pandas as pd
import numpy as np
import requests
import os
from datetime import timedelta
from typing import Tuple

# --- CONFIGURAZIONE ---
SYMBOL = 'BTCUSDT' 
CRASH_THRESHOLD = 0.005 # 0.5% (50 bps) di caduta: rende Y=1 raro
LOOK_AHEAD_WINDOW_SECONDS = 60 
SEQUENCE_LENGTH = 5 # <--- MODIFICATO A 5
FEAR_GREED_API_URL = "https://api.alternative.me/fng/"

TRADE_FILES_BASE = [f'{SYMBOL}-trades-2021-05-18', f'{SYMBOL}-trades-2021-05-19']
BUYER_MAKER_COL = 'isBuyerMaker' 

# --- FUNZIONI DI CARICAMENTO DATI ---

def load_and_combine_data(trade_files_base: list) -> pd.DataFrame:
    """Carica e unisce i trade tick dai file compressi specifici."""
    all_trades = []
    print("Inizio caricamento Trade Tick Data...")
    
    for base_name in trade_files_base:
        file_name = f"{base_name}.zip" 
        if not os.path.exists(file_name): print(f"⚠️ ERRORE: File non trovato: {file_name}."); continue
        df = pd.read_csv(
            file_name, compression='zip', header=None, 
            names=['id', 'price', 'qty', 'quote_qty', 'timestamp', BUYER_MAKER_COL, 'isBestMatch']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['price'] = df['price'].astype(float)
        df['qty'] = df['qty'].astype(float)
        all_trades.append(df)
        print(f"✅ Caricato {file_name}: {len(df):,} trades.")
        
    if not all_trades: raise FileNotFoundError("Nessun file di trade valido trovato o caricato.")
    combined_df = pd.concat(all_trades, ignore_index=True).sort_values(by='timestamp').drop_duplicates(subset=['timestamp', 'price']).reset_index(drop=True)
    return combined_df

def download_historical_fng() -> pd.DataFrame:
    """Scarica l'intera cronologia del Fear & Greed Index."""
    try:
        print("🌍 Scaricando l'indice storico Fear & Greed...")
        response = requests.get(FEAR_GREED_API_URL, params={'limit': 0}); response.raise_for_status() 
        df = pd.DataFrame(response.json().get('data', []))
        df['timestamp'] = df['timestamp'].astype(int); df['Fear_Greed_Index'] = df['value'].astype(int)
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('date').sort_index(); df = df[['Fear_Greed_Index']]
        return df
    except Exception as e: print(f"ERRORE API F&G: {e}"); return pd.DataFrame()

# --- FUNZIONE PRINCIPALE DI ELABORAZIONE ---

def process_historical_data(trades_df: pd.DataFrame, fng_df: pd.DataFrame) -> pd.DataFrame:
    """Esegue Feature Engineering Avanzato e Etichettatura."""
    
    trades_df['side'] = np.where(trades_df[BUYER_MAKER_COL], 'SELL', 'BUY') 
    trades_df['second'] = trades_df['timestamp'].dt.floor('1S')
    
    # 1. Aggregazione Microstrutturale (Base 1s)
    second_data = trades_df.groupby('second').agg(
        price=('price', 'last'), 
        OFI=('qty', lambda x: trades_df.loc[x.index, 'qty'][trades_df.loc[x.index, 'side'] == 'BUY'].sum() - trades_df.loc[x.index, 'qty'][trades_df.loc[x.index, 'side'] == 'SELL'].sum()),
        trade_count=('price', 'count'),
        price_min=('price', 'min'),
        price_max=('price', 'max'),
        total_buy_qty=('qty', lambda x: trades_df.loc[x.index, 'qty'][trades_df.loc[x.index, 'side'] == 'BUY'].sum()),
        total_sell_qty=('qty', lambda x: trades_df.loc[x.index, 'qty'][trades_df.loc[x.index, 'side'] == 'SELL'].sum())
    ).reset_index().rename(columns={'second': 'timestamp'})
    
    second_data = second_data.set_index('timestamp').asfreq('1S').ffill().fillna(0).reset_index()
    
    # Calcolo Features Avanced
    second_data['Hawkes_Intensity'] = (second_data['trade_count'].rolling(window=10).mean() * 0.5) + 0.5 
    second_data['Spread_BPS'] = ((second_data['price_max'] - second_data['price_min']) / second_data['price']) * 10000 
    
    # ⭐ NUOVE FEATURE PREDITTIVE PER AUC:
    
    # 1. Delta OFI (Velocità di sbilanciamento)
    second_data['Delta_OFI'] = second_data['OFI'] - second_data['OFI'].rolling(window=5).mean().shift(1).fillna(0)
    
    # 2. Range Volatility (Proxy della Frizione di Mercato)
    second_data['Range_Volatility'] = (second_data['price_max'] - second_data['price_min']).rolling(window=60).mean().fillna(0)
    
    # 3. Volume Aggressivo Totale (Misura l'impatto)
    second_data['Aggressive_Volume'] = (second_data['total_buy_qty'] + second_data['total_sell_qty']).rolling(window=5).sum().fillna(0)
    
    # 4. Imbalance Ratio Assoluto (Unidirezionalità Estrema)
    second_data['Imbalance_Ratio_Abs'] = np.abs(second_data['OFI']) / second_data['Aggressive_Volume']
    
    # 5. Etichettatura Target (Y=Crash)
    second_data['future_min'] = second_data['price'].iloc[::-1].rolling(window=LOOK_AHEAD_WINDOW_SECONDS, min_periods=1).min().iloc[::-1]
    second_data['Target_Y'] = np.where(second_data['future_min'] / second_data['price'] < (1 - CRASH_THRESHOLD), 1, 0)
    
    # 6. Merge Sentiment
    fng_daily = fng_df[['Fear_Greed_Index']].resample('D').mean()
    second_data['date_only'] = second_data['timestamp'].dt.normalize()
    second_data = second_data.merge(fng_daily, left_on='date_only', right_index=True, how='left')
    second_data.rename(columns={'Fear_Greed_Index': 'Fear_Greed'}, inplace=True)
    second_data['Twitter_Sentiment'] = (second_data['Fear_Greed'] - 50) / 50 
    
    # Pulizia Finale
    final_data = second_data.dropna()
    final_data = final_data.iloc[:-LOOK_AHEAD_WINDOW_SECONDS]
    
    return final_data

def prepare_lstm_sequences(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Prepara le sequenze 3D (X) e i target (Y) per la LSTM."""
    
    # Vettore di 10 Feature Totali
    features = [
        'OFI', 'Hawkes_Intensity', 'price', 
        'Fear_Greed', 'Twitter_Sentiment', 
        'Spread_BPS', 'Delta_OFI', 
        'Range_Volatility', 'Aggressive_Volume', 'Imbalance_Ratio_Abs' 
    ]
    
    X = df[features].values
    Y = df['Target_Y'].values
    
    X_sequences = []; Y_targets = []
    for i in range(len(X) - SEQUENCE_LENGTH):
        X_sequences.append(X[i:i + SEQUENCE_LENGTH])
        Y_targets.append(Y[i + SEQUENCE_LENGTH - 1]) 
        
    return np.array(X_sequences), np.array(Y_targets)

# --- ESECUZIONE PRINCIPALE ---

if __name__ == '__main__':
    try:
        # Chiamate alle funzioni (allineate correttamente)
        trades_data = load_and_combine_data(TRADE_FILES_BASE) 
        fng_data = download_historical_fng()
        
        if fng_data.empty or trades_data.empty: raise Exception("Impossibile procedere senza dati completi.")

        processed_df = process_historical_data(trades_data, fng_data)
        
        X_train, Y_train = prepare_lstm_sequences(processed_df)

        # Normalizzazione
        MEAN = X_train.mean(axis=(0, 1)); STD = X_train.std(axis=(0, 1))
        X_train_normalized = (X_train - MEAN) / (STD + 1e-6)

        # Salvataggio
        np.save('X_train.npy', X_train_normalized); np.save('Y_train.npy', Y_train)
        np.save('normalization_mean.npy', MEAN); np.save('normalization_std.npy', STD)
        
        crash_count = np.sum(Y_train)
        print(f"--- Dataset pronto. Crash Rilevati (Y=1): {crash_count/len(Y_train)*100:.4f}% ---")
        
    except FileNotFoundError as e: print(f"\nFATALE: File non trovato. {e}")
    except Exception as e: print(f"\nFATALE: Errore nella pipeline di dati. {e}")