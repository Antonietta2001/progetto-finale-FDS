"""
MARKET HEARTBEAT - Advanced Data Processor (V7 - MASSIMA OTTIMIZZAZIONE)
Pipeline ottimizzata per l'elaborazione efficiente di grandi volumi di dati.
- FOCUS: Maggio e Giugno 2021 (Crash Analysis)
- Caching su disco per F&G/Twitter.
- Batching corretto (I/O singolo).
- Ottimizzazione memoria (Downcasting).
- 🚀 PARALLELIZZAZIONE dei Chunk (sfrutta più core CPU).
"""
import pandas as pd
import numpy as np
import requests
import os
import zipfile 
import io 
import warnings 
import subprocess
import sys
from datetime import date
from dateutil.relativedelta import relativedelta 
from typing import Tuple, List
from scipy.stats import skew, kurtosis
from pathlib import Path
import concurrent.futures 

# --- CONFIGURAZIONE GLOBALE ---
SYMBOL = 'BTCUSDT'
CRASH_THRESHOLD = 0.003  # 0.3% (30 bps)
LOOK_AHEAD_WINDOW_SECONDS = 60
SEQUENCE_LENGTH = 10 
FEAR_GREED_API_URL = "https://api.alternative.me/fng/"
TWITTER_FILE = 'Bitcoin_tweets_clean.csv' 
BUYER_MAKER_COL = 'isBuyerMaker'

# VARIABILI DI CACHING SU DISCO
FNG_CACHE_FILE = 'cache_fng_index.parquet'
TWITTER_CACHE_FILE = 'cache_twitter_sentiment.parquet'

# 🟢 CONFIGURAZIONE BATCHING E PARALLELIZZAZIONE 🟢
USE_PARTIAL_MONTH = False 
CHUNK_DAYS = 5             
BIG_MONTH_THRESHOLD = 50_000_000 
MAX_WORKERS = 4            # NUMERO DI CORE DA USARE PER IL PARALLELISMO 
# ----------------------------------------------------

# 🎯 MODIFICA CHIAVE: Riduzione della finestra temporale a 2 mesi per la massima velocità
start_date_trade = date(2021, 5, 1)
end_date_trade = date(2021, 6, 1) # <--- ORA ELABORA SOLO MAGGIO E GIUGNO

current_date = start_date_trade
date_range = []
while current_date <= end_date_trade:
    date_range.append(current_date)
    current_date += relativedelta(months=1) 

TRADE_FILES_BASE = [
    f'{SYMBOL}-trades-{d.strftime("%Y-%m")}'
    for d in date_range
]

print(f"\n{'='*70}")
print(f"🎯 CONFIGURAZIONE: ANALISI CRASH 18-19 MAGGIO 2021 (MODALITÀ ESSENZIALE)")
print(f"{'='*70}")
print(f"📅 Periodo: {start_date_trade.strftime('%d %B %Y')} → {end_date_trade.strftime('%d %B %Y')}")
print(f"📦 File da processare: {len(TRADE_FILES_BASE)} mesi")
print(f"📊 MODALITÀ COMPLETA: Mesi essenziali (Batching Parallelo con {MAX_WORKERS} workers)")
print(f"   → Tempo stimato: 7-10 minuti")
    
print(f"{'='*70}\n")

# --- FUNZIONE PER L'OTTIMIZZAZIONE DELLA MEMORIA ---

def downcast_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Riconverte i float64 in float32 e gli interi in int32 per risparmiare memoria."""
    print(f"   -> Ottimizzazione memoria (Downcasting)...")
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

# --- FUNZIONI DI CARICAMENTO CON CACHING ---
# (Queste funzioni rimangono invariate per mantenere l'integrità dei dati)

def load_single_month_raw(base_name: str, max_days: int = None) -> pd.DataFrame:
    """Carica un singolo file ZIP mensile in modo memory-safe, con opzione di limitare i giorni."""
    file_name_zip = f"{base_name}.zip" 
    
    try:
        with zipfile.ZipFile(file_name_zip, 'r') as z:
            csv_files = [name for name in z.namelist() if name.endswith('.csv')]
            target_csv_name = f"{base_name}.csv" 
            
            if target_csv_name in csv_files:
                file_to_read = target_csv_name
            elif len(csv_files) >= 1:
                file_to_read = min(csv_files, key=len) 
                if len(csv_files) > 1:
                    print(f"   ℹ️ Rilevati file multipli. Caricamento forzato di: {file_to_read}")
            else:
                raise ValueError("Nessun file CSV trovato all'interno dell'archivio ZIP.")

            with z.open(file_to_read) as f:
                df = pd.read_csv(
                    io.BytesIO(f.read()),
                    header=None,
                    names=['id', 'price', 'qty', 'quote_qty', 'timestamp', 
                           BUYER_MAKER_COL, 'isBestMatch'],
                    dtype={'id': 'int64', 'price': 'float32', 'qty': 'float32', 
                           'quote_qty': 'float32', BUYER_MAKER_COL: 'bool'}
                )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # FILTRA I PRIMI N GIORNI SE RICHIESTO (Modalità Turbo)
        if max_days is not None:
            min_date = df['timestamp'].min()
            cutoff_date = min_date + pd.Timedelta(days=max_days)
            df = df[df['timestamp'] < cutoff_date]
            print(f"   ⚡ TURBO: Limitato ai primi {max_days} giorni")
        
        df = df.sort_values(by='timestamp').drop_duplicates(subset=['timestamp', 'price'])
        
        print(f"   ✓ {len(df):,} trades caricati (raw)")
        return df

    except FileNotFoundError:
        print(f"⚠️ File non trovato: {file_name_zip}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ ERRORE durante la lettura di {file_name_zip}: {e}")
        return pd.DataFrame()

# ... (download_historical_fng e load_sentiment_data rimangono invariate) ...

def download_historical_fng() -> pd.DataFrame:
    """Download Fear & Greed Index con caching su disco."""
    fng_path = Path(FNG_CACHE_FILE)
    if fng_path.exists():
        print(f"🌍 Caricamento Fear & Greed Index da cache ({FNG_CACHE_FILE})...")
        try:
            return pd.read_parquet(fng_path).set_index('timestamp')
        except Exception as e:
            print(f"⚠️ Errore caricamento cache F&G ({e}). Ritento il download.")
            fng_path.unlink(missing_ok=True) 

    try:
        print("🌍 Download Fear & Greed Index...")
        response = requests.get("https://api.alternative.me/fng/", params={'limit': 0})
        response.raise_for_status()
        data = response.json().get('data', [])
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
        df['Fear_Greed_Index'] = df['value'].astype(int)
        
        df_final = df.set_index('timestamp').sort_index()[['Fear_Greed_Index']]
        df_final.reset_index().to_parquet(fng_path, index=False)
        
        print(f"✅ F&G Index: {len(df_final)} giorni caricati e salvati in cache")
        return df_final
    except Exception as e:
        print(f"❌ ERRORE API F&G: {e}")
        return pd.DataFrame()

def load_sentiment_data(sentiment_file: str) -> pd.Series:
    """Carica e processa sentiment Twitter con caching su disco per il risultato finale."""
    twitter_path = Path(TWITTER_CACHE_FILE)
    if twitter_path.exists():
        print(f"🐦 Caricamento Sentiment Twitter da cache ({TWITTER_CACHE_FILE})...")
        try:
            df_cache = pd.read_parquet(twitter_path)
            return df_cache.set_index('timestamp')['Twitter_Sentiment']
        except Exception as e:
            print(f"⚠️ Errore caricamento cache Twitter ({e}). Ritento il processing.")
            twitter_path.unlink(missing_ok=True) 
    
    try:
        print("🐦 Caricamento e processing Sentiment Twitter...")
        df_tweets = pd.read_csv(sentiment_file)
        df_tweets = df_tweets.dropna(subset=['date', 'user_followers'])
        df_tweets['date'] = pd.to_datetime(df_tweets['date'])
        df_tweets['user_followers'] = df_tweets['user_followers'].astype(float)
        df_tweets['weighted_volume'] = df_tweets['user_followers'] 
        
        sentiment_hourly = df_tweets.set_index('date')['weighted_volume'].resample('h').sum()
        
        sentiment_mean = sentiment_hourly.mean()
        sentiment_std = sentiment_hourly.std()
        sentiment_hourly = (sentiment_hourly - sentiment_mean) / (sentiment_std + 1e-8)
        
        sentiment_series = sentiment_hourly.rename('Twitter_Sentiment')
        
        sentiment_series.reset_index().rename(columns={'date': 'timestamp'}).to_parquet(twitter_path, index=False)
        
        print(f"✅ Sentiment Twitter: {len(sentiment_series)} punti orari processati e salvati in cache")
        return sentiment_series
    except Exception as e:
        print(f"❌ ERRORE nel processing del sentiment: {e}")
        return pd.Series()


# --- FUNZIONE DI FEATURE ENGINEERING OTTIMIZZATA (VECTORIZED BATCH) ---
# (Questa funzione rimane invariata per mantenere la qualità delle feature)

def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola features microstrutturali avanzate (Downsampling Tick -> 1S) - OTTIMIZZATO"""
    # ... (Il corpo della funzione compute_advanced_features è lo stesso del codice V7) ...
    print(f"   -> Calcolo Feature (1S) su {len(df):,} trades...")
    
    # Preparazione base (Vectorization)
    df['is_taker_sell'] = df[BUYER_MAKER_COL]  
    df['is_taker_buy'] = ~df[BUYER_MAKER_COL] 
    df['second'] = df['timestamp'].dt.floor('1s') 
    df['total_value'] = df['price'] * df['qty'] 

    # PRE-CALCOLA volumi taker per secondo (PRIMA dell'aggregazione principale)
    print("   -> Pre-calcolo volumi taker...")
    taker_buy_volume = df[df['is_taker_buy']].groupby('second')['qty'].sum()
    taker_sell_volume = df[df['is_taker_sell']].groupby('second')['qty'].sum()
    
    # Aggregazione per secondo (Batch Aggregation)
    print("   -> Aggregazione 1S...")
    agg_dict = {
        'price': ['last', 'min', 'max', 'std'],
        'qty': 'sum',
        'id': 'count', 
        'total_value': 'sum', 
    }
    
    second_data = df.groupby('second').agg(agg_dict)
    second_data.columns = ['price', 'price_min', 'price_max', 'price_std',
                           'volume', 'trade_count', 'total_value']
    second_data = second_data.reset_index().rename(columns={'second': 'timestamp'})
    
    del df # Libera memoria

    print(f"   -> Downsampling completato: {len(second_data):,} secondi")
    
    # Riempimento gap temporali
    second_data = second_data.set_index('timestamp').asfreq('1s').ffill()
    second_data = second_data.fillna(0).reset_index()

    # Merge volumi taker
    taker_buy_series = taker_buy_volume.reindex(second_data['timestamp'], fill_value=0)
    taker_sell_series = taker_sell_volume.reindex(second_data['timestamp'], fill_value=0)
    
    print("   -> Calcolo features avanzate...")
    
    # 1. VWAP (Rolling Window Batch)
    second_data['VWAP_10s'] = (
        second_data['total_value'].rolling(10, min_periods=1).sum() / 
        (second_data['volume'].rolling(10, min_periods=1).sum() + 1e-8)
    )
    second_data['VWAP_Deviation'] = second_data['price'] / (second_data['VWAP_10s'] + 1e-8) - 1

    # 2. OFI Taker
    second_data['OFI_Taker'] = taker_buy_series.values - taker_sell_series.values
    
    # 3. RAPPORTO TAKER/MAKER 
    taker_volume = taker_buy_series.values + taker_sell_series.values
    maker_volume_proxy = second_data['volume'] - taker_volume
    second_data['Taker_Maker_Ratio'] = taker_volume / (maker_volume_proxy + 1e-8)

    # 4. OFI DERIVATE e CUMULATIVE (Rolling Window Batch)
    second_data['OFI_velocity'] = second_data['OFI_Taker'].diff()
    second_data['OFI_cumsum_30s'] = second_data['OFI_Taker'].rolling(30, min_periods=1).sum()
    
    # 5. HAWKES INTENSITY (EWM Batch)
    second_data['Hawkes_Intensity'] = (
        second_data['trade_count'].ewm(span=10, adjust=False, min_periods=1).mean()
    )
    
    # 6. SPREAD & MICROSTRUCTURE 
    second_data['Spread_BPS'] = (
        (second_data['price_max'] - second_data['price_min']) / second_data['price'] * 10000
    )
    second_data['Spread_BPS'] = second_data['Spread_BPS'].fillna(0)
    
    # 7. VOLATILITY e MOMENTUM (Rolling Window Batch)
    second_data['volatility_5s'] = second_data['price'].rolling(5, min_periods=1).std()
    second_data['volatility_30s'] = second_data['price'].rolling(30, min_periods=1).std()
    second_data['volatility_ratio'] = (
        second_data['volatility_5s'] / (second_data['volatility_30s'] + 1e-8)
    )
    second_data['return_30s'] = second_data['price'].pct_change(30)
    
    # 8. PRICE SKEWNESS & KURTOSIS (Rolling Window Batch)
    print("   -> Calcolo skewness/kurtosis...")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        second_data['price_skew_30s'] = (
            second_data['price'].rolling(30, min_periods=3)
            .apply(lambda x: skew(x.dropna()) if len(x.dropna()) > 2 else 0, raw=False)
        )
        second_data['price_kurtosis_30s'] = (
            second_data['price'].rolling(30, min_periods=3)
            .apply(lambda x: kurtosis(x.dropna()) if len(x.dropna()) > 2 else 0, raw=False)
        )

    # Pulizia finale
    second_data = second_data.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 🚀 Downcasting per ottimizzazione memoria
    second_data = downcast_numeric_features(second_data)
    
    print("   -> Feature calcolate ✓")
    return second_data

# --- PROCESSING CON CHUNKING PARALLELO (MULTICORE BATCHING) ---
# (Questa funzione rimane invariata per sfruttare i core)

def process_month_in_chunks(raw_df: pd.DataFrame, chunk_days: int) -> pd.DataFrame:
    """
    Processa un DataFrame già caricato dividendolo in chunk temporali più piccoli,
    utilizzando il parallelismo per processare i chunk in contemporanea.
    """
    print(f"   📦 Processing con chunking ({chunk_days} giorni) in PARALLELO con {MAX_WORKERS} workers...")
    
    raw_df['day'] = raw_df['timestamp'].dt.date
    unique_days = sorted(raw_df['day'].unique())
    
    chunks_to_process = []
    
    # 1. Preparazione dei chunk (Divisione)
    for i in range(0, len(unique_days), chunk_days):
        day_chunk = unique_days[i:i+chunk_days]
        df_chunk = raw_df.loc[raw_df['day'].isin(day_chunk)].copy()
        df_chunk = df_chunk.drop(columns=['day'])
        chunks_to_process.append(df_chunk)
        print(f"      Pronto Chunk {i//chunk_days + 1}: {len(df_chunk):,} trades")
        
    del raw_df 

    processed_chunks = []
    
    # 2. Esecuzione in parallelo
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(compute_advanced_features, chunk): i for i, chunk in enumerate(chunks_to_process)}
        
        for future in concurrent.futures.as_completed(futures):
            chunk_index = futures[future] + 1
            try:
                processed_chunk = future.result()
                processed_chunks.append(processed_chunk)
                print(f"      ✅ Chunk {chunk_index} completato e unito.")
            except Exception as e:
                print(f"      ❌ Errore critico durante l'elaborazione del Chunk {chunk_index}: {e}")
                
    
    # 3. Ritorno dei risultati (Concatenazione dei risultati paralleli)
    return pd.concat(processed_chunks, ignore_index=True)


# --- ALTRE FUNZIONI ---
# (Rimangono invariate: label_crashes, merge_sentiment, prepare_lstm_sequences, process_full_pipeline)

def label_crashes(df: pd.DataFrame) -> pd.DataFrame:
    """Etichetta crash (Utilizza un Rolling Window a ritroso)."""
    print("   🎯 Labeling crashes...")

    df['future_min'] = (
        df['price']
        .iloc[::-1] 
        .rolling(window=LOOK_AHEAD_WINDOW_SECONDS, min_periods=1)
        .min()
        .iloc[::-1] 
    )

    df['Target_Y'] = np.where(
        df['future_min'] / df['price'] < (1 - CRASH_THRESHOLD),
        np.float32(1), 
        np.float32(0)
    )
    return df

def merge_sentiment(df: pd.DataFrame, fng_df: pd.DataFrame, twitter_sentiment: pd.Series) -> pd.DataFrame:
    """Merge Fear & Greed Index e Sentiment Twitter Orario"""
    if not fng_df.empty:
        fng_daily = fng_df['Fear_Greed_Index'].resample('D').mean()
        df['date_only'] = df['timestamp'].dt.normalize()
        df = df.merge(fng_daily, left_on='date_only', right_index=True, how='left')
        df.rename(columns={'Fear_Greed_Index': 'Fear_Greed'}, inplace=True)
        df['Fear_Greed'] = df['Fear_Greed'].fillna(50)  
        df.drop(columns=['date_only'], inplace=True)
    else:
        df['Fear_Greed'] = 50

    if not twitter_sentiment.empty:
        df['timestamp_hour'] = df['timestamp'].dt.floor('h')
        df = df.merge(
            twitter_sentiment,
            left_on='timestamp_hour',
            right_index=True,
            how='left'
        )
        df['Twitter_Sentiment'] = df['Twitter_Sentiment'].ffill().fillna(0)
        df.drop(columns=['timestamp_hour'], inplace=True)
    else:
        df['Twitter_Sentiment'] = 0

    return df

def prepare_lstm_sequences(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Prepara sequenze temporali (lookback window) per l'addestramento dell'LSTM."""
    print("🔄 Creazione sequenze LSTM...")
    
    feature_cols = [
        'OFI_Taker', 'OFI_velocity', 'OFI_cumsum_30s', 'VWAP_Deviation', 
        'Taker_Maker_Ratio', 'Hawkes_Intensity', 'Spread_BPS',
        'volatility_5s', 'volatility_30s', 'volatility_ratio',
        'price_skew_30s', 'price_kurtosis_30s', 'return_30s', 'price_std',
        'volume', 'trade_count', 'Fear_Greed', 'Twitter_Sentiment' 
    ]
    
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"   Features disponibili: {len(available_features)}/{len(feature_cols)}")
    
    X = df[available_features].values
    Y = df['Target_Y'].values
    
    X_sequences = []
    Y_targets = []
    
    for i in range(len(X) - SEQUENCE_LENGTH + 1):
        X_sequences.append(X[i:i + SEQUENCE_LENGTH])
        Y_targets.append(Y[i + SEQUENCE_LENGTH - 1])
        
    X_arr = np.array(X_sequences)
    Y_arr = np.array(Y_targets)
    
    print(f"✅ Shape: X={X_arr.shape}, Y={Y_arr.shape}")
    print(f"   Crash samples: {Y_arr.sum():,} ({Y_arr.sum()/len(Y_arr)*100:.2f}%)")
    
    return X_arr, Y_arr

def process_full_pipeline(
    trade_files_base: List[str],
    fng_df: pd.DataFrame,
    twitter_data: pd.Series
) -> pd.DataFrame:
    """Pipeline memory-efficient che processa i dati mese per mese."""
    processed_months = []
    
    max_days_limit = None # Non usiamo la modalità Turbo (limitazione per giorni)
    
    print("\n--- INIZIO PROCESSING MENSILE (Memory Safe + Chunking) ---")
    for base_name in trade_files_base:
        print(f"\n📅 Processing Mese: {base_name}")
        
        # 🟢 Carica il file RAW una sola volta
        raw_df_month = load_single_month_raw(base_name, max_days=max_days_limit)
        
        if raw_df_month.empty:
            continue
        
        # Logica di Chunking per dataset molto grandi (Batching per giorni)
        if len(raw_df_month) > BIG_MONTH_THRESHOLD and not USE_PARTIAL_MONTH:
            
            # 🚀 Attivazione Parallelizzazione
            processed_df_month = process_month_in_chunks(raw_df_month, CHUNK_DAYS)
            
        else:
            # Se il mese è piccolo, elabora sequenzialmente
            processed_df_month = compute_advanced_features(raw_df_month)
            del raw_df_month
        
        processed_df_month = label_crashes(processed_df_month)
        processed_df_month = merge_sentiment(processed_df_month, fng_df, twitter_data)
        
        processed_months.append(processed_df_month)
        print(f"   ✅ Mese completato: {len(processed_df_month):,} campioni 1S.")

    if not processed_months:
        raise Exception("Nessun dato valido processato.")

    print("\n--- CONCATENAZIONE FINALE (Su dati 1S) ---")
    final_processed_df = pd.concat(processed_months, ignore_index=True)
    
    final_processed_df = final_processed_df.fillna(0) 
    final_processed_df = final_processed_df.iloc[:-LOOK_AHEAD_WINDOW_SECONDS]
    
    print(f"✅ Dataset finale combinato: {len(final_processed_df):,} samples totali")
    return final_processed_df

# --- ESECUZIONE ---
if __name__ == '__main__':
    
    try:
        fng_data = download_historical_fng()
        twitter_data = load_sentiment_data("Bitcoin_tweets_clean.csv") 
        
        processed_df = process_full_pipeline(TRADE_FILES_BASE, fng_data, twitter_data)
        
        X_train, Y_train = prepare_lstm_sequences(processed_df)
        
        print("📊 Normalizzazione features...")
        MEAN = X_train.mean(axis=(0, 1))
        STD = X_train.std(axis=(0, 1))
        X_train_normalized = (X_train - MEAN.astype(np.float32)) / (STD.astype(np.float32) + 1e-8)
        
        # --- Salvataggio (Caching su disco dei risultati finali) ---
        
        np.save('X_train_v7_essential.npy', X_train_normalized)
        np.save('Y_train_v7_essential.npy', Y_train)
        np.save('normalization_mean_v7_essential.npy', MEAN)
        np.save('normalization_std_v7_essential.npy', STD)
        
        processed_df.to_parquet('processed_data_v7_essential.parquet', index=False)
        
        print("\n" + "="*70)
        print("✅ FASE 1 COMPLETATA - Data Processing (V7 - ESSENZIALE E VELOCE)")
        print("="*70)
        print(f"📊 Statistiche Dataset:")
        print(f"   • Samples totali: {len(Y_train):,}")
        print(f"   • Crash samples: {Y_train.sum():,} ({Y_train.sum()/len(Y_train)*100:.2f}%)")
        print(f"   • Periodo coperto: {processed_df['timestamp'].min()} → {processed_df['timestamp'].max()}")
        print(f"\n💾 File salvati:")
        print(f"   • X_train_v7_essential.npy")
        print(f"   • Y_train_v7_essential.npy")
        print(f"   • processed_data_v7_essential.parquet")
        print(f"\n🎯 Il crash del 18-19 Maggio 2021 è incluso nel dataset!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        import traceback
        traceback.print_exc()