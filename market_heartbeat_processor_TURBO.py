"""
MARKET HEARTBEAT - Ultra-Fast Data Processor (TURBO MODE)
Pipeline MASSIMAMENTE ottimizzata per velocità estrema.
- AGGREGAZIONE: 30 secondi (30x riduzione dati vs 1s)
- FOCUS: Maggio e Giugno 2021 (Crash Analysis)
- Parallelizzazione massiva
- Riduzione drastica finestre temporali
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

# --- CONFIGURAZIONE ULTRA-VELOCE ---
SYMBOL = 'BTCUSDT'
CRASH_THRESHOLD = 0.003  # 0.3% (30 bps)

# 🚀 OTTIMIZZAZIONI CHIAVE PER VELOCITÀ MASSIMA
AGGREGATION_WINDOW = '30s'  # Da 1s a 30s = 30x meno dati
LOOK_AHEAD_WINDOW_BUCKETS = 2  # 2 buckets di 30s = 60s totali
SEQUENCE_LENGTH = 3  # 3 buckets di 30s = 90s di lookback (vs 10s prima)

# Finestre feature (in numero di buckets di 30s)
VWAP_WINDOW = 1       # 1 bucket = 30s
OFI_CUMSUM_WINDOW = 2 # 2 buckets = 60s (era 30s con 1s)
VOLATILITY_SHORT = 1  # 30s
VOLATILITY_LONG = 2   # 60s (era 30s con 1s)
MOMENTUM_WINDOW = 2   # 60s
SKEW_KURTOSIS_WINDOW = 2  # 60s

FEAR_GREED_API_URL = "https://api.alternative.me/fng/"
TWITTER_FILE = 'Bitcoin_tweets_clean.csv' 
BUYER_MAKER_COL = 'isBuyerMaker'

# CACHING
FNG_CACHE_FILE = 'cache_fng_index_turbo.parquet'
TWITTER_CACHE_FILE = 'cache_twitter_sentiment_turbo.parquet'

# PARALLELIZZAZIONE
USE_PARTIAL_MONTH = False 
CHUNK_DAYS = 7  # Chunk più grandi = meno overhead
BIG_MONTH_THRESHOLD = 50_000_000 
MAX_WORKERS = 6  # Usa più core se disponibili

# Range temporale
start_date_trade = date(2021, 5, 1)
end_date_trade = date(2021, 6, 1)

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
print(f"🚀 CONFIGURAZIONE: TURBO MODE - VELOCITÀ MASSIMA")
print(f"{'='*70}")
print(f"📅 Periodo: {start_date_trade.strftime('%d %B %Y')} → {end_date_trade.strftime('%d %B %Y')}")
print(f"⚡ Aggregazione: {AGGREGATION_WINDOW} (30x riduzione vs 1s)")
print(f"📦 Sequence Length: {SEQUENCE_LENGTH} buckets ({SEQUENCE_LENGTH * 30}s lookback)")
print(f"🔧 Workers paralleli: {MAX_WORKERS}")
print(f"⏱️  Tempo stimato: 30-60 secondi")
print(f"{'='*70}\n")

# --- OTTIMIZZAZIONE MEMORIA ---
def downcast_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Conversione aggressiva per minimizzare memoria."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            # Usa int16 quando possibile per conteggi
            if df[col].max() < 32767 and df[col].min() > -32768:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
    return df

# --- CARICAMENTO FILE ---
def load_single_month_raw(base_name: str, max_days: int = None) -> pd.DataFrame:
    """Carica singolo file ZIP mensile."""
    file_name_zip = f"{base_name}.zip" 
    
    try:
        with zipfile.ZipFile(file_name_zip, 'r') as z:
            csv_files = [name for name in z.namelist() if name.endswith('.csv')]
            target_csv_name = f"{base_name}.csv" 
            
            if target_csv_name in csv_files:
                file_to_read = target_csv_name
            elif len(csv_files) >= 1:
                file_to_read = min(csv_files, key=len)
            else:
                raise ValueError("Nessun file CSV trovato.")

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
        
        if max_days is not None:
            min_date = df['timestamp'].min()
            cutoff_date = min_date + pd.Timedelta(days=max_days)
            df = df[df['timestamp'] < cutoff_date]
            print(f"   ⚡ Limitato ai primi {max_days} giorni")
        
        df = df.sort_values(by='timestamp').drop_duplicates(subset=['timestamp', 'price'])
        
        print(f"   ✓ {len(df):,} trades caricati")
        return df

    except FileNotFoundError:
        print(f"⚠️ File non trovato: {file_name_zip}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        return pd.DataFrame()

# --- FEAR & GREED INDEX ---
def download_historical_fng() -> pd.DataFrame:
    """Download Fear & Greed Index con caching."""
    fng_path = Path(FNG_CACHE_FILE)
    if fng_path.exists():
        print(f"🌍 Caricamento F&G da cache...")
        try:
            return pd.read_parquet(fng_path).set_index('timestamp')
        except Exception as e:
            print(f"⚠️ Errore cache: {e}")
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
        
        print(f"✅ F&G: {len(df_final)} giorni")
        return df_final
    except Exception as e:
        print(f"❌ ERRORE F&G: {e}")
        return pd.DataFrame()

# --- TWITTER SENTIMENT ---
def load_sentiment_data(sentiment_file: str) -> pd.Series:
    """Carica sentiment Twitter con caching."""
    twitter_path = Path(TWITTER_CACHE_FILE)
    if twitter_path.exists():
        print(f"🐦 Caricamento Sentiment da cache...")
        try:
            df_cache = pd.read_parquet(twitter_path)
            return df_cache.set_index('timestamp')['Twitter_Sentiment']
        except Exception as e:
            print(f"⚠️ Errore cache: {e}")
            twitter_path.unlink(missing_ok=True)
    
    try:
        print("🐦 Processing Sentiment Twitter...")
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
        
        print(f"✅ Sentiment: {len(sentiment_series)} punti orari")
        return sentiment_series
    except Exception as e:
        print(f"❌ ERRORE sentiment: {e}")
        return pd.Series()

# --- FEATURE ENGINEERING ULTRA-VELOCE ---
def compute_advanced_features_turbo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering ULTRA-OTTIMIZZATO con aggregazione a 30s.
    Riduzione 30x del numero di samples rispetto alla versione 1s.
    """
    print(f"   -> 🚀 TURBO Feature Engineering ({AGGREGATION_WINDOW}) su {len(df):,} trades...")
    
    # Preparazione base
    df['is_taker_sell'] = df[BUYER_MAKER_COL]  
    df['is_taker_buy'] = ~df[BUYER_MAKER_COL] 
    df['time_bucket'] = df['timestamp'].dt.floor(AGGREGATION_WINDOW)  # 🔥 CHIAVE: 30s
    df['total_value'] = df['price'] * df['qty']

    # Pre-calcola volumi taker
    taker_buy_volume = df[df['is_taker_buy']].groupby('time_bucket')['qty'].sum()
    taker_sell_volume = df[df['is_taker_sell']].groupby('time_bucket')['qty'].sum()
    
    # Aggregazione principale (batch)
    agg_dict = {
        'price': ['last', 'min', 'max', 'std'],
        'qty': 'sum',
        'id': 'count', 
        'total_value': 'sum', 
    }
    
    bucket_data = df.groupby('time_bucket').agg(agg_dict)
    bucket_data.columns = ['price', 'price_min', 'price_max', 'price_std',
                           'volume', 'trade_count', 'total_value']
    bucket_data = bucket_data.reset_index().rename(columns={'time_bucket': 'timestamp'})
    
    del df  # Libera memoria

    print(f"   -> Downsampling: {len(bucket_data):,} buckets di {AGGREGATION_WINDOW}")
    
    # Riempimento gap temporali
    bucket_data = bucket_data.set_index('timestamp').asfreq(AGGREGATION_WINDOW).ffill()
    bucket_data = bucket_data.fillna(0).reset_index()

    # Merge volumi taker
    taker_buy_series = taker_buy_volume.reindex(bucket_data['timestamp'], fill_value=0)
    taker_sell_series = taker_sell_volume.reindex(bucket_data['timestamp'], fill_value=0)
    
    print("   -> Calcolo features ottimizzate...")
    
    # 1. VWAP (1 bucket = 30s)
    bucket_data['VWAP_30s'] = (
        bucket_data['total_value'].rolling(VWAP_WINDOW, min_periods=1).sum() / 
        (bucket_data['volume'].rolling(VWAP_WINDOW, min_periods=1).sum() + 1e-8)
    )
    bucket_data['VWAP_Deviation'] = bucket_data['price'] / (bucket_data['VWAP_30s'] + 1e-8) - 1

    # 2. OFI Taker
    bucket_data['OFI_Taker'] = taker_buy_series.values - taker_sell_series.values
    
    # 3. Taker/Maker Ratio
    taker_volume = taker_buy_series.values + taker_sell_series.values
    maker_volume_proxy = bucket_data['volume'] - taker_volume
    bucket_data['Taker_Maker_Ratio'] = taker_volume / (maker_volume_proxy + 1e-8)

    # 4. OFI Derivatives (2 buckets = 60s)
    bucket_data['OFI_velocity'] = bucket_data['OFI_Taker'].diff()
    bucket_data['OFI_cumsum_60s'] = bucket_data['OFI_Taker'].rolling(OFI_CUMSUM_WINDOW, min_periods=1).sum()
    
    # 5. Hawkes Intensity (finestra ridotta)
    bucket_data['Hawkes_Intensity'] = (
        bucket_data['trade_count'].ewm(span=2, adjust=False, min_periods=1).mean()
    )
    
    # 6. Spread
    bucket_data['Spread_BPS'] = (
        (bucket_data['price_max'] - bucket_data['price_min']) / bucket_data['price'] * 10000
    )
    bucket_data['Spread_BPS'] = bucket_data['Spread_BPS'].fillna(0)
    
    # 7. Volatility e Momentum (finestre ridotte)
    bucket_data['volatility_30s'] = bucket_data['price'].rolling(VOLATILITY_SHORT, min_periods=1).std()
    bucket_data['volatility_60s'] = bucket_data['price'].rolling(VOLATILITY_LONG, min_periods=1).std()
    bucket_data['volatility_ratio'] = (
        bucket_data['volatility_30s'] / (bucket_data['volatility_60s'] + 1e-8)
    )
    bucket_data['return_60s'] = bucket_data['price'].pct_change(MOMENTUM_WINDOW)
    
    # 8. Skewness & Kurtosis (finestre minime)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        bucket_data['price_skew_60s'] = (
            bucket_data['price'].rolling(SKEW_KURTOSIS_WINDOW, min_periods=2)
            .apply(lambda x: skew(x.dropna()) if len(x.dropna()) >= 2 else 0, raw=False)
        )
        bucket_data['price_kurtosis_60s'] = (
            bucket_data['price'].rolling(SKEW_KURTOSIS_WINDOW, min_periods=2)
            .apply(lambda x: kurtosis(x.dropna()) if len(x.dropna()) >= 2 else 0, raw=False)
        )

    # Pulizia e downcasting
    bucket_data = bucket_data.replace([np.inf, -np.inf], np.nan).fillna(0)
    bucket_data = downcast_numeric_features(bucket_data)
    
    print("   -> ✅ Features calcolate (TURBO)")
    return bucket_data

# --- CHUNKING PARALLELO ---
def process_month_in_chunks(raw_df: pd.DataFrame, chunk_days: int) -> pd.DataFrame:
    """Processa chunk in parallelo."""
    print(f"   📦 Chunking parallelo ({chunk_days}gg) con {MAX_WORKERS} workers...")
    
    raw_df['day'] = raw_df['timestamp'].dt.date
    unique_days = sorted(raw_df['day'].unique())
    
    chunks_to_process = []
    
    for i in range(0, len(unique_days), chunk_days):
        day_chunk = unique_days[i:i+chunk_days]
        df_chunk = raw_df.loc[raw_df['day'].isin(day_chunk)].copy()
        df_chunk = df_chunk.drop(columns=['day'])
        chunks_to_process.append(df_chunk)
        
    del raw_df 

    processed_chunks = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(compute_advanced_features_turbo, chunk): i for i, chunk in enumerate(chunks_to_process)}
        
        for future in concurrent.futures.as_completed(futures):
            chunk_index = futures[future] + 1
            try:
                processed_chunk = future.result()
                processed_chunks.append(processed_chunk)
                print(f"      ✅ Chunk {chunk_index} completato")
            except Exception as e:
                print(f"      ❌ Errore Chunk {chunk_index}: {e}")
    
    return pd.concat(processed_chunks, ignore_index=True)

# --- LABELING CRASHES ---
def label_crashes(df: pd.DataFrame) -> pd.DataFrame:
    """Etichetta crash (2 buckets = 60s con aggregazione 30s)."""
    print("   🎯 Labeling crashes...")

    df['future_min'] = (
        df['price']
        .iloc[::-1] 
        .rolling(window=LOOK_AHEAD_WINDOW_BUCKETS, min_periods=1)
        .min()
        .iloc[::-1] 
    )

    df['Target_Y'] = np.where(
        df['future_min'] / df['price'] < (1 - CRASH_THRESHOLD),
        np.float32(1), 
        np.float32(0)
    )
    return df

# --- MERGE SENTIMENT ---
def merge_sentiment(df: pd.DataFrame, fng_df: pd.DataFrame, twitter_sentiment: pd.Series) -> pd.DataFrame:
    """Merge Fear & Greed e Sentiment."""
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

# --- SEQUENZE LSTM ---
def prepare_lstm_sequences(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Prepara sequenze LSTM."""
    print("🔄 Creazione sequenze LSTM (TURBO)...")
    
    feature_cols = [
        'OFI_Taker', 'OFI_velocity', 'OFI_cumsum_60s', 'VWAP_Deviation', 
        'Taker_Maker_Ratio', 'Hawkes_Intensity', 'Spread_BPS',
        'volatility_30s', 'volatility_60s', 'volatility_ratio',
        'price_skew_60s', 'price_kurtosis_60s', 'return_60s', 'price_std',
        'volume', 'trade_count', 'Fear_Greed', 'Twitter_Sentiment' 
    ]
    
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"   Features: {len(available_features)}/{len(feature_cols)}")
    
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
    print(f"   Crash: {Y_arr.sum():,} ({Y_arr.sum()/len(Y_arr)*100:.2f}%)")
    
    return X_arr, Y_arr

# --- PIPELINE COMPLETA ---
def process_full_pipeline(
    trade_files_base: List[str],
    fng_df: pd.DataFrame,
    twitter_data: pd.Series
) -> pd.DataFrame:
    """Pipeline ultra-veloce."""
    processed_months = []
    
    print("\n--- PROCESSING TURBO MODE ---")
    for base_name in trade_files_base:
        print(f"\n📅 Mese: {base_name}")
        
        raw_df_month = load_single_month_raw(base_name, max_days=None)
        
        if raw_df_month.empty:
            continue
        
        if len(raw_df_month) > BIG_MONTH_THRESHOLD:
            processed_df_month = process_month_in_chunks(raw_df_month, CHUNK_DAYS)
        else:
            processed_df_month = compute_advanced_features_turbo(raw_df_month)
            del raw_df_month
        
        processed_df_month = label_crashes(processed_df_month)
        processed_df_month = merge_sentiment(processed_df_month, fng_df, twitter_data)
        
        processed_months.append(processed_df_month)
        print(f"   ✅ Completato: {len(processed_df_month):,} samples")

    if not processed_months:
        raise Exception("Nessun dato processato.")

    print("\n--- CONCATENAZIONE FINALE ---")
    final_processed_df = pd.concat(processed_months, ignore_index=True)
    
    final_processed_df = final_processed_df.fillna(0) 
    final_processed_df = final_processed_df.iloc[:-LOOK_AHEAD_WINDOW_BUCKETS]
    
    print(f"✅ Dataset finale: {len(final_processed_df):,} samples")
    return final_processed_df

# --- MAIN ---
if __name__ == '__main__':
    
    try:
        fng_data = download_historical_fng()
        twitter_data = load_sentiment_data("Bitcoin_tweets_clean.csv") 
        
        processed_df = process_full_pipeline(TRADE_FILES_BASE, fng_data, twitter_data)
        
        X_train, Y_train = prepare_lstm_sequences(processed_df)
        
        print("📊 Normalizzazione...")
        MEAN = X_train.mean(axis=(0, 1))
        STD = X_train.std(axis=(0, 1))
        X_train_normalized = (X_train - MEAN.astype(np.float32)) / (STD.astype(np.float32) + 1e-8)
        
        # Salvataggio con nomi univoci
        np.save('X_train_TURBO.npy', X_train_normalized)
        np.save('Y_train_TURBO.npy', Y_train)
        np.save('normalization_mean_TURBO.npy', MEAN)
        np.save('normalization_std_TURBO.npy', STD)
        
        processed_df.to_parquet('processed_data_TURBO.parquet', index=False)
        
        print("\n" + "="*70)
        print("✅ TURBO MODE COMPLETATO - Data Processing")
        print("="*70)
        print(f"📊 Statistiche:")
        print(f"   • Samples: {len(Y_train):,}")
        print(f"   • Crash: {Y_train.sum():,} ({Y_train.sum()/len(Y_train)*100:.2f}%)")
        print(f"   • Aggregazione: {AGGREGATION_WINDOW}")
        print(f"   • Lookback: {SEQUENCE_LENGTH} buckets ({SEQUENCE_LENGTH * 30}s)")
        print(f"   • Periodo: {processed_df['timestamp'].min()} → {processed_df['timestamp'].max()}")
        print(f"\n💾 File salvati:")
        print(f"   • X_train_TURBO.npy")
        print(f"   • Y_train_TURBO.npy")
        print(f"   • processed_data_TURBO.parquet")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()