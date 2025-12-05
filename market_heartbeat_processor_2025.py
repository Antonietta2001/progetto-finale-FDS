"""
MARKET HEARTBEAT - Processor 2024 (ATTUALE)
Dataset: Gennaio-Novembre 2024 per trading Dicembre 2024 / Gennaio 2025
"""
import numpy as np
import pandas as pd
import zipfile
import io
from concurrent import futures
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURAZIONE ---
AGGREGATION_WINDOW = '30s'
PREDICTION_HORIZON = 2
MOVEMENT_THRESHOLD = 0.001
SEQUENCE_LENGTH = 3
CHUNK_DAYS = 7
BIG_MONTH_THRESHOLD = 50_000_000
MAX_WORKERS = 6

# Feature windows
VWAP_WINDOW = 1
OFI_CUMSUM_WINDOW = 2
VOLATILITY_SHORT = 1
VOLATILITY_LONG = 2
SKEW_KURTOSIS_WINDOW = 2
MOMENTUM_WINDOW = 4
TREND_WINDOW = 6

def load_single_month_raw(base_name: str):
    """Carica file ZIP mensile Binance."""
    file_name_zip = f"{base_name}.zip"
    
    print(f"   📦 Loading {base_name}...")
    
    with zipfile.ZipFile(file_name_zip, 'r') as z:
        csv_files = [name for name in z.namelist() if name.endswith('.csv')]
        target_csv_name = f"{base_name}.csv"
        
        if target_csv_name in csv_files:
            file_to_read = target_csv_name
        else:
            file_to_read = min(csv_files, key=len) if csv_files else csv_files[0]
        
        with z.open(file_to_read) as f:
            df = pd.read_csv(
                io.BytesIO(f.read()),
                header=None,
                names=['id', 'price', 'qty', 'quote_qty', 'timestamp', 'isBuyerMaker', 'isBestMatch'],
                dtype={'id': 'int64', 'price': 'float32', 'qty': 'float32', 
                       'quote_qty': 'float32', 'isBuyerMaker': 'bool'}
            )
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values(by='timestamp')
    df = df.drop_duplicates(subset=['timestamp', 'price'])
    
    print(f"      ✅ {len(df):,} trades loaded")
    return df

def compute_enhanced_features(df: pd.DataFrame):
    """Feature engineering ottimizzato."""
    print(f"   🔨 Computing features...")
    
    df['is_taker_sell'] = df['isBuyerMaker']
    df['is_taker_buy'] = ~df['isBuyerMaker']
    df['time_bucket'] = df['timestamp'].dt.floor(AGGREGATION_WINDOW)
    df['total_value'] = df['price'] * df['qty']
    
    taker_buy_volume = df[df['is_taker_buy']].groupby('time_bucket')['qty'].sum()
    taker_sell_volume = df[df['is_taker_sell']].groupby('time_bucket')['qty'].sum()
    
    agg_dict = {
        'price': ['last', 'min', 'max', 'std'],
        'qty': 'sum',
        'id': 'count',
        'total_value': 'sum'
    }
    
    bucket_data = df.groupby('time_bucket').agg(agg_dict)
    bucket_data.columns = ['price', 'price_min', 'price_max', 'price_std', 
                           'volume', 'trade_count', 'total_value']
    bucket_data = bucket_data.reset_index()
    bucket_data = bucket_data.rename(columns={'time_bucket': 'timestamp'})
    bucket_data = bucket_data.set_index('timestamp').asfreq(AGGREGATION_WINDOW).ffill().fillna(0).reset_index()
    
    taker_buy_series = taker_buy_volume.reindex(bucket_data['timestamp'], fill_value=0)
    taker_sell_series = taker_sell_volume.reindex(bucket_data['timestamp'], fill_value=0)
    
    # Core features
    bucket_data['VWAP_30s'] = (
        bucket_data['total_value'].rolling(VWAP_WINDOW, min_periods=1).sum() / 
        (bucket_data['volume'].rolling(VWAP_WINDOW, min_periods=1).sum() + 1e-8)
    )
    bucket_data['VWAP_Deviation'] = bucket_data['price'] / (bucket_data['VWAP_30s'] + 1e-8) - 1
    bucket_data['OFI_Taker'] = taker_buy_series.values - taker_sell_series.values
    
    taker_volume = taker_buy_series.values + taker_sell_series.values
    maker_volume_proxy = bucket_data['volume'] - taker_volume
    bucket_data['Taker_Maker_Ratio'] = taker_volume / (maker_volume_proxy + 1e-8)
    
    bucket_data['OFI_velocity'] = bucket_data['OFI_Taker'].diff()
    bucket_data['OFI_cumsum_60s'] = bucket_data['OFI_Taker'].rolling(OFI_CUMSUM_WINDOW, min_periods=1).sum()
    bucket_data['Hawkes_Intensity'] = bucket_data['trade_count'].ewm(span=2, adjust=False, min_periods=1).mean()
    
    bucket_data['Spread_BPS'] = (
        (bucket_data['price_max'] - bucket_data['price_min']) / bucket_data['price'] * 10000
    )
    bucket_data['Spread_BPS'] = bucket_data['Spread_BPS'].fillna(0)
    
    bucket_data['volatility_30s'] = bucket_data['price'].rolling(VOLATILITY_SHORT, min_periods=1).std()
    bucket_data['volatility_60s'] = bucket_data['price'].rolling(VOLATILITY_LONG, min_periods=1).std()
    bucket_data['volatility_ratio'] = bucket_data['volatility_30s'] / (bucket_data['volatility_60s'] + 1e-8)
    bucket_data['return_60s'] = bucket_data['price'].pct_change(OFI_CUMSUM_WINDOW)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        from scipy.stats import skew, kurtosis
        
        bucket_data['price_skew_60s'] = (
            bucket_data['price'].rolling(SKEW_KURTOSIS_WINDOW, min_periods=2)
            .apply(lambda x: skew(x.dropna()) if len(x.dropna()) >= 2 else 0, raw=False)
        )
        
        bucket_data['price_kurtosis_60s'] = (
            bucket_data['price'].rolling(SKEW_KURTOSIS_WINDOW, min_periods=2)
            .apply(lambda x: kurtosis(x.dropna()) if len(x.dropna()) >= 2 else 0, raw=False)
        )
    
    # Dynamic features
    bucket_data['momentum_30s'] = bucket_data['price'].diff(1)
    bucket_data['momentum_60s'] = bucket_data['price'].diff(2)
    bucket_data['momentum_120s'] = bucket_data['price'].diff(4)
    bucket_data['acceleration'] = bucket_data['momentum_30s'].diff()
    
    def compute_trend_strength(prices):
        if len(prices) < 3:
            return 0
        x = np.arange(len(prices))
        try:
            slope, intercept = np.polyfit(x, prices, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((prices - y_pred) ** 2)
            ss_tot = np.sum((prices - np.mean(prices)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            return r2 * np.sign(slope)
        except:
            return 0
    
    bucket_data['trend_strength'] = (
        bucket_data['price'].rolling(TREND_WINDOW, min_periods=3)
        .apply(compute_trend_strength, raw=True)
    )
    
    volume_ma = bucket_data['volume'].rolling(MOMENTUM_WINDOW, min_periods=1).mean()
    bucket_data['volume_surge'] = bucket_data['volume'] / (volume_ma + 1e-8)
    
    bucket_data['avg_trade_size'] = bucket_data['volume'] / (bucket_data['trade_count'] + 1e-8)
    avg_size_ma = bucket_data['avg_trade_size'].rolling(MOMENTUM_WINDOW, min_periods=1).mean()
    bucket_data['trade_size_anomaly'] = bucket_data['avg_trade_size'] / (avg_size_ma + 1e-8)
    
    bucket_data['price_range'] = (bucket_data['price_max'] - bucket_data['price_min']) / (bucket_data['price'] + 1e-8)
    range_ma = bucket_data['price_range'].rolling(MOMENTUM_WINDOW, min_periods=1).mean()
    bucket_data['range_expansion'] = bucket_data['price_range'] / (range_ma + 1e-8)
    
    bucket_data['OFI_momentum'] = bucket_data['OFI_Taker'].diff(2)
    
    vol_percentile_95 = bucket_data['volatility_30s'].rolling(20, min_periods=5).quantile(0.95)
    bucket_data['vol_breakout'] = (bucket_data['volatility_30s'] > vol_percentile_95).astype(float)
    
    bucket_data['cum_return_120s'] = (
        bucket_data['price'] / bucket_data['price'].shift(MOMENTUM_WINDOW) - 1
    )
    
    bucket_data['taker_sell_pct'] = taker_sell_series.values / (taker_volume + 1e-8)
    
    bucket_data = bucket_data.fillna(0)
    bucket_data = bucket_data.replace([np.inf, -np.inf], 0)
    
    print(f"      ✅ {len(bucket_data):,} buckets, {bucket_data.shape[1]} features")
    return bucket_data

def label_directional_movement(df: pd.DataFrame) -> pd.DataFrame:
    """Labeling directional."""
    print(f"\n🎯 Labeling directional movement...")
    
    df['future_price'] = df['price'].shift(-PREDICTION_HORIZON)
    df['future_return'] = (df['future_price'] / df['price']) - 1
    
    df['Target_Y'] = np.nan
    df.loc[df['future_return'] > MOVEMENT_THRESHOLD, 'Target_Y'] = 1.0
    df.loc[df['future_return'] < -MOVEMENT_THRESHOLD, 'Target_Y'] = 0.0
    
    df = df.dropna(subset=['Target_Y'])
    
    up_count = (df['Target_Y'] == 1).sum()
    down_count = (df['Target_Y'] == 0).sum()
    total = len(df)
    
    print(f"   UP: {up_count:,} ({up_count/total*100:.2f}%)")
    print(f"   DOWN: {down_count:,} ({down_count/total*100:.2f}%)")
    
    return df

def process_month_in_chunks(raw_df: pd.DataFrame, chunk_days: int):
    """Processa mese in chunk paralleli."""
    raw_df['day'] = raw_df['timestamp'].dt.date
    unique_days = sorted(raw_df['day'].unique())
    
    chunks_to_process = []
    for i in range(0, len(unique_days), chunk_days):
        day_chunk = unique_days[i:i+chunk_days]
        df_chunk = raw_df.loc[raw_df['day'].isin(day_chunk)].copy()
        df_chunk = df_chunk.drop(columns=['day'])
        chunks_to_process.append(df_chunk)
    
    processed_chunks = []
    
    with futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(compute_enhanced_features, chunk): i 
                     for i, chunk in enumerate(chunks_to_process)}
        
        for future in futures.as_completed(future_map):
            chunk_index = future_map[future] + 1
            try:
                processed_chunk = future.result()
                processed_chunks.append(processed_chunk)
                print(f"      ✅ Chunk {chunk_index} done")
            except Exception as e:
                print(f"      ❌ Error Chunk {chunk_index}: {e}")
    
    processed_chunks = sorted(processed_chunks, key=lambda x: x['timestamp'].min())
    final_df = pd.concat(processed_chunks, ignore_index=True)
    final_df = final_df.sort_values(by='timestamp').reset_index(drop=True)
    
    return final_df

def process_and_save_2024(months, output_prefix='DIRECTIONAL_2023_OTT'):
    """Pipeline completa per 2024."""
    print("="*80)
    print("🚀 MARKET HEARTBEAT - PROCESSOR 2024")
    print("   Dataset: Gennaio-Novembre 2024 (11 mesi)")
    print("   Target: Trading Dicembre 2024 / Gennaio 2025")
    print("="*80)
    
    all_data = []
    
    for month_base in months:
        print(f"\n📅 Processing: {month_base}")
        raw_df = load_single_month_raw(month_base)
        
        if len(raw_df) > BIG_MONTH_THRESHOLD:
            print(f"   → Large month, using chunking...")
            processed_df = process_month_in_chunks(raw_df, CHUNK_DAYS)
        else:
            processed_df = compute_enhanced_features(raw_df)
        
        all_data.append(processed_df)
    
    # Concatena
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(by='timestamp').reset_index(drop=True)
    
    print(f"\n📊 Combined Dataset 2024:")
    print(f"   Total buckets: {len(combined_df):,}")
    print(f"   Period: {combined_df['timestamp'].min()} → {combined_df['timestamp'].max()}")
    print(f"   Months: {len(months)}")
    
    # Labeling
    combined_df = label_directional_movement(combined_df)
    
    # Save parquet
    output_parquet = f'processed_data_{output_prefix}.parquet'
    combined_df.to_parquet(output_parquet, index=False)
    print(f"\n💾 Saved: {output_parquet}")
    
    # Prepare sequences
    print(f"\n🔄 Creating sequences...")
    
    feature_cols = [
        'OFI_Taker', 'OFI_velocity', 'OFI_cumsum_60s', 'VWAP_Deviation',
        'Taker_Maker_Ratio', 'Hawkes_Intensity', 'Spread_BPS',
        'volatility_30s', 'volatility_60s', 'volatility_ratio',
        'price_skew_60s', 'price_kurtosis_60s', 'return_60s', 'price_std',
        'volume', 'trade_count',
        'momentum_30s', 'momentum_60s', 'momentum_120s', 'acceleration',
        'trend_strength', 'volume_surge', 'trade_size_anomaly',
        'range_expansion', 'OFI_momentum', 'vol_breakout',
        'cum_return_120s', 'taker_sell_pct'
    ]
    
    X_features = combined_df[feature_cols].values
    Y_labels = combined_df['Target_Y'].values
    
    # Normalizzazione
    MEAN = X_features.mean(axis=0)
    STD = X_features.std(axis=0)
    X_normalized = (X_features - MEAN) / (STD + 1e-8)
    
    # Sequenze
    X_sequences = []
    Y_sequences = []
    
    for i in range(len(X_normalized) - SEQUENCE_LENGTH + 1):
        X_sequences.append(X_normalized[i:i + SEQUENCE_LENGTH])
        Y_sequences.append(Y_labels[i + SEQUENCE_LENGTH - 1])
    
    X_sequences = np.array(X_sequences, dtype=np.float32)
    Y_sequences = np.array(Y_sequences, dtype=np.float32)
    
    print(f"   X: {X_sequences.shape}")
    print(f"   Y: {Y_sequences.shape}")
    
    # Save
    np.save(f'X_train_{output_prefix}.npy', X_sequences)
    np.save(f'Y_train_{output_prefix}.npy', Y_sequences)
    np.save(f'normalization_mean_{output_prefix}.npy', MEAN)
    np.save(f'normalization_std_{output_prefix}.npy', STD)
    
    print(f"\n✅ PROCESSING 2024 COMPLETE")
    print(f"   Features: {len(feature_cols)} (NO sentiment - solo microstructure)")
    print(f"   Samples: {len(X_sequences):,}")
    print(f"   Ready for training!")
    
    return X_sequences, Y_sequences

if __name__ == '__main__':
    # ⭐ CONFIGURAZIONE 2024 (11 MESI)
    MONTHS = ['BTCUSDT-trades-2023-03']
    
    print("\n💡 IMPORTANTE:")
    print("   Training su Jan-Nov 2024 (11 mesi)")
    print("   Test su Dicembre 2024 (out-of-sample)")
    print("   Deploy su Gennaio 2025")
    print("\n⏱️  Tempo stimato: 2-3 ore\n")
    
    X, Y = process_and_save_2024(MONTHS, output_prefix='DIRECTIONAL_2023_MAR')
    
    print(f"\n🎯 Dataset 2024 pronto!")
    print(f"   Next: python market_heartbeat_trainer_2024.py")