# =========================
# DATA PROCESSING 
# =========================

import numpy as np
import pandas as pd
import zipfile
import io
import warnings
from concurrent import futures
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")

# CONFIG
AGGREGATION_WINDOW = "30s"
PREDICTION_HORIZON = 10
MOVEMENT_THRESHOLD = 0.001
SEQUENCE_LENGTH = 3

CHUNK_DAYS = 7
BIG_MONTH_THRESHOLD = 50_000_000
MAX_WORKERS = 6

VWAP_WINDOW = 1
OFI_CUMSUM_WINDOW = 2
VOLATILITY_SHORT = 1
VOLATILITY_LONG = 2
SKEW_KURTOSIS_WINDOW = 2
MOMENTUM_WINDOW = 4
TREND_WINDOW = 6

OUTPUT_PREFIX = "DIRECTIONAL_2023_2024_FULLSPAN"

# FEATURES USED FOR ML (ORDER MATTERS)
FEATURE_COLS = [
    "OFI_Taker","OFI_velocity","OFI_cumsum_60s","VWAP_Deviation",
    "Taker_Maker_Ratio","Hawkes_Intensity","Spread_BPS",
    "volatility_30s","volatility_60s","volatility_ratio",
    "price_skew_60s","price_kurtosis_60s","return_60s","price_std",
    "volume","trade_count",
    "momentum_30s","momentum_60s","momentum_120s","acceleration",
    "trend_strength","volume_surge","trade_size_anomaly",
    "range_expansion","OFI_momentum","vol_breakout",
    "cum_return_120s","taker_sell_pct",
]

# LOAD RAW
def load_single_month_raw(base_name: str) -> pd.DataFrame:
    file_zip = f"{base_name}.zip"
    print(f"  Loading {base_name}...")

    with zipfile.ZipFile(file_zip, "r") as z:
        csv_files = [n for n in z.namelist() if n.endswith(".csv")]
        target = f"{base_name}.csv"
        if target in csv_files:
            chosen = target
        else:
            chosen = min(csv_files, key=len) if csv_files else csv_files[0]
        with z.open(chosen) as f:
            raw = f.read()

    df = pd.read_csv(
        io.BytesIO(raw),
        header=None,
        names=["id", "price", "qty", "quote_qty", "timestamp", "isBuyerMaker", "isBestMatch"],
        usecols=["id", "price", "qty", "timestamp", "isBuyerMaker"],
        dtype={"id": "int64", "price": "float32", "qty": "float32", "isBuyerMaker": "bool"},
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp", "price"], inplace=True)

    print(f" {len(df):,} trades loaded")
    return df

# FEATURES
def compute_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  Computing features...")

    df["is_taker_sell"] = df["isBuyerMaker"]
    df["is_taker_buy"] = ~df["isBuyerMaker"]
    df["time_bucket"] = df["timestamp"].dt.floor(AGGREGATION_WINDOW)
    df["total_value"] = df["price"] * df["qty"]

    is_taker_buy = df["is_taker_buy"]
    is_taker_sell = df["is_taker_sell"]
    time_bucket = df["time_bucket"]

    taker_buy_volume = df.loc[is_taker_buy].groupby(time_bucket)["qty"].sum()
    taker_sell_volume = df.loc[is_taker_sell].groupby(time_bucket)["qty"].sum()

    agg_dict = {
        "price": ["last", "min", "max", "std"],
        "qty": "sum",
        "id": "count",
        "total_value": "sum",
    }

    bucket = df.groupby("time_bucket").agg(agg_dict)
    bucket.columns = [
        "price",
        "price_min",
        "price_max",
        "price_std",
        "volume",
        "trade_count",
        "total_value",
    ]

    bucket.reset_index(inplace=True)
    bucket.rename(columns={"time_bucket": "timestamp"}, inplace=True)

    bucket = (
        bucket.set_index("timestamp")
              .asfreq(AGGREGATION_WINDOW)
              .ffill()
              .fillna(0)
              .reset_index()
    )

    ts = bucket["timestamp"]
    taker_buy_series = taker_buy_volume.reindex(ts, fill_value=0)
    taker_sell_series = taker_sell_volume.reindex(ts, fill_value=0)

    vol_roll = bucket["volume"].rolling(VWAP_WINDOW, min_periods=1)
    val_roll = bucket["total_value"].rolling(VWAP_WINDOW, min_periods=1)
    bucket["VWAP_30s"] = val_roll.sum() / (vol_roll.sum() + 1e-8)
    bucket["VWAP_Deviation"] = bucket["price"] / (bucket["VWAP_30s"] + 1e-8) - 1

    bucket["OFI_Taker"] = taker_buy_series.values - taker_sell_series.values
    taker_volume = taker_buy_series.values + taker_sell_series.values
    maker_volume_proxy = bucket["volume"].values - taker_volume
    bucket["Taker_Maker_Ratio"] = taker_volume / (maker_volume_proxy + 1e-8)

    bucket["OFI_velocity"] = bucket["OFI_Taker"].diff()
    bucket["OFI_cumsum_60s"] = bucket["OFI_Taker"].rolling(OFI_CUMSUM_WINDOW, min_periods=1).sum()
    bucket["Hawkes_Intensity"] = bucket["trade_count"].ewm(span=2, adjust=False, min_periods=1).mean()

    bucket["Spread_BPS"] = (bucket["price_max"] - bucket["price_min"]) / bucket["price"] * 10000
    bucket["Spread_BPS"].fillna(0, inplace=True)

    bucket["volatility_30s"] = bucket["price"].rolling(VOLATILITY_SHORT, min_periods=1).std()
    bucket["volatility_60s"] = bucket["price"].rolling(VOLATILITY_LONG, min_periods=1).std()
    bucket["volatility_ratio"] = bucket["volatility_30s"] / (bucket["volatility_60s"] + 1e-8)

    bucket["return_60s"] = bucket["price"].pct_change(OFI_CUMSUM_WINDOW)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        bucket["price_skew_60s"] = bucket["price"].rolling(SKEW_KURTOSIS_WINDOW, min_periods=2).apply(
            lambda x: skew(x.dropna()) if len(x.dropna()) >= 2 else 0,
            raw=False
        )

        bucket["price_kurtosis_60s"] = bucket["price"].rolling(SKEW_KURTOSIS_WINDOW, min_periods=2).apply(
            lambda x: kurtosis(x.dropna()) if len(x.dropna()) >= 2 else 0,
            raw=False
        )

    bucket["momentum_30s"] = bucket["price"].diff(1)
    bucket["momentum_60s"] = bucket["price"].diff(2)
    bucket["momentum_120s"] = bucket["price"].diff(4)
    bucket["acceleration"] = bucket["momentum_30s"].diff()

    def compute_trend_strength(prices):
        if len(prices) < 3:
            return 0.0
        x = np.arange(len(prices), dtype=np.float32)
        try:
            slope, intercept = np.polyfit(x, prices, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((prices - y_pred) ** 2)
            ss_tot = np.sum((prices - np.mean(prices)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            return r2 * np.sign(slope)
        except Exception:
            return 0.0

    bucket["trend_strength"] = bucket["price"].rolling(TREND_WINDOW, min_periods=3).apply(
        compute_trend_strength, raw=True
    )

    volume_ma = bucket["volume"].rolling(MOMENTUM_WINDOW, min_periods=1).mean()
    bucket["volume_surge"] = bucket["volume"] / (volume_ma + 1e-8)

    bucket["avg_trade_size"] = bucket["volume"] / (bucket["trade_count"] + 1e-8)
    avg_size_ma = bucket["avg_trade_size"].rolling(MOMENTUM_WINDOW, min_periods=1).mean()
    bucket["trade_size_anomaly"] = bucket["avg_trade_size"] / (avg_size_ma + 1e-8)

    bucket["price_range"] = (bucket["price_max"] - bucket["price_min"]) / (bucket["price"] + 1e-8)
    range_ma = bucket["price_range"].rolling(MOMENTUM_WINDOW, min_periods=1).mean()
    bucket["range_expansion"] = bucket["price_range"] / (range_ma + 1e-8)

    bucket["OFI_momentum"] = bucket["OFI_Taker"].diff(2)

    vol_p95 = bucket["volatility_30s"].rolling(20, min_periods=5).quantile(0.95)
    bucket["vol_breakout"] = (bucket["volatility_30s"] > vol_p95).astype(float)

    bucket["cum_return_120s"] = bucket["price"] / bucket["price"].shift(MOMENTUM_WINDOW) - 1
    bucket["taker_sell_pct"] = taker_sell_series.values / (taker_volume + 1e-8)

    bucket.replace([np.inf, -np.inf], 0, inplace=True)
    bucket.fillna(0, inplace=True)

    print(f" {len(bucket):,} buckets, {bucket.shape[1]} features")
    return bucket

# LABEL (H10)
def label_directional_movement(df: pd.DataFrame) -> pd.DataFrame:
    print("\n Labeling directional movement...")

    df["future_price"] = df["price"].shift(-PREDICTION_HORIZON)
    df["future_return"] = (df["future_price"] / df["price"]) - 1

    df["Target_Y"] = np.nan
    df.loc[df["future_return"] > MOVEMENT_THRESHOLD, "Target_Y"] = 1.0
    df.loc[df["future_return"] < -MOVEMENT_THRESHOLD, "Target_Y"] = 0.0

    df = df.dropna(subset=["Target_Y"])

    up = int((df["Target_Y"] == 1).sum())
    down = int((df["Target_Y"] == 0).sum())
    total = len(df)

    print(f"   UP:   {up:,} ({up / total * 100:.2f}%)")
    print(f"   DOWN: {down:,} ({down / total * 100:.2f}%)")
    return df

# CHUNKING
def process_month_in_chunks(raw_df: pd.DataFrame, chunk_days: int) -> pd.DataFrame:
    raw_df["day"] = raw_df["timestamp"].dt.date
    unique_days = sorted(raw_df["day"].unique())

    chunks = []
    for i in range(0, len(unique_days), chunk_days):
        day_chunk = unique_days[i:i + chunk_days]
        part = raw_df.loc[raw_df["day"].isin(day_chunk)].copy()
        part.drop(columns=["day"], inplace=True)
        chunks.append(part)

    processed = []
    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut_map = {ex.submit(compute_enhanced_features, c): idx for idx, c in enumerate(chunks)}
        for fut in futures.as_completed(fut_map):
            idx = fut_map[fut] + 1
            out = fut.result()
            processed.append(out)
            print(f" Chunk {idx} done")

    processed.sort(key=lambda x: x["timestamp"].min())
    final_df = pd.concat(processed, ignore_index=True)
    final_df.sort_values("timestamp", inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    return final_df

# PIPELINE
def process_and_save_2023_2024(months, output_prefix=OUTPUT_PREFIX):
    print("=" * 80)
    print(" MARKET HEARTBEAT - PROCESSOR 2023-2024 (FULL SPAN)")
    print("   Dataset: Jan 2023 – Dec 2024")
    print(f"   Target:  LSTM intraday con horizon H10 = {PREDICTION_HORIZON} barre")
    print("=" * 80)

    all_months = []
    for month_base in months:
        print(f"\n Processing: {month_base}")
        raw_df = load_single_month_raw(month_base)

        if len(raw_df) > BIG_MONTH_THRESHOLD:
            print("   → Large month, using chunking...")
            month_df = process_month_in_chunks(raw_df, CHUNK_DAYS)
        else:
            month_df = compute_enhanced_features(raw_df)

        all_months.append(month_df)

    combined = pd.concat(all_months, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    print("\n Combined Dataset (2023JAN–2024DEC):")
    print(f"   Total buckets: {len(combined):,}")
    print(f"   Period: {combined['timestamp'].min()} → {combined['timestamp'].max()}")
    print(f"   Months: {len(months)}")

    combined = label_directional_movement(combined)

    out_parquet = f"processed_data_{output_prefix}.parquet"
    combined.to_parquet(out_parquet, index=False)
    print(f"\n Saved: {out_parquet}")

    print("\n Creating sequences...")

    X = combined[FEATURE_COLS].values
    Y = combined["Target_Y"].values

    MEAN = X.mean(axis=0)
    STD = X.std(axis=0)
    Xn = (X - MEAN) / (STD + 1e-8)

    X_seq, Y_seq = [], []
    limit = len(Xn) - SEQUENCE_LENGTH + 1
    for i in range(limit):
        X_seq.append(Xn[i:i + SEQUENCE_LENGTH])
        Y_seq.append(Y[i + SEQUENCE_LENGTH - 1])

    X_seq = np.array(X_seq, dtype=np.float32)
    Y_seq = np.array(Y_seq, dtype=np.float32)

    n_total = len(X_seq)
    print("\n Sequenze generate (full dataset):")
    print(f"   X (full): {X_seq.shape}")
    print(f"   Y (full): {Y_seq.shape}")
    print(f"   Totale campioni (sequenze): {n_total:,}")

    np.save(f"X_all_{output_prefix}.npy", X_seq)
    np.save(f"Y_all_{output_prefix}.npy", Y_seq)
    np.save(f"normalization_mean_{output_prefix}.npy", MEAN)
    np.save(f"normalization_std_{output_prefix}.npy", STD)

    train_ratio = 0.70
    val_ratio = 0.15

    train_end = int(n_total * train_ratio)
    val_end = train_end + int(n_total * val_ratio)

    X_train = X_seq[:train_end]
    Y_train = Y_seq[:train_end]

    X_val = X_seq[train_end:val_end]
    Y_val = Y_seq[train_end:val_end]

    X_test = X_seq[val_end:]
    Y_test = Y_seq[val_end:]

    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)

    print("\n Dataset split (sequenze):")
    print(f"   Train: {n_train:,}  ({n_train / n_total * 100:.2f}%)")
    print(f"   Val:   {n_val:,}  ({n_val / n_total * 100:.2f}%)")
    print(f"   Test:  {n_test:,}  ({n_test / n_total * 100:.2f}%)")
    print(f"   Check: {n_train + n_val + n_test:,} == {n_total:,}")

    np.save(f"X_train_{output_prefix}.npy", X_train)
    np.save(f"Y_train_{output_prefix}.npy", Y_train)
    np.save(f"X_val_{output_prefix}.npy", X_val)
    np.save(f"Y_val_{output_prefix}.npy", Y_val)
    np.save(f"X_test_{output_prefix}.npy", X_test)
    np.save(f"Y_test_{output_prefix}.npy", Y_test)

    print("\n PROCESSING 2023-2024 COMPLETE (FULL SPAN + SPLIT 70/15/15)")
    print(
        f"   Features: {len(FEATURE_COLS)} (NO sentiment - solo microstructure)\n"
        f"   Samples total: {n_total:,}\n"
        f"   Train/Val/Test: {n_train:,} / {n_val:,} / {n_test:,}\n"
        "   Ready for training!"
    )

    return X_seq, Y_seq

# MAIN
if __name__ == "__main__":
    MONTHS = [
        "BTCUSDT-trades-2023-03","BTCUSDT-trades-2023-04","BTCUSDT-trades-2023-05",
        "BTCUSDT-trades-2023-06","BTCUSDT-trades-2023-07","BTCUSDT-trades-2023-08",
        "BTCUSDT-trades-2023-09","BTCUSDT-trades-2023-10","BTCUSDT-trades-2023-11",
        "BTCUSDT-trades-2023-12",
        "BTCUSDT-trades-2024-01","BTCUSDT-trades-2024-02","BTCUSDT-trades-2024-03",
        "BTCUSDT-trades-2024-04","BTCUSDT-trades-2024-05","BTCUSDT-trades-2024-06",
        "BTCUSDT-trades-2024-07","BTCUSDT-trades-2024-08","BTCUSDT-trades-2024-09",
        "BTCUSDT-trades-2024-10","BTCUSDT-trades-2024-11","BTCUSDT-trades-2024-12",
    ]

    print("\n💡 IMPORTANTE:")
    print("   Dataset multi-anno FULL SPAN per TRAIN+VAL+TEST")
    print("   2023-01 → 2023-12 + 2024-01 → 2024-12")
    print("   (logica/matematica invariata)")
    print("\n Tempo di esecuzione: dipende dalla macchina\n")

    X, Y = process_and_save_2023_2024(MONTHS, output_prefix=OUTPUT_PREFIX)

    print("\n Dataset 2023-2024 FULL SPAN pronto!")
    print("   Next: training script (es. market_heartbeat_trainer_2023_2024.py)")
