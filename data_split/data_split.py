# =========================
# DATA SPLIT
# =========================

import os, glob, json
import numpy as np
import pandas as pd
from datetime import datetime

PREDICTION_HORIZON = 10

POSSIBLE_SOURCE_FILES = [
    "processed_data_DIRECTIONAL_2023_2024_FULLSPAN.parquet",
    "processed_data_OPTIMIZED_2023_2024.parquet",
    "processed_data_CORRECT.parquet",
    "processed_data.parquet",
]

SPLITS = {
    "TRAIN":      ("2023-01-01", "2024-06-30", "Model training"),
    "OPTIMIZE":   ("2024-07-01", "2024-09-30", "Grid search hyperparameters"),
    "VALIDATION": ("2024-10-01", "2024-12-31", "Final test (was called TEST before)"),
}

SEQUENCE_LENGTH = 3

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
    "regime_bull","regime_bear","regime_sideways",
    "trend_strength_10d","volatility_percentile",
    "price_distance_ma5","price_distance_ma10","volume_regime",
    "rsi_7d","rsi_14d","rsi_21d",
    "return_3d","return_5d","return_7d","return_10d","return_14d",
    "momentum_strength","directional_bias",
    "ma_cross","near_high","near_low",
]

# FILE FIND
def find_source_file():
    for f in POSSIBLE_SOURCE_FILES:
        if os.path.exists(f):
            return f
    cands = glob.glob("processed_data*.parquet")
    return cands[0] if cands else None

# VISUAL
def visualize_split():
    print("\n" + "="*80)
    print(" SPLIT STRATEGY (H10)")
    print("="*80)


# SEQUENCES
def create_sequences(X, Y, T, seq_len):
    Xs, Ys, Ts = [], [], []
    n = len(X)
    limit = n - seq_len + 1
    for i in range(limit):
        Xs.append(X[i:i+seq_len])
        Ys.append(Y[i+seq_len-1])
        Ts.append(T[i+seq_len-1])
    return (np.array(Xs, np.float32),
            np.array(Ys, np.float32),
            np.array(Ts))

# MAIN SPLIT
def create_splits():
    print("="*80)
    print(" CREATING CORRECT SPLITS (H10)")
    print("="*80)

    src = find_source_file()
    if not src:
        print("\n ERROR: no processed parquet found.")
        print("Expected (first match wins):")
        for f in POSSIBLE_SOURCE_FILES:
            print(" -", f)
        return None

    print("\n Source:", src)
    print(" Loading...")
    try:
        df = pd.read_parquet(src)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception as e:
        print(" Load error:", e)
        return None

    print(f" Loaded: {len(df):,} rows")
    print(f"Period: {df['timestamp'].min()} → {df['timestamp'].max()}")

    base_required = ["timestamp", "Target_Y"]
    missing_base = [c for c in base_required if c not in df.columns]
    if missing_base:
        print(" Missing required columns:", missing_base)
        return None

    available_features = [f for f in FEATURE_COLS if f in df.columns]
    if len(available_features) < len(FEATURE_COLS):
        miss = len(FEATURE_COLS) - len(available_features)
        print(f"  Missing {miss} feature(s). Using {len(available_features)}/{len(FEATURE_COLS)} available features.")
    else:
        print(f" All features present: {len(available_features)}")

    print("\n" + "="*80)
    print("SPLITTING DATA")
    print("="*80)

    splits = {}
    for name, (start_s, end_s, purpose) in SPLITS.items():
        start = pd.to_datetime(start_s)
        end = pd.to_datetime(end_s)

        part = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
        splits[name] = part

        icon = "TRAIN" if name == "TRAIN" else ("OPTIMIZE" if name == "OPTIMIZE" else "GOAL")
        pct = (len(part) / len(df) * 100) if len(df) else 0.0
        print(f"\n{icon} {name}")
        print(f"  rows:   {len(part):,} ({pct:.1f}%)")
        print(f"  period: {start_s} → {end_s}")
        print(f"  use:    {purpose}")

        if len(part) and "regime" in part.columns:
            vc = part["regime"].value_counts()
            print("  regimes:")
            for k, v in vc.items():
                rp = v / len(part) * 100
                ri = "BULL" if k == "BULL" else ("BEAR" if k == "BEAR" else "📊")
                print(f"    {ri} {k}: {v:,} ({rp:.1f}%)")

    print("\n" + "="*80)
    print(" SAVING SPLIT PARQUETS")
    print("="*80)

    for name, part in splits.items():
        out = "data_test_CORRECT.parquet" if name == "VALIDATION" else f"data_{name.lower()}_CORRECT.parquet"
        part.to_parquet(out, index=False)
        size_mb = os.path.getsize(out) / (1024**2)
        print(f" {out} ({size_mb:.1f} MB)")

    print("\n" + "="*80)
    print(" PREPARING TRAINING SEQUENCES (TRAIN only)")
    print("="*80)

    df_tr = splits["TRAIN"]
    if len(df_tr) < SEQUENCE_LENGTH:
        print(" Not enough TRAIN rows to create sequences.")
        return None

    X_tr = df_tr[available_features].values
    Y_tr = df_tr["Target_Y"].values
    T_tr = df_tr["timestamp"].values

    MEAN = X_tr.mean(axis=0)
    STD = X_tr.std(axis=0)
    X_tr_norm = (X_tr - MEAN) / (STD + 1e-8)

    X_seq, Y_seq, T_seq = create_sequences(X_tr_norm, Y_tr, T_tr, SEQUENCE_LENGTH)

    print("X:", X_seq.shape, "Y:", Y_seq.shape, "T:", T_seq.shape)
    up = int((Y_seq == 1).sum())
    dn = int((Y_seq == 0).sum())
    tot = len(Y_seq)
    print(f"Target: UP {up:,} ({up/tot*100:.1f}%) | DOWN {dn:,} ({dn/tot*100:.1f}%)")

    print("\n Saving arrays...")
    np.save("X_train_CORRECT.npy", X_seq)
    np.save("Y_train_CORRECT.npy", Y_seq)
    np.save("timestamps_train_CORRECT.npy", T_seq)
    np.save("normalization_mean_CORRECT.npy", MEAN)
    np.save("normalization_std_CORRECT.npy", STD)

    with open("feature_list_CORRECT.txt", "w") as f:
        for i, feat in enumerate(available_features):
            f.write(f"{i}: {feat}\n")

    cfg = {
        "split_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": src,
        "prediction_horizon": "H10",
        "total_samples": int(len(df)),
        "sequence_length": SEQUENCE_LENGTH,
        "features": {"total": len(available_features), "list": available_features},
        "splits": {
            k: {
                "samples": int(len(splits[k])),
                "percentage": (len(splits[k]) / len(df) * 100) if len(df) else 0.0,
                "start": SPLITS[k][0],
                "end": SPLITS[k][1],
                "purpose": SPLITS[k][2],
            }
            for k in SPLITS
        },
        "normalization": {
            "method": "standardization",
            "fitted_on": "TRAIN only",
            "mean_file": "normalization_mean_CORRECT.npy",
            "std_file": "normalization_std_CORRECT.npy",
        },
        "files_created": {
            "parquet": [
                "data_train_CORRECT.parquet",
                "data_optimize_CORRECT.parquet",
                "data_test_CORRECT.parquet",
            ],
            "arrays": [
                "X_train_CORRECT.npy",
                "Y_train_CORRECT.npy",
                "timestamps_train_CORRECT.npy",
                "normalization_mean_CORRECT.npy",
                "normalization_std_CORRECT.npy",
            ],
            "other": ["feature_list_CORRECT.txt", "SPLIT_CONFIG_CORRECT.json"],
        },
        "critical_notes": [
            "Grid search ONLY on data_optimize_CORRECT.parquet",
            "Final test ONLY on data_test_CORRECT.parquet (Q4 2024)",
            "NO leakage: never tune on test/validation",
        ],
    }

    with open("SPLIT_CONFIG_CORRECT.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print("\n" + "="*80)
    print(" DONE")
    print("="*80)
    print(f"Source: {src}")
    print(f"TRAIN: {len(splits['TRAIN']):,} | OPT: {len(splits['OPTIMIZE']):,} | VAL(test): {len(splits['VALIDATION']):,}")
    print("Files: data_*_CORRECT.parquet + X_train_CORRECT.npy/Y_train_CORRECT.npy + normalization_* + config")
    print("="*80 + "\n")

    return splits

if __name__ == "__main__":
    visualize_split()
    out = create_splits()
    print(" SUCCESS!\n" if out else " FAILED!\n")

