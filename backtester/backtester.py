# =========================
# BACKTEST 
# =========================

import sys, json, warnings
from pathlib import Path
from math import e
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from scipy.stats import norm

warnings.filterwarnings("ignore")

# CONFIG
INITIAL_CAPITAL = 100_000
POSITION_SIZE = 0.35
FEE_BPS = 2.5
SLIPPAGE_BPS = 2.5

BAR_SECONDS = 30
SMOOTHING_WINDOW = 60
SEQUENCE_LENGTH = 3

MIN_HOLD = 10
MAX_HOLD = 90

L_THR_BASE = 0.50
S_THR_BASE = 0.50

RANDOM_SEARCH_STEPS = 40

SL_SIGMA_MIN, SL_SIGMA_MAX = 1.5, 2.7
ENTRY_THR_MIN, ENTRY_THR_MAX = 0.52, 0.64
NB_MIN, NB_MAX = 0.03, 0.09

TRAIL_SIGMA_START = 1.5
COOLDOWN_BARS = 10

DATA_FILE_ALL = "processed_data_DIRECTIONAL_2023_2024_FULLSPAN.parquet"

PREFIX = "CORRECT_H10"
MODEL_FILE = f"best_model_{PREFIX}.keras"
NORM_MEAN_FILE = "normalization_mean_CORRECT.npy"
NORM_STD_FILE = "normalization_std_CORRECT.npy"

TOP_K_PREFILTER_B = 15
TOP_N_FINAL = 5

OUT_JSON = "backtest_H10_criterionB_DSRVAL_top5_ensemble_report.json"

FEATURE_COLS = [
    "OFI_Taker","OFI_velocity","OFI_cumsum_60s","VWAP_Deviation",
    "Taker_Maker_Ratio","Hawkes_Intensity","Spread_BPS",
    "volatility_30s","volatility_60s","volatility_ratio",
    "price_skew_60s","price_kurtosis_60s","return_60s","price_std",
    "volume","trade_count",
    "momentum_30s","momentum_60s","momentum_120s","acceleration",
    "trend_strength","volume_surge","trade_size_anomaly",
    "range_expansion","OFI_momentum","vol_breakout",
    "cum_return_120s","taker_sell_pct"
]

# ATTENTION
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        e_ = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        e_ = tf.reduce_sum(e_, axis=-1, keepdims=True)
        alpha = tf.nn.softmax(e_, axis=1)
        return tf.reduce_sum(x * alpha, axis=1)

    def get_config(self):
        return super().get_config()

# LOAD
def load_model_and_normalization():
    print("=" * 80)
    print(" BACKTEST H10 – CRITERIO B (train+val) + DSR su VAL + report TEST")
    print("=" * 80)

    print("\n Loading model:", MODEL_FILE)
    model = keras.models.load_model(MODEL_FILE, custom_objects={"AttentionLayer": AttentionLayer})
    print(" Model loaded.")

    print("\n Loading normalization:", NORM_MEAN_FILE, NORM_STD_FILE)
    mean = np.load(NORM_MEAN_FILE)
    std = np.load(NORM_STD_FILE)
    print(" Normalization loaded.")
    return model, mean, std

# SEQ + PRED
def build_sequences_and_predictions(df_raw, model, mean, std):
    df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

    X = df_raw[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
    mean = mean.astype(np.float32, copy=False)
    std = std.astype(np.float32, copy=False)

    Xn = (X - mean) / (std + 1e-8)

    from numpy.lib.stride_tricks import sliding_window_view
    X_seq = sliding_window_view(Xn, (SEQUENCE_LENGTH, Xn.shape[1]))[:, 0, :, :]
    X_seq = np.asarray(X_seq, dtype=np.float32)

    print(f"   🤖 Predicting {len(X_seq):,} sequences...")
    probs = model.predict(X_seq, batch_size=2048, verbose=0).reshape(-1)

    df_full = df_raw.iloc[SEQUENCE_LENGTH - 1:].reset_index(drop=True)
    df_full["log_ret"] = np.log(df_full["price"]).diff().fillna(0.0)
    return df_full, probs

# STRATEGY
class VolatilityAdaptiveLongShort:
    def __init__(self, initial_capital, position_size, fee_bps, slippage_bps,
                 min_hold, max_hold, sl_sigma, tp_sigma, l_thr_exit, s_thr_exit,
                 trail_sigma_start, cooldown_bars, vol_series):
        self.capital = float(initial_capital)
        self.invested_capital = 0.0
        self.position_state = 0
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.entry_bucket = 0
        self.entry_vol = 0.0
        self.total_fees = 0.0
        self.trades_log = []
        self._eq_ts, self._eq_val = [], []
        self.cooldown_until = -1

        self.fee_multiplier = (fee_bps + slippage_bps) / 10000.0
        self.sl_sigma = float(sl_sigma)
        self.tp_sigma = float(tp_sigma)
        self.trail_sigma_start = float(trail_sigma_start)

        self.min_hold_buckets = int(min_hold)
        self.max_hold_buckets = int(max_hold)

        self.l_thr_exit = float(l_thr_exit)
        self.s_thr_exit = float(s_thr_exit)

        self.vol_series = vol_series
        self.cooldown_bars = int(cooldown_bars)

    def _pay_fee_notional(self, notional):
        fee = float(notional) * self.fee_multiplier
        self.total_fees += fee
        return fee

    def _mark_equity(self, price, timestamp):
        price = float(price)
        if self.position_state == 1:
            eq = self.capital + self.position_qty * price
        elif self.position_state == -1:
            eq = self.capital + self.position_qty * (2 * self.entry_price - price)
        else:
            eq = self.capital
        self._eq_ts.append(timestamp)
        self._eq_val.append(float(eq))

    def open_position(self, price, idx, timestamp, direction, position_size):
        if self.position_state != 0:
            return
        if idx < self.cooldown_until:
            return

        invest = self.capital * float(position_size)
        if invest <= 0:
            return

        fee = self._pay_fee_notional(invest)
        self.position_qty = (invest - fee) / float(price)

        self.position_state = 1 if direction == "LONG" else -1
        self.entry_price = float(price)
        self.entry_bucket = int(idx)
        self.entry_vol = float(self.vol_series[idx])

        self.capital -= invest
        self.invested_capital = invest
        self._mark_equity(price, timestamp)

    def close_position(self, price, idx, timestamp, reason):
        if self.position_state == 0:
            return

        direction = "LONG" if self.position_state == 1 else "SHORT"
        price = float(price)

        if direction == "LONG":
            pnl_gross = self.position_qty * (price - self.entry_price)
        else:
            pnl_gross = self.position_qty * (self.entry_price - price)

        notional_exit = abs(self.position_qty * price)
        fee = self._pay_fee_notional(notional_exit)
        pnl_net = pnl_gross - fee

        self.capital += self.invested_capital + pnl_net

        self.trades_log.append({
            "pnl_net": pnl_net,
            "winning": pnl_net > 0,
            "direction": direction,
            "reason": reason,
        })

        self.position_state = 0
        self.position_qty = 0.0
        self.invested_capital = 0.0
        self.entry_price = 0.0
        self.entry_vol = 0.0
        self.cooldown_until = int(idx) + self.cooldown_bars
        self._mark_equity(price, timestamp)

    def _signed_return(self, price):
        if self.position_state == 0 or self.entry_price <= 0:
            return 0.0
        r = float(price) / self.entry_price - 1.0
        return r if self.position_state == 1 else -r

    def check_exit(self, price, idx, timestamp, current_pred):
        if self.position_state == 0:
            return False, None

        hold = int(idx) - int(self.entry_bucket)
        if hold >= self.max_hold_buckets:
            return True, "TIME"

        sigma = max(float(self.entry_vol), 1e-8)
        signed_ret = self._signed_return(price)

        tp = self.tp_sigma * sigma
        sl = -self.sl_sigma * sigma

        if signed_ret >= self.trail_sigma_start * sigma and signed_ret <= 0.0:
            return True, "TRAIL"

        if signed_ret >= tp:
            return True, "TP"
        if signed_ret <= sl:
            return True, "SL"

        if hold >= self.min_hold_buckets:
            if self.position_state == 1 and float(current_pred) <= self.s_thr_exit:
                return True, "SIGNAL"
            if self.position_state == -1 and float(current_pred) >= self.l_thr_exit:
                return True, "SIGNAL"

        return False, None

# METRICS
EULER_GAMMA = 0.5772156649015329

def sharpe_daily(returns_daily, rf=0.0, periods_per_year=252):
    r = np.asarray(returns_daily, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0, 0.0
    ex = r - rf
    mu = ex.mean()
    sig = ex.std(ddof=1)
    if sig <= 0:
        return 0.0, 0.0
    sr_non = mu / sig
    sr_ann = sr_non * np.sqrt(periods_per_year)
    return float(sr_ann), float(sr_non)

def _moment_stats(x):
    x = np.asarray(x, dtype=np.float64)
    m = np.mean(x)
    xc = x - m
    mu2 = np.mean(xc ** 2)
    mu3 = np.mean(xc ** 3)
    mu4 = np.mean(xc ** 4)
    sk = mu3 / (mu2 ** 1.5 + 1e-16)
    ku = mu4 / (mu2 ** 2 + 1e-16)
    return float(sk), float(ku)

def deflated_sharpe_ratio(returns, sharpe_all_non_ann):
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 3:
        return 0.0, 0.0, 0.0

    mu = r.mean()
    sig = r.std(ddof=1) + 1e-16
    sr_hat = mu / sig

    sharpe_all = np.asarray(sharpe_all_non_ann, dtype=np.float64)
    N = len(sharpe_all)
    if N < 2:
        return 0.0, float(sr_hat), 0.0

    var_sr = np.var(sharpe_all, ddof=1)
    sigma_sr = np.sqrt(var_sr + 1e-16)

    term1 = (1 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / N)
    term2 = EULER_GAMMA * norm.ppf(1.0 - 1.0 / (N * e))
    sr0 = sigma_sr * (term1 + term2)

    g3, g4 = _moment_stats(r)
    sigma_sr0 = np.sqrt((1.0 - g3 * sr0 + ((g4 - 1.0) / 4.0) * (sr0 ** 2)) / max(T - 1, 1))
    z = (sr_hat - sr0) / (sigma_sr0 + 1e-16)
    dsr = norm.cdf(z)
    return float(dsr), float(sr_hat), float(sr0)

# CAGR
def _segment_days(start_ts, end_ts):
    start = pd.Timestamp(start_ts)
    end = pd.Timestamp(end_ts)
    return float((end - start).total_seconds() / 86400.0)

def annualized_ric(total_ret_pct, start_ts, end_ts):
    r = float(total_ret_pct) / 100.0
    days = _segment_days(start_ts, end_ts)
    if not np.isfinite(days) or days <= 0:
        return float("nan")
    return float(((1.0 + r) ** (365.0 / days) - 1.0) * 100.0)

# JSON HELPERS
def _clean_numeric(x):
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, (float, int)):
        return x
    return x

def clean_metrics_for_json(res):
    out = {}
    for k, v in res.items():
        if k in ("daily_returns", "daily_returns_s", "daily_equity_s"):
            continue
        out[k] = _clean_numeric(v)
    return out

def json_default(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    try:
        return float(o)
    except Exception:
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

# SMOOTHING
def rolling_mean_min_periods_1(x, window):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    c = np.cumsum(x, dtype=np.float64)
    w = int(window)
    for i in range(n):
        s = i - w + 1
        if s <= 0:
            out[i] = c[i] / (i + 1)
        else:
            out[i] = (c[i] - c[s - 1]) / w
    return out

# BACKTEST SEGMENT
def run_backtest_segment(df_seg, preds_seg, entry_thr, sl_sigma, tp_sigma, no_trade_band):
    preds_sm = rolling_mean_min_periods_1(preds_seg, SMOOTHING_WINDOW)

    vol_lookback = 60
    vol_series = (
        df_seg["log_ret"].rolling(window=vol_lookback, min_periods=10).std()
        .bfill().replace(0, np.nan).fillna(df_seg["log_ret"].std())
        .to_numpy(dtype=np.float64, copy=False)
    )

    strat = VolatilityAdaptiveLongShort(
        INITIAL_CAPITAL, POSITION_SIZE, FEE_BPS, SLIPPAGE_BPS,
        MIN_HOLD, MAX_HOLD,
        sl_sigma, tp_sigma,
        L_THR_BASE, S_THR_BASE,
        TRAIL_SIGMA_START, COOLDOWN_BARS,
        vol_series
    )

    long_entry_thr = max(float(entry_thr), 0.5 + float(no_trade_band))
    base_short_thr = 1.0 - float(entry_thr) + 0.02
    short_entry_thr = min(base_short_thr, 0.5 - float(no_trade_band))

    price_arr = df_seg["price"].to_numpy(dtype=np.float64, copy=False)
    ts_arr = df_seg["timestamp"].to_numpy(copy=False)

    mom_arr = df_seg["momentum_60s"].to_numpy(dtype=np.float64, copy=False) if "momentum_60s" in df_seg.columns else None
    trend_arr = df_seg["trend_strength"].to_numpy(dtype=np.float64, copy=False) if "trend_strength" in df_seg.columns else None

    for idx in range(len(price_arr)):
        price = float(price_arr[idx])
        ts = ts_arr[idx]
        pred = float(preds_sm[idx])

        should_exit, reason = strat.check_exit(price, idx, ts, pred)
        if strat.position_state != 0 and should_exit:
            strat.close_position(price, idx, ts, reason)

        if strat.position_state == 0 and idx >= vol_lookback:
            m60 = float(mom_arr[idx]) if mom_arr is not None else 0.0
            tr = float(trend_arr[idx]) if trend_arr is not None else 0.0

            if pred >= long_entry_thr and tr >= 0:
                strat.open_position(price, idx, ts, "LONG", POSITION_SIZE)
            elif pred <= short_entry_thr and m60 < 0 and tr <= 0:
                strat.open_position(price, idx, ts, "SHORT", POSITION_SIZE)

        strat._mark_equity(price, ts)

    if strat.position_state != 0:
        strat.close_position(float(price_arr[-1]), len(price_arr) - 1, ts_arr[-1], "END")

    if len(strat._eq_ts) == 0:
        daily_returns = np.array([], dtype=np.float64)
        daily_returns_s = pd.Series(dtype=float)
        daily_equity_s = pd.Series(dtype=float)
        max_dd = 0.0
    else:
        eq = pd.DataFrame({"timestamp": strat._eq_ts, "equity": strat._eq_val})
        eq = eq.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()

        daily_equity_s = eq["equity"].resample("1D").last().dropna()
        daily_returns_s = daily_equity_s.pct_change().dropna()
        daily_returns = daily_returns_s.to_numpy(dtype=np.float64)

        eq["cummax"] = eq["equity"].cummax()
        dd = eq["equity"] / eq["cummax"] - 1.0
        max_dd = float(dd.min() * 100.0)

    final_cap = float(strat.capital)
    total_ret = (final_cap / INITIAL_CAPITAL - 1.0) * 100.0
    bh_return = (float(price_arr[-1]) / float(price_arr[0]) - 1.0) * 100.0
    alpha = total_ret - bh_return

    trades = len(strat.trades_log)
    wins = [t["pnl_net"] for t in strat.trades_log if t["pnl_net"] > 0]
    losses = [t["pnl_net"] for t in strat.trades_log if t["pnl_net"] < 0]

    winning = len(wins)
    win_rate = (winning / trades * 100.0) if trades > 0 else 0.0

    sharpe_ann, sharpe_non = sharpe_daily(daily_returns)

    win_avg = float(np.mean(wins)) if wins else 0.0
    loss_avg = float(-np.mean(losses)) if losses else 0.0
    wl_ratio = (win_avg / loss_avg) if (win_avg > 0 and loss_avg > 0) else np.nan

    start_ts = df_seg["timestamp"].iloc[0]
    end_ts = df_seg["timestamp"].iloc[-1]
    days = _segment_days(start_ts, end_ts)

    ann_ret = annualized_ric(total_ret, start_ts, end_ts)
    ann_bh = annualized_ric(bh_return, start_ts, end_ts)
    ann_alpha = ann_ret - ann_bh

    return {
        "total_ret": float(total_ret),
        "alpha": float(alpha),
        "bh_return": float(bh_return),
        "win_rate": float(win_rate),
        "trades": int(trades),
        "winning": int(winning),
        "losing": int(trades - winning),
        "costs": float(strat.total_fees),
        "sharpe_ann": float(sharpe_ann),
        "sharpe_non_ann": float(sharpe_non),
        "max_dd": float(max_dd),
        "daily_returns": daily_returns,
        "daily_returns_s": daily_returns_s,
        "daily_equity_s": daily_equity_s,
        "win_pnl_avg": float(win_avg),
        "loss_pnl_avg": float(loss_avg),
        "wl_ratio": float(wl_ratio) if np.isfinite(wl_ratio) else np.nan,
        "days": float(days),
        "ann_return": float(ann_ret),
        "ann_bh_return": float(ann_bh),
        "ann_alpha": float(ann_alpha),
    }

# PRINT
def _print_block(label, res):
    print(f" {label}")
    print("-" * 80)
    print(f"Total Return:           {res['total_ret']:+7.2f}%")
    print(f"Buy & Hold Return:      {res['bh_return']:+7.2f}%")
    print(f"Alpha vs B&H:           {res['alpha']:+7.2f}%")
    print(f"Sharpe (ann.):          {res['sharpe_ann']:+7.2f}")
    print(f"Max Drawdown:           {res['max_dd']:+7.2f}%")
    print(f"Win Rate:               {res['win_rate']:5.1f}%")
    print(f"Trades:                 {res['trades']}")
    print(f"Costs (fees+slip):      {res['costs']:.2f}")
    print(f"WL ratio:               {res['wl_ratio']:.4f}")
    print(f"Segment days:           {res['days']:.2f}")
    print(f"RIC Ann. Return (CAGR): {res['ann_return']:+7.2f}%\n")

# ENSEMBLE
def _align_returns_series(series_list):
    idx = None
    for s in series_list:
        if s is None or len(s) == 0:
            return None
        idx = s.index if idx is None else idx.intersection(s.index)
    if idx is None or len(idx) < 10:
        return None
    return [s.loc[idx].astype(float) for s in series_list]

def risk_parity_weights_inv_vol(train_val_returns_series_list):
    aligned = _align_returns_series(train_val_returns_series_list)
    if aligned is None:
        n = len(train_val_returns_series_list)
        return np.ones(n) / max(n, 1)

    vols = np.array([float(np.std(s.values, ddof=1)) for s in aligned], dtype=np.float64)
    vols = np.where(vols <= 1e-12, np.nan, vols)
    inv = 1.0 / vols
    if np.all(~np.isfinite(inv)):
        n = len(train_val_returns_series_list)
        return np.ones(n) / max(n, 1)

    inv = np.where(np.isfinite(inv), inv, 0.0)
    return inv / (inv.sum() + 1e-16)

def portfolio_metrics_from_returns(r_s):
    r = r_s.dropna()
    if len(r) < 2:
        return {"total_ret": 0.0, "sharpe_ann": 0.0, "max_dd": 0.0, "ann_return": np.nan}

    equity = (1.0 + r).cumprod()
    total_ret = (equity.iloc[-1] - 1.0) * 100.0

    sr_ann, _ = sharpe_daily(r.values)

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min() * 100.0)

    days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0
    ann_return = float(((1.0 + total_ret / 100.0) ** (365.0 / days) - 1.0) * 100.0) if days > 0 else float("nan")

    return {"total_ret": float(total_ret), "sharpe_ann": float(sr_ann), "max_dd": float(max_dd), "ann_return": float(ann_return)}

def ensemble_backtest_segment(strategies_res_list, weights, label):
    rets = [r["daily_returns_s"] for r in strategies_res_list]
    aligned = _align_returns_series(rets)
    if aligned is None:
        print(f" Ensemble {label}: non abbastanza giorni in comune.")
        return {"total_ret": 0.0, "sharpe_ann": 0.0, "max_dd": 0.0, "ann_return": np.nan}

    mat = np.vstack([s.values for s in aligned])
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    port = (w * mat).sum(axis=0)
    port_s = pd.Series(port, index=aligned[0].index)
    return portfolio_metrics_from_returns(port_s)

# MAIN
if __name__ == "__main__":
    model, mean, std = load_model_and_normalization()

    print("\n Loading FULL-SPAN dataset:", DATA_FILE_ALL)
    if not Path(DATA_FILE_ALL).exists():
        print(" File non trovato:", DATA_FILE_ALL)
        sys.exit(1)

    df_raw = pd.read_parquet(DATA_FILE_ALL)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])

    df_full, preds_full = build_sequences_and_predictions(df_raw, model, mean, std)

    print(f"   FULL samples: {len(df_full):,} ({df_full['timestamp'].min()} → {df_full['timestamp'].max()})")

    N = len(df_full)
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)
    n_test = N - n_train - n_val

    df_train = df_full.iloc[:n_train].copy()
    df_val   = df_full.iloc[n_train:n_train + n_val].copy()
    df_test  = df_full.iloc[n_train + n_val:].copy()

    preds_train = preds_full[:n_train]
    preds_val   = preds_full[n_train:n_train + n_val]
    preds_test  = preds_full[n_train + n_val:]

    print("\n" + "="*80)
    print(" DATA SPLIT (70/15/15 su sample, ordinato nel tempo)")
    print("="*80)
    print(f"   TRAIN: {len(df_train):,} ({df_train['timestamp'].min()} → {df_train['timestamp'].max()})")
    print(f"   VAL:   {len(df_val):,} ({df_val['timestamp'].min()} → {df_val['timestamp'].max()})")
    print(f"   TEST:  {len(df_test):,} ({df_test['timestamp'].min()} → {df_test['timestamp'].max()})")

    print("\n" + "="*80)
    print(f" RANDOM SEARCH SU TRAIN + VAL – {RANDOM_SEARCH_STEPS} turni (NO TEST)")
    print("="*80 + "\n")

    rng = np.random.default_rng(seed=42)
    strategies = []

    for i in range(RANDOM_SEARCH_STEPS):
        entry_thr = float(rng.uniform(ENTRY_THR_MIN, ENTRY_THR_MAX))
        sl_sigma = float(rng.uniform(SL_SIGMA_MIN, SL_SIGMA_MAX))
        tp_sigma = float(rng.uniform(sl_sigma + 0.25, min(sl_sigma + 1.75, 4.5)))
        no_trade_band = float(rng.uniform(NB_MIN, NB_MAX))

        print(f"[Random {i+1}/{RANDOM_SEARCH_STEPS}] L{entry_thr:.2f} / SL{sl_sigma:.2f}σ / TP{tp_sigma:.2f}σ / NB{no_trade_band:.2f}")

        res_tr = run_backtest_segment(df_train, preds_train, entry_thr, sl_sigma, tp_sigma, no_trade_band)
        res_va = run_backtest_segment(df_val, preds_val, entry_thr, sl_sigma, tp_sigma, no_trade_band)

        strategies.append({
            "strategy_name": f"L{entry_thr:.2f} SL{sl_sigma:.2f}σ TP{tp_sigma:.2f}σ NB{no_trade_band:.2f}",
            "entry_thr": entry_thr,
            "sl_sigma": sl_sigma,
            "tp_sigma": tp_sigma,
            "no_trade_band": no_trade_band,
            "train": res_tr,
            "val": res_va,
            "test": None,
        })

    print("\n" + "="*80)
    print(f" CRITERIO B – PRE-FILTRO (WR+WL train/val) -> TOP {TOP_K_PREFILTER_B}")
    print("="*80 + "\n")

    rows = []
    for idx, s in enumerate(strategies):
        wr_t = float(s["train"]["win_rate"])
        wr_v = float(s["val"]["win_rate"])
        wr_mean = 0.5 * (wr_t + wr_v)
        wr_diff = abs(wr_t - wr_v)

        wl_t = s["train"]["wl_ratio"]
        wl_v = s["val"]["wl_ratio"]
        if np.isnan(wl_t) or np.isnan(wl_v):
            wl_mean, wl_diff = 0.0, 999.0
            wl_tf, wl_vf = float("nan"), float("nan")
        else:
            wl_tf, wl_vf = float(wl_t), float(wl_v)
            wl_mean = 0.5 * (wl_tf + wl_vf)
            wl_diff = abs(wl_tf - wl_vf)

        score_b = (wr_mean - wr_diff + 50.0 * wl_mean - 20.0 * wl_diff)

        rows.append({
            "idx": idx,
            "strategy_name": s["strategy_name"],
            "wr_train": wr_t, "wr_val": wr_v,
            "wl_train": wl_tf, "wl_val": wl_vf,
            "score_b": score_b
        })

    df_b = pd.DataFrame(rows).sort_values("score_b", ascending=False).reset_index(drop=True)

    for r in df_b.itertuples():
        print(f"{r.Index+1:2d}) {r.strategy_name:<35} WR_T={r.wr_train:5.1f}% WR_V={r.wr_val:5.1f}% "
              f"WL_T={r.wl_train:6.3f} WL_V={r.wl_val:6.3f} score={r.score_b:8.3f}")

    cand_idx = df_b["idx"].iloc[:TOP_K_PREFILTER_B].tolist()

    print(f"\n Candidati (TOP {TOP_K_PREFILTER_B}):")
    for j, idx in enumerate(cand_idx, 1):
        print(f"  {j:2d}) {strategies[idx]['strategy_name']}")

    print("\n" + "="*80)
    print(f" TOP {TOP_N_FINAL} – ordinati per DSR su VALIDATION (NO TEST)")
    print("="*80 + "\n")

    sharpe_non_ann_val_all = np.array([s["val"]["sharpe_non_ann"] for s in strategies], dtype=np.float64)

    scored = []
    for idx in cand_idx:
        va = strategies[idx]["val"]
        dsr_val, sr_hat, sr0 = deflated_sharpe_ratio(va["daily_returns"], sharpe_non_ann_val_all)

        scored.append({
            "idx": idx,
            "strategy_name": strategies[idx]["strategy_name"],
            "dsr_val": dsr_val,
            "sharpe_val": float(va["sharpe_ann"]),
            "max_dd_val": float(va["max_dd"]),
            "costs_val": float(va["costs"]),
            "sr_hat_non_ann_val": sr_hat,
            "sr0_val": sr0,
        })

    df_top = (pd.DataFrame(scored)
              .sort_values(by=["dsr_val", "sharpe_val", "max_dd_val"], ascending=[False, False, False])
              .reset_index(drop=True))

    df_top5 = df_top.iloc[:TOP_N_FINAL].copy()
    top5_idx = df_top5["idx"].tolist()

    print(" TOP 5 (decise SOLO su VALIDATION):\n")
    for r in df_top5.itertuples():
        print(f"{r.Index+1:2d}) {r.strategy_name:<35} DSR_VAL={r.dsr_val:6.4f} "
              f"Sharpe_VAL={r.sharpe_val:6.2f} MaxDD_VAL={r.max_dd_val:7.2f}% Costs_VAL={r.costs_val:10.2f}")

    best_idx = top5_idx[0]
    best = strategies[best_idx]

    print("\n" + "="*80)
    print(" REPORT: calcolo TEST per TOP 5 (TEST NON usato per selezione)")
    print("="*80 + "\n")

    for idx in top5_idx:
        s = strategies[idx]
        s["test"] = run_backtest_segment(df_test, preds_test, s["entry_thr"], s["sl_sigma"], s["tp_sigma"], s["no_trade_band"])

    top5 = [strategies[i] for i in top5_idx]

    train_val_series = []
    for s in top5:
        tv = pd.concat([s["train"]["daily_returns_s"], s["val"]["daily_returns_s"]]).dropna()
        train_val_series.append(tv)

    w = risk_parity_weights_inv_vol(train_val_series)

    print(" Pesi risk-parity (da TRAIN+VAL):")
    for si, wi in zip(top5, w):
        print(f"  {si['strategy_name']:<35} w={wi:7.3f}")

    ens_train = ensemble_backtest_segment([s["train"] for s in top5], w, "TRAIN")
    ens_val   = ensemble_backtest_segment([s["val"]   for s in top5], w, "VAL")
    ens_test  = ensemble_backtest_segment([s["test"]  for s in top5], w, "TEST")

    print("\n" + "="*80)
    print(" MIGLIORE SINGOLA (#1 per DSR_VAL) – TRAIN / VAL / TEST")
    print("="*80 + "\n")

    print("STRATEGY:", best["strategy_name"])
    print(f"Params: entry_thr={best['entry_thr']:.3f}, sl_sigma={best['sl_sigma']:.2f}, "
          f"tp_sigma={best['tp_sigma']:.2f}, no_trade_band={best['no_trade_band']:.3f}\n")

    _print_block("TRAIN", best["train"])
    _print_block("VALIDATION", best["val"])
    _print_block("TEST (solo report)", best["test"])

    print("\n" + "="*80)
    print(" ENSEMBLE risk-parity (TOP 5) – TRAIN / VAL / TEST")
    print("="*80 + "\n")

    print(f"ENSEMBLE TRAIN: Ret={ens_train['total_ret']:+7.2f}% | Sharpe={ens_train['sharpe_ann']:+6.2f} | "
          f"MaxDD={ens_train['max_dd']:+7.2f}% | AnnRet={ens_train['ann_return']:+7.2f}%")
    print(f"ENSEMBLE VAL:   Ret={ens_val['total_ret']:+7.2f}% | Sharpe={ens_val['sharpe_ann']:+6.2f} | "
          f"MaxDD={ens_val['max_dd']:+7.2f}% | AnnRet={ens_val['ann_return']:+7.2f}%")
    print(f"ENSEMBLE TEST:  Ret={ens_test['total_ret']:+7.2f}% | Sharpe={ens_test['sharpe_ann']:+6.2f} | "
          f"MaxDD={ens_test['max_dd']:+7.2f}% | AnnRet={ens_test['ann_return']:+7.2f}%\n")

    out = {
        "split_info": {
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "train_start": str(df_train["timestamp"].iloc[0]),
            "train_end": str(df_train["timestamp"].iloc[-1]),
            "val_start": str(df_val["timestamp"].iloc[0]),
            "val_end": str(df_val["timestamp"].iloc[-1]),
            "test_start": str(df_test["timestamp"].iloc[0]),
            "test_end": str(df_test["timestamp"].iloc[-1]),
        },
        "random_search_steps": int(RANDOM_SEARCH_STEPS),
        "criterion": "B_only",
        "prefilter_top_k": int(TOP_K_PREFILTER_B),
        "selected_top5_by": "DSR_on_validation_only",
        "top5_indices": [int(x) for x in top5_idx],
        "top5_strategies": [
            {
                "rank": int(i + 1),
                "strategy_name": strategies[idx]["strategy_name"],
                "params": {
                    "entry_thr": float(strategies[idx]["entry_thr"]),
                    "sl_sigma": float(strategies[idx]["sl_sigma"]),
                    "tp_sigma": float(strategies[idx]["tp_sigma"]),
                    "no_trade_band": float(strategies[idx]["no_trade_band"]),
                },
                "train": clean_metrics_for_json(strategies[idx]["train"]),
                "val": clean_metrics_for_json(strategies[idx]["val"]),
                "test_report_only": clean_metrics_for_json(strategies[idx]["test"]),
            }
            for i, idx in enumerate(top5_idx)
        ],
        "ensemble": {
            "weights_from_train_val": [
                {"strategy_name": s["strategy_name"], "w": float(wi)}
                for s, wi in zip(top5, w)
            ],
            "train": ens_train,
            "val": ens_val,
            "test_report_only": ens_test,
        },
        "best_single": {
            "strategy_name": best["strategy_name"],
            "train": clean_metrics_for_json(best["train"]),
            "val": clean_metrics_for_json(best["val"]),
            "test_report_only": clean_metrics_for_json(best["test"]),
        }
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=json_default)

    print(" RISULTATI SALVATI IN:", OUT_JSON)

