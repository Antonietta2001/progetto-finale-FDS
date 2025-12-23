# =========================
# ECG LIVE VISUALIZER
# =========================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import tensorflow as tf
import keras
from collections import deque
import time
import warnings
warnings.filterwarnings("ignore")

DATA_FILE_ALL = "processed_data_DIRECTIONAL_2023_2024_FULLSPAN.parquet"
MODEL_FILE = "best_model_CORRECT_H10.keras"
MEAN_FILE = "normalization_mean_CORRECT.npy"
STD_FILE = "normalization_std_CORRECT.npy"

INITIAL_CAPITAL = 100_000
POSITION_SIZE = 0.35
FEE_BPS = 2.5
SLIPPAGE_BPS = 2.5

BAR_SECONDS = 30
SEQUENCE_LENGTH = 3
SMOOTHING_WINDOW = 60

MIN_HOLD = 10
MAX_HOLD = 90

ENTRY_THR = 0.523
NO_TRADE_BAND = 0.058
SL_SIGMA = 1.61
TP_SIGMA = 2.94

TRAIL_SIGMA_START = 1.5
COOLDOWN_BARS = 10
VOLATILITY_LOOKBACK = 60

REPLAY_SPEED_MS = 40
SKIP_SAMPLES = 8
FLASH_INTENSITY = 0.90
FLASH_DECAY = 0.20

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

def safe_value(v, default=0.0):
    try:
        if v is None:
            return default
        if np.isnan(v) or np.isinf(v):
            return default
        return float(v)
    except Exception:
        return default

def safe_ylim(values, default_range=(-1, 1), margin_pct=0.1):
    if len(values) == 0:
        return default_range
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return default_range
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmin == vmax:
        m = abs(vmin) * 0.2 if vmin != 0 else 1.0
        return (vmin - m, vmax + m)
    m = (vmax - vmin) * margin_pct
    return (vmin - m, vmax + m)

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
        self.position_size = float(position_size)

        self.tp_exits = 0
        self.sl_exits = 0
        self.timeout_exits = 0
        self.signal_exits = 0
        self.trail_exits = 0

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
        return float(eq)

    def open_position(self, price, idx, timestamp, direction):
        if self.position_state != 0:
            return None
        if idx < self.cooldown_until:
            return None

        invest = self.capital * self.position_size
        if invest <= 0:
            return None

        fee = self._pay_fee_notional(invest)
        self.position_qty = (invest - fee) / float(price)

        self.position_state = 1 if direction == "LONG" else -1
        self.entry_price = float(price)
        self.entry_bucket = int(idx)
        self.entry_vol = float(self.vol_series[idx])

        self.capital -= invest
        self.invested_capital = invest
        self._mark_equity(price, timestamp)
        return direction

    def close_position(self, price, idx, timestamp, reason):
        if self.position_state == 0:
            return None

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

        if reason == "TP": self.tp_exits += 1
        elif reason == "SL": self.sl_exits += 1
        elif reason == "TIME": self.timeout_exits += 1
        elif reason == "SIGNAL": self.signal_exits += 1
        elif reason == "TRAIL": self.trail_exits += 1

        self.position_state = 0
        self.position_qty = 0.0
        self.invested_capital = 0.0
        self.entry_price = 0.0
        self.entry_vol = 0.0
        self.cooldown_until = int(idx) + self.cooldown_bars
        self._mark_equity(price, timestamp)
        return reason

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

class ECGVisualizer:
    def __init__(self, df_test, preds_test_sm, vol_series):
        self.df = df_test
        self.preds = preds_test_sm
        self.vol = vol_series

        self.strat = VolatilityAdaptiveLongShort(
            INITIAL_CAPITAL, POSITION_SIZE, FEE_BPS, SLIPPAGE_BPS,
            MIN_HOLD, MAX_HOLD,
            SL_SIGMA, TP_SIGMA,
            0.50, 0.50,
            TRAIL_SIGMA_START, COOLDOWN_BARS,
            self.vol
        )

        self.i = 0
        self.flash_alpha = 0.0
        self.flash_count = 0

        self.times = deque(maxlen=300)
        self.prices = deque(maxlen=300)
        self.preds_w = deque(maxlen=300)
        self.returns_bps = deque(maxlen=300)
        self.equity = deque(maxlen=300)

        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(20, 11))

        gs = self.fig.add_gridspec(
            4, 1, height_ratios=[1, 1, 1, 1],
            hspace=0.15, left=0.12, right=0.98,
            top=0.95, bottom=0.12
        )

        self.ax_price = self.fig.add_subplot(gs[0])
        self.ax_pred  = self.fig.add_subplot(gs[1])
        self.ax_ret   = self.fig.add_subplot(gs[2])
        self.ax_eq    = self.fig.add_subplot(gs[3])

        self.fig.suptitle(
            ' MARKET HEARTBEAT - Strategy Live',
            fontsize=24, fontweight='bold', color='#00FF00', y=0.995
        )

        self.line_price, = self.ax_price.plot([], [], '#00FF00', linewidth=3, label='BTC Price')
        self.line_pred,  = self.ax_pred.plot([], [], '#00FFFF', linewidth=3, label='ML Prediction')
        self.line_ret,   = self.ax_ret.plot([], [], '#FFFF00', linewidth=2.5, label='Returns 30s')
        self.line_eq,    = self.ax_eq.plot([], [], '#00FF00', linewidth=3.5, label='Portfolio Value')

        long_thr = max(ENTRY_THR, 0.5 + NO_TRADE_BAND)
        short_thr = min(1.0 - ENTRY_THR + 0.02, 0.5 - NO_TRADE_BAND)

        self.ax_pred.axhline(long_thr, color='lime', linestyle='--',
                             alpha=0.6, linewidth=2, label=f'Long Entry {long_thr:.3f}')
        self.ax_pred.axhline(short_thr, color='red', linestyle='--',
                             alpha=0.6, linewidth=2, label=f'Short Entry {short_thr:.3f}')
        self.ax_pred.axhline(0.5, color='gray', linestyle=':', alpha=0.3, label='Neutral')

        self.ax_pred.axhspan(short_thr, long_thr, alpha=0.1, color='yellow', label='No-Trade Band')

        self.ax_ret.axhline(0, color='gray', linestyle=':', alpha=0.4)
        self.ax_eq.axhline(INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.4, label='Initial')

        self.ax_price.set_ylabel('Price (USD)', fontweight='bold', fontsize=14, color='#00FF00')
        self.ax_pred.set_ylabel('ML Signal',   fontweight='bold', fontsize=14, color='#00FFFF')
        self.ax_ret.set_ylabel('Returns (BPS)',fontweight='bold', fontsize=14, color='#FFFF00')
        self.ax_eq.set_ylabel('Equity (USD)',  fontweight='bold', fontsize=14, color='#00FF00')

        for ax in [self.ax_price, self.ax_pred, self.ax_ret, self.ax_eq]:
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#003300')
            ax.set_facecolor('#000000')
            for spine in ax.spines.values():
                spine.set_color('#00FF00')
                spine.set_linewidth(2)
            ax.legend(loc='upper left', fontsize=9, framealpha=0.7)

        for ax in [self.ax_price, self.ax_pred, self.ax_ret]:
            ax.set_xticklabels([])

        self.ax_eq.set_xlabel('Time (samples)', fontweight='bold', fontsize=14, color='white')

        self.metrics_text = self.fig.text(
            0.5, 0.02, '', fontsize=14, color='white',
            ha='center', va='bottom', family='monospace',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.9,
                      edgecolor='lime', linewidth=3)
        )

        self.flash_patches = []
        for ax in [self.ax_price, self.ax_pred, self.ax_ret, self.ax_eq]:
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                             facecolor='red', edgecolor='none',
                             alpha=0, zorder=100)
            ax.add_patch(rect)
            self.flash_patches.append(rect)

        print(" ECG ready - BACKTEST ENGINE + OLD LOOK")

    def update(self, frame):
        if self.i >= len(self.df):
            return []

        steps = min(SKIP_SAMPLES, len(self.df) - self.i)
        for _ in range(steps):
            row = self.df.iloc[self.i]
            price = float(row["price"])
            pred = float(self.preds[self.i])
            ts = row["timestamp"]

            should_exit, reason = self.strat.check_exit(price, self.i, ts, pred)
            if self.strat.position_state != 0 and should_exit:
                self.strat.close_position(price, self.i, ts, reason)

            if self.strat.position_state == 0 and self.i >= VOLATILITY_LOOKBACK and self.i >= self.strat.cooldown_until:
                long_thr = max(ENTRY_THR, 0.5 + NO_TRADE_BAND)
                short_thr = min(1.0 - ENTRY_THR + 0.02, 0.5 - NO_TRADE_BAND)

                m60 = float(row["momentum_60s"])
                tr  = float(row["trend_strength"])

                if pred >= long_thr and tr >= 0:
                    act = self.strat.open_position(price, self.i, ts, "LONG")
                    if act:
                        self.flash_alpha = FLASH_INTENSITY
                        self.flash_count += 1
                elif pred <= short_thr and m60 < 0 and tr <= 0:
                    act = self.strat.open_position(price, self.i, ts, "SHORT")
                    if act:
                        self.flash_alpha = FLASH_INTENSITY
                        self.flash_count += 1

            eq = self.strat._mark_equity(price, ts)

            if self.i > 0:
                prev_price = float(self.df.iloc[self.i - 1]["price"])
                rbps = (price / prev_price - 1.0) * 10000.0
            else:
                rbps = 0.0

            self.times.append(self.i)
            self.prices.append(price)
            self.preds_w.append(pred)
            self.returns_bps.append(rbps)
            self.equity.append(eq)

            self.i += 1

        self.line_price.set_data(list(self.times), list(self.prices))
        self.line_pred.set_data(list(self.times), list(self.preds_w))
        self.line_ret.set_data(list(self.times), list(self.returns_bps))
        self.line_eq.set_data(list(self.times), list(self.equity))

        if len(self.times) > 1:
            xmin, xmax = min(self.times), max(self.times)
            for ax in [self.ax_price, self.ax_pred, self.ax_ret, self.ax_eq]:
                ax.set_xlim(xmin, xmax)

            ymin, ymax = safe_ylim(self.prices, default_range=(45000, 55000), margin_pct=0.15)
            self.ax_price.set_ylim(ymin, ymax)

            self.ax_pred.set_ylim(0, 1)

            ymin, ymax = safe_ylim(self.returns_bps, default_range=(-200, 200), margin_pct=0.3)
            self.ax_ret.set_ylim(ymin, ymax)

            ymin, ymax = safe_ylim(self.equity, default_range=(INITIAL_CAPITAL*0.9, INITIAL_CAPITAL*1.1), margin_pct=0.1)
            self.ax_eq.set_ylim(ymin, ymax)

        if self.flash_alpha > 0:
            for patch in self.flash_patches:
                patch.set_alpha(self.flash_alpha)
            self.flash_alpha -= FLASH_DECAY
            if self.flash_alpha < 0:
                self.flash_alpha = 0
        else:
            for patch in self.flash_patches:
                patch.set_alpha(0)

        curr_eq = float(self.equity[-1]) if len(self.equity) else INITIAL_CAPITAL
        total_ret = safe_value((curr_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0, 0.0)

        total_trades = len(self.strat.trades_log)

        if total_trades > 0:
            wins = sum(t["pnl_net"] for t in self.strat.trades_log if t["pnl_net"] > 0)
            losses = abs(sum(t["pnl_net"] for t in self.strat.trades_log if t["pnl_net"] <= 0))
            profit_factor = (wins / losses) if losses > 0 else (999.9 if wins > 0 else 0.0)
        else:
            profit_factor = 0.0

        total_exits = self.strat.tp_exits + self.strat.sl_exits + self.strat.timeout_exits + self.strat.signal_exits + self.strat.trail_exits
        if total_exits > 0:
            tp_pct = self.strat.tp_exits / total_exits * 100.0
            sl_pct = self.strat.sl_exits / total_exits * 100.0
        else:
            tp_pct = sl_pct = 0.0

        metrics = f"Sample: {self.i:,}/{len(self.df):,}  |  "
        metrics += f"Return: {total_ret:+.2f}%  |  "
        metrics += f"Trades: {total_trades}  |  "
        metrics += f"PF: {profit_factor:.2f}  |  "
        metrics += f"TP: {tp_pct:.0f}%  SL: {sl_pct:.0f}%"
        if self.flash_count > 0:
            metrics += f"  | {self.flash_count}"

        self.metrics_text.set_text(metrics)

        return (self.line_price, self.line_pred, self.line_ret, self.line_eq, self.metrics_text, *self.flash_patches)

def main():
    print("="*80)
    print(" ECG - ENGINE 1:1 BACKTEST (solo segmento TEST 70/15/15)")
    print("="*80)

    print("\n Loading model + normalization...")
    model = keras.models.load_model(MODEL_FILE, custom_objects={"AttentionLayer": AttentionLayer})
    mean = np.load(MEAN_FILE).astype(np.float32, copy=False)
    std = np.load(STD_FILE).astype(np.float32, copy=False)

    print("\n Loading FULL-SPAN dataset...")
    df_raw = pd.read_parquet(DATA_FILE_ALL)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

    missing = [c for c in FEATURE_COLS if c not in df_raw.columns]
    if missing:
        raise RuntimeError(f"Dataset missing features: {missing}")

    X = df_raw[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
    Xn = (X - mean) / (std + 1e-8)

    from numpy.lib.stride_tricks import sliding_window_view
    X_seq = sliding_window_view(Xn, (SEQUENCE_LENGTH, Xn.shape[1]))[:, 0, :, :]
    X_seq = np.asarray(X_seq, dtype=np.float32)

    print(f" Predicting {len(X_seq):,} sequences...")
    probs = model.predict(X_seq, batch_size=2048, verbose=0).reshape(-1)

    df_full = df_raw.iloc[SEQUENCE_LENGTH - 1:].reset_index(drop=True).copy()
    df_full["log_ret"] = np.log(df_full["price"]).diff().fillna(0.0)

    N = len(df_full)
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)
    n_test = N - n_train - n_val

    df_test = df_full.iloc[n_train + n_val:].copy().reset_index(drop=True)
    preds_test = probs[n_train + n_val:]

    preds_test_sm = rolling_mean_min_periods_1(preds_test, SMOOTHING_WINDOW)

    vol_series = (
        df_test["log_ret"].rolling(window=VOLATILITY_LOOKBACK, min_periods=10).std()
        .bfill().replace(0, np.nan).fillna(df_test["log_ret"].std())
        .to_numpy(dtype=np.float64, copy=False)
    )

    print("\n" + "="*80)
    print(f"TEST segment: {len(df_test):,} samples ({df_test['timestamp'].min()} → {df_test['timestamp'].max()})")
    print(f"Params: entry_thr={ENTRY_THR:.3f} sl={SL_SIGMA:.2f} tp={TP_SIGMA:.2f} nb={NO_TRADE_BAND:.3f}")
    print("="*80)

    time.sleep(1)
    vis = ECGVisualizer(df_test, preds_test_sm, vol_series)

    ani = FuncAnimation(
        vis.fig,
        vis.update,
        frames=max(1, len(df_test) // SKIP_SAMPLES),
        interval=REPLAY_SPEED_MS,
        repeat=False,
        blit=False
    )

    plt.show(block=True)

    final_eq = vis.strat._eq_val[-1] if vis.strat._eq_val else INITIAL_CAPITAL
    final_ret = (final_eq / INITIAL_CAPITAL - 1.0) * 100.0
    print("\n" + "="*80)
    print(" ECG COMPLETE (TEST)")
    print(f"  Total flashes: {vis.flash_count}")
    print(f"  Total trades: {len(vis.strat.trades_log)}")
    total_exits = vis.strat.tp_exits + vis.strat.sl_exits + vis.strat.timeout_exits + vis.strat.signal_exits + vis.strat.trail_exits
    print(f"  TP exits: {vis.strat.tp_exits} ({vis.strat.tp_exits/total_exits*100 if total_exits>0 else 0:.0f}%)")
    print(f"  SL exits: {vis.strat.sl_exits} ({vis.strat.sl_exits/total_exits*100 if total_exits>0 else 0:.0f}%)")
    print(f"  Signal exits (Hidden): {vis.strat.signal_exits}")
    print(f"  Trail exits (Hidden): {vis.strat.trail_exits}")
    print(f"  Timeout exits (Hidden): {vis.strat.timeout_exits}")
    print(f"\n  Final Return: {final_ret:+.2f}%")
    print("="*80)

if __name__ == "__main__":
    main()

