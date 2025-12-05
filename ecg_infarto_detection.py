"""
🫀 MARKET HEARTBEAT - INFARTO DETECTION SYSTEM
Sistema di monitoraggio ECG con rilevamento crash ("infarti") del mercato
Performance: Alpha +10.21%, Win Rate 54.8%
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
import keras
from collections import deque
import time
from datetime import datetime

# ==========================================
# CONFIGURAZIONE OTTIMALE (VALIDATED)
# ==========================================

INITIAL_CAPITAL = 100_000
POSITION_SIZE = 0.85
FEE_BPS = 2.0
SLIPPAGE_BPS = 1.5

# BEST STRATEGY: L0.54 SL120 TP100BPS
ENTRY_LONG = 0.54
ENTRY_SHORT = 0.46  # 1.0 - 0.54
EXIT_LONG = 0.50    # Neutral threshold
EXIT_SHORT = 0.50

STOP_LOSS_BPS = 120.0   # 1.2%
TAKE_PROFIT_BPS = 100.0  # 1.0%

MIN_HOLD = 80   # 40 minuti
MAX_HOLD = 400  # 200 minuti
SMOOTHING_WINDOW = 60
SEQUENCE_LENGTH = 3

# 🚨 CRASH DETECTION THRESHOLDS
CRASH_THRESHOLD = 0.35  # Predizione < 0.35 = CRASH ALERT!
CRASH_VELOCITY = -0.05  # Caduta > 5% in 5 minuti = INFARTO!

# Visualization
REPLAY_SPEED = 50  # ms per frame
MAX_POINTS = 200

# ==========================================
# ATTENTION LAYER
# ==========================================

@tf.keras.utils.register_keras_serializable()
class AttentionLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        e = tf.reduce_sum(e, axis=-1, keepdims=True)
        alpha = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * alpha, axis=1)
    
    def get_config(self):
        return super().get_config()

# ==========================================
# LONG/SHORT TRADING SYSTEM
# ==========================================

class HeartbeatTrader:
    """Long/Short trader con crash detection."""
    
    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.position_state = 0  # 0=CASH, 1=LONG, -1=SHORT
        self.position_qty = 0
        self.entry_price = 0
        self.entry_idx = 0
        self.total_pnl = 0
        self.trades = []
        self.equity_history = []
        
        self.sl_mult = STOP_LOSS_BPS / 10000.0
        self.tp_mult = TAKE_PROFIT_BPS / 10000.0
        self.fee_mult = (FEE_BPS + SLIPPAGE_BPS) / 10000.0
        
        # 🚨 Crash tracking
        self.crash_alerts = []
        self.last_crash_idx = -999
    
    def detect_crash(self, idx, pred, pred_velocity):
        """Rileva crash imminente (infarto)."""
        
        # Alert già recente?
        if idx - self.last_crash_idx < 60:  # No spam (30 min)
            return False
        
        # 🚨 CRASH ALERT CONDITIONS
        crash_detected = (
            pred < CRASH_THRESHOLD or 
            pred_velocity < CRASH_VELOCITY
        )
        
        if crash_detected:
            self.crash_alerts.append({
                'idx': idx,
                'pred': pred,
                'velocity': pred_velocity,
                'severity': 'HIGH' if pred < 0.30 else 'MEDIUM'
            })
            self.last_crash_idx = idx
            return True
        
        return False
    
    def open_position(self, price, idx, direction):
        """Apri posizione LONG o SHORT."""
        
        if self.position_state != 0:
            return False
        
        invest = self.capital * POSITION_SIZE
        cost = invest * self.fee_mult
        
        self.position_qty = (invest - cost) / price
        self.entry_price = price
        self.entry_idx = idx
        self.position_state = 1 if direction == 'LONG' else -1
        self.capital -= invest
        
        self.trades.append({
            'idx': idx,
            'action': 'BUY' if direction == 'LONG' else 'SHORT',
            'direction': direction,
            'price': price,
            'qty': self.position_qty
        })
        
        return True
    
    def close_position(self, price, idx, reason):
        """Chiudi posizione."""
        
        if self.position_state == 0:
            return False
        
        direction = 'LONG' if self.position_state == 1 else 'SHORT'
        
        # Calcola PnL
        if direction == 'LONG':
            pnl_gross = self.position_qty * (price - self.entry_price)
        else:  # SHORT
            pnl_gross = self.position_qty * (self.entry_price - price)
        
        # Costi
        proceeds = self.position_qty * price
        cost = proceeds * self.fee_mult
        pnl_net = pnl_gross - cost
        
        # Restituisci capitale investito + PnL
        invested = INITIAL_CAPITAL * POSITION_SIZE
        self.capital += invested + pnl_net
        self.total_pnl += pnl_net
        
        self.trades.append({
            'idx': idx,
            'action': 'SELL' if direction == 'LONG' else 'COVER',
            'direction': direction,
            'price': price,
            'qty': self.position_qty,
            'pnl': pnl_net,
            'reason': reason
        })
        
        self.position_state = 0
        self.position_qty = 0
        
        return True
    
    def should_exit(self, price, idx, pred):
        """Check exit conditions."""
        
        if self.position_state == 0:
            return False, None
        
        hold_time = idx - self.entry_idx
        
        # Time exit
        if hold_time >= MAX_HOLD:
            return True, 'MAX_HOLD'
        
        # Calculate unrealized return
        if self.position_state == 1:  # LONG
            unreal_ret = (price / self.entry_price) - 1
            
            # Risk exits
            if unreal_ret >= self.tp_mult:
                return True, 'TAKE_PROFIT'
            if unreal_ret <= -self.sl_mult:
                return True, 'STOP_LOSS'
            
            # Signal exit
            if pred <= EXIT_LONG and hold_time >= MIN_HOLD:
                return True, 'SIGNAL'
        
        else:  # SHORT
            unreal_ret = 1 - (price / self.entry_price)
            
            # Risk exits
            if unreal_ret >= self.tp_mult:
                return True, 'TAKE_PROFIT'
            if unreal_ret <= -self.sl_mult:
                return True, 'STOP_LOSS'
            
            # Signal exit
            if pred >= EXIT_SHORT and hold_time >= MIN_HOLD:
                return True, 'SIGNAL'
        
        return False, None
    
    def get_equity(self, price):
        """Calcola equity totale."""
        
        if self.position_state == 0:
            return self.capital
        
        # Valore posizione (quanto vale ora)
        position_value = self.position_qty * price
        
        # Capitale investito
        invested = INITIAL_CAPITAL * POSITION_SIZE
        
        if self.position_state == 1:  # LONG
            # Equity = cash + valore posizione
            return self.capital + position_value
        
        else:  # SHORT
            # PnL da SHORT = (entry_price - current_price) * qty
            pnl_short = self.position_qty * (self.entry_price - price)
            # Equity = cash + capitale investito + PnL
            return self.capital + invested + pnl_short

# ==========================================
# ECG VISUALIZATION CON CRASH DETECTION
# ==========================================

class ECGHeartbeatMonitor:
    """Monitor ECG con infarto detection."""
    
    def __init__(self, df_test, predictions_smooth):
        self.df_test = df_test
        self.predictions = predictions_smooth
        self.trader = HeartbeatTrader()
        self.current_idx = 0
        
        # Data buffers
        self.times = deque(maxlen=MAX_POINTS)
        self.prices = deque(maxlen=MAX_POINTS)
        self.preds = deque(maxlen=MAX_POINTS)
        self.equity = deque(maxlen=MAX_POINTS)
        
        # Trade markers
        self.long_entries = []
        self.long_exits = []
        self.short_entries = []
        self.short_exits = []
        self.crash_markers = []
        
        # Setup plot (stile ospedale)
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(18, 12))
        
        # Grid layout
        gs = self.fig.add_gridspec(4, 2, height_ratios=[2, 2, 1.5, 1], hspace=0.3, wspace=0.25)
        
        self.ax_price = self.fig.add_subplot(gs[0, :])
        self.ax_ecg = self.fig.add_subplot(gs[1, :])
        self.ax_position = self.fig.add_subplot(gs[2, 0])
        self.ax_equity = self.fig.add_subplot(gs[2, 1])
        self.ax_status = self.fig.add_subplot(gs[3, :])
        self.ax_status.axis('off')
        
        # Title
        self.fig.suptitle('🫀 MARKET HEARTBEAT - INFARTO DETECTION SYSTEM', 
                         fontsize=20, fontweight='bold', color='#00FF00')
        
        # Lines
        self.line_price, = self.ax_price.plot([], [], '#00FF00', linewidth=2, label='BTC Price')
        self.line_ecg, = self.ax_ecg.plot([], [], 'cyan', linewidth=2, label='ML Prediction (ECG)')
        self.line_pos, = self.ax_position.plot([], [], 'yellow', linewidth=2, label='Position')
        self.line_equity, = self.ax_equity.plot([], [], 'lime', linewidth=2, label='Portfolio Value')
        
        # Thresholds sul ECG
        self.ax_ecg.axhline(ENTRY_LONG, color='g', linestyle='--', alpha=0.6, linewidth=1.5, label=f'LONG Entry {ENTRY_LONG}')
        self.ax_ecg.axhline(ENTRY_SHORT, color='orange', linestyle='--', alpha=0.6, linewidth=1.5, label=f'SHORT Entry {ENTRY_SHORT}')
        self.ax_ecg.axhline(CRASH_THRESHOLD, color='red', linestyle=':', alpha=0.8, linewidth=2, label=f'🚨 CRASH ZONE {CRASH_THRESHOLD}')
        
        self.ax_equity.axhline(INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.5, label='Initial')
        
        # Labels
        self.ax_price.set_ylabel('Price (USD)', fontweight='bold', fontsize=12, color='white')
        self.ax_ecg.set_ylabel('Prediction (Heart Rate)', fontweight='bold', fontsize=12, color='cyan')
        self.ax_position.set_ylabel('Position Value ($)', fontweight='bold', fontsize=12, color='yellow')
        self.ax_equity.set_ylabel('Equity ($)', fontweight='bold', fontsize=12, color='lime')
        self.ax_equity.set_xlabel('Time (buckets)', fontweight='bold', fontsize=12, color='white')
        
        for ax in [self.ax_price, self.ax_ecg, self.ax_position, self.ax_equity]:
            ax.grid(True, alpha=0.2, linestyle='--', color='gray')
            ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
        
        # Status text
        self.status_text = self.ax_status.text(
            0.5, 0.5, '', 
            fontsize=14, 
            color='white',
            ha='center', 
            va='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='black', alpha=0.9, edgecolor='lime', linewidth=2)
        )
    
    def update(self, frame):
        """Update frame."""
        
        if self.current_idx >= len(self.df_test):
            return
        
        # Get data
        row = self.df_test.iloc[self.current_idx]
        price = row['price']
        pred = self.predictions[self.current_idx]
        
        # Prediction velocity (crash detection)
        if self.current_idx >= 10:
            pred_velocity = pred - self.predictions[self.current_idx - 10]
        else:
            pred_velocity = 0
        
        # 🚨 CRASH DETECTION
        crash_detected = self.trader.detect_crash(self.current_idx, pred, pred_velocity)
        if crash_detected:
            self.crash_markers.append((self.current_idx, pred))
        
        # Trading logic
        should_exit, exit_reason = self.trader.should_exit(price, self.current_idx, pred)
        
        if self.trader.position_state != 0 and should_exit:
            if self.trader.position_state == 1:
                self.long_exits.append((self.current_idx, price))
            else:
                self.short_exits.append((self.current_idx, price))
            
            self.trader.close_position(price, self.current_idx, exit_reason)
        
        if self.trader.position_state == 0:
            if pred >= ENTRY_LONG:
                if self.trader.open_position(price, self.current_idx, 'LONG'):
                    self.long_entries.append((self.current_idx, price))
            
            elif pred <= ENTRY_SHORT:
                if self.trader.open_position(price, self.current_idx, 'SHORT'):
                    self.short_entries.append((self.current_idx, price))
        
        # Update buffers
        self.times.append(self.current_idx)
        self.prices.append(price)
        self.preds.append(pred)
        
        current_equity = self.trader.get_equity(price)
        self.equity.append(current_equity)
        
        # Update lines
        self.line_price.set_data(list(self.times), list(self.prices))
        self.line_ecg.set_data(list(self.times), list(self.preds))
        self.line_equity.set_data(list(self.times), list(self.equity))
        
        # Position value
        if self.trader.position_state != 0:
            pos_value = self.trader.position_qty * price
            position_series = [pos_value if t >= self.trader.entry_idx else 0 for t in self.times]
            self.line_pos.set_data(list(self.times), position_series)
        else:
            self.line_pos.set_data(list(self.times), [0] * len(self.times))
        
        # Update axes
        if len(self.times) > 0:
            xmin, xmax = min(self.times), max(self.times)
            
            for ax in [self.ax_price, self.ax_ecg, self.ax_position, self.ax_equity]:
                ax.set_xlim(xmin, xmax)
            
            # Y limits
            if len(self.prices) > 0:
                price_margin = (max(self.prices) - min(self.prices)) * 0.05
                self.ax_price.set_ylim(min(self.prices) - price_margin, max(self.prices) + price_margin)
            
            self.ax_ecg.set_ylim(0, 1)
            
            if len(self.equity) > 0:
                eq_margin = (max(self.equity) - min(self.equity)) * 0.1
                self.ax_equity.set_ylim(min(self.equity) - eq_margin, max(self.equity) + eq_margin)
        
        # Update markers
        if self.long_entries:
            x_long, y_long = zip(*self.long_entries)
            self.ax_price.scatter(x_long, y_long, c='lime', s=200, marker='^', zorder=5, edgecolors='white', linewidths=2, label='LONG Entry')
        
        if self.long_exits:
            x_long_exit, y_long_exit = zip(*self.long_exits)
            self.ax_price.scatter(x_long_exit, y_long_exit, c='green', s=150, marker='v', zorder=5, edgecolors='white', linewidths=2)
        
        if self.short_entries:
            x_short, y_short = zip(*self.short_entries)
            self.ax_price.scatter(x_short, y_short, c='orange', s=200, marker='v', zorder=5, edgecolors='black', linewidths=2, label='SHORT Entry')
        
        if self.short_exits:
            x_short_exit, y_short_exit = zip(*self.short_exits)
            self.ax_price.scatter(x_short_exit, y_short_exit, c='darkorange', s=150, marker='^', zorder=5, edgecolors='black', linewidths=2)
        
        # 🚨 CRASH MARKERS
        if self.crash_markers:
            x_crash, y_crash = zip(*self.crash_markers)
            self.ax_ecg.scatter(x_crash, y_crash, c='red', s=300, marker='X', zorder=10, edgecolors='yellow', linewidths=3, label='🚨 CRASH ALERT!')
        
        # Status text
        pnl_pct = (current_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        position_str = {
            0: "💵 CASH",
            1: f"📈 LONG {self.trader.position_qty:.4f} BTC",
            -1: f"📉 SHORT {self.trader.position_qty:.4f} BTC"
        }[self.trader.position_state]
        
        winning = sum(1 for t in self.trader.trades if 'pnl' in t and t['pnl'] > 0)
        total_trades = len([t for t in self.trader.trades if 'pnl' in t])
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
        
        bh_return = (price / self.df_test.iloc[0]['price'] - 1) * 100
        alpha = pnl_pct - bh_return
        
        # 🚨 Crash alert status
        crash_status = ""
        if crash_detected:
            crash_status = " | 🚨 INFARTO ALERT! 🚨"
        elif len(self.trader.crash_alerts) > 0:
            last_crash = self.trader.crash_alerts[-1]
            if self.current_idx - last_crash['idx'] < 120:  # Ultimo crash < 1 ora fa
                crash_status = f" | ⚠️  Last Alert: {self.current_idx - last_crash['idx']} buckets ago"
        
        status = f"Time: {self.current_idx}/{len(self.df_test)} | "
        status += f"Price: ${price:,.2f} | "
        status += f"ECG: {pred:.3f} | "
        status += f"{position_str}\n"
        status += f"Equity: ${current_equity:,.2f} | "
        status += f"PnL: {pnl_pct:+.2f}% | "
        status += f"Alpha: {alpha:+.2f}% | "
        status += f"Trades: {total_trades} | "
        status += f"Win Rate: {win_rate:.1f}%"
        status += crash_status
        
        self.status_text.set_text(status)
        
        # Color status box based on alerts
        if crash_detected:
            self.status_text.set_bbox(dict(boxstyle='round,pad=0.8', facecolor='darkred', alpha=0.9, edgecolor='red', linewidth=3))
        else:
            self.status_text.set_bbox(dict(boxstyle='round,pad=0.8', facecolor='black', alpha=0.9, edgecolor='lime', linewidth=2))
        
        self.current_idx += 1

# ==========================================
# MAIN
# ==========================================

if __name__ == '__main__':
    print("="*80)
    print("🫀 MARKET HEARTBEAT - INFARTO DETECTION SYSTEM")
    print("="*80)
    print("\n📊 Configuration:")
    print(f"   Strategy: LONG/SHORT")
    print(f"   Entry LONG: {ENTRY_LONG}")
    print(f"   Entry SHORT: {ENTRY_SHORT}")
    print(f"   Stop Loss: {STOP_LOSS_BPS} BPS")
    print(f"   Take Profit: {TAKE_PROFIT_BPS} BPS")
    print(f"   🚨 Crash Threshold: {CRASH_THRESHOLD}")
    
    print("\n📦 Loading model...")
    model = keras.models.load_model(
        'best_model_2024.keras',
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    
    print("📦 Loading data...")
    df = pd.read_parquet('processed_data_DIRECTIONAL_2024_DEC.parquet')
    mean = np.load('normalization_mean_DIRECTIONAL_2024.npy')
    std = np.load('normalization_std_DIRECTIONAL_2024.npy')
    
    # Prepare features
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
    
    X_features = df[feature_cols].values
    X_normalized = (X_features - mean) / (std + 1e-8)
    
    # Sequences
    X_sequences = []
    for i in range(len(X_normalized) - SEQUENCE_LENGTH + 1):
        X_sequences.append(X_normalized[i:i + SEQUENCE_LENGTH])
    X_sequences = np.array(X_sequences, dtype=np.float32)
    
    print("🤖 Generating predictions...")
    predictions = model.predict(X_sequences, batch_size=2048, verbose=0).flatten()
    predictions_smooth = pd.Series(predictions).rolling(
        window=SMOOTHING_WINDOW, min_periods=1
    ).mean().values
    
    df_test = df.iloc[SEQUENCE_LENGTH-1:].reset_index(drop=True)
    
    print(f"✅ Ready: {len(df_test):,} samples")
    print(f"\n⏱️  Estimated time: {len(df_test) * REPLAY_SPEED / 1000 / 60:.1f} minutes")
    print("\n🚨 CRASH DETECTION ACTIVE")
    print(f"   Alert when prediction < {CRASH_THRESHOLD}")
    print(f"   Alert when velocity < {CRASH_VELOCITY}")
    
    print("\n🎬 Starting in 3 seconds...")
    time.sleep(3)
    
    # Create monitor
    monitor = ECGHeartbeatMonitor(df_test, predictions_smooth)
    
    # Animation
    ani = FuncAnimation(
        monitor.fig,
        monitor.update,
        frames=len(df_test),
        interval=REPLAY_SPEED,
        repeat=False,
        blit=False
    )
    
    plt.show()
    
    # Final summary
    print("\n" + "="*80)
    print("📊 INFARTO DETECTION - FINAL SUMMARY")
    print("="*80)
    
    final_equity = monitor.trader.get_equity(df_test.iloc[-1]['price'])
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    bh_return = (df_test['price'].iloc[-1] / df_test['price'].iloc[0] - 1) * 100
    alpha = total_return - bh_return
    
    winning = sum(1 for t in monitor.trader.trades if 'pnl' in t and t['pnl'] > 0)
    total_trades = len([t for t in monitor.trader.trades if 'pnl' in t])
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\n💰 PERFORMANCE:")
    print(f"   Initial:     ${INITIAL_CAPITAL:,.2f}")
    print(f"   Final:       ${final_equity:,.2f}")
    print(f"   Return:      {total_return:+.2f}%")
    print(f"   Buy & Hold:  {bh_return:+.2f}%")
    print(f"   Alpha:       {alpha:+.2f}%")
    
    print(f"\n📈 TRADES:")
    print(f"   Total:       {total_trades}")
    print(f"   Winning:     {winning}")
    print(f"   Win Rate:    {win_rate:.1f}%")
    
    print(f"\n🚨 CRASH ALERTS:")
    print(f"   Total:       {len(monitor.trader.crash_alerts)}")
    
    if monitor.trader.crash_alerts:
        print(f"\n   Recent alerts:")
        for alert in monitor.trader.crash_alerts[-5:]:
            severity_icon = "🔴" if alert['severity'] == 'HIGH' else "🟠"
            print(f"   {severity_icon} Bucket {alert['idx']}: Pred={alert['pred']:.3f}, Velocity={alert['velocity']:+.3f}")
    
    print("\n" + "="*80)
    print("✅ MONITORING COMPLETE")
    print("="*80)