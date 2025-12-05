"""
🫀 MARKET HEARTBEAT - UNIVERSAL MONTH TESTER
Test su qualsiasi mese con configurazione ottimale da Grid Search

Usage:
    python universal_tester.py --month SEP_2023
    python universal_tester.py --month JAN_2025
    python universal_tester.py --month FEB_2024
"""
import numpy as np
import pandas as pd
import keras
import json
import argparse
import sys
from pathlib import Path

# ==========================================
# CONFIGURAZIONE
# ==========================================

INITIAL_CAPITAL = 100_000
SEQUENCE_LENGTH = 3

# ==========================================
# ATTENTION LAYER
# ==========================================

@keras.utils.register_keras_serializable()
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
        import tensorflow as tf
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        e = tf.reduce_sum(e, axis=-1, keepdims=True)
        alpha = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * alpha, axis=1)
    
    def get_config(self):
        return super().get_config()

# ==========================================
# BACKTESTER
# ==========================================

class UniversalBacktester:
    """Backtester che usa configurazione da JSON."""
    
    def __init__(self, config):
        self.config = config
        
        self.capital = INITIAL_CAPITAL
        self.position_state = 0
        self.position_qty = 0
        self.entry_price = 0
        self.entry_idx = 0
        self.total_fees = 0
        self.trades = []
        self.equity_history = []
        
        # Load config
        self.entry_long = config['entry_long']
        self.entry_short = config['entry_short']
        self.exit_long = config['exit_long']
        self.exit_short = config['exit_short']
        
        self.sl_mult = config['stop_loss_bps'] / 10000.0
        self.tp_mult = config['take_profit_bps'] / 10000.0
        
        self.smoothing_window = config['smoothing_window']
        self.min_hold = config['min_hold']
        self.max_hold = config['max_hold']
        self.position_size = config['position_size']
        
        self.fee_mult = 0.00035  # 2.0 + 1.5 BPS
    
    def open_position(self, price, idx, direction):
        if self.position_state != 0:
            return
        
        invest = self.capital * self.position_size
        cost = invest * self.fee_mult
        
        self.position_qty = (invest - cost) / price
        self.entry_price = price
        self.entry_idx = idx
        self.position_state = 1 if direction == 'LONG' else -1
        
        self.capital -= invest
        self.total_fees += cost
        
        self.trades.append({
            'idx': idx,
            'action': 'BUY' if direction == 'LONG' else 'SHORT',
            'direction': direction,
            'price': price
        })
    
    def close_position(self, price, idx, reason):
        if self.position_state == 0:
            return
        
        direction = 'LONG' if self.position_state == 1 else 'SHORT'
        
        if direction == 'LONG':
            pnl_gross = self.position_qty * (price - self.entry_price)
        else:
            pnl_gross = self.position_qty * (self.entry_price - price)
        
        proceeds = self.position_qty * price
        cost = proceeds * self.fee_mult
        pnl_net = pnl_gross - cost
        
        invest = INITIAL_CAPITAL * self.position_size
        self.capital += invest + pnl_net
        self.total_fees += cost
        
        self.trades.append({
            'idx': idx,
            'action': 'SELL' if direction == 'LONG' else 'COVER',
            'direction': direction,
            'price': price,
            'pnl': pnl_net,
            'reason': reason,
            'winning': pnl_net > 0
        })
        
        self.position_state = 0
        self.position_qty = 0
    
    def should_exit(self, price, idx, pred):
        if self.position_state == 0:
            return False, None
        
        hold_time = idx - self.entry_idx
        
        # Time exit
        if hold_time >= self.max_hold:
            return True, 'MAX_HOLD'
        
        # Risk exits
        if self.position_state == 1:  # LONG
            unreal = (price / self.entry_price) - 1
            
            if unreal >= self.tp_mult:
                return True, 'TAKE_PROFIT'
            if unreal <= -self.sl_mult:
                return True, 'STOP_LOSS'
            
            if pred <= self.exit_long and hold_time >= self.min_hold:
                return True, 'SIGNAL'
        
        else:  # SHORT
            unreal = 1 - (price / self.entry_price)
            
            if unreal >= self.tp_mult:
                return True, 'TAKE_PROFIT'
            if unreal <= -self.sl_mult:
                return True, 'STOP_LOSS'
            
            if pred >= self.exit_short and hold_time >= self.min_hold:
                return True, 'SIGNAL'
        
        return False, None
    
    def run(self, df, predictions):
        """Run backtest."""
        
        print(f"\n🔄 Smoothing predictions (window={self.smoothing_window})...")
        preds_smooth = pd.Series(predictions).rolling(
            window=self.smoothing_window, min_periods=1
        ).mean().values
        
        print(f"🚀 Running backtest...")
        
        for idx in range(len(df)):
            price = df.iloc[idx]['price']
            pred = preds_smooth[idx]
            
            # Check exit
            should_exit, reason = self.should_exit(price, idx, pred)
            
            if self.position_state != 0 and should_exit:
                self.close_position(price, idx, reason)
            
            # Check entry
            if self.position_state == 0:
                if pred >= self.entry_long:
                    self.open_position(price, idx, 'LONG')
                elif pred <= self.entry_short:
                    self.open_position(price, idx, 'SHORT')
            
            # Track equity
            if self.position_state != 0:
                equity = self.capital + (self.position_qty * price)
            else:
                equity = self.capital
            
            self.equity_history.append(equity)
        
        # Close final
        if self.position_state != 0:
            self.close_position(df.iloc[-1]['price'], len(df)-1, 'END')
        
        return self.compute_metrics(df)
    
    def compute_metrics(self, df):
        """Compute performance metrics."""
        
        final_capital = self.capital
        total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
        bh_return = (df['price'].iloc[-1] / df['price'].iloc[0] - 1) * 100
        alpha = total_return - bh_return
        
        closed_trades = [t for t in self.trades if 'pnl' in t]
        winning = sum(1 for t in closed_trades if t['winning'])
        total_trades = len(closed_trades)
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
        
        # Sharpe
        equity_series = pd.Series(self.equity_history)
        returns = equity_series.pct_change().fillna(0)
        sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252 * 48)
        
        # Exit reasons
        exit_reasons = {}
        for t in closed_trades:
            reason = t.get('reason', 'UNKNOWN')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return {
            'final_capital': final_capital,
            'total_return': total_return,
            'bh_return': bh_return,
            'alpha': alpha,
            'sharpe': sharpe,
            'total_trades': total_trades,
            'winning_trades': winning,
            'losing_trades': total_trades - winning,
            'win_rate': win_rate,
            'total_costs': self.total_fees,
            'exit_reasons': exit_reasons
        }

# ==========================================
# MAIN TESTER
# ==========================================

def test_month(month_prefix, model_file='best_model_2024.keras', 
               config_file='best_strategy_config.json'):
    """
    Test su qualsiasi mese.
    
    Args:
        month_prefix: e.g., 'SEP_2023', 'JAN_2025', 'FEB_2024'
        model_file: Path al modello
        config_file: Path alla configurazione ottimale
    """
    
    print("="*80)
    print(f"🫀 MARKET HEARTBEAT - UNIVERSAL TESTER")
    print(f"   Testing: {month_prefix}")
    print("="*80)
    
    # 1. Load config
    print(f"\n📦 Loading best configuration...")
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"   ✅ Loaded config from: {config_file}")
        print(f"   Optimized on: {config['optimization_period']}")
        print(f"   Entry: {config['entry_long']:.2f}")
        print(f"   SL: {config['stop_loss_bps']:.0f} BPS")
        print(f"   TP: {config['take_profit_bps']:.0f} BPS")
    
    except FileNotFoundError:
        print(f"   ❌ Config file not found: {config_file}")
        print(f"   Run grid_search_with_save.py first!")
        sys.exit(1)
    
    # 2. Load model
    print(f"\n📦 Loading model...")
    try:
        model = keras.models.load_model(
            model_file,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        print(f"   ✅ Model loaded: {model_file}")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        sys.exit(1)
    
    # 3. Load data
    print(f"\n📂 Loading test data...")
    
    data_file = f'processed_data_DIRECTIONAL_{month_prefix}.parquet'
    
    if not Path(data_file).exists():
        print(f"   ❌ Data file not found: {data_file}")
        print(f"   Process the month first with processor!")
        sys.exit(1)
    
    df = pd.read_parquet(data_file)
    print(f"   ✅ Loaded: {len(df):,} samples")
    print(f"   Period: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    # 4. Load normalization
    print(f"\n📦 Loading normalization...")
    try:
        mean = np.load('normalization_mean_DIRECTIONAL_2024.npy')
        std = np.load('normalization_std_DIRECTIONAL_2024.npy')
        print(f"   ✅ Using 2024 normalization stats")
    except:
        print(f"   ⚠️  Normalization files not found, computing new ones")
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
        X = df[feature_cols].values
        mean = X.mean(axis=0)
        std = X.std(axis=0)
    
    # 5. Generate predictions
    print(f"\n🤖 Generating predictions...")
    
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
    
    X = df[feature_cols].values
    X_norm = (X - mean) / (std + 1e-8)
    
    X_seq = []
    for i in range(len(X_norm) - SEQUENCE_LENGTH + 1):
        X_seq.append(X_norm[i:i + SEQUENCE_LENGTH])
    X_seq = np.array(X_seq, dtype=np.float32)
    
    predictions = model.predict(X_seq, batch_size=2048, verbose=0).flatten()
    
    df_test = df.iloc[SEQUENCE_LENGTH-1:].reset_index(drop=True)
    
    print(f"   ✅ Generated {len(predictions):,} predictions")
    
    # 6. Run backtest
    print(f"\n" + "="*80)
    print(f"📊 BACKTESTING WITH OPTIMAL CONFIG")
    print(f"="*80)
    
    backtester = UniversalBacktester(config)
    metrics = backtester.run(df_test, predictions)
    
    # 7. Report
    print(f"\n" + "="*80)
    print(f"📊 RESULTS: {month_prefix}")
    print(f"="*80)
    
    print(f"\n💰 PERFORMANCE:")
    print(f"   Initial Capital:  ${INITIAL_CAPITAL:,.2f}")
    print(f"   Final Capital:    ${metrics['final_capital']:,.2f}")
    print(f"   Total Return:     {metrics['total_return']:+.2f}%")
    print(f"   Buy & Hold:       {metrics['bh_return']:+.2f}%")
    print(f"   Alpha:            {metrics['alpha']:+.2f}%")
    print(f"   Sharpe Ratio:     {metrics['sharpe']:.2f}")
    
    # Status
    if metrics['alpha'] > 5:
        status = "✅✅ EXCELLENT"
    elif metrics['alpha'] > 2:
        status = "✅ GOOD"
    elif metrics['alpha'] > 0:
        status = "⚠️  POSITIVE but low"
    else:
        status = "❌ NEGATIVE"
    
    print(f"\n   Status: {status}")
    
    print(f"\n📈 TRADING:")
    print(f"   Total Trades:     {metrics['total_trades']}")
    print(f"   Winning:          {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)")
    print(f"   Losing:           {metrics['losing_trades']}")
    print(f"   Total Costs:      ${metrics['total_costs']:,.2f}")
    
    wr_status = "✅" if metrics['win_rate'] >= 50 else "⚠️"
    print(f"   Win Rate Status:  {wr_status}")
    
    print(f"\n🎯 EXIT REASONS:")
    for reason, count in sorted(metrics['exit_reasons'].items(), key=lambda x: -x[1]):
        pct = count / metrics['total_trades'] * 100
        print(f"   {reason}: {count} ({pct:.1f}%)")
    
    print(f"\n" + "="*80)
    print(f"📊 COMPARISON WITH OPTIMIZATION PERIOD")
    print(f"="*80)
    
    opt_alpha = config['performance']['alpha']
    opt_wr = config['performance']['win_rate']
    
    alpha_diff = metrics['alpha'] - opt_alpha
    wr_diff = metrics['win_rate'] - opt_wr
    
    print(f"\n   Optimization (Dec 2024): Alpha {opt_alpha:+.2f}%, WR {opt_wr:.1f}%")
    print(f"   This Period ({month_prefix}): Alpha {metrics['alpha']:+.2f}%, WR {metrics['win_rate']:.1f}%")
    print(f"   Difference: Alpha {alpha_diff:+.2f}%, WR {wr_diff:+.1f}%")
    
    # Diagnosis
    print(f"\n💡 INTERPRETATION:")
    
    if metrics['alpha'] > 0 and metrics['win_rate'] >= 48:
        print(f"   ✅ Config generalizes WELL to this period")
        print(f"   ✅ Model is ROBUST")
        print(f"   ✅ Safe for deployment")
    
    elif metrics['alpha'] < 0 and metrics['win_rate'] < 45:
        print(f"   ❌ Config does NOT work on this period")
        print(f"   ❌ Possible MODEL OVERFITTING")
        print(f"   💡 Consider: Re-train with TimeSeriesSplit")
    
    else:
        print(f"   ⚠️  Mixed results")
        print(f"   ⚠️  Performance degraded but not catastrophic")
        print(f"   💡 Monitor more periods before deployment")
    
    print(f"\n" + "="*80)
    
    return metrics

# ==========================================
# CLI
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test Market Heartbeat su qualsiasi mese',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Esempi:
  python universal_tester.py AUG_2023
  python universal_tester.py JAN_2025
  python universal_tester.py SEP_2023
        '''
    )
    
    parser.add_argument('month', type=str,
                       help='Mese da testare (e.g., SEP_2023, JAN_2025, AUG_2023)')
    parser.add_argument('--model', type=str, default='best_model_2024.keras',
                       help='Path al modello (default: best_model_2024.keras)')
    parser.add_argument('--config', type=str, default='best_strategy_config.json',
                       help='Path alla configurazione (default: best_strategy_config.json)')
    
    args = parser.parse_args()
    
    try:
        metrics = test_month(args.month, args.model, args.config)
        print(f"\n✅ TEST COMPLETATO")
    
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)