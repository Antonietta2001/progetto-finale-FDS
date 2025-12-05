"""
================================================================================
MARKET HEARTBEAT - ULTIMA CALIBRAZIONE E GRID SEARCH FINALE (2024)
Correzione Definitiva: Risolto AttributeError su 'equity_history'.
================================================================================
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import sys
import os
import glob
import warnings
import matplotlib.pyplot as plt
from keras import layers, Model
from sklearn.metrics import accuracy_score
import json

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURAZIONE
# ==========================================

INITIAL_CAPITAL = 100_000
POSITION_SIZE = 0.85
FEE_BPS = 2.0
SLIPPAGE_BPS = 1.5

# Parametri HFT
SMOOTHING_WINDOW = 60
SEQUENCE_LENGTH = 3
MIN_HOLD = 80
MAX_HOLD = 400

# Parametri Fissi Ottimali
OPTIMAL_ENTRY_THR = 0.54 
L_THR_BASE = 0.50 
S_THR_BASE = 0.50 
TAKE_PROFIT_BPS = 150.0

# GRIGLIA SL/TP DA TESTARE
TP_BPS_TESTS = [100.0, 150.0]
SL_BPS_TESTS = [75.0, 100.0, 120.0] 

DATA_FILE_DEC = 'processed_data_DIRECTIONAL_2024_DEC.parquet'
MODEL_FILE = 'best_model_2024.keras'

# ==========================================
# MODELLO E FUNZIONI DI SUPPORTO
# ==========================================

@tf.keras.utils.register_keras_serializable()
class AttentionLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        super().build(input_shape)
    
    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        e = tf.reduce_sum(e, axis=-1, keepdims=True)
        alpha = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * alpha, axis=1)
    
    def get_config(self):
        return super().get_config()

def load_assets_for_backtest():
    """Carica modello, normalizzazione e dati di Dicembre."""
    print("="*80)
    print("🎯 MARKET HEARTBEAT - FINAL BACKTEST START")
    print("="*80)
    
    try:
        print("\n📦 Loading model...")
        model = keras.models.load_model(MODEL_FILE, custom_objects={'AttentionLayer': AttentionLayer})
        
        print("\n📦 Loading normalization...")
        mean = np.load('normalization_mean_DIRECTIONAL_2024.npy')
        std = np.load('normalization_std_DIRECTIONAL_2024.npy')
        
        print("\n📂 Loading data...")
        df_raw = pd.read_parquet(DATA_FILE_DEC)
        
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
        
        X_features = df_raw[feature_cols].values
        X_normalized = (X_features - mean) / (std + 1e-8)
        
        X_sequences = []
        for i in range(len(X_normalized) - SEQUENCE_LENGTH + 1):
            X_sequences.append(X_normalized[i:i + SEQUENCE_LENGTH])
        X_sequences = np.array(X_sequences, dtype=np.float32)

        print("\n🤖 Generating predictions...")
        predictions_raw = model.predict(X_sequences, batch_size=2048, verbose=0).flatten()
        
        df_test = df_raw.iloc[SEQUENCE_LENGTH-1:].reset_index(drop=True)
        
        return df_test, predictions_raw
        
    except Exception as e:
        print(f"❌ ERRORE CRITICO CARICAMENTO ASSET: {e}")
        print("   Verificare che Trainer e Processor 2024 abbiano salvato tutti i file richiesti.")
        sys.exit(1)


# --- CLASSE STRATEGIA LONG/SHORT ---

class LongShortOptimizer:
    
    def __init__(self, initial_capital, position_size, fee_bps, slippage_bps, 
                 min_hold, max_hold, sl_bps, tp_bps, l_thr_exit, s_thr_exit):
        
        self.capital = initial_capital 
        self.invested_capital = 0 
        self.position_state = 0 
        self.position_qty = 0
        self.entry_price = 0
        self.entry_bucket = 0
        self.total_fees = 0
        self.trades_log = []
        self.equity_history = [] # 🎯 CORREZIONE: Inizializza l'history dell'equity

        self.fee_multiplier = (fee_bps + slippage_bps) / 10000
        self.sl_multiplier = sl_bps / 10000.0
        self.tp_multiplier = tp_bps / 10000.0
        
        self.min_hold_buckets = min_hold
        self.max_hold_buckets = max_hold
        
        self.l_thr_exit = l_thr_exit
        self.s_thr_exit = s_thr_exit

    def open_position(self, price, idx, direction, position_size):
        if self.position_state != 0: return
        invest_amount = self.capital * position_size
        cost_entry = invest_amount * self.fee_multiplier
        
        self.position_qty = (invest_amount - cost_entry) / price 
        
        self.position_state = 1 if direction == 'LONG' else -1
        self.entry_price = price
        self.entry_bucket = idx
        self.capital -= invest_amount 
        self.invested_capital = invest_amount
        self.total_fees += cost_entry
        # self.equity_history verrà aggiornata nel loop principale

    def close_position(self, price, idx, exit_reason):
        if self.position_state == 0: return
        
        direction = 'LONG' if self.position_state == 1 else 'SHORT'
        
        if direction == 'LONG':
            pnl_gross = self.position_qty * (price - self.entry_price)
        else:
            pnl_gross = self.position_qty * (self.entry_price - price)

        cost_exit = self.position_qty * price * self.fee_multiplier
        pnl_net = pnl_gross - cost_exit
        
        self.capital += self.invested_capital + pnl_net
        self.total_fees += cost_exit
        
        self.trades_log.append({'pnl_net': pnl_net, 'winning': pnl_net > 0})
        
        self.position_state = 0
        self.position_qty = 0
        self.invested_capital = 0

    def check_exit(self, price, idx, current_pred):
        if self.position_state == 0: return False, None
        
        hold_time = idx - self.entry_bucket
        
        time_exit = hold_time >= self.max_hold_buckets
        
        if self.position_state == 1: # Long
            unrealized_return = (price / self.entry_price) - 1
            risk_exit = (unrealized_return >= self.tp_multiplier) or (unrealized_return <= -self.sl_multiplier)
            signal_exit = (current_pred <= self.s_thr_exit) and (hold_time >= self.min_hold_buckets)
        else: # Short
            unrealized_return = 1 - (price / self.entry_price)
            risk_exit = (unrealized_return >= self.tp_multiplier) or (unrealized_return <= -self.sl_multiplier)
            signal_exit = (current_pred >= self.l_thr_exit) and (hold_time >= self.min_hold_buckets)

        if time_exit: return True, 'TIME'
        if risk_exit: return True, 'RISK'
        if signal_exit: return True, 'SIGNAL'
        return False, None


def run_backtest_optimized(df_test, predictions_test, entry_thr, sl_bps, tp_bps):
    
    predictions_smoothed = pd.Series(predictions_test).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean().values
    
    strategy = LongShortOptimizer(
        INITIAL_CAPITAL, POSITION_SIZE, FEE_BPS, SLIPPAGE_BPS, 
        MIN_HOLD, MAX_HOLD, sl_bps, tp_bps,
        L_THR_BASE, S_THR_BASE 
    )
    
    L_thr_entry = entry_thr
    S_thr_entry = 1.0 - entry_thr 

    for idx in range(len(df_test)):
        row = df_test.iloc[idx]
        price = row['price']
        pred = predictions_smoothed[idx] 
        
        should_exit, reason = strategy.check_exit(price, idx, pred)
        
        if strategy.position_state != 0 and should_exit:
            strategy.close_position(price, idx, reason)

        if strategy.position_state == 0:
            if pred >= L_thr_entry:
                strategy.open_position(price, idx, 'LONG', POSITION_SIZE)
            elif pred <= S_thr_entry:
                strategy.open_position(price, idx, 'SHORT', POSITION_SIZE)
        
        # 🎯 CORREZIONE: Aggiorna l'equity ad ogni passo del tempo
        current_equity = strategy.capital + (strategy.position_qty * price) if strategy.position_state != 0 else strategy.capital
        strategy.equity_history.append({'time': idx, 'equity': current_equity})


    # Chiudi posizione finale
    if strategy.position_state != 0:
        price = df_test.iloc[-1]['price']
        strategy.close_position(price, len(df_test)-1, 'END')
    
    # Metrics
    final_capital = strategy.capital
    total_ret = (final_capital / INITIAL_CAPITAL - 1) * 100
    bh_return = (df_test['price'].iloc[-1] / df_test['price'].iloc[0] - 1) * 100
    alpha = total_ret - bh_return
    
    total_trades = len(strategy.trades_log)
    winning = sum(t['winning'] for t in strategy.trades_log)
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
    
    # Calcolo Sharpe (richiede l'history di equity)
    equity_df = pd.DataFrame(strategy.equity_history)
    equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
    sharpe = (equity_df['returns'].mean() / (equity_df['returns'].std() + 1e-8)) * np.sqrt(252 * 48)

    
    return {
        'strategy_name': f"L{entry_thr:.2f} SL{sl_bps:.0f} TP{tp_bps:.0f}BPS",
        'total_ret': total_ret,
        'alpha': alpha,
        'win_rate': win_rate,
        'trades': total_trades,
        'winning': winning,
        'losing': total_trades - winning,
        'costs': strategy.total_fees,
        'bh_return': bh_return,
        'sharpe': sharpe
    }

def generate_comparative_report(all_results, df_test, bh_return_pct, save_path='final_sl_optimization_2024.png'):
    """Genera un report comparativo e grafico per l'ottimizzazione SL."""
    
    df_results = pd.DataFrame(all_results)
    
    # 🎯 Selezione Automatica Migliore Strategia (Basata su Alpha)
    df_results = df_results.sort_values(by='alpha', ascending=False).reset_index(drop=True)
    best_robust = df_results.iloc[0]
    
    # 1. Grafico a barre (Alpha e Win Rate)
    strategies_labels = df_results['strategy_name'].apply(lambda x: x.split(' ')[1] + ' ' + x.split(' ')[2])
    alpha_values = df_results['alpha']
    win_rate_values = df_results['win_rate']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Subplot 1: Alpha ---
    colors_alpha = ['green' if x > 0 else 'red' for x in alpha_values]
    axes[0].bar(strategies_labels, alpha_values, color=colors_alpha)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[0].axhline(bh_return_pct, color='blue', linestyle=':', linewidth=1, label=f'B&H ({bh_return_pct:+.2f}%)')
    
    axes[0].set_title(f'Alpha Return ({OPTIMAL_ENTRY_THR} Entry)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Alpha Return (%)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.5)
    
    # --- Subplot 2: Win Rate ---
    colors_wr = ['darkgreen' if x >= 50 else 'orange' for x in win_rate_values]
    
    axes[1].bar(strategies_labels, win_rate_values, color=colors_wr)
    axes[1].axhline(50, color='red', linestyle='--', linewidth=1, label='Break-Even (50%)')
    
    axes[1].set_title('Win Rate (Affidabilità)')
    axes[1].set_ylabel('Win Rate (%)')
    axes[1].set_ylim(40, 60)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.5)
    
    plt.suptitle(f"Market Heartbeat: Ottimizzazione Finale SL (Dicembre 2024)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    
    print(f"\n📈 Grafico salvato: {save_path}")
    
    # 2. Output testuale finale (CORRETTO)
    
    print("\n" + "="*80)
    print("🏆 RISULTATI GRID SEARCH FINALE (Dicembre 2024)")
    print("================================================================================\n")
    
    print(f"{'STRATEGY (L/SL/TP)':<25} | {'ALPHA':>8} | {'RETURN':>8} | {'WIN RATE':>8} | {'SHARPE':>7} | {'COSTS (K)':>10}")
    print("-" * 90)
    
    for r in df_results.itertuples():
        # ✅ CORREZIONE FINALE: Uso del metodo .format() per evitare il conflitto di sintassi
        status = '🥇' if r.strategy_name == best_robust['strategy_name'] else ' '
        costs_k_str = f"${r.costs/1000:,.1f}K"
        
        output_line = "{:<25} {:<2} | {:>+8.2f} | {:>+8.2f} | {:>8.1f}% | {:>7.2f} | {:>10}".format(
            r.strategy_name, status, r.alpha, r.total_ret, r.win_rate, r.sharpe, costs_k_str
        )
        print(output_line)

    print("\n" + "="*80)
    print(f"✅ BEST STRATEGY FINALE: {best_robust['strategy_name']} (Alpha: {best_robust['alpha']:+.2f}%)")
    
    if best_robust['win_rate'] >= 50.0:
        print("🎉 ROBUSTEZZA CONFERMATA: Win Rate > 50%. Strategia ad alta affidabilità per il Deployment.")
    else:
        print("⚠️ ATTENZIONE: Win Rate < 50%. L'Alpha è generato da grandi vincite. La prudenza è d'obbligo.")
    print("================================================================================")


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    
    # 1. Caricamento Asset e Predizione
    df_test_data, predictions_raw = load_assets_for_backtest()
    
    # 2. Esecuzione della Grid Search
    
    print("\n" + "="*80)
    print("📈 INIZIO GRID SEARCH SL/TP FINALE")
    print("================================================================================\n")
    
    all_results = []
    
    # Entry Threshold fissato al punto di stabilità ottimale
    L_thr_entry = OPTIMAL_ENTRY_THR 
    
    for sl_bps in SL_BPS_TESTS:
        for tp_bps in TP_BPS_TESTS:
            print(f"Testing L{L_thr_entry:.2f} / SL{sl_bps:.0f} / TP{tp_bps:.0f} BPS...")
            
            results = run_backtest_optimized(
                df_test_data, predictions_raw, L_thr_entry, sl_bps, tp_bps
            )
            all_results.append(results)

    # 3. Report Finale (Genera grafico)
    
    bh_return_pct = all_results[0]['bh_return']
    generate_comparative_report(all_results, df_test_data, bh_return_pct)