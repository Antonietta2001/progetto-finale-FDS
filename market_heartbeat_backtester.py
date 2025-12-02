"""
MARKET HEARTBEAT - Backtesting Engine (Simplified)
Versione semplificata senza import TensorFlow pesanti.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path

# --- CONFIGURAZIONE ---
PROCESSED_DATA_FILE = 'processed_data_TURBO.parquet'
MODEL_FILE = 'best_model_TURBO_FIXED.keras'
THRESHOLD_FILE = 'optimal_threshold_TURBO_FIXED.npy'

# Parametri economici
INITIAL_CAPITAL = 100000
POSITION_SIZE = 0.95
TRANSACTION_FEE_BPS = 7.5
SLIPPAGE_BPS = 5

# Threshold configurations
THRESHOLD_CONFIGS = {
    'conservative': 0.95,
    'balanced': 0.88,
    'aggressive': 0.75,
    'very_aggressive': 0.60
}

# --- CARICAMENTO DATI ---
def load_backtest_data():
    """Carica dati processati."""
    print("📂 Caricamento dati...")
    
    df = pd.read_parquet(PROCESSED_DATA_FILE)
    print(f"   ✓ Dataset: {len(df):,} samples")
    
    # Threshold ottimale
    try:
        optimal_threshold = np.load(THRESHOLD_FILE)
        print(f"   ✓ Threshold ottimale: {optimal_threshold:.4f}")
    except:
        optimal_threshold = 0.88
        print(f"   ⚠️ Threshold non trovato, uso default: {optimal_threshold:.4f}")
    
    return df, optimal_threshold

# --- PREDIZIONI ---
def load_predictions():
    """
    Carica o genera predizioni.
    Se il modello non si carica, usa predizioni dummy basate su features.
    """
    print("\n🤖 Caricamento predizioni...")
    
    # Prova a caricare modello TensorFlow
    try:
        from tensorflow import keras
        
        # Custom objects per caricare il modello
        class AttentionLayer(keras.layers.Layer):
            def __init__(self, **kwargs):
                super(AttentionLayer, self).__init__(**kwargs)
            
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
                super(AttentionLayer, self).build(input_shape)
            
            def call(self, inputs):
                import tensorflow as tf
                e = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
                e = tf.reduce_sum(e, axis=-1, keepdims=True)
                alpha = tf.nn.softmax(e, axis=1)
                context = inputs * alpha
                context = tf.reduce_sum(context, axis=1)
                return context
        
        def weighted_binary_crossentropy(pos_weight):
            def loss(y_true, y_pred):
                import tensorflow as tf
                epsilon = tf.keras.backend.epsilon()
                y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
                bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
                weight = y_true * pos_weight + (1 - y_true) * 1.0
                weighted_bce = weight * bce
                return tf.reduce_mean(weighted_bce)
            return loss
        
        model = keras.models.load_model(
            MODEL_FILE,
            custom_objects={
                'AttentionLayer': AttentionLayer,
                'loss': weighted_binary_crossentropy(160)
            }
        )
        print("   ✓ Modello TensorFlow caricato")
        return model, 'tensorflow'
        
    except Exception as e:
        print(f"   ⚠️ Impossibile caricare modello TensorFlow: {e}")
        print("   → Uso predizioni basate su rule-based system")
        return None, 'rule_based'

def generate_predictions(df, model, model_type):
    """Genera predizioni dal modello o con regole."""
    
    # 🎯 PRIORITÀ 1: Carica predizioni salvate da file
    try:
        print("   🔍 Tentativo caricamento predizioni LSTM salvate...")
        predictions_full = np.load('predictions_LSTM_TURBO.npy')
        print(f"   ✅ Predizioni LSTM caricate da file! ({len(predictions_full):,} samples)")
        
        # Verifica compatibilità lunghezza
        if len(predictions_full) >= len(df):
            return predictions_full[:len(df)]
        else:
            print(f"   ⚠️ Lunghezza incompatibile: {len(predictions_full)} vs {len(df)} richiesti")
    except FileNotFoundError:
        print(f"   ⚠️ File 'predictions_LSTM_TURBO.npy' non trovato")
        print(f"   → Esegui prima: python save_predictions.py")
    except Exception as e:
        print(f"   ⚠️ Errore caricamento: {e}")
    
    # PRIORITÀ 2: Usa modello TensorFlow se disponibile
    if model_type == 'tensorflow' and model is not None:
        # Usa modello LSTM
        print("   Generazione predizioni con LSTM...")
        
        # Prepara sequenze
        X = prepare_sequences(df)
        predictions = model.predict(X, batch_size=1024, verbose=0).flatten()
        
        # Allinea con dataframe
        sequence_length = 3
        predictions_aligned = np.concatenate([
            np.zeros(sequence_length - 1),  # Primi timesteps senza predizione
            predictions
        ])
        
        return predictions_aligned[:len(df)]
    
    # PRIORITÀ 3: Fallback a rule-based system
    else:
        # Sistema rule-based
        print("   ⚠️ Generazione predizioni con rule-based system (meno accurato)...")
        
        # Logica semplice: crash se OFI molto negativo + volatility spike
        crash_score = (
            (df['OFI_Taker'] < df['OFI_Taker'].quantile(0.05)).astype(float) * 0.4 +
            (df['volatility_ratio'] > df['volatility_ratio'].quantile(0.95)).astype(float) * 0.3 +
            (df['Spread_BPS'] > df['Spread_BPS'].quantile(0.90)).astype(float) * 0.3
        )
        
        return crash_score.values

def prepare_sequences(df):
    """Prepara sequenze per LSTM."""
    feature_cols = [
        'OFI_Taker', 'OFI_velocity', 'OFI_cumsum_60s', 'VWAP_Deviation', 
        'Taker_Maker_Ratio', 'Hawkes_Intensity', 'Spread_BPS',
        'volatility_30s', 'volatility_60s', 'volatility_ratio',
        'price_skew_60s', 'price_kurtosis_60s', 'return_60s', 'price_std',
        'volume', 'trade_count', 'Fear_Greed', 'Twitter_Sentiment' 
    ]
    
    available_features = [f for f in feature_cols if f in df.columns]
    X = df[available_features].values
    
    # Normalizzazione
    try:
        MEAN = np.load('normalization_mean_TURBO.npy')
        STD = np.load('normalization_std_TURBO.npy')
        X_normalized = (X - MEAN) / (STD + 1e-8)
    except:
        MEAN = X.mean(axis=0)
        STD = X.std(axis=0)
        X_normalized = (X - MEAN) / (STD + 1e-8)
    
    # Sequenze
    sequence_length = 3
    X_sequences = []
    
    for i in range(len(X_normalized) - sequence_length + 1):
        X_sequences.append(X_normalized[i:i + sequence_length])
    
    return np.array(X_sequences)

# --- TRADING STRATEGY ---
class TradingStrategy:
    """Strategia di trading basata su crash signals."""
    
    def __init__(self, initial_capital, position_size, tx_fee_bps, slippage_bps):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.tx_fee_bps = tx_fee_bps
        self.slippage_bps = slippage_bps
        
        self.capital = initial_capital
        self.position = 0
        self.in_position = False
        
        self.trades = []
        self.equity_curve = []
        
    def calculate_costs(self, trade_value):
        """Calcola costi transazione."""
        tx_cost = trade_value * (self.tx_fee_bps / 10000)
        slippage_cost = trade_value * (self.slippage_bps / 10000)
        return tx_cost + slippage_cost
    
    def open_position(self, price, timestamp):
        """Apre posizione long."""
        if self.in_position:
            return
        
        trade_capital = self.capital * self.position_size
        costs = self.calculate_costs(trade_capital)
        net_capital = trade_capital - costs
        
        self.position = net_capital / price
        self.in_position = True
        
        self.trades.append({
            'timestamp': timestamp,
            'action': 'BUY',
            'price': price,
            'quantity': self.position,
            'value': trade_capital,
            'costs': costs,
            'capital': self.capital
        })
    
    def close_position(self, price, timestamp, reason='SIGNAL'):
        """Chiude posizione."""
        if not self.in_position:
            return
        
        trade_value = self.position * price
        costs = self.calculate_costs(trade_value)
        net_value = trade_value - costs
        
        self.capital = self.capital * (1 - self.position_size) + net_value
        
        self.trades.append({
            'timestamp': timestamp,
            'action': 'SELL',
            'price': price,
            'quantity': self.position,
            'value': trade_value,
            'costs': costs,
            'capital': self.capital,
            'reason': reason
        })
        
        self.position = 0
        self.in_position = False
    
    def update_equity(self, price, timestamp):
        """Aggiorna equity."""
        if self.in_position:
            current_value = self.position * price
            total_equity = self.capital * (1 - self.position_size) + current_value
        else:
            total_equity = self.capital
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'in_position': self.in_position
        })

# --- BACKTESTING ---
def run_backtest(df, predictions, threshold):
    """Esegue backtest."""
    print(f"\n🔄 Backtest threshold={threshold:.4f}")
    
    strategy = TradingStrategy(INITIAL_CAPITAL, POSITION_SIZE, TRANSACTION_FEE_BPS, SLIPPAGE_BPS)
    
    # Inizia in posizione
    strategy.open_position(df.iloc[0]['price'], df.iloc[0]['timestamp'])
    
    for idx in range(len(predictions)):
        price = df.iloc[idx]['price']
        timestamp = df.iloc[idx]['timestamp']
        crash_signal = predictions[idx] >= threshold
        
        if crash_signal and strategy.in_position:
            strategy.close_position(price, timestamp, reason='CRASH_SIGNAL')
        elif not crash_signal and not strategy.in_position:
            strategy.open_position(price, timestamp)
        
        strategy.update_equity(price, timestamp)
    
    if strategy.in_position:
        strategy.close_position(df.iloc[-1]['price'], df.iloc[-1]['timestamp'], reason='END')
    
    return strategy

# --- ANALISI PERFORMANCE ---
def analyze_performance(strategy, df, threshold_name):
    """Calcola metriche."""
    
    equity_df = pd.DataFrame(strategy.equity_curve)
    trades_df = pd.DataFrame(strategy.trades)
    
    final_capital = strategy.capital
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    buy_hold_return = (df.iloc[-1]['price'] / df.iloc[0]['price']) - 1
    buy_hold_final = INITIAL_CAPITAL * (1 + buy_hold_return)
    
    equity_df['returns'] = equity_df['equity'].pct_change()
    sharpe = (equity_df['returns'].mean() / equity_df['returns'].std()) * np.sqrt(len(equity_df)) if equity_df['returns'].std() > 0 else 0
    
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
    max_drawdown = equity_df['drawdown'].min()
    
    num_trades = len(trades_df)
    total_costs = trades_df['costs'].sum()
    
    results = {
        'threshold_name': threshold_name,
        'final_capital': final_capital,
        'total_return_pct': total_return * 100,
        'buy_hold_return_pct': buy_hold_return * 100,
        'alpha_pct': (total_return - buy_hold_return) * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown * 100,
        'num_trades': num_trades,
        'total_costs': total_costs,
        'buy_hold_final': buy_hold_final
    }
    
    return results, equity_df, trades_df

# --- VISUALIZZAZIONI ---
def plot_results(all_results, all_equity_curves, df):
    """Crea grafici."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Equity curves
    ax1 = axes[0, 0]
    for threshold_name, equity_df in all_equity_curves.items():
        ax1.plot(equity_df['timestamp'], equity_df['equity'], label=threshold_name, linewidth=2, alpha=0.8)
    
    buy_hold_equity = INITIAL_CAPITAL * (df['price'] / df['price'].iloc[0])
    ax1.plot(df['timestamp'].iloc[:len(buy_hold_equity)], buy_hold_equity, 
             label='Buy & Hold', linestyle='--', linewidth=2, color='black', alpha=0.6)
    
    ax1.set_title('Equity Curves Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.5)
    
    # Returns
    ax2 = axes[0, 1]
    threshold_names = [r['threshold_name'] for r in all_results]
    returns = [r['total_return_pct'] for r in all_results]
    colors = ['green' if r > 0 else 'red' for r in returns]
    
    ax2.barh(threshold_names, returns, color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(x=all_results[0]['buy_hold_return_pct'], color='orange', linestyle='--', linewidth=2, label='Buy & Hold')
    ax2.set_xlabel('Total Return (%)')
    ax2.set_title('Strategy Returns', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Risk metrics
    ax3 = axes[1, 0]
    sharpe_ratios = [r['sharpe_ratio'] for r in all_results]
    max_drawdowns = [abs(r['max_drawdown_pct']) for r in all_results]
    
    x = np.arange(len(threshold_names))
    width = 0.35
    
    ax3.bar(x - width/2, sharpe_ratios, width, label='Sharpe Ratio', alpha=0.8)
    ax3_twin = ax3.twinx()
    ax3_twin.bar(x + width/2, max_drawdowns, width, label='Max Drawdown (%)', alpha=0.8, color='red')
    
    ax3.set_xlabel('Strategy')
    ax3.set_ylabel('Sharpe Ratio', color='blue')
    ax3_twin.set_ylabel('Max Drawdown (%)', color='red')
    ax3.set_title('Risk Metrics', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(threshold_names, rotation=45, ha='right')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Costs
    ax4 = axes[1, 1]
    total_costs = [r['total_costs'] for r in all_results]
    
    ax4.bar(threshold_names, total_costs, alpha=0.7, color='coral')
    ax4.set_xlabel('Strategy')
    ax4.set_ylabel('Total Costs ($)')
    ax4.set_title('Transaction Costs', fontsize=12, fontweight='bold')
    ax4.set_xticklabels(threshold_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Market Heartbeat - Backtesting Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('backtest_analysis_SIMPLE.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Grafici salvati: backtest_analysis_SIMPLE.png")
    plt.close()

# --- REPORT ---
def generate_report(all_results):
    """Genera report testuale."""
    
    report = []
    report.append("="*80)
    report.append("MARKET HEARTBEAT - BACKTESTING REPORT")
    report.append("="*80)
    report.append(f"\nInitial Capital: ${INITIAL_CAPITAL:,.2f}")
    report.append("\n" + "="*80)
    
    best_strategy = max(all_results, key=lambda x: x['total_return_pct'])
    
    report.append(f"\n🏆 BEST STRATEGY: {best_strategy['threshold_name'].upper()}")
    report.append(f"   Final Capital: ${best_strategy['final_capital']:,.2f}")
    report.append(f"   Total Return: {best_strategy['total_return_pct']:.2f}%")
    report.append(f"   Alpha vs B&H: {best_strategy['alpha_pct']:.2f}%")
    
    report.append("\n" + "-"*80)
    report.append("\nDETAILED COMPARISON:")
    report.append("-"*80)
    
    for result in all_results:
        report.append(f"\n📊 {result['threshold_name'].upper()}")
        report.append(f"   Final Capital:    ${result['final_capital']:,.2f}")
        report.append(f"   Total Return:     {result['total_return_pct']:+.2f}%")
        report.append(f"   Buy & Hold:       {result['buy_hold_return_pct']:+.2f}%")
        report.append(f"   Alpha:            {result['alpha_pct']:+.2f}%")
        report.append(f"   Sharpe Ratio:     {result['sharpe_ratio']:.2f}")
        report.append(f"   Max Drawdown:     {result['max_drawdown_pct']:.2f}%")
        report.append(f"   Number of Trades: {result['num_trades']}")
        report.append(f"   Total Costs:      ${result['total_costs']:,.2f}")
    
    report.append("\n" + "="*80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open('backtest_report_SIMPLE.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n💾 Report salvato: backtest_report_SIMPLE.txt")

# --- MAIN ---
if __name__ == '__main__':
    print("="*80)
    print("🫀 MARKET HEARTBEAT - BACKTESTING ENGINE (SIMPLE)")
    print("="*80)
    
    try:
        # Carica dati
        df, optimal_threshold = load_backtest_data()
        
        # Carica modello o usa rule-based
        model, model_type = load_predictions()
        
        # Genera predizioni
        predictions = generate_predictions(df, model, model_type)
        print(f"   ✓ {len(predictions):,} predizioni generate")
        
        # Split out-of-sample (ultimi 30%)
        split_idx = int(len(df) * 0.70)
        df_test = df.iloc[split_idx:].reset_index(drop=True)
        predictions_test = predictions[split_idx:]
        
        print(f"\n📊 Test Set: {len(df_test):,} samples")
        print(f"   Periodo: {df_test['timestamp'].min()} → {df_test['timestamp'].max()}")
        
        # Backtesting
        all_results = []
        all_equity_curves = {}
        
        THRESHOLD_CONFIGS['optimal'] = float(optimal_threshold)
        
        for name, threshold in THRESHOLD_CONFIGS.items():
            strategy = run_backtest(df_test, predictions_test, threshold)
            results, equity_df, trades_df = analyze_performance(strategy, df_test, name)
            
            all_results.append(results)
            all_equity_curves[name] = equity_df
            
            print(f"   ✓ {name}: Return={results['total_return_pct']:.2f}%, Trades={results['num_trades']}")
        
        # Visualizzazioni
        print("\n📊 Generazione visualizzazioni...")
        plot_results(all_results, all_equity_curves, df_test)
        
        # Report
        print("\n📄 Generazione report...")
        generate_report(all_results)
        
        print("\n" + "="*80)
        print("✅ BACKTESTING COMPLETATO")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()