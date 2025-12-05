"""
MARKET HEARTBEAT - Trainer 2024 (ATTUALE)
Training su Jan-Nov 2024 per trading Dic 2024 / Jan 2025
"""
import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import json

# --- CONFIG ---
X_FILE = 'X_train_DIRECTIONAL_2024.npy'
Y_FILE = 'Y_train_DIRECTIONAL_2024.npy'

CNN_FILTERS = 64
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DROPOUT = 0.3

EPOCHS = 50
BATCH_SIZE = 1024
INITIAL_LR = 0.001

TARGET_ACCURACY = 0.63  # Target realistico per 2024

# --- ATTENTION ---
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(layers.Layer):
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

# --- METRICS CALLBACK ---
class DirectionalMetricsCallback(Callback):
    def __init__(self, val_data):
        super().__init__()
        self.X_val, self.Y_val = val_data
        self.best_accuracy = 0
        self.history = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    def on_epoch_end(self, epoch, logs=None):
        y_pred_proba = self.model.predict(self.X_val, verbose=0).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        accuracy = accuracy_score(self.Y_val, y_pred)
        precision = precision_score(self.Y_val, y_pred, zero_division=0)
        recall = recall_score(self.Y_val, y_pred, zero_division=0)
        f1 = f1_score(self.Y_val, y_pred, zero_division=0)
        
        self.history['accuracy'].append(accuracy)
        self.history['precision'].append(precision)
        self.history['recall'].append(recall)
        self.history['f1'].append(f1)
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.model.save('best_model_2024.keras')
            status = "⭐ SAVED"
        else:
            status = ""
        
        profit_status = "✅" if accuracy >= TARGET_ACCURACY else "⚠️"
        
        print(f"\n   📊 Acc:{accuracy:.3f}{profit_status} P:{precision:.3f} R:{recall:.3f} F1:{f1:.3f} {status}")

# --- MODEL ---
def build_directional_model(timesteps, features):
    """Modello ottimizzato per 2024."""
    inp = layers.Input((timesteps, features))
    
    # CNN
    x = layers.Conv1D(CNN_FILTERS, 3, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(CNN_FILTERS, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # BiLSTM
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS_1, return_sequences=True, dropout=DROPOUT)
    )(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS_2, return_sequences=True, dropout=DROPOUT)
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Attention
    x = AttentionLayer()(x)
    
    # Dense
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    out = layers.Dense(1, activation='sigmoid', name='direction')(x)
    
    return Model(inp, out, name='MarketHeartbeat_2024')

# --- PLOTTING ---
def plot_training(history, callback, save_path='training_2024.png'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss
    axes[0,0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0,0].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[0,0].set_title('Loss', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # AUC
    axes[0,1].plot(history.history['auc'], label='Train', linewidth=2)
    axes[0,1].plot(history.history['val_auc'], label='Val', linewidth=2)
    axes[0,1].set_title('AUC', fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Built-in Accuracy
    axes[0,2].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0,2].plot(history.history['val_accuracy'], label='Val', linewidth=2)
    axes[0,2].axhline(TARGET_ACCURACY, color='red', linestyle='--', label=f'Target {TARGET_ACCURACY}')
    axes[0,2].set_title('Accuracy (Built-in)', fontweight='bold')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Callback Accuracy
    axes[1,0].plot(callback.history['accuracy'], linewidth=2, color='blue')
    axes[1,0].axhline(TARGET_ACCURACY, color='red', linestyle='--', label=f'Profit Threshold')
    axes[1,0].axhline(0.5, color='gray', linestyle=':', label='Random')
    axes[1,0].set_title('Validation Accuracy (Callback)', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Precision & Recall
    axes[1,1].plot(callback.history['precision'], linewidth=2, color='blue', label='Precision')
    axes[1,1].plot(callback.history['recall'], linewidth=2, color='orange', label='Recall')
    axes[1,1].set_title('Precision & Recall', fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # F1-Score
    axes[1,2].plot(callback.history['f1'], linewidth=2, color='green')
    axes[1,2].set_title('F1-Score', fontweight='bold')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Plot: {save_path}")
    plt.close()

# --- MAIN ---
if __name__ == '__main__':
    print("="*80)
    print("🎯 MARKET HEARTBEAT - TRAINER 2024")
    print("   Dataset: Gennaio-Novembre 2024 (11 mesi)")
    print("   Target: Trading Dicembre 2024 / Gennaio 2025")
    print("="*80)
    
    # Load
    X = np.load(X_FILE)
    Y = np.load(Y_FILE)
    print(f"\n✅ X={X.shape}, Y={Y.shape}")
    print(f"   Features: {X.shape[2]} (microstructure only)")
    print(f"   Dataset: 2024 (current market)")
    
    # Balance check
    up_count = (Y==1).sum()
    down_count = (Y==0).sum()
    print(f"\n📊 Target Distribution:")
    print(f"   UP: {up_count:,} ({up_count/len(Y)*100:.2f}%)")
    print(f"   DOWN: {down_count:,} ({down_count/len(Y)*100:.2f}%)")
    print(f"   Balance Ratio: {min(up_count,down_count)/max(up_count,down_count):.2%}")
    
    # Split
    n_train = int(0.70 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train, X_temp = X[:n_train], X[n_train:]
    Y_train, Y_temp = Y[:n_train], Y[n_train:]
    X_val, X_test = X_temp[:n_val], X_temp[n_val:]
    Y_val, Y_test = Y_temp[:n_val], Y_temp[n_val:]
    
    print(f"\n📦 Train={len(X_train):,} Val={len(X_val):,} Test={len(X_test):,}")
    
    # Build
    model = build_directional_model(X_train.shape[1], X_train.shape[2])
    
    # Class weights
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(Y_train),
        y=Y_train
    )
    class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}
    
    print(f"   Class weights: DOWN={class_weights[0]:.3f}, UP={class_weights[1]:.3f}")
    
    model.compile(
        optimizer=Adam(INITIAL_LR),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')]
    )
    
    print(f"\n🏗️ Params: {model.count_params():,}")
    print(f"🎯 Target Accuracy: ≥{TARGET_ACCURACY*100:.0f}% (profitable)")
    
    # Callbacks
    dir_callback = DirectionalMetricsCallback((X_val, Y_val))
    early_stop = EarlyStopping(monitor='val_auc', patience=15, 
                               restore_best_weights=True, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, 
                                  patience=5, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint('lstm_model_2024.keras', 
                                monitor='val_accuracy', save_best_only=True, 
                                mode='max', verbose=0)
    
    print(f"\n{'='*80}")
    print("🚀 TRAINING START - 2024 DATASET")
    print(f"{'='*80}\n")
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[dir_callback, early_stop, reduce_lr, checkpoint],
        verbose=1
    )
    
    plot_training(history, dir_callback)
    
    # Test evaluation
    print(f"\n{'='*80}")
    print("📊 TEST SET EVALUATION")
    print(f"{'='*80}")
    
    try:
        model = keras.models.load_model('best_model_2024.keras',
                                       custom_objects={'AttentionLayer': AttentionLayer})
        print("✅ Best model loaded")
    except:
        print("⚠️ Using final model")
    
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metriche
    accuracy = accuracy_score(Y_test, y_pred)
    auc = roc_auc_score(Y_test, y_pred_proba)
    precision = precision_score(Y_test, y_pred, zero_division=0)
    recall = recall_score(Y_test, y_pred, zero_division=0)
    f1 = f1_score(Y_test, y_pred, zero_division=0)
    
    print(f"\n🎯 FINAL METRICS:")
    print(f"   Accuracy: {accuracy:.4f} {'✅ PROFITABLE!' if accuracy >= TARGET_ACCURACY else '⚠️ Below target'}")
    print(f"   AUC: {auc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1: {f1:.4f}")
    
    cm = confusion_matrix(Y_test, y_pred)
    
    print(f"\n📋 Confusion Matrix:")
    print(f"   TN={cm[0,0]:,} FP={cm[0,1]:,}")
    print(f"   FN={cm[1,0]:,} TP={cm[1,1]:,}")
    
    print(f"\n📄 Classification Report:")
    print(classification_report(Y_test, y_pred, target_names=['DOWN','UP'], digits=4))
    
    # Expected performance
    if accuracy > 0.55:
        edge = (accuracy - 0.5) * 2
        expected_annual_return = edge * 100 * 252
        print(f"\n💰 TRADING POTENTIAL:")
        print(f"   Edge over random: {edge*100:.2f}%")
        print(f"   Expected annual return: {expected_annual_return:.1f}%")
        print(f"   (Assuming 1 trade/day, no costs)")
    
    # Save
    print("\n💾 Saving...")
    X_full = np.load(X_FILE)
    predictions_full = model.predict(X_full, batch_size=2048, verbose=1).flatten()
    np.save('predictions_LSTM_2024.npy', predictions_full)
    np.save('optimal_threshold_2024.npy', 0.5)
    model.save('lstm_model_2024_FINAL.keras')
    
    with open('model_config_2024.json', 'w') as f:
        json.dump({
            'threshold': 0.5,
            'accuracy': float(accuracy),
            'auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'profitable': bool(accuracy >= TARGET_ACCURACY),
            'dataset': '2024_jan_nov',
            'target_period': 'dec_2024_jan_2025'
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ TRAINING 2024 COMPLETE")
    print(f"{'='*80}")
    print(f"\n💡 Next steps:")
    print(f"   1. Download BTCUSDT-trades-2024-12.zip")
    print(f"   2. Process Dicembre 2024")
    print(f"   3. Valida con: python validate_dec_2024.py")
    print(f"   4. Se alpha > 0% → Deploy Gennaio 2025!")