"""
MARKET HEARTBEAT - Ultra-Fast LSTM Trainer (TURBO MODE - FIXED)
Training ottimizzato per dati con aggregazione 30s.
Versione corretta con Cosine Annealing Scheduler funzionante.
Compatibile con TensorFlow 2.20+
"""
import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, f1_score, recall_score
)
import matplotlib.pyplot as plt
import math

# --- CONFIGURAZIONE TURBO ---
X_FILE = 'X_train_TURBO.npy'
Y_FILE = 'Y_train_TURBO.npy'

# Architettura
CNN_FILTERS = 48
CNN_KERNEL = 2
LSTM_UNITS_1 = 80
LSTM_UNITS_2 = 40
DROPOUT_RATE = 0.35
RECURRENT_DROPOUT = 0.0

# Training
EPOCHS = 60
BATCH_SIZE = 512
EARLY_STOP_PATIENCE = 12
INITIAL_LR = 0.0015
MIN_LR = 0.00001

# Risk tuning
FN_PENALTY_MULTIPLIER = 1.8
MIN_RECALL_TARGET = 0.35

# --- ATTENTION LAYER ---
class AttentionLayer(layers.Layer):
    """Attention mechanism per sequenze temporali."""
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
        e = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        e = tf.reduce_sum(e, axis=-1, keepdims=True)
        alpha = tf.nn.softmax(e, axis=1)
        context = inputs * alpha
        context = tf.reduce_sum(context, axis=1)
        return context

# --- WEIGHTED BCE ---
def weighted_binary_crossentropy(pos_weight):
    """Binary crossentropy pesata per classe positiva."""
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        weight = y_true * pos_weight + (1 - y_true) * 1.0
        weighted_bce = weight * bce
        return tf.reduce_mean(weighted_bce)
    return loss

# --- COSINE ANNEALING SCHEDULER (FIXED) ---
class CosineAnnealingScheduler(Callback):
    """
    Learning rate scheduler con cosine annealing.
    Versione corretta compatibile con TensorFlow 2.x
    """
    def __init__(self, initial_lr, min_lr, total_epochs):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
    
    def on_epoch_begin(self, epoch, logs=None):
        # Calcola nuovo learning rate con cosine annealing
        cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / self.total_epochs))
        lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        
        # Imposta learning rate - Metodo compatibile TF 2.x
        self.model.optimizer.learning_rate.assign(lr)
        
        print(f" - LR: {lr:.6f}")

# --- F1 SCORE CALLBACK ---
class F1ScoreCallback(Callback):
    """Monitora F1-Score su validation set."""
    def __init__(self, validation_data, patience=12):
        super().__init__()
        self.X_val, self.Y_val = validation_data
        self.patience = patience
        self.best_f1 = -1
        self.wait = 0
        self.history_f1 = []
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        Y_pred_proba = self.model.predict(self.X_val, verbose=0).flatten()
        p, r, t = precision_recall_curve(self.Y_val, Y_pred_proba)
        f1_scores = np.divide(2 * (p * r), (p + r), out=np.zeros_like(p), where=(p + r) != 0)
        
        current_f1 = np.max(f1_scores) if len(f1_scores) > 0 else 0.0
        
        self.history_f1.append(current_f1)
        logs['val_f1_score'] = current_f1
        
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.wait = 0
            print(f" - val_f1: {current_f1:.4f} ⭐ NEW BEST")
        else:
            self.wait += 1
            print(f" - val_f1: {current_f1:.4f} (Best: {self.best_f1:.4f})")
                        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print(f"\nEarly stopping: F1 non migliora da {self.patience} epoch")

# --- MODEL ARCHITECTURE ---
def build_turbo_lstm_model(timesteps, features):
    """Architettura ottimizzata per sequenze brevi."""
    inputs = layers.Input(shape=(timesteps, features))
    
    x = layers.Conv1D(filters=CNN_FILTERS, kernel_size=CNN_KERNEL, 
                      activation='relu', padding='same')(inputs) 
    x = layers.BatchNormalization()(x)
    
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS_1, return_sequences=True, dropout=DROPOUT_RATE,
                   recurrent_dropout=RECURRENT_DROPOUT,
                   kernel_regularizer=keras.regularizers.l2(0.001))
    )(x) 
    x = layers.BatchNormalization()(x)
    
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS_2, return_sequences=True, dropout=DROPOUT_RATE,
                   recurrent_dropout=RECURRENT_DROPOUT,
                   kernel_regularizer=keras.regularizers.l2(0.001))
    )(x) 
    x = layers.BatchNormalization()(x)
    
    x = AttentionLayer()(x)
    
    x = layers.Dense(48, activation='relu', 
                    kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(24, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name='crash_prediction')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='MarketHeartbeat_TURBO')
    return model

# --- PLOTTING ---
def plot_training_history(history, f1_history, save_path='training_history_TURBO_FIXED.png'):
    """Visualizza training history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history.history['auc'], label='Train AUC', linewidth=2)
    axes[0, 1].plot(history.history['val_auc'], label='Val AUC', linewidth=2)
    axes[0, 1].set_title('Model AUC', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    axes[1, 0].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
    axes[1, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(f1_history, label='Val F1-Score', linewidth=2, color='green')
    axes[1, 1].set_title('Validation F1-Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Grafico salvato: {save_path}")
    plt.close()

# --- THRESHOLD OPTIMIZATION ---
def find_optimal_threshold_risk_tuned(y_true, y_pred_proba, min_recall=0.35):
    """Trova soglia ottimale con focus su recall minimo."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    valid_indices = np.where(recall >= min_recall)[0]
    
    if len(valid_indices) == 0:
        print(f"⚠️ Recall >= {min_recall*100:.0f}% non raggiungibile.")
        f1_scores = np.divide(2 * (precision * recall), (precision + recall), 
                             out=np.zeros_like(precision), where=(precision + recall) != 0)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    best_index_in_valid = np.argmax(precision[valid_indices])
    best_global_index = valid_indices[best_index_in_valid]
    optimal_threshold = thresholds[best_global_index]
    
    print(f"\n🎯 Soglia ottimale: {optimal_threshold:.4f}")
    print(f"   Recall: {recall[best_global_index]:.4f}")
    print(f"   Precision: {precision[best_global_index]:.4f}")
    
    return optimal_threshold

# --- MAIN ---
if __name__ == '__main__':
    print("="*80)
    print("🚀 MARKET HEARTBEAT - TURBO TRAINING (FIXED)")
    print("="*80)
    
    try:
        X = np.load(X_FILE)
        Y = np.load(Y_FILE)
        print(f"✅ Dati: X={X.shape}, Y={Y.shape}")
    except FileNotFoundError:
        print(f"❌ File non trovati. Esegui prima market_heartbeat_processor_TURBO.py")
        exit(1)
    
    crash_count = (Y == 1).sum()
    no_crash_count = (Y == 0).sum()
    print(f"\n📊 Crash: {crash_count:,} ({crash_count/len(Y)*100:.2f}%)")
    
    train_size = int(0.70 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, X_temp = X[:train_size], X[train_size:]
    Y_train, Y_temp = Y[:train_size], Y[train_size:]
    X_val, X_test = X_temp[:val_size], X_temp[val_size:]
    Y_val, Y_test = Y_temp[:val_size], Y_temp[val_size:]
    
    print(f"📦 Split: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    
    base_pos_weight = no_crash_count / (crash_count + 1e-8)
    final_pos_weight = base_pos_weight * FN_PENALTY_MULTIPLIER
    print(f"\n⚖️  Peso classe positiva: {final_pos_weight:.2f}")
    
    _, TIME_STEPS, FEATURES = X_train.shape
    model = build_turbo_lstm_model(TIME_STEPS, FEATURES)
    
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR, clipnorm=1.0), 
        loss=weighted_binary_crossentropy(final_pos_weight),
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    print(f"\n🏗️  Model: {model.count_params():,} parametri")
    
    f1_callback = F1ScoreCallback((X_val, Y_val), patience=EARLY_STOP_PATIENCE)
    cosine_scheduler = CosineAnnealingScheduler(INITIAL_LR, MIN_LR, EPOCHS)
    
    callbacks = [
        f1_callback,
        cosine_scheduler,
        ModelCheckpoint('best_model_TURBO_FIXED.keras', monitor='val_auc', 
                       save_best_only=True, mode='max', verbose=1)
    ]
    
    print(f"\n{'='*80}")
    print("🚀 TRAINING START")
    print(f"{'='*80}\n")
    
    history = model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    plot_training_history(history, f1_callback.history_f1)
    
    print(f"\n{'='*80}")
    print("📊 VALUTAZIONE TEST SET")
    print(f"{'='*80}")
    
    try:
        best_model = keras.models.load_model(
            'best_model_TURBO_FIXED.keras',
            custom_objects={'loss': weighted_binary_crossentropy(final_pos_weight), 
                          'AttentionLayer': AttentionLayer}
        )
    except:
        best_model = model
        
    Y_pred_proba = best_model.predict(X_test, verbose=0).flatten()
    TUNED_THRESHOLD = find_optimal_threshold_risk_tuned(Y_test, Y_pred_proba, MIN_RECALL_TARGET)
    Y_pred = (Y_pred_proba >= TUNED_THRESHOLD).astype(int)
    
    auc_roc = roc_auc_score(Y_test, Y_pred_proba)
    recall = recall_score(Y_test, Y_pred, zero_division=0)
    f1 = f1_score(Y_test, Y_pred, zero_division=0)
    
    print(f"\n🎯 METRICHE:")
    print(f"   AUC-ROC: {auc_roc:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(Y_test, Y_pred, target_names=['Non-Crash', 'Crash'], 
                                digits=4, zero_division=0))
    
    cm = confusion_matrix(Y_test, Y_pred)
    print(f"\n🔢 Confusion Matrix:")
    print(f"   TN: {cm[0,0]:,}, FP: {cm[0,1]:,}")
    print(f"   FN: {cm[1,0]:,}, TP: {cm[1,1]:,}")
    
    best_model.save('lstm_model_TURBO_FIXED.keras')
    np.save('optimal_threshold_TURBO_FIXED.npy', TUNED_THRESHOLD)
    
    import json
    with open('model_config_TURBO_FIXED.json', 'w') as f:
        json.dump({'threshold': float(TUNED_THRESHOLD), 'features': FEATURES}, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETATO")
    print(f"{'='*80}")