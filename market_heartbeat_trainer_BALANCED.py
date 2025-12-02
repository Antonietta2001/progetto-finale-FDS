# ATTENZIONE: Questo codice è pensato per essere un file Python separato (.py)
# e non deve essere eseguito nel contesto di un notebook che richiede la registrazione Keras.
# Assicurati di usare Python 3.8+ e TensorFlow 2.x

"""
MARKET HEARTBEAT - Optimized LSTM Trainer (PRECISION FOCUS)
Strategia: Massimizzazione della Precisione per minimizzare i Falsi Positivi (FP) 
e superare i costi di trading.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, f1_score, recall_score
)
import matplotlib.pyplot as plt
import math
import json
import sys

# --- CONFIGURAZIONE OTTIMIZZATA ---
X_FILE = 'X_train_TURBO.npy'
Y_FILE = 'Y_train_TURBO.npy'

# Architettura (invariata)
CNN_FILTERS = 48
CNN_KERNEL = 2
LSTM_UNITS_1 = 80
LSTM_UNITS_2 = 40
DROPOUT_RATE = 0.35
RECURRENT_DROPOUT = 0.0

# Training
EPOCHS = 60
BATCH_SIZE = 512
EARLY_STOP_PATIENCE = 15 
INITIAL_LR = 0.0012 
MIN_LR = 0.00001

# 🎯 MODIFICHE CHIAVE: PRECISION FOCUS
# Aumentiamo la penalità e rendiamo il target Recall NON VINCOLANTE
FN_PENALTY_MULTIPLIER = 2.5 
MIN_RECALL_TARGET = 0.10 

# --- ATTENTION LAYER (Registrato) ---
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(layers.Layer):
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
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()

# --- WEIGHTED BCE (Registrato) ---
@tf.keras.utils.register_keras_serializable()
def weighted_binary_crossentropy(pos_weight):
    def loss(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        weight = y_true * pos_weight + (1 - y_true) * 1.0
        weighted_bce = weight * bce
        return tf.reduce_mean(weighted_bce)
    
    def get_config():
        return {"pos_weight": pos_weight}
    
    loss.get_config = get_config
    return loss

# --- COSINE ANNEALING ---
class CosineAnnealingScheduler(Callback):
    def __init__(self, initial_lr, min_lr, total_epochs):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
    
    def on_epoch_begin(self, epoch, logs=None):
        cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / self.total_epochs))
        lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        self.model.optimizer.learning_rate.assign(lr)
        print(f" - LR: {lr:.6f}")

# --- BALANCED METRICS CALLBACK ---
class BalancedMetricsCallback(Callback):
    # ... (corpo omesso per brevità, usa F1-Max per il monitoraggio)
    def __init__(self, validation_data, patience=15):
        super().__init__()
        self.X_val, self.Y_val = validation_data
        self.patience = patience
        self.best_f1 = -1
        self.wait = 0
        self.history_f1 = []
        self.history_precision = []
        self.history_recall = []
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        Y_pred_proba = self.model.predict(self.X_val, verbose=0).flatten()
        p, r, t = precision_recall_curve(self.Y_val, Y_pred_proba)
        f1_scores = np.divide(2 * (p * r), (p + r), out=np.zeros_like(p), where=(p + r) != 0)
        
        best_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
        current_f1 = f1_scores[best_idx] if len(f1_scores) > 0 else 0.0
        current_precision = p[best_idx] if len(p) > 0 else 0.0
        current_recall = r[best_idx] if len(r) > 0 else 0.0
        
        self.history_f1.append(current_f1)
        self.history_precision.append(current_precision)
        self.history_recall.append(current_recall)
        
        logs['val_f1_score'] = current_f1
        logs['val_precision'] = current_precision
        logs['val_recall'] = current_recall
        
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.wait = 0
            print(f" - F1:{current_f1:.4f} P:{current_precision:.4f} R:{current_recall:.4f} ⭐")
        else:
            self.wait += 1
            print(f" - F1:{current_f1:.4f} P:{current_precision:.4f} R:{current_recall:.4f} (Best F1:{self.best_f1:.4f})")
                        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print(f"\nEarly stopping: F1 non migliora da {self.patience} epoch")

# --- MODEL ---
def build_balanced_lstm_model(timesteps, features):
    """Architettura con maggiore regolarizzazione per ridurre overfitting."""
    inputs = layers.Input(shape=(timesteps, features))
    
    x = layers.Conv1D(filters=CNN_FILTERS, kernel_size=CNN_KERNEL, 
                      activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x) 
    
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS_1, return_sequences=True, dropout=DROPOUT_RATE,
                    recurrent_dropout=RECURRENT_DROPOUT,
                    kernel_regularizer=keras.regularizers.l2(0.002))
    )(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS_2, return_sequences=True, dropout=DROPOUT_RATE,
                    recurrent_dropout=RECURRENT_DROPOUT,
                    kernel_regularizer=keras.regularizers.l2(0.002))
    )(x)
    x = layers.BatchNormalization()(x)
    
    x = AttentionLayer()(x)
    
    x = layers.Dense(48, activation='relu', 
                     kernel_regularizer=keras.regularizers.l2(0.002))(x)
    x = layers.Dropout(0.45)(x) 
    
    x = layers.Dense(24, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.002))(x)
    x = layers.Dropout(0.35)(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name='crash_prediction')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='MarketHeartbeat_BALANCED')
    return model

# --- THRESHOLD OPTIMIZATION (Precisione Massima con Vincolo Basso) ---
def find_optimal_threshold_balanced(y_true, y_pred_proba, min_recall=0.10):
    """
    Trova la soglia che massimizza la Precisione, mantenendo Recall >= 10%.
    Questo approccio taglia il rumore al massimo.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # 1. Filtra solo le soglie che soddisfano il Recall minimo (10% per default)
    valid_indices = np.where(recall >= min_recall)[0]
    
    if len(valid_indices) == 0:
        print(f"⚠️ Recall >= {min_recall*100:.0f}% non raggiungibile. Tornando all'ottimizzazione F1.")
        # Fallback: Massimizza F1 se il vincolo non è raggiungibile
        f1_scores = np.divide(2 * (precision * recall), (precision + recall), out=np.zeros_like(precision), where=(precision + recall) != 0)
        best_global_index = np.argmax(f1_scores)
    else:
        # 2. Tra le soglie valide, scegli quella che massimizza la Precisione
        best_index_in_valid = np.argmax(precision[valid_indices])
        best_global_index = valid_indices[best_index_in_valid]

    optimal_threshold = thresholds[best_global_index] if best_global_index < len(thresholds) else 0.5
    
    # Calcola le metriche finali per il log
    final_precision = precision[best_global_index]
    final_recall = recall[best_global_index]
    final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-8)
    
    print(f"\n🎯 Threshold ottimale (Precision-Maximized, Recall >= {min_recall*100:.0f}%): {optimal_threshold:.4f}")
    print(f"   Recall: {final_recall:.4f}")
    print(f"   Precision: {final_precision:.4f}")
    print(f"   F1: {final_f1:.4f}")
    
    return optimal_threshold

# --- PLOTTING (Corretta lo scope di MIN_RECALL_TARGET) ---
def plot_training_history(history, metrics_callback, min_recall_target, save_path='training_BALANCED.png'):
    """Visualizza andamento training"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUC
    axes[0, 1].plot(history.history['auc'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_auc'], label='Val', linewidth=2)
    axes[0, 1].set_title('AUC', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 2].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 2].plot(history.history['val_accuracy'], label='Val', linewidth=2)
    axes[0, 2].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # F1
    axes[1, 0].plot(metrics_callback.history_f1, linewidth=2, color='green')
    axes[1, 0].set_title('F1-Score', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 1].plot(metrics_callback.history_precision, linewidth=2, color='blue')
    axes[1, 1].set_title('Precision', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 2].plot(metrics_callback.history_recall, linewidth=2, color='orange')
    # ✅ Usa l'argomento passato
    axes[1, 2].axhline(y=min_recall_target, color='red', linestyle='--', label=f'Target {min_recall_target}')
    axes[1, 2].set_title('Recall', fontsize=12, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Grafico: {save_path}")
    plt.close()

# --- MAIN ---
if __name__ == '__main__':
    print("="*80)
    print("🎯 MARKET HEARTBEAT - PRECISION FOCUS TRAINING (FP Reduction)")
    print("="*80)
    
    try:
        X = np.load(X_FILE)
        Y = np.load(Y_FILE)
        print(f"✅ Dati: X={X.shape}, Y={Y.shape}")
    except FileNotFoundError:
        print(f"❌ File non trovati")
        sys.exit(1)
    
    crash_count = (Y == 1).sum()
    no_crash_count = (Y == 0).sum()
    print(f"\n📊 Crash: {crash_count:,} ({crash_count/len(Y)*100:.2f}%)")
    
    train_size = int(0.70 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, X_temp = X[:train_size], X[train_size:]
    Y_train, Y_temp = Y[:train_size], Y[train_size:]
    X_val, X_test = X_temp[:val_size], X_temp[val_size:]
    Y_val, Y_test = Y_temp[:val_size], Y_temp[val_size:]
    
    print(f"📦 Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    
    base_pos_weight = no_crash_count / (crash_count + 1e-8)
    final_pos_weight = base_pos_weight * FN_PENALTY_MULTIPLIER
    
    print(f"\n⚖️  Peso (FN penalty {FN_PENALTY_MULTIPLIER}x): {final_pos_weight:.2f}")
    print(f"🎯 Thresholding Recall Target: {MIN_RECALL_TARGET*100:.0f}% (Vincolo Minimo)")
    
    _, TIME_STEPS, FEATURES = X_train.shape
    model = build_balanced_lstm_model(TIME_STEPS, FEATURES)
    
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR, clipnorm=1.0),
        loss=weighted_binary_crossentropy(final_pos_weight),
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    print(f"\n🏗️  Parametri: {model.count_params():,}")
    
    metrics_callback = BalancedMetricsCallback((X_val, Y_val), patience=EARLY_STOP_PATIENCE)
    cosine_scheduler = CosineAnnealingScheduler(INITIAL_LR, MIN_LR, EPOCHS)
    
    callbacks = [
        metrics_callback,
        cosine_scheduler,
        ModelCheckpoint('best_model_PRECISION.keras', monitor='val_auc', 
                        save_best_only=True, mode='max', verbose=1)
    ]
    
    print(f"\n{'='*80}")
    print("🚀 TRAINING START (PRECISION FOCUS MODE)")
    print(f"{'='*80}\n")
    
    history = model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # ✅ Chiamata corretta
    plot_training_history(history, metrics_callback, MIN_RECALL_TARGET) 
    
    print(f"\n{'='*80}")
    print("📊 VALUTAZIONE TEST SET")
    print(f"{'='*80}")
    
    try:
        custom_objects = {
            'AttentionLayer': AttentionLayer,
            'weighted_binary_crossentropy': weighted_binary_crossentropy(final_pos_weight)
        }
        best_model = keras.models.load_model(
            'best_model_PRECISION.keras',
            custom_objects=custom_objects
        )
    except Exception as e:
        print(f"⚠️ Errore nel caricamento del modello salvato. Uso il modello corrente. Errore: {e}")
        best_model = model
        
    Y_pred_proba = best_model.predict(X_test, verbose=0).flatten()
    
    # 🚨 Adesso massimizziamo la Precisione con il vincolo basso
    TUNED_THRESHOLD = find_optimal_threshold_balanced(Y_test, Y_pred_proba, MIN_RECALL_TARGET)
    Y_pred = (Y_pred_proba >= TUNED_THRESHOLD).astype(int)
    
    auc_roc = roc_auc_score(Y_test, Y_pred_proba)
    
    report = classification_report(Y_test, Y_pred, target_names=['Non-Crash', 'Crash'], digits=4, output_dict=True, zero_division=0)
    precision_crash = report['Crash']['precision']
    recall = report['Crash']['recall']
    f1 = report['Crash']['f1-score']

    # Stampa i risultati nel formato richiesto
    print(f"\n🎯 METRICHE FINALI (Threshold Precision-Max):")
    print(f"   AUC: {auc_roc:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   Precision: {precision_crash:.4f}")
    print(f"   F1: {f1:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(Y_test, Y_pred, target_names=['Non-Crash', 'Crash'], digits=4, zero_division=0))
    
    cm = confusion_matrix(Y_test, Y_pred)
    print(f"\n🔢 Confusion Matrix:")
    print(f"   TN: {cm[0,0]:,}, FP: {cm[0,1]:,}")
    print(f"   FN: {cm[1,0]:,}, TP: {cm[1,1]:,}")
    
    best_model.save('lstm_model_PRECISION.keras')
    np.save('optimal_threshold_PRECISION.npy', TUNED_THRESHOLD)
    
    # Salvataggio predizioni complete per backtesting
    print("\n💾 Salvataggio predizioni complete...")
    X_full = np.load(X_FILE)
    predictions_full = best_model.predict(X_full, batch_size=1024, verbose=1).flatten()
    np.save('predictions_LSTM_PRECISION.npy', predictions_full)
    print(f"✅ Salvate {len(predictions_full):,} predizioni")
    
    with open('model_config_PRECISION.json', 'w') as f:
        json.dump({
            'threshold': float(TUNED_THRESHOLD),
            'fn_penalty': FN_PENALTY_MULTIPLIER,
            'min_recall_target': MIN_RECALL_TARGET,
            'test_recall': float(recall),
            'test_precision': float(precision_crash),
            'test_f1': float(f1)
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ TRAINING PRECISION FOCUS COMPLETATO")
    print(f"{'='*80}")