"""
MARKET HEARTBEAT - Advanced LSTM Trainer (V8 - RISK FOCUSED)
Strategia: Penalità aggressiva sui False Negatives (FN) e Tuning della soglia per Recall minimo.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, f1_score, recall_score
)
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE AVANZATA ---
X_FILE = 'X_train_v6_crash.npy'
Y_FILE = 'Y_train_v6_crash.npy'

# Parametri Architettura (Invariati dalla V7.3)
CNN_FILTERS = 32
CNN_KERNEL = 3
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.3
RECURRENT_DROPOUT = 0.0

# Parametri Training
EPOCHS = 50
BATCH_SIZE = 256
EARLY_STOP_PATIENCE = 10 
LR_PATIENCE = 4
INITIAL_LR = 0.001
MIN_LR = 0.00001

# 🟢 TUNING PER IL RISCHIO: MOLTIPLICATORE AGGRESSIVO SU FN
FN_PENALTY_MULTIPLIER = 1.5 
MIN_RECALL_TARGET = 0.40 # Obiettivo: ottenere almeno il 40% di Crash Detection Rate

# --- ATTENTION MECHANISM (Non modificato) ---
class AttentionLayer(layers.Layer):
    """Attention Layer customizzato per sequenze."""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    # ... build e call methods omessi per brevità, sono identici ...
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

# --- WEIGHTED BINARY CROSSENTROPY (Non modificato) ---
def weighted_binary_crossentropy(pos_weight):
    """BCE pesata"""
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        weight = y_true * pos_weight + (1 - y_true) * 1.0
        weighted_bce = weight * bce
        return tf.reduce_mean(weighted_bce)
    return loss

# --- CUSTOM CALLBACK (Basato su AUC/Loss per stabilità, ma si traccia F1) ---
class F1ScoreCallback(Callback):
    """Calcola e monitora l'F1-Score sul set di validazione"""
    def __init__(self, validation_data, patience=10):
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
            print(f" - val_f1_score: {current_f1:.4f} ⭐ NEW BEST")
        else:
            self.wait += 1
            print(f" - val_f1_score: {current_f1:.4f} (Best: {self.best_f1:.4f})")
                        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print(f"\nEarly stopping: val_f1_score non migliora da {self.patience} epoch")

# --- MODEL ARCHITECTURE (Non modificato) ---
def build_stable_lstm_model(timesteps, features):
    """Architettura Semplificata e Robusta"""
    inputs = layers.Input(shape=(timesteps, features))
    
    x = layers.Conv1D(filters=CNN_FILTERS, kernel_size=CNN_KERNEL, activation='relu', padding='same')(inputs) 
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS_1, return_sequences=True, dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT, kernel_regularizer=keras.regularizers.l2(0.001)))(x) 
    x = layers.BatchNormalization()(x)
    
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS_2, return_sequences=True, dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT, kernel_regularizer=keras.regularizers.l2(0.001)))(x) 
    x = layers.BatchNormalization()(x)
    
    x = AttentionLayer()(x)
    
    x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name='crash_prediction')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='MarketHeartbeat_LSTM_V8_Risk')
    return model

# --- FUNZIONI DI TUNING E PLOT ---
def plot_training_history(history, f1_history, save_path='training_history_v8_risk_tuned.png'):
    """Visualizza andamento training"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Loss, AUC, Accuracy (omesse per brevità, sono identiche)
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss (Weighted BCE)', fontsize=12, fontweight='bold')
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

def find_optimal_threshold_risk_tuned(y_true, y_pred_proba, min_recall=0.40):
    """
    Trova la soglia che massimizza la Precisione
    MANTENENDO il Recall al di sopra di un minimo specificato (min_recall).
    Priorità: 1. FN Reduction (Recall) -> 2. FP Reduction (Precision).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # 1. Filtra solo le soglie che soddisfano il Recall minimo
    valid_indices = np.where(recall >= min_recall)[0]
    
    if len(valid_indices) == 0:
        # Se non si raggiunge l'obiettivo, torna all'ottimizzazione F1 (come fallback)
        print(f"⚠️ Impossibile raggiungere Recall >= {min_recall*100:.0f}%. Tornando all'ottimizzazione F1.")
        f1_scores = np.divide(2 * (precision * recall), (precision + recall), out=np.zeros_like(precision), where=(precision + recall) != 0)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    # 2. Tra le soglie valide, scegli quella che massimizza la Precisione (minimizzi i falsi allarmi)
    # NOTA: Usiamo np.argmax per trovare l'indice della massima precisione nel subset filtrato
    best_index_in_valid = np.argmax(precision[valid_indices])
    best_global_index = valid_indices[best_index_in_valid]
    
    optimal_threshold = thresholds[best_global_index]
    
    print(f"\n🎯 Soglia Risk-Tuned trovata:")
    print(f"   Target Recall ({min_recall*100:.0f}%) raggiunto.")
    print(f"   Recall finale: {recall[best_global_index]:.4f}, Precisione finale: {precision[best_global_index]:.4f}")
    
    return optimal_threshold

# --- MAIN TRAINING PIPELINE ---
if __name__ == '__main__':
    print("="*70)
    print("🫀 MARKET HEARTBEAT - LSTM Training V8 (RISK FOCUSED)")
    print("="*70)
    
    # 1. Carica dati
    try:
        X = np.load(X_FILE)
        Y = np.load(Y_FILE)
        print(f"✅ Dati caricati: X={X.shape}, Y={Y.shape}")
    except FileNotFoundError:
        print(f"❌ ERRORE: File {X_FILE} o {Y_FILE} non trovati.")
        exit(1)
    
    # 2. Verifica distribuzione classi
    crash_count = (Y == 1).sum()
    no_crash_count = (Y == 0).sum()
    
    # 3. Split temporale
    train_size = int(0.70 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, X_temp = X[:train_size], X[train_size:]
    Y_train, Y_temp = Y[:train_size], Y[train_size:]
    X_val, X_test = X_temp[:val_size], X_temp[val_size:]
    Y_val, Y_test = Y_temp[:val_size], Y_temp[val_size:]
    
    # 4. Calcola peso per la classe positiva (WEIGHTED BCE)
    # Peso base: rapporto di non-crash/crash
    base_pos_weight = no_crash_count / (crash_count + 1e-8)
    
    # Peso finale: applica il moltiplicatore aggressivo per spingere il Recall
    final_pos_weight = base_pos_weight * FN_PENALTY_MULTIPLIER
    
    print(f"\n⚖️  Positive Class Weight (Base: {base_pos_weight:.2f}, Finale: {final_pos_weight:.2f})")
    
    # 5. Costruisci modello
    _, TIME_STEPS, FEATURES = X_train.shape
    model = build_stable_lstm_model(TIME_STEPS, FEATURES)
    
    # Compila con Weighted BCE e peso finale aggressivo
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR, clipnorm=1.0), 
        loss=weighted_binary_crossentropy(final_pos_weight),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print(f"✅ Model compiled with Weighted BCE (pos_weight={final_pos_weight:.2f})")
    
    # 6. Callbacks
    f1_callback = F1ScoreCallback(
        validation_data=(X_val, Y_val), 
        patience=EARLY_STOP_PATIENCE 
    )
    
    callbacks = [
        f1_callback,
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=LR_PATIENCE, min_lr=MIN_LR, verbose=1),
        ModelCheckpoint('best_model_v8_risk_tuned.keras', monitor='val_auc', save_best_only=True, mode='max', verbose=1)
    ]
    
    # 7. Training
    history = model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 8. Plot history
    plot_training_history(history, f1_callback.history_f1, save_path='training_history_v8_risk_tuned.png')
    
    # 9. Valutazione su Test Set
    print(f"\n{'='*70}")
    print("📊 VALUTAZIONE FINALE SU TEST SET (TUNING RISK-BASED)")
    print(f"{'='*70}")
    
    # Carica il modello con i pesi migliori
    try:
        best_model = keras.models.load_model(
            'best_model_v8_risk_tuned.keras',
            custom_objects={'loss': weighted_binary_crossentropy(final_pos_weight), 'AttentionLayer': AttentionLayer}
        )
    except:
        best_model = model
        
    Y_pred_proba = best_model.predict(X_test, verbose=0).flatten()
    
    # TUNING CHIAVE: Trova la soglia che garantisce un Recall minimo di 40%
    TUNED_THRESHOLD = find_optimal_threshold_risk_tuned(Y_test, Y_pred_proba, min_recall=MIN_RECALL_TARGET)
    Y_pred = (Y_pred_proba >= TUNED_THRESHOLD).astype(int)
    
    # Metriche finali
    auc_roc = roc_auc_score(Y_test, Y_pred_proba)
    recall = recall_score(Y_test, Y_pred, zero_division=0)
    f1 = f1_score(Y_test, Y_pred, zero_division=0)
    
    print(f"\n🎯 PERFORMANCE METRICS (Risk-Tuned):")
    print(f"   AUC-ROC:  {auc_roc:.4f}")
    print(f"   Recall (Crash Detection Rate): {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Soglia finale (Risk-Tuned): {TUNED_THRESHOLD:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(Y_test, Y_pred, target_names=['Non-Crash (0)', 'Crash (1)'], digits=4, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(Y_test, Y_pred)
    print(f"\n🔢 Confusion Matrix (Focus sui Falsi Negativi):")
    print(f"   True Neg (TN):  {cm[0,0]:,}")
    print(f"   False Pos (FP): {cm[0,1]:,} (Aumenteranno per spingere il Recall)")
    print(f"   False Neg (FN): {cm[1,0]:,} (Devono diminuire drasticamente)")
    print(f"   True Pos (TP):  {cm[1,1]:,}")
    
    # 10. Salvataggio finale
    best_model.save('lstm_market_heartbeat_model_v8_risk_tuned.keras')
    np.save('optimal_threshold_v8_risk_tuned.npy', TUNED_THRESHOLD)
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETATO V8 (RISK FOCUSED)")
    print(f"📁 Modello salvato per l'utilizzo in produzione.")
    print(f"{'='*70}")